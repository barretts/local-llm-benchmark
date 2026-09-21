import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from localbench.config import load, digest
from localbench.state import State, gpu_lock
from localbench.paths import safe_source
from localbench.ownership import check_port, validate_process, validate_container
from localbench.prompts import pack, validate_context


class CoreTests(unittest.TestCase):
    def config(self, root):
        c = load()
        for name in ("project","state","logs","artifacts"):
            c["paths"][name] = str(Path(root)/name)
        return c

    def key(self, **changes):
        result = dict(engine="binarySHA",model="weightSHA",effective_settings={"slots":1},profile="default",fixture_prompt_hash="fixtureSHA",tokenizer="version",seed=42,mode="fresh",replicate=0)
        result.update(changes)
        return result

    def test_resume_deadline_and_finished_jobs_and_weight_reservations(self):
        with tempfile.TemporaryDirectory() as root:
            c = self.config(root)
            now = [1000.]
            s = State(c,lambda:now[0])
            with self.assertRaises(RuntimeError): s.start_execution("probe")
            s.control("harness_verified",True)
            s.start_execution("probe")
            deadline = s.run["deadline"]
            a,b = s.enqueue(self.key()),s.enqueue(self.key(replicate=1))
            s.finish(s.begin(a),"passed",{"test":True})
            s.begin(b) # simulated crash.
            s.reserve_weight("partial",100,"download","file")
            s.close()
            now[0] += 600
            s = State(c,lambda:now[0]); s.recover(); s.start_execution("resume")
            self.assertEqual(s.run["deadline"],deadline)
            self.assertEqual(s.db.execute("SELECT status FROM jobs WHERE id=?",(a,)).fetchone()[0],"passed")
            self.assertEqual(s.db.execute("SELECT status FROM jobs WHERE id=?",(b,)).fetchone()[0],"pending")
            s.reserve_weight("partial",100,"download","file")
            self.assertEqual(s.snapshot()["new_weight_bytes_reserved"],100)
            s.close()

    def test_stop_and_budget_and_cumulative_cap(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.config(root); now=[0.]; s=State(c,lambda:now[0]); s.control("harness_verified",True);s.start_execution("probe")
            s.stop()
            with self.assertRaisesRegex(RuntimeError,"stop_after_current"):s.check_budget()
            s.recover(); now[0]=s.run["deadline"]-7000
            with self.assertRaisesRegex(RuntimeError,"budget_exhausted"):s.check_budget()
            s.reserve_weight("a",c["limits"]["new_model_weight_bytes"],"download","a")
            with self.assertRaisesRegex(RuntimeError,"weight_budget"):s.reserve_weight("b",1,"download","b")
            s.close()

    def test_transient_two_retries_not_model_oom(self):
        with tempfile.TemporaryDirectory() as root:
            s=State(self.config(root)); job=s.enqueue(self.key())
            for _ in range(3):s.finish(s.begin(job),"invalid",reason="network_timeout",transient=True)
            self.assertEqual(s.db.execute("SELECT status FROM jobs WHERE id=?",(job,)).fetchone()[0],"invalid")
            with self.assertRaises(RuntimeError):s.begin(job)
            job=s.enqueue(self.key(replicate=2));s.finish(s.begin(job),"failed",reason="model_oom")
            self.assertEqual(s.db.execute("SELECT status FROM jobs WHERE id=?",(job,)).fetchone()[0],"failed")
            s.close()

    def test_paths_and_grader_protection(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(safe_source(root,"src/cache.py",{"src/cache.py"}),Path(root).resolve()/"src/cache.py")
            for p in ("../x","src/../grader.py","C:/x.py","//host/x.py","src\\x.py","tests/test.py","grader/x.py","src/NUL.py","src/a.py "):
                with self.subTest(path=p),self.assertRaises(ValueError):safe_source(root,p)
            with self.assertRaises(ValueError):safe_source(root,"src/errors.ts",{"src/request.ts"})

    def test_symlink_escape_if_platform_supports_it(self):
        with tempfile.TemporaryDirectory() as root,tempfile.TemporaryDirectory() as outside:
            src=Path(root)/"src";src.mkdir()
            try:(src/"link").symlink_to(outside,target_is_directory=True)
            except OSError:self.skipTest("Windows symlink privilege unavailable; junction handled by production path check")
            with self.assertRaises(ValueError):safe_source(root,"src/link/x.py")

    def test_ownership_stale_pid_containers_ports(self):
        saved={"owner":"localbench","pid":7,"created":10,"command_hash":"sha"}
        self.assertTrue(validate_process(saved,saved))
        self.assertFalse(validate_process(saved,{**saved,"created":11}))
        self.assertFalse(validate_container({},"r","j"))
        self.assertTrue(validate_container({"localbench.owner":"localbench","localbench.run":"r","localbench.job":"j"},"r","j"))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1",0));listener.listen()
            with self.assertRaises(RuntimeError):check_port(listener.getsockname()[1])

    def test_lock_excludes_second_controller(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.config(root)
            with gpu_lock(c):
                with self.assertRaises(RuntimeError):
                    with gpu_lock(c):pass

    def test_exact_template_packing_and_invalid_context(self):
        observed=[]
        def tokenize(messages):
            count=10+sum(len(m["content"]) for m in messages) # deliberately fake exact tokenizer, includes fixed template/tool cost.
            observed.append(count);return {"count":count,"provenance":"mock-template+tools"}
        p=pack(tokenize,lambda filler:[{"role":"system","content":"fixed-tools"},{"role":"user","content":filler+"end"}],42,4096)
        self.assertEqual(p["expected_tokens"],4096);self.assertGreater(len(observed),3)
        m=load()["measurement"]
        self.assertEqual(validate_context(61440,{"prompt_tokens":61440},m,65536),[])
        self.assertIn("actual_prompt_count_mismatch",validate_context(61440,{"prompt_tokens":61441},m,65536))
        self.assertIn("truncated_context",validate_context(61440,{"prompt_tokens":61440},m,65536,True))

    def test_600_second_fake_sweep_stop_resume_resources_report(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.config(root);now=[1000.];s=State(c,lambda:now[0]);s.control("harness_verified",True);s.start_execution("mock-clock-test")
            keys=[s.enqueue(self.key(replicate=i)) for i in range(20)]
            resource_log=Path(root)/"resources.jsonl"
            for i,job in enumerate(keys):
                attempt=s.begin(job);now[0]+=30
                with resource_log.open("a") as f:f.write(json.dumps({"time":now[0],"mock_gpu_mib":1000+i})+"\n")
                if i==5:
                    s.close();s=State(c,lambda:now[0]);s.recover();attempt=s.begin(job)
                s.finish(attempt,"passed",{"mock_action_seconds":i/10})
                if i==10:s.stop();s.close();s=State(c,lambda:now[0]);s.recover()
            summary=s.snapshot();self.assertEqual(summary["jobs"],{"passed":20});self.assertEqual(now[0]-1000,600)
            self.assertEqual(len(resource_log.read_text().splitlines()),20)
            self.assertEqual(len(s.db.execute("SELECT * FROM attempts WHERE reason='controller_interrupted'").fetchall()),1)
            self.assertEqual(digest(self.key()),digest(dict(reversed(list(self.key().items())))))
            self.assertNotEqual(digest(self.key()),digest(self.key(engine="other-version")))
            s.close()
