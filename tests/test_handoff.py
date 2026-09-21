"""Mock-only endpoint handoff tests; no model, engine, download or GPU activity."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import hashlib
import json
import tempfile
import unittest

from localbench.config import atomic_json, digest, file_hash
from localbench.handoff import demo_and_prepare, load_plan, launch_plan, _pin, _save_immutable
from localbench.state import State


class FakeProcess:
    def __init__(self, pid):
        self.pid, self.stopped = pid, False

    def poll(self):
        return 0 if self.stopped else None


class FakeAdapter:
    def __init__(self, fixture, runtime, model, settings, number):
        self.fixture = fixture
        self.engine = runtime["engine"]
        self.model = copy.deepcopy(model)
        self.requested = copy.deepcopy(settings)
        self.process = FakeProcess(1000 + number)
        self.http = SimpleNamespace(port=38201)
        self.effective = {"effective_context": 65536, "effective_slots": 1, "context_shift_disabled": True,
                          "launch_argv": [str(fixture.binary), "--model", model["path"], "--host", "127.0.0.1", "--port", "38201"]}
        self.saved = {"owner": "localbench", "run_id": fixture.state.run["id"], "pid": self.process.pid,
                      "created": 1234 + number, "argv": self.effective["launch_argv"], "endpoint": "http://127.0.0.1:38201"}
        self.runtime = runtime
        if runtime["kind"] == "ollama":
            self.local_name = "localbench-mock-ollama"

    def metadata(self):
        return {"binary": str(self.fixture.binary), "binary_sha256": file_hash(self.fixture.binary),
                "adjacent_dlls": {"mock.dll": file_hash(self.fixture.dll)}, "version_output": "mock pinned runtime 1",
                "model_sha256": self.model["sha256"], "effective_settings": self.effective}

    def unload_owned(self):
        self.fixture.events.append(("stop", self.process.pid))
        self.process.stopped = True


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.binary = self.root / "runtime" / "llama-server.exe"
        self.binary.parent.mkdir()
        self.binary.write_bytes(b"nonexecutable fixture runtime")
        self.dll = self.binary.parent / "mock.dll"
        self.dll.write_bytes(b"fixture DLL")
        self.weight = self.root / "model.gguf"
        self.weight.write_bytes(b"fixture model; not real weights")
        self.controller = self.root / "controller.py"
        self.controller.write_text("# fixture controller", encoding="utf-8")
        self.config = {"_spec_hash": "temporary-handoff-spec", "paths": {"state": str(self.root / "state"),
                       "artifacts": str(self.root / "artifacts"), "logs": str(self.root / "logs"),
                       "agent_workspaces": str(self.root / "workspaces")},
                       "installed_tools": {"docker": "mock-existing-docker", "bundled_llama_server": str(self.binary)},
                       "private_ports": {"native": 38201, "ollama": 38207}, "default_sampling": {"temperature": 0.6, "top_k": 20},
                       "limits": {"benchmark_elapsed_seconds": 604800, "new_model_weight_bytes": 300000000000,
                                  "retry_transient_attempts": 2, "server_start_timeout_seconds": 300, "per_task_timeout_seconds": 900}}
        for name in ("artifacts", "logs", "workspaces"):
            (self.root / name).mkdir()
        atomic_json(self.root / "artifacts" / "grader-images.json", {"python": {"digest": "python@sha256:" + "e" * 64}})
        self.state = State(self.config, clock=lambda: 1000)
        self.state.control("harness_verified", True)
        self.state.control("fixture_version", "mock-fixture-version")
        self.model = {"id": "mock-model", "path": str(self.weight), "sha256": file_hash(self.weight), "bytes": self.weight.stat().st_size}
        self.state.entity("model", self.model["id"], self.model)
        self.identifier = "c" * 64
        self.candidate = {"configuration_id": self.identifier, "engine": "bundled-llama", "model": "mock-model",
                          "settings": {"context": 65536, "slots": 1, "reasoning": "default"},
                          "runtime": {"engine": "bundled-llama", "kind": "native", "binary_path": str(self.binary)}}
        self.events, self.adapters, self.coding_calls = [], [], []
        self.changed_identity = False
        self.unsafe_isolation = False
        self.omit_wire = False
        self.grade_passes = True
        self.source_patch = patch("localbench.handoff._source_pins", side_effect=lambda state=None: [_pin(self.controller, "controller", state=state)])
        self.factory_patch = patch("localbench.handoff.make_adapter", side_effect=self.factory)
        self.coding_patch = patch("localbench.handoff.coding_job", side_effect=self.coding)
        self.source_patch.start(); self.factory_mock = self.factory_patch.start(); self.coding_mock = self.coding_patch.start()

    def tearDown(self):
        self.coding_patch.stop(); self.factory_patch.stop(); self.source_patch.stop()
        self.state.close()
        self.tmp.cleanup()

    def factory(self, config, state, collector, runtime, model, settings):
        if self.adapters and self.adapters[-1].process.poll() is None:
            raise AssertionError("the prior GPU instance was not stopped")
        adapter = FakeAdapter(self, runtime, model, settings, len(self.adapters))
        self.adapters.append(adapter)
        self.events.append(("launch", adapter.process.pid))
        return adapter, "d" * 64 if self.changed_identity and len(self.adapters) > 1 else self.identifier

    def coding(self, config, state, adapter, collector, identifier, fixture, seed, **kwargs):
        self.coding_calls.append((identifier, fixture, seed, kwargs))
        self.events.append(("coding", adapter.process.pid))
        workspace = self.root / "workspaces" / "owned-demo"
        workspace.mkdir(exist_ok=True)
        prefix = self.root / "logs" / "quality-mock-a1"
        diff = Path(str(prefix) + "-source.diff")
        diff.write_text("--- src/cache.py\n+++ src/cache.py\n+ corrected cache behavior\n", encoding="utf-8")
        if not self.omit_wire:
            turn = str(prefix) + "-turn0"
            atomic_json(turn + "-request.json", {"model": "local-model", "messages": [{"role": "user", "content": "public fixture task"}], "tools": [{"type": "function", "function": {"name": "read_file"}}]})
            Path(turn + "-raw.bin").write_bytes(b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
            Path(turn + "-events.jsonl").write_text(json.dumps({"event": {"type": "tool_call", "name": "read_file", "arguments": {"path": "src/cache.py"}}}) + "\n", encoding="utf-8")
            atomic_json(turn + "-result.json", {"valid_stream": True})
            atomic_json(str(prefix) + "-checkpoint.json", {"messages": [{"role": "user", "content": "public task"}, {"role": "assistant", "tool_calls": []}, {"role": "tool", "content": "public source contents"}]})
        argv = [config["installed_tools"]["docker"], "run", "--label", "localbench.owner=localbench", "--network", "none",
                "--read-only", "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--memory", "2g", "--cpus", "2", "--pids-limit", "128", "python@sha256:" + "e" * 64, "python", "/driver/grade_python.py"]
        if self.unsafe_isolation:
            argv[argv.index("none")] = "host"
        grade = {"passed": self.grade_passes, "complete": True, "launch_argv": argv}
        return {"kind": "quality", "fixture": fixture, "seed": seed, "configuration_id": identifier,
                "passed": self.grade_passes, "valid": True, "reason": "passed" if self.grade_passes else "test_failure",
                "workspace": str(workspace), "source_diff": str(diff), "public_grading": grade, "hidden_grading": grade}

    def prepared(self):
        result = demo_and_prepare(self.config, self.state, object(), self.candidate)
        return result, Path(result["launch_plan"])

    def test_stops_then_relaunches_exact_saved_plan_and_real_demo_contract(self):
        result, path = self.prepared()
        self.assertTrue(result["passed"])
        self.assertTrue(result["isolated"])
        self.assertTrue(result["launch_tested_from_stopped_owned_instance"])
        self.assertEqual([event[0] for event in self.events], ["launch", "stop", "launch", "coding", "stop"])
        self.assertEqual(self.coding_calls, [(self.identifier, "py02", 42, {"replicate": 1, "mode": "isolated-demo"})])
        self.assertTrue(all(adapter.process.poll() == 0 for adapter in self.adapters))
        envelope = load_plan(path)
        self.assertEqual(envelope["plan"]["runtime"], self.candidate["runtime"])
        self.assertEqual(envelope["plan"]["requested_settings"], self.candidate["settings"])
        self.assertTrue(result["transcript"]["native_tool_call_observed"])
        self.assertTrue(any(entry["path"].endswith("-raw.bin") for entry in result["transcript"]["files"]))
        self.assertEqual(json.loads(self.state.db.execute("SELECT key_json FROM jobs").fetchone()[0])["kind"], "demo")
        self.assertIsNone(self.state.run["started"])
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM weights").fetchone()[0], 0)

    def test_verified_resume_uses_checkpoint_without_another_launch(self):
        first, path = self.prepared()
        content = path.read_bytes()
        second = demo_and_prepare(self.config, self.state, object(), self.candidate)
        self.assertEqual(first, second)
        self.assertEqual(len(self.adapters), 2)
        self.assertEqual(path.read_bytes(), content)

    def test_launch_plan_returns_owned_live_adapter_for_cli_lifecycle(self):
        result, path = self.prepared()
        adapter, identifier = launch_plan(self.config, self.state, object(), path)
        self.assertEqual(identifier, self.identifier)
        self.assertIsNone(adapter.process.poll())
        self.assertEqual(adapter.handoff_launch_evidence["interface"]["chat_path"], "/v1/chat/completions")
        adapter.unload_owned()

    def test_plan_tampering_rejected_before_launch(self):
        result, path = self.prepared()
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["plan"]["requested_settings"]["context"] = 1024
        atomic_json(path, envelope)
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            launch_plan(self.config, self.state, object(), path)
        self.assertEqual(len(self.adapters), 2)

    def test_changed_model_or_runtime_file_rejected_before_launch(self):
        result, path = self.prepared()
        for target in (self.weight, self.binary, self.dll):
            original = target.read_bytes()
            target.write_bytes(original + b"changed")
            with self.assertRaisesRegex(RuntimeError, "pinned_file_changed"):
                launch_plan(self.config, self.state, object(), path)
            target.write_bytes(original)
        self.assertEqual(len(self.adapters), 2)

    def test_changed_controller_source_rejected_before_launch(self):
        result, path = self.prepared()
        self.controller.write_text("# changed protocol", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "pinned_file_changed:controller"):
            launch_plan(self.config, self.state, object(), path)
        self.assertEqual(len(self.adapters), 2)

    def test_changed_profile_or_private_port_rejected_before_launch(self):
        result, path = self.prepared()
        self.config["default_sampling"]["temperature"] = 0.7
        with self.assertRaisesRegex(RuntimeError, "launch_config_changed"):
            launch_plan(self.config, self.state, object(), path)
        self.assertEqual(len(self.adapters), 2)

    def test_relaunched_identity_change_stops_instance_and_records_invalid_demo(self):
        self.changed_identity = True
        result, path = self.prepared()
        self.assertFalse(result["passed"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "handoff_effective_configuration_identity_changed")
        self.assertEqual([event[0] for event in self.events], ["launch", "stop", "launch", "stop"])
        self.assertEqual(len(self.coding_calls), 0)

    def test_host_test_execution_cannot_count_as_isolated_demo(self):
        self.unsafe_isolation = True
        result, path = self.prepared()
        self.assertFalse(result["passed"])
        self.assertFalse(result["valid"])
        self.assertFalse(result["isolated"])
        self.assertEqual(result["reason"], "isolated_demo_grading_evidence_missing")

    def test_missing_actual_wire_logs_cannot_count_as_successful_demo(self):
        self.omit_wire = True
        result, path = self.prepared()
        self.assertFalse(result["passed"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "handoff_demo_actual_wire_transcript_missing")
        self.assertTrue(result["demo_instance"]["stopped"])

    def test_real_quality_failure_retains_failure_without_declaring_demo_success(self):
        self.grade_passes = False
        result, path = self.prepared()
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "test_failure")

    def test_ollama_records_and_demonstrates_its_actual_native_interface(self):
        self.candidate["engine"] = "ollama"
        self.candidate["runtime"] = {"engine": "ollama", "kind": "ollama"}
        result, path = self.prepared()
        self.assertTrue(result["passed"])
        self.assertEqual(result["interface"]["chat_path"], "/api/chat")
        self.assertEqual(result["interface"]["wire_protocol"], "Ollama native streaming NDJSON")
        self.assertEqual(load_plan(path)["plan"]["interface"]["model"], "localbench-mock-ollama")

    def test_immutable_plan_cannot_be_replaced(self):
        result, path = self.prepared()
        envelope = load_plan(path)
        _save_immutable(path, envelope)
        changed = copy.deepcopy(envelope)
        changed["plan"]["engine"] = "different"
        with self.assertRaisesRegex(RuntimeError, "immutable_plan_already_differs"):
            _save_immutable(path, changed)

    def test_url_credentials_rejected_even_with_recomputed_content_hash(self):
        result, path = self.prepared()
        envelope = load_plan(path)
        envelope["plan"]["runtime"]["public_url"] = "https://public.example/artifact?secret=must-not-be-saved"
        envelope["plan_id"] = digest(envelope["plan"])
        atomic_json(path, envelope)
        with self.assertRaisesRegex(ValueError, "credential_or_signed_url"):
            load_plan(path)

    def test_recomputed_unregistered_plan_cannot_launch(self):
        result, path = self.prepared()
        envelope = load_plan(path)
        envelope["plan"]["requested_settings"]["reasoning"] = "disabled"
        envelope["plan_id"] = digest(envelope["plan"])
        forged = path.with_name("forged.json")
        atomic_json(forged, envelope)
        with self.assertRaisesRegex(RuntimeError, "not_registered_and_stopped"):
            launch_plan(self.config, self.state, object(), forged)
        self.assertEqual(len(self.adapters), 2)

    def test_added_runtime_dll_is_rejected_before_an_engine_can_load_it(self):
        result, path = self.prepared()
        (self.binary.parent / "unreviewed.dll").write_bytes(b"unreviewed fixture DLL")
        with self.assertRaisesRegex(RuntimeError, "dependency_file_set_changed"):
            launch_plan(self.config, self.state, object(), path)
        self.assertEqual(len(self.adapters), 2)

    def test_missing_successful_transcript_invalidates_current_demo_gate_on_resume(self):
        result, path = self.prepared()
        raw = next(pin for pin in result["transcript"]["files"] if pin["path"].endswith("-raw.bin"))
        Path(raw["path"]).unlink()
        resumed = demo_and_prepare(self.config, self.state, object(), self.candidate)
        self.assertFalse(resumed["passed"])
        self.assertFalse(resumed["valid"])
        self.assertEqual(len(self.adapters), 2)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM jobs WHERE status='passed'").fetchone()[0], 0)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM attempts WHERE status='passed'").fetchone()[0], 1)

    def test_missing_isolation_images_stops_original_and_records_invalid_setup(self):
        (self.root / "artifacts" / "grader-images.json").unlink()
        result = demo_and_prepare(self.config, self.state, object(), self.candidate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "handoff_pinned_isolation_images_missing")
        self.assertEqual([event[0] for event in self.events], ["launch", "stop"])
        self.assertEqual(len(self.coding_calls), 0)

    def test_missing_model_registry_does_not_abort_other_finalists(self):
        self.state.db.execute("DELETE FROM entities WHERE kind='model'")
        self.state.db.commit()
        result = demo_and_prepare(self.config, self.state, object(), self.candidate)
        self.assertFalse(result["passed"])
        self.assertIn("model_registry_entry_missing", result["reason"])
        self.assertEqual(len(self.adapters), 0)

    def test_hf_subset_git_blob_and_supporting_tokenizer_files_are_pinned(self):
        root = self.root / "hf-checkpoint"
        root.mkdir()
        weights = root / "model.safetensors"
        weights.write_bytes(b"tiny mock tensor shard")
        configuration = root / "config.json"
        configuration.write_text('{"max_position_embeddings":65536}', encoding="utf-8")
        raw = configuration.read_bytes()
        blob = hashlib.sha1(("blob " + str(len(raw)) + "\0").encode() + raw).hexdigest()
        self.model = {"id": "mock-model", "path": str(root), "repo": "public/mock-model", "revision": "a" * 40,
                      "sha256": digest({"pinned_model_fixture": True}), "files": [
                          {"filename": "model.safetensors", "bytes": weights.stat().st_size, "sha256": file_hash(weights), "weight": True},
                          {"filename": "config.json", "bytes": len(raw), "git_blob_sha1": blob, "weight": False}]}
        self.state.entity("model", "mock-model", self.model)
        result, path = self.prepared()
        self.assertTrue(result["passed"])
        pins = load_plan(path)["plan"]["file_pins"]
        self.assertTrue(any(pin["path"] == str(configuration) and pin["sha256"] == file_hash(configuration) for pin in pins))
        (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "support_file_set_changed"):
            launch_plan(self.config, self.state, object(), path)
        self.assertEqual(len(self.adapters), 2)


if __name__ == "__main__":
    unittest.main()
