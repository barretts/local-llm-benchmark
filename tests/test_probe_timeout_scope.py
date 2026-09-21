"""Exercise the staged scheduler correction without applying shared source.

Only the scheduler's actual scope/measured_job functions are extracted from AST.
The real NativeAdapter->LocalHTTP measurement path uses fake sockets and a
monotonic fake clock. No network, GPU, generated code, state DB or server launch.
After maintenance applies the patch these tests use the live corrected source.
"""

import ast
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock

from localbench.adapters import base
from localbench.adapters.native import NativeAdapter
from localbench.metrics import Measurement
from localbench.schema import validate_tool
from localbench.tools import apply_unified_patch


ROOT=Path(__file__).resolve().parents[1]


def staged_scheduler():
    source=(ROOT/"localbench/scheduler.py").read_text(encoding="utf-8")
    if "def probe_request_timeout(" not in source:
        patch=(ROOT/"staged/probe-timeout-scheduler.patch").read_text(encoding="utf-8")
        source=apply_unified_patch(source,patch,"localbench/scheduler.py")
    tree=ast.parse(source)
    names={"CachedJob","result_status","probe_request_timeout","measured_job"}
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    namespace={"contextmanager":contextmanager,"time":time,"digest":lambda value:"fixture-digest"}
    exec(compile(ast.Module(body=selected,type_ignores=[]),"staged scheduler functions","exec"),namespace)
    return namespace


def config(root):
    return {"limits":{"per_16k_request_timeout_seconds":300,"per_64k_request_timeout_seconds":900,
                      "per_task_timeout_seconds":900},
            "paths":{"logs":str(root)},"private_ports":{"native":38201},
            "installed_tools":{"bundled_llama_server":str(root/"not-executed-server.exe"),
                               "ollama":str(root/"not-executed-ollama.exe")}}


class Clock:
    def __init__(self):self.value=0.0
    def monotonic(self):return self.value
    def perf_ns(self):return int(self.value*1e9)


class Socket:
    def __init__(self):self.timeout=None;self.observed=[]
    def gettimeout(self):return self.timeout
    def settimeout(self,value):self.timeout=value;self.observed.append(value)


class Connection:
    def __init__(self):self.sock=Socket();self.closed=False
    def close(self):self.closed=True


class Response:
    def __init__(self,clock,steps):self.clock=clock;self.steps=iter(steps);self.closed=False
    def read1(self,size):
        try:
            elapsed,chunk=next(self.steps)
        except StopIteration:return b""
        self.clock.value=elapsed
        return chunk
    def close(self):self.closed=True


def record(delta,finish=None):
    value={"choices":[{"index":0,"delta":delta,"finish_reason":finish}]}
    return ("data: "+json.dumps(value)+"\n\n").encode()


class TimeoutScopeTests(unittest.TestCase):
    def setUp(self):
        self.stage=staged_scheduler()
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.config=config(self.root)
        self.adapter=NativeAdapter(self.config,None,"bundled-llama",{"id":"mock","path":"unused.gguf"})

    def tearDown(self):self.temporary.cleanup()

    def test_known_small_and_full_targets_then_restore_previous(self):
        self.adapter.http.timeout=117
        for target,wanted in ((4096,300),(16384,300),(61440,900)):
            with self.stage["probe_request_timeout"](self.config,self.adapter,"capacity",target):
                self.assertEqual(self.adapter.http.timeout,wanted)
            self.assertEqual(self.adapter.http.timeout,117)

    def test_quality_remaining_total_deadline_and_unknown_target_unchanged(self):
        self.adapter.http.timeout=117
        for kind,target in (("quality",16384),("quality",61440),("capacity",None),
                            ("capacity",0),("capacity",-1),("capacity",True)):
            with self.stage["probe_request_timeout"](self.config,self.adapter,kind,target):
                self.assertEqual(self.adapter.http.timeout,117)
            self.assertEqual(self.adapter.http.timeout,117)
        self.assertEqual(self.config["limits"]["per_task_timeout_seconds"],900)

    def test_restore_after_error_and_control_stop(self):
        self.adapter.http.timeout=117
        for error in (ValueError("operation_failed"),RuntimeError("stop_after_current"),
                      RuntimeError("budget_exhausted"),KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__),self.assertRaises(type(error)):
                with self.stage["probe_request_timeout"](self.config,self.adapter,"timing",16384):
                    self.assertEqual(self.adapter.http.timeout,300)
                    raise error
            self.assertEqual(self.adapter.http.timeout,117)

    def _actual_stream(self,target,mode,final_time):
        clock=Clock()
        conn=Connection()
        first=record({"tool_calls":[{"index":0,"id":"call","function":{
            "name":"read_file","arguments":'{"path":"src/cache.py"}'}}]})
        steps=[(1,first),(100,record({"content":"streaming"})),
               (200,record({"content":"still streaming"})),
               (final_time,record({},"tool_calls")+b"data: [DONE]\n\n")]
        response=Response(clock,steps)
        def request(path,body,deadline=None):
            expected=300 if target<=16384 else 900
            self.assertEqual(deadline,expected)
            conn.sock.settimeout(deadline)
            return conn,response
        def timed_measurement(**kwargs):return Measurement(clock_ns=clock.perf_ns,**kwargs)
        previous=self.adapter.http.timeout
        with mock.patch.object(base.time,"monotonic",clock.monotonic), \
             mock.patch.object(base,"Measurement",timed_measurement), \
             mock.patch.object(self.adapter.http,"request",request):
            with self.stage["probe_request_timeout"](self.config,self.adapter,"timing",target):
                result=self.adapter.stream_chat({"stream":True},log_prefix=self.root/(mode+str(target)),
                                                validator=validate_tool,mode=mode)
        self.assertEqual(self.adapter.http.timeout,previous)
        self.assertTrue(conn.closed)
        self.assertTrue(response.closed)
        return result,conn

    def test_actual_native_stream_reaches_300_boundary_despite_continuous_events(self):
        for mode in ("fresh","cached"):
            result,conn=self._actual_stream(16384,mode,300)
            self.assertFalse(result["valid_stream"])
            self.assertIn("local_request_deadline_exceeded",result["error"])
            self.assertFalse(result["metrics"]["first_action_valid"])
            self.assertEqual(result["metrics"]["wall_seconds"],300)
            self.assertEqual(result["calls"],[])
            self.assertLessEqual(conn.sock.observed[-1],100)

    def test_actual_full_context_stream_allows_more_than_300_and_stops_at_900(self):
        for mode in ("fresh","cached"):
            result,_=self._actual_stream(61440,mode,350)
            self.assertTrue(result["valid_stream"])
            self.assertTrue(result["metrics"]["first_action_valid"])
            result,_=self._actual_stream(61440,mode,900)
            self.assertFalse(result["valid_stream"])
            self.assertFalse(result["metrics"]["first_action_valid"])
            self.assertEqual(result["metrics"]["wall_seconds"],900)

    def test_actual_small_stream_completes_before_limit(self):
        result,_=self._actual_stream(16384,"fresh",299)
        self.assertTrue(result["valid_stream"])
        self.assertTrue(result["metrics"]["first_action_valid"])
        self.assertEqual(result["metrics"]["wall_seconds"],299)

    def test_measured_job_scopes_operation_in_both_modes_and_restores(self):
        class DB:
            def execute(self,*args):return self
            def fetchone(self):return {"status":"pending"}
        class State:
            def __init__(self):self.db=DB();self.finished=[]
            def check_budget(self,*args):pass
            def enqueue(self,key):return "job"
            def begin(self,job):return 1
            def control(self,*args):pass
            def finish(self,*args,**kwargs):self.finished.append((args,kwargs))
        collector=types.SimpleNamespace(snapshot=lambda *args:{"foreign_workload_overlap":False})
        self.stage["key_base"]=lambda *args:{}
        self.stage["prior_result"]=lambda *args:None
        self.adapter.model={"id":"mock"}
        self.adapter.engine="bundled-llama"
        self.adapter.http.timeout=117
        for mode in ("fresh","cached"):
            for target,limit in ((16384,300),(61440,900)):
                for behavior in ("pass","error","stop"):
                    state=State()
                    def operation(callback):
                        self.assertEqual(self.adapter.http.timeout,limit)
                        callback({"prompt_hash":"packed"})
                        if behavior=="error":raise ValueError("operation_failure")
                        if behavior=="stop":raise RuntimeError("stop_after_current")
                        return {"passed":True,"valid":True}
                    args=(self.config,state,self.adapter,collector,"configuration","timing",42,mode,0,target,operation)
                    if behavior=="stop":
                        with self.assertRaisesRegex(RuntimeError,"stop_after_current"):
                            self.stage["measured_job"](*args)
                    else:
                        result=self.stage["measured_job"](*args)
                        self.assertEqual(result["passed"],behavior=="pass")
                        self.assertEqual(len(state.finished),1)
                    self.assertEqual(self.adapter.http.timeout,117)


if __name__=="__main__":unittest.main()
