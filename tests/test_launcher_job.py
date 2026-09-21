"""CPU-only Windows ownership proofs, confined to disposable test directories."""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from launcher import windows_job


# The child duplicates its grandchild's original process handle directly into
# this test runner. The test never discovers processes or terminates by PID.
_CHILD = r'''
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

api = ctypes.WinDLL("kernel32", use_last_error=True)
api.GetCurrentProcess.argtypes = []
api.GetCurrentProcess.restype = wintypes.HANDLE
api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
api.OpenProcess.restype = wintypes.HANDLE
api.DuplicateHandle.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
    wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
    wintypes.BOOL, wintypes.DWORD]
api.DuplicateHandle.restype = wintypes.BOOL
api.CloseHandle.argtypes = [wintypes.HANDLE]
api.CloseHandle.restype = wintypes.BOOL

grandchild = subprocess.Popen([getattr(sys, "_base_executable", sys.executable),
                              "-c", "import time; time.sleep(120)"],
                             creationflags=0x08000000)
runner = api.OpenProcess(0x0040, False, int(sys.argv[2]))
if not runner:
    raise ctypes.WinError(ctypes.get_last_error())
retained = wintypes.HANDLE()
actual_child = wintypes.HANDLE()
try:
    if not api.DuplicateHandle(api.GetCurrentProcess(), int(grandchild._handle),
            runner, ctypes.byref(retained), 0x00100000, False, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    if not api.DuplicateHandle(api.GetCurrentProcess(), api.GetCurrentProcess(),
            runner, ctypes.byref(actual_child), 0x00100000, False, 0):
        raise ctypes.WinError(ctypes.get_last_error())
finally:
    api.CloseHandle(runner)
print("owned-child-output", flush=True)
print("owned-child-error", file=sys.stderr, flush=True)
Path(sys.argv[1]).write_text(json.dumps({
    "handle": retained.value,
    "child_handle": actual_child.value,
    "pid": os.getpid(),
    "arguments": sys.argv[3:],
    "environment": os.environ.get("LAUNCHER_TEST_UNICODE"),
    "stdin": sys.stdin.read(),
}), encoding="utf-8")
time.sleep(120)
'''


@unittest.skipUnless(os.name == "nt", "Windows Job Objects require Windows")
class OwnedJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="launcher-job-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.GetCurrentProcess.argtypes = []
        self.api.GetCurrentProcess.restype = wintypes.HANDLE
        self.api.DuplicateHandle.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
            ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
            wintypes.BOOL, wintypes.DWORD,
        ]
        self.api.DuplicateHandle.restype = wintypes.BOOL
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.WaitForSingleObject.restype = wintypes.DWORD
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.api.GetHandleInformation.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.api.GetHandleInformation.restype = wintypes.BOOL

    def retain(self, original):
        retained = wintypes.HANDLE()
        current = self.api.GetCurrentProcess()
        self.assertTrue(self.api.DuplicateHandle(
            current, original, current, ctypes.byref(retained),
            0x00100000, False, 0,
        ))
        self.addCleanup(self.api.CloseHandle, retained.value)
        return retained.value

    def read_ready(self, path, job):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            self.assertTrue(job.alive(), "owned child exited before readiness")
            time.sleep(0.02)
        self.fail("owned child did not become ready")

    def assert_exited(self, handle):
        self.assertEqual(self.api.WaitForSingleObject(handle, 10000), 0)

    def test_close_terminates_child_and_grandchild_using_retained_handles(self):
        ready = self.root / "ready.json"
        arguments = ["with spaces", 'a"quoted"argument', "trailing slash\\", ""]
        env = dict(os.environ, LAUNCHER_TEST_UNICODE="café 雪")
        job = windows_job.OwnedJob()
        self.addCleanup(job.close)
        inherited = wintypes.DWORD()
        self.assertTrue(self.api.GetHandleInformation(job._job_handle, ctypes.byref(inherited)))
        self.assertEqual(inherited.value & 1, 0, "job handle must not be inherited")
        with (self.root / "child.log").open("wb") as log:
            pid = job.spawn([sys.executable, "-c", _CHILD, str(ready), str(os.getpid()), *arguments], env, log)
            original = self.retain(job._process_handle)
            record = self.read_ready(ready, job)
            grandchild = record["handle"]
            self.addCleanup(self.api.CloseHandle, grandchild)
            actual_child = record["child_handle"]
            self.addCleanup(self.api.CloseHandle, actual_child)
            # Venv Python can launch a redirector as the initial process.
            # Retain and verify both it and the actual interpreter separately.
            self.assertEqual(job.pid, pid)
            self.assertGreater(record["pid"], 0)
            self.assertEqual(record["arguments"], arguments)
            self.assertEqual(record["environment"], "café 雪")
            self.assertEqual(record["stdin"], "")
            self.assertTrue(job.alive())
            self.assertEqual(self.api.WaitForSingleObject(actual_child, 0), 258)
            self.assertEqual(self.api.WaitForSingleObject(grandchild, 0), 258)
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                job.spawn([sys.executable, "-c", "pass"], stdout=log)
            job.close()
            self.assert_exited(original)
            self.assert_exited(actual_child)
            self.assert_exited(grandchild)
            self.assertFalse(job.alive())
            self.assertIsNone(job._job_handle)
            self.assertIsNone(job._process_handle)
            job.close()
        output = (self.root / "child.log").read_text(encoding="utf-8")
        self.assertIn("owned-child-output", output)
        self.assertIn("owned-child-error", output)

    def test_exit_code_retains_normal_result_across_python_redirector(self):
        job = windows_job.OwnedJob()
        self.addCleanup(job.close)
        with self.assertRaisesRegex(RuntimeError, "no retained launched process"):
            job.exit_code()
        gate = self.root / "allow-exit.txt"
        source = (
            "from pathlib import Path; import sys, time\n"
            "gate = Path(sys.argv[1])\n"
            "while not gate.exists(): time.sleep(0.01)\n"
            "raise SystemExit(7)\n"
        )
        job.spawn([sys.executable, "-c", source, str(gate)])
        self.assertIsNone(job.exit_code())
        gate.write_text("exit", encoding="utf-8")
        self.assert_exited(job._process_handle)
        self.assertFalse(job.alive())
        self.assertEqual(job.exit_code(), 7)
        self.assertEqual(job.exit_code(), 7)
        job.close()
        with self.assertRaisesRegex(RuntimeError, "no retained launched process"):
            job.exit_code()

    def assert_failed_launch_is_closed(self, operation):
        job = windows_job.OwnedJob()
        self.addCleanup(job.close)
        retained = []
        sentinel = self.root / "must-not-run.txt"
        source = "from pathlib import Path; import time; Path(%r).write_text('ran'); time.sleep(120)" % str(sentinel)

        def fail(*args):
            retained.append(self.retain(job._process_handle))
            ctypes.set_last_error(5)
            return 0xFFFFFFFF if operation == "ResumeThread" else 0

        with mock.patch.object(windows_job._kernel32, operation, side_effect=fail):
            with self.assertRaisesRegex(OSError, operation):
                job.spawn([sys.executable, "-c", source])
        self.assertEqual(len(retained), 1)
        self.assert_exited(retained[0])
        self.assertFalse(sentinel.exists(), "failed launch must never resume")
        self.assertIsNone(job._process_handle)
        self.assertIsNone(job._job_handle)

    def test_assignment_failure_terminates_the_suspended_original_child(self):
        self.assert_failed_launch_is_closed("AssignProcessToJobObject")

    def test_resume_failure_terminates_the_suspended_owned_child(self):
        self.assert_failed_launch_is_closed("ResumeThread")

    def test_interrupt_after_creation_terminates_the_suspended_original_child(self):
        job = windows_job.OwnedJob()
        self.addCleanup(job.close)
        create = windows_job._kernel32.CreateProcessW
        retained = []
        sentinel = self.root / "interrupt-must-not-run.txt"
        source = "from pathlib import Path; Path(%r).write_text('ran')" % str(sentinel)

        def interrupted(*args):
            self.assertTrue(create(*args))
            process = args[-1]._obj
            retained.append(self.retain(process.hProcess))
            raise KeyboardInterrupt("simulated interruption")

        with mock.patch.object(windows_job._kernel32, "CreateProcessW", side_effect=interrupted):
            with self.assertRaisesRegex(KeyboardInterrupt, "simulated interruption"):
                job.spawn([sys.executable, "-c", source])
        self.assertEqual(len(retained), 1)
        self.assert_exited(retained[0])
        self.assertFalse(sentinel.exists())
        self.assertIsNone(job._process_handle)
        self.assertIsNone(job._job_handle)

    def test_creation_failure_closes_job_without_a_process(self):
        with windows_job.OwnedJob() as job:
            with self.assertRaisesRegex(OSError, "CreateProcessW"):
                job.spawn([str(self.root / "missing.exe")])
            self.assertFalse(job.alive())
            self.assertIsNone(job._job_handle)
            self.assertIsNone(job._process_handle)


if __name__ == "__main__":
    unittest.main()
