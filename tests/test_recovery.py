"""Mock-only orphan recovery; no CIM process query or real termination."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from localbench.config import digest
from localbench.recovery import recover_process, recover_processes, recover_lm_studio, _split_windows_command_line


class FakeProcess:
    def __init__(self, argv, created=100, alive=True, stop_ok=True):
        self.expected_argv = argv
        self.creation, self.running, self.stop_ok = created, alive, stop_ok
        self.stops, self.closed = [], False

    def alive(self):
        return self.running

    def created(self):
        return self.creation

    def argv(self):
        return self.expected_argv

    def terminate(self, timeout):
        self.stops.append(timeout)
        return self.stop_ok

    def close(self):
        self.closed = True


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/"state").mkdir()
        (self.root/"runtime").mkdir()
        self.binary = self.root/"runtime"/"owned-engine.exe"
        self.binary.write_bytes(b"inert executable artifact")
        self.config = {
            "paths": {"state": str(self.root/"state"), "runtime_root": str(self.root/"runtime")},
            "private_ports": {"native": 38201, "ollama": 38207},
            "installed_tools": {"bundled_llama_server": str(self.binary), "ollama": str(self.binary),
                                "lms": str(self.root/"lms.exe")}}
        self.state = SimpleNamespace(run={"id": "mock-run"})
        self.argv = [str(self.binary), "--host", "127.0.0.1", "--port", "38201"]
        self.saved = {"owner": "localbench", "run_id": "mock-run", "pid": 123456,
                      "created": 100, "argv": self.argv, "command_hash": digest(self.argv),
                      "endpoint": "http://127.0.0.1:38201"}
        self.path = self.root/"state"/"owned-38201.json"
        self.identifier = "localbench-"+"a"*32

    def tearDown(self):
        self.temp.cleanup()

    def save(self, saved=None):
        self.path.write_text(json.dumps(saved or self.saved), encoding="utf-8")

    def save_lm(self):
        value = {"owner": "localbench", "run_id": "mock-run", "instance_id": self.identifier,
                 "model_key": "owned-model-key", "load_config": {"context_length": 65536, "parallel": 1},
                 "endpoint": "http://127.0.0.1:1234", "original_loaded_instance_ids": [],
                 "argv": [self.config["installed_tools"]["lms"], "load", "owned-model-key",
                          "--identifier", self.identifier]}
        path = self.root/"state"/"owned-lm-studio-instance.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return value, path

    def inventory(self, owned=True, config=None):
        models = [{"key": "unrelated-key", "loaded_instances": [{"id": "user-model", "config": {"context_length": 4096}}]}]
        if owned:
            models.append({"key": "owned-model-key", "loaded_instances": [{"id": self.identifier,
                           "config": config or {"context_length": 65536, "parallel": 1}}]})
        return {"models": models}

    def test_exact_retained_original_process_stops_once_and_marks_handle(self):
        self.save()
        process = FakeProcess(self.argv)
        factory = Mock(return_value=process)
        result = recover_process(self.config, self.state, self.path, factory)
        self.assertEqual(result["status"], "recovered")
        factory.assert_called_once_with(123456)
        self.assertEqual(process.stops, [15])
        self.assertTrue(process.closed)
        self.assertTrue(json.loads(self.path.read_text(encoding="utf-8"))["stopped"])

    def test_recycled_pid_or_changed_windows_argv_refuses_termination(self):
        for process in (FakeProcess(self.argv+["--personal-work"]),):
            with self.subTest(process=process):
                self.save()
                with self.assertRaisesRegex(RuntimeError, "stale_process"):
                    recover_process(self.config, self.state, self.path, lambda pid: process)
                self.assertEqual(process.stops, [])
                self.assertTrue(process.closed)
                self.assertNotIn("stopped", json.loads(self.path.read_text(encoding="utf-8")))

    def test_identity_change_after_command_read_refuses_termination(self):
        self.save()
        process = FakeProcess(self.argv)
        process.created = Mock(side_effect=[100, 101])
        with self.assertRaisesRegex(RuntimeError, "stale_process_creation"):
            recover_process(self.config, self.state, self.path, lambda pid: process)
        self.assertEqual(process.stops, [])

    def test_invalid_owner_run_command_endpoint_binary_and_legacy_time_never_open_process(self):
        changes = [{"owner": "other"}, {"run_id": "other-run"}, {"created": None},
                   {"command_hash": "changed"}, {"endpoint": "http://127.0.0.1:11434"},
                   {"argv": ["C:/personal/agent.exe"], "command_hash": digest(["C:/personal/agent.exe"])}]
        for change in changes:
            with self.subTest(change=change):
                self.save({**self.saved, **change})
                factory = Mock()
                with self.assertRaises(RuntimeError):
                    recover_process(self.config, self.state, self.path, factory)
                factory.assert_not_called()

    def test_exit_or_bounded_timeout_does_not_kill_by_name_or_claim_success(self):
        self.save()
        process = FakeProcess(self.argv, alive=False)
        result = recover_process(self.config, self.state, self.path, lambda pid: process)
        self.assertEqual(result["status"], "already_exited")
        self.assertEqual(process.stops, [])
        self.save()
        process = FakeProcess(self.argv, stop_ok=False)
        with self.assertRaisesRegex(RuntimeError, "identity_retained"):
            recover_process(self.config, self.state, self.path, lambda pid: process)
        self.assertEqual(process.stops, [15])
        self.assertNotIn("stopped", json.loads(self.path.read_text(encoding="utf-8")))

    def test_unfingerprinted_managed_descendants_remain_pending_without_parent_kill(self):
        self.save({**self.saved, "children": [{"pid": 42, "parent": 123456, "created": 200}]})
        factory = Mock()
        result = recover_process(self.config, self.state, self.path, factory)
        self.assertEqual(result["status"], "pending_descendants")
        factory.assert_not_called()

    def test_missing_lm_credential_leaves_unique_model_and_app_untouched(self):
        self.save_lm()
        http = Mock()
        with patch.dict(os.environ, {}, clear=True):
            result = recover_lm_studio(self.config, self.state, http)
        self.assertEqual(result["status"], "pending_auth")
        http.json.assert_not_called()

    def test_lm_recovery_unloads_only_exact_original_instance_and_keeps_unrelated_model(self):
        self.save_lm()
        http = Mock()
        http.json.side_effect = [self.inventory(), self.inventory(),
                                 {"instance_id": self.identifier}, self.inventory(owned=False)]
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "in-memory-only-test-key"}):
            result = recover_lm_studio(self.config, self.state, http)
        self.assertEqual(result["status"], "recovered")
        unloads = [call.args for call in http.json.call_args_list if call.args[0].endswith("/unload")]
        self.assertEqual(unloads, [("/api/v1/models/unload", {"instance_id": self.identifier})])
        record = (self.root/"state"/"owned-lm-studio-instance.json").read_text(encoding="utf-8")
        self.assertNotIn("in-memory-only-test-key", record)
        self.assertTrue(json.loads(record)["unloaded"])

    def test_lm_changed_configuration_or_second_snapshot_never_unloads(self):
        for inventories in ([self.inventory(config={"context_length": 4096})],
                            [self.inventory(), self.inventory(config={"context_length": 32768})]):
            with self.subTest(inventories=inventories):
                self.save_lm()
                http = Mock()
                http.json.side_effect = inventories
                with patch.dict(os.environ, {"LM_BENCH_TOKEN": "in-memory-only"}):
                    with self.assertRaisesRegex(RuntimeError, "stale_managed"):
                        recover_lm_studio(self.config, self.state, http)
                self.assertFalse(any(call.args[0].endswith("/unload") for call in http.json.call_args_list))

    def test_lm_credential_rejection_is_pending_with_no_app_termination(self):
        self.save_lm()
        http = Mock()
        http.json.side_effect = RuntimeError("local_http_status_401")
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "rejected-in-memory-key"}):
            result = recover_lm_studio(self.config, self.state, http)
        self.assertEqual(result["status"], "pending_auth")
        self.assertFalse(any(call.args[0].endswith("/unload") for call in http.json.call_args_list))

    @unittest.skipUnless(os.name == "nt", "Windows command-line parser")
    def test_windows_exact_argv_roundtrip_handles_spaces_quotes_and_trailing_slashes(self):
        argv = ["C:/Owned Runtime/server.exe", "--model", "C:/Owned Models/a file.gguf",
                "--literal", 'text with "quotes"', "--cache", "C:\\owned folder\\cache\\"]
        self.assertEqual(_split_windows_command_line(subprocess.list2cmdline(argv)), argv)

    def test_bulk_recovery_finishes_unaffected_handles_and_reports_refusal(self):
        self.save({**self.saved, "owner": "unrelated"})
        argv = [str(self.binary), "--host", "127.0.0.1", "--port", "38207"]
        second = {**self.saved, "pid": 123457, "argv": argv, "command_hash": digest(argv),
                  "endpoint": "http://127.0.0.1:38207"}
        (self.root/"state"/"owned-38207.json").write_text(json.dumps(second), encoding="utf-8")
        process = FakeProcess(argv)
        factory = Mock(return_value=process)
        result = recover_processes(self.config, self.state, factory, include_lm_studio=False)
        self.assertEqual([item["status"] for item in result], ["refused", "recovered"])
        factory.assert_called_once_with(123457)
        self.assertEqual(process.stops, [15])


if __name__ == "__main__":
    unittest.main()
