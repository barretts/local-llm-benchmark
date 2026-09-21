"""Mock-only original-container cleanup proofs; no Docker/GPU/process calls."""
import copy
import json
import unittest

from localbench.handoff import _close_and_prove
from tests import test_container_adapter as fixtures


class ContainerCleanupTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ContainerAdapterTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.fixture.state.clock = lambda: 1000
        self.fixture.state.control = lambda key, value: self.fixture.state.controls.__setitem__(key, value)
        self.adapter, self.docker = self.fixture.prepared()
        self.fixture.launch_mock(self.adapter)
        self.original = self.adapter.process

    def tearDown(self):
        self.adapter.close_logs()
        self.fixture.tearDown()

    def test_original_handle_polls_verified_exit_after_adapter_is_cleared(self):
        self.adapter.unload_owned()
        calls = len(self.docker.calls)
        self.assertIsNone(self.adapter.saved)
        self.assertIsNone(self.adapter.process)
        self.assertEqual(self.original.poll(), 0)
        self.assertEqual(self.original.poll(), 0)
        self.assertEqual(len(self.docker.calls), calls)
        proof = self.original.removal_evidence
        self.assertEqual(proof["container_id"], fixtures.CID)
        self.assertTrue(proof["removed"])
        self.assertTrue(proof["absence_verified"])
        self.assertTrue(proof["stopped_verified"])
        record = json.loads((self.fixture.root / "state" / "owned-container-38202.json").read_text(encoding="utf-8"))
        self.assertEqual(record["cleanup_completion"], proof)

    def test_handoff_close_proves_original_container_stop_without_inspecting_cleared_adapter(self):
        proof = _close_and_prove(self.adapter, self.fixture.state)
        self.assertTrue(proof["stopped"])
        self.assertEqual(proof["owned_handle"]["container_id"], fixtures.CID)
        self.assertTrue(proof["completion_evidence"]["absence_verified"])
        self.assertEqual(proof["completion_evidence"]["stop_argv"][-1], fixtures.CID)
        self.assertEqual(self.original.poll(), 0)
        self.assertTrue(self.original.removal_evidence["absence_verified"])

    def test_completed_handle_cannot_poll_a_newly_launched_instance(self):
        self.adapter.unload_owned()
        self.fixture.launch_mock(self.adapter)
        self.assertIsNone(self.adapter.process.poll())
        calls = len(self.docker.calls)
        self.assertEqual(self.original.poll(), 0)
        self.assertEqual(len(self.docker.calls), calls)
        self.adapter.unload_owned()

    def test_completed_removal_evidence_cannot_be_mutated_through_property(self):
        self.adapter.unload_owned()
        external = self.original.removal_evidence
        external["removed"] = False
        external["exit_code"] = 137
        self.assertTrue(self.original.removal_evidence["removed"])
        self.assertEqual(self.original.poll(), 0)

    def test_stop_failure_retains_original_identity_without_completion(self):
        self.docker.stop_code = 1
        with self.assertRaisesRegex(RuntimeError, "owned_container_stop_failed"):
            self.adapter.unload_owned()
        self.assertIs(self.adapter.process, self.original)
        self.assertIsNotNone(self.adapter.saved)
        self.assertIsNone(self.original.removal_evidence)
        self.assertIsNone(self.original.poll())
        self.assertFalse(any(call[0] == "remove-owned" for call in self.docker.calls))

    def test_stop_command_success_without_stopped_state_never_removes(self):
        def unchanged_stop(label, args, **kwargs):
            if label == "stop-owned":
                return {"exit_code": 0, "output": fixtures.CID, "argv": ["docker"] + args, "log": "mock-stop.log"}
            return self.docker(label, args, **kwargs)
        self.adapter._docker = unchanged_stop
        with self.assertRaisesRegex(RuntimeError, "stopped_state_unverified"):
            self.adapter.unload_owned()
        self.assertIs(self.adapter.process, self.original)
        self.assertIsNone(self.original.removal_evidence)
        self.assertFalse(any(call[0] == "remove-owned" for call in self.docker.calls))

    def test_remove_failure_retains_identity_and_no_completed_flag(self):
        def failed_remove(label, args, **kwargs):
            if label == "remove-owned":
                return {"exit_code": 1, "output": "mock failure", "argv": ["docker"] + args, "log": "mock-rm.log"}
            return self.docker(label, args, **kwargs)
        self.adapter._docker = failed_remove
        with self.assertRaisesRegex(RuntimeError, "owned_container_remove_failed"):
            self.adapter.unload_owned()
        self.assertIsNotNone(self.adapter.saved)
        self.assertIs(self.adapter.process, self.original)
        self.assertIsNone(self.original.removal_evidence)

    def test_remove_success_without_actual_absence_is_not_completion(self):
        def unchanged_remove(label, args, **kwargs):
            if label == "remove-owned":
                return {"exit_code": 0, "output": fixtures.CID, "argv": ["docker"] + args, "log": "mock-rm.log"}
            return self.docker(label, args, **kwargs)
        self.adapter._docker = unchanged_remove
        with self.assertRaisesRegex(RuntimeError, "owned_container_removal_unverified"):
            self.adapter.unload_owned()
        self.assertIsNotNone(self.adapter.saved)
        self.assertIs(self.adapter.process, self.original)
        self.assertIsNone(self.original.removal_evidence)

    def test_daemon_error_after_rm_cannot_be_treated_as_verified_absence(self):
        def daemon_failure(label, args, **kwargs):
            if label == "container-inspect" and self.docker.current is None:
                return {"exit_code": 1, "output": "Error: daemon unreachable", "argv": ["docker"] + args, "log": "mock-inspect.log"}
            return self.docker(label, args, **kwargs)
        self.adapter._docker = daemon_failure
        with self.assertRaisesRegex(RuntimeError, "owned_container_removal_unverified"):
            self.adapter.unload_owned()
        self.assertIsNone(self.original.removal_evidence)
        self.assertIsNotNone(self.adapter.saved)
        self.assertIs(self.adapter.process, self.original)

    def test_wrong_missing_object_id_does_not_prove_original_absence(self):
        def wrong_absence(label, args, **kwargs):
            if label == "container-inspect" and self.docker.current is None:
                return {"exit_code": 1, "output": "Error: No such object: " + "d" * 64, "argv": ["docker"] + args, "log": "mock-inspect.log"}
            return self.docker(label, args, **kwargs)
        self.adapter._docker = wrong_absence
        with self.assertRaisesRegex(RuntimeError, "owned_container_removal_unverified"):
            self.adapter.unload_owned()
        self.assertIsNone(self.original.removal_evidence)
        self.assertIsNotNone(self.adapter.saved)

    def test_stale_original_identity_refuses_stop_and_completion(self):
        self.docker.current["started"] = "unrelated-new-start"
        with self.assertRaisesRegex(RuntimeError, "stale_container_identity"):
            self.adapter.unload_owned()
        self.assertIsNone(self.original.removal_evidence)
        self.assertIsNotNone(self.adapter.saved)
        self.assertFalse(any(call[0] in ("stop-owned", "remove-owned") for call in self.docker.calls))

    def test_uncompleted_handle_cannot_silently_follow_a_replaced_adapter_identity(self):
        self.adapter.saved["container_id"] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "stale_container_handle"):
            self.original.poll()
        self.assertIsNone(self.original.removal_evidence)

    def test_completion_marker_rejects_wrong_original_identity(self):
        observed = copy.deepcopy(self.docker.current)
        observed.update(id="d" * 64, running=False)
        proof = {"container_id": fixtures.CID, "removed": True, "absence_verified": True, "exit_code": 0}
        with self.assertRaisesRegex(RuntimeError, "completion_identity_unverified"):
            self.original._complete_removal(observed, proof)
        self.assertIsNone(self.original.removal_evidence)


if __name__ == "__main__":
    unittest.main()
