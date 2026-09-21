"""Stop/elapsed-budget races use only tiny temporary State and inert adapters."""
import json
import unittest
from unittest.mock import patch

from localbench.config import canonical, digest
from localbench.scheduler import key_base, measured_job
from localbench.state import State
from tests import test_quality_controller as controller
from tests.test_quality_controller import FakeAdapter


class Collector:
    def __init__(self):
        self.calls = []

    def snapshot(self, *args):
        self.calls.append(args)
        return {"foreign_workload_overlap": False}


class MeasurementStopTests(unittest.TestCase):
    setUpFixture = controller.QualityControllerTests.setUp

    def setUp(self):
        self.setUpFixture()
        self.now = 1000.0
        self.state = State(self.config, clock=lambda: self.now)
        self.state.control("harness_verified", True)
        self.state.start_execution("mock-only measurement planning")
        self.adapter = FakeAdapter([])
        self.adapter.state = self.state
        self.collector = Collector()
        self.packed = {"prompt_hash": "stable-inert-packed-prompt"}

    def tearDown(self):
        self.state.close()
        controller.QualityControllerTests.tearDown(self)

    def measure(self, operation):
        return measured_job(self.config, self.state, self.adapter, self.collector,
            "mock-configuration", "capacity", 17, "fresh", 0, 61440, operation)

    def success(self, on_packed):
        on_packed(self.packed)
        return {"kind": "capacity", "configuration_id": "mock-configuration", "passed": True, "valid": True}

    def counts(self):
        return tuple(self.state.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
            for table in ('jobs', 'attempts'))

    def test_stop_during_packing_does_not_create_packed_or_unpacked_ghost(self):
        def operation(on_packed):
            self.state.stop()
            return self.success(on_packed)
        with self.assertRaisesRegex(RuntimeError, "stop_after_current"):
            self.measure(operation)
        self.assertEqual(self.counts(), (0, 0))
        self.assertEqual(self.collector.calls, [])

    def test_remaining_request_budget_is_rechecked_after_packing(self):
        def operation(on_packed):
            self.now = self.state.run["deadline"]-7200-self.config["limits"]["per_64k_request_timeout_seconds"]+1
            return self.success(on_packed)
        with self.assertRaisesRegex(RuntimeError, "budget_exhausted"):
            self.measure(operation)
        self.assertEqual(self.counts(), (0, 0))
        self.assertEqual(self.collector.calls, [])

    def test_stop_before_packed_callback_is_control_flow_without_fake_measurement(self):
        def operation(on_packed):
            self.state.stop()
            raise RuntimeError("stop_after_current")
        with self.assertRaisesRegex(RuntimeError, "stop_after_current"):
            self.measure(operation)
        self.assertEqual(self.counts(), (0, 0))

    def test_stop_after_enqueue_preserves_one_real_pending_job_then_resumes_it(self):
        original_run = self.state.run
        begin = self.state.begin
        def stopped_begin(job):
            self.state.stop()
            return begin(job)
        with patch.object(self.state, "begin", stopped_begin), self.assertRaisesRegex(RuntimeError, "stop_after_current"):
            self.measure(self.success)
        self.assertEqual(self.counts(), (1, 0))
        pending = self.state.db.execute('SELECT * FROM jobs').fetchone()
        self.assertEqual(pending['status'], 'pending')
        self.assertEqual(json.loads(pending['key_json'])['fixture_prompt_hash'], self.packed['prompt_hash'])
        self.state.recover()
        result = self.measure(self.success)
        self.assertTrue(result['passed'])
        self.assertEqual(self.counts(), (1, 1))
        completed = self.state.db.execute('SELECT * FROM jobs').fetchone()
        self.assertEqual(completed['id'], pending['id'])
        self.assertEqual(completed['status'], 'passed')
        self.assertEqual(self.state.run['id'], original_run['id'])
        self.assertEqual(self.state.run['started'], original_run['started'])
        self.assertEqual(self.state.run['deadline'], original_run['deadline'])

    def test_real_packing_error_still_has_archived_invalid_attempt(self):
        def operation(on_packed):
            raise ValueError('inert_packing_failure')
        result = self.measure(operation)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'inert_packing_failure')
        self.assertEqual(self.counts(), (1, 1))
        self.assertEqual(self.state.db.execute('SELECT status FROM jobs').fetchone()[0], 'invalid')
        reused = self.measure(lambda callback: self.fail('real terminal packing failure must be reused'))
        self.assertEqual(reused, result)
        self.assertEqual(self.counts(), (1, 1))

    def test_retired_unstarted_placeholder_cannot_suppress_real_packed_measurement(self):
        base = key_base(self.config, self.adapter, 'mock-configuration', 'capacity', 17, 'fresh', 0, 61440)
        packed_id = self.state.enqueue({**base, 'fixture_prompt_hash': self.packed['prompt_hash']})
        orphan_id = self.state.enqueue({**base, 'fixture_prompt_hash': digest({'unpacked_plan': base})})
        receipt = {'kind': 'maintenance', 'unstarted': True, 'reason': 'unstarted_cancellation_placeholder'}
        # This is the exact retirement shape in temporary test State only;
        # production reconciliation remains owned by the root controller.
        with self.state.db:
            self.state.db.execute("UPDATE jobs SET status='skipped',reason=?,result=? WHERE id=?",
                (receipt['reason'], canonical(receipt), orphan_id))
        result = self.measure(self.success)
        self.assertTrue(result['passed'])
        self.assertEqual(self.state.db.execute('SELECT status FROM jobs WHERE id=?', (packed_id,)).fetchone()[0], 'passed')
        orphan = self.state.db.execute('SELECT status,result FROM jobs WHERE id=?', (orphan_id,)).fetchone()
        self.assertEqual(orphan['status'], 'skipped')
        self.assertEqual(json.loads(orphan['result']), receipt)
        self.assertEqual(self.state.db.execute('SELECT COUNT(*) FROM attempts WHERE job_id=?', (orphan_id,)).fetchone()[0], 0)
        reused = self.measure(lambda callback: self.fail('completed packed measurement must be reused'))
        self.assertEqual(reused, result)
        self.assertEqual(self.counts(), (2, 1))


if __name__ == '__main__':
    unittest.main()
