"""Regression checks for a CLI picker blocking an unattended managed load."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from tests import test_managed_adapters as fixtures
from localbench.recovery import recover_lm_studio


class NoninteractiveLoadTests(unittest.TestCase):
    setUp = fixtures.ManagedAdapterTests.setUp
    tearDown = fixtures.ManagedAdapterTests.tearDown
    lms = fixtures.ManagedAdapterTests.lms
    inventory = fixtures.ManagedAdapterTests.inventory

    def failure(self, code, output):
        adapter = self.lms()
        adapter.http = Mock()
        adapter.http.json.return_value = self.inventory()
        result = {'exit_code': code, 'output': output, 'log': 'mock-load.log'}
        with patch.dict(os.environ, {'LM_BENCH_TOKEN': 'mock-memory-only'}), patch(
                'localbench.adapters.lms._logged_command', return_value=result) as cli:
            with self.assertRaises(RuntimeError) as raised:
                adapter.launch()
        self.assertIn('--yes', cli.call_args.args[2])
        saved = json.loads((Path(self.config['paths']['state']) / 'owned-lm-studio-instance.json').read_text())
        return adapter, saved, str(raised.exception)

    def test_no_match_never_submitted_load_and_repeated_absence_can_retire(self):
        adapter, saved, reason = self.failure(1,
            'Model not found\nNo model found that matches model key "test-model".')
        self.assertTrue(saved['load_not_started'])
        self.assertTrue(saved['unloaded'])
        self.assertIsNone(adapter.instance_id)
        self.assertIn('no_load_submitted', reason)
        self.assertGreaterEqual(adapter.http.json.call_count, 3)
        self.assertFalse(any(call.args[0].endswith('/unload') for call in adapter.http.json.call_args_list))

    def test_timeout_remains_pending(self):
        _, saved, reason = self.failure(None, 'TimeoutExpired')
        self.assertFalse(saved['load_not_started'])
        self.assertNotIn('unloaded', saved)
        self.assertIn('completion_unknown', reason)

    def test_other_failure_or_wrong_model_does_not_prove_no_submission(self):
        for code, output in [(137, 'Model not found No model found that matches model key test-model'),
                (1, 'Model not found No model found that matches model key other-model'),
                (1, 'Backend failed after submitting test-model')]:
            with self.subTest(code=code, output=output):
                _, saved, reason = self.failure(code, output)
                self.assertFalse(saved['load_not_started'])
                self.assertNotIn('unloaded', saved)
                self.assertIn('completion_unknown', reason)

    def test_recovery_accepts_proven_non_submission_but_not_unknown_timeout(self):
        _, saved, _ = self.failure(1,
            'Model not found\nNo model found that matches model key "test-model".')
        saved.pop('unloaded')
        path = Path(self.config['paths']['state']) / 'owned-lm-studio-instance.json'
        path.write_text(json.dumps(saved))
        client = Mock()
        client.json.return_value = self.inventory()
        with patch.dict(os.environ, {'LM_BENCH_TOKEN': 'mock-memory-only'}):
            result = recover_lm_studio(self.config, self.state, client)
        self.assertEqual(result['status'], 'already_unloaded')
        self.assertTrue(json.loads(path.read_text())['unloaded'])


if __name__ == '__main__':
    unittest.main()
