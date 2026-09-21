"""Refused orphan recovery must precede any new measurement or runtime call."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
from localbench.scheduler import run


class ResumeGpuGateTests(unittest.TestCase):
    def test_unresolved_original_ownership_prevents_new_gpu_work(self):
        for status in ('refused','pending_descendants','pending_auth','pending_load'):
            with self.subTest(status=status):
                state=SimpleNamespace(run={'id':'mock-run'},entity=Mock(),control=Mock())
                with patch('localbench.recovery.recover_processes',return_value=[{'status':status}]),patch('localbench.scheduler.doctor') as doctor:
                    with self.assertRaisesRegex(RuntimeError,'owned_recovery_unresolved'):
                        run({},state,SimpleNamespace(command='resume'))
                doctor.assert_not_called()
                state.control.assert_called_once_with('ownership_recovery',[{'status':status}])


if __name__=='__main__':unittest.main()
