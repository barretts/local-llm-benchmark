"""Original handle completion and stable PID-reuse proof, using mocks only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from localbench.adapters.native import NativeAdapter
from localbench.recovery import recover_process
from tests import test_recovery as recovery_fixtures
from tests.test_recovery import FakeProcess


class NativeCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.adapter=NativeAdapter.__new__(NativeAdapter)
        self.adapter.config={'paths':{'state':str(self.root)}};self.adapter.port=38201
        self.adapter.saved={'owner':'localbench','created':100,'pid':123456,'argv':['inert.exe']}
        self.adapter.handles=[];self.adapter.process=Mock(pid=123456)
        self.process=self.adapter.process

    def tearDown(self):self.temp.cleanup()

    def test_original_exit_is_persisted(self):
        self.process.poll.side_effect=[None,0]
        with patch('localbench.adapters.native.creation_time',return_value=100):self.adapter.unload_owned()
        record=json.loads((self.root/'owned-38201.json').read_text())
        self.assertIs(record['stopped'],True);self.assertEqual(record['created'],100)
        self.assertEqual(record['exit_code'],0);self.process.terminate.assert_called_once()
        self.assertIsNone(self.adapter.process)

    def test_recycled_identity_or_failed_exit_never_creates_completed_proof(self):
        for created,polls in ((101,[None]),(100,[None,None])):
            with self.subTest(created=created):
                self.adapter.process=self.process;self.process.poll.side_effect=polls
                with patch('localbench.adapters.native.creation_time',return_value=created):
                    with self.assertRaises(RuntimeError):self.adapter.unload_owned()
                self.assertFalse((self.root/'owned-38201.json').exists())
                self.assertIs(self.adapter.process,self.process)


class PidReuseProofTests(unittest.TestCase):
    setUp=recovery_fixtures.RecoveryTests.setUp
    tearDown=recovery_fixtures.RecoveryTests.tearDown
    save=recovery_fixtures.RecoveryTests.save
    def test_proved_pid_reuse_retires_original_without_touching_new_instance(self):
        self.save();process=FakeProcess(self.argv+['new unrelated app'],created=101)
        result=recover_process(self.config,self.state,self.path,lambda pid:process)
        self.assertEqual(result['status'],'original_process_replaced');self.assertEqual(process.stops,[])
        evidence=json.loads(Path(result['evidence']).read_text())
        self.assertEqual(evidence['original_handle'],self.saved)
        self.assertEqual(evidence['observed_creation'],101);self.assertIs(evidence['termination_performed'],False)
        self.assertTrue(process.closed)

    def test_missing_or_unstable_replacement_creation_fails_closed(self):
        for values in ((None,),(101,102)):
            with self.subTest(values=values):
                self.save();process=FakeProcess(self.argv);process.created=Mock(side_effect=values)
                with self.assertRaisesRegex(RuntimeError,'stale_process_creation'):
                    recover_process(self.config,self.state,self.path,lambda pid:process)
                self.assertEqual(process.stops,[])
                self.assertNotIn('stopped',json.loads(self.path.read_text()))


if __name__=='__main__':unittest.main()
