"""A phase that expires while packing must not create or grade a measurement."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from localbench.config import load
from localbench.scheduler import measured_job
from localbench.state import State
from localbench.speculation_search import PhaseDeadline,phase_budget

class PhaseMeasurementTests(unittest.TestCase):
    def test_expiry_before_packed_request_preserves_an_unmeasured_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg=load();now=[1000]
            for name in ('state','logs','artifacts'):cfg['paths'][name]=str(Path(temporary)/name)
            state=State(cfg,clock=lambda:now[0])
            try:
                adapter=SimpleNamespace(engine='inert',model={'id':'inert'},http=SimpleNamespace(timeout=123))
                base=dict(configuration_id='inert',kind='timing',seed=42,mode='fresh',replicate=0,target_tokens=61440)
                def operation(callback):
                    now[0]=1500
                    callback({'prompt_hash':'inert-prompt'})
                    raise AssertionError('must not send a request after phase budget expires')
                original=state.check_budget
                with patch('localbench.scheduler.key_base',return_value=base),phase_budget(state,2000):
                    with self.assertRaises(PhaseDeadline):
                        measured_job(cfg,state,adapter,None,'inert','timing',42,'fresh',0,61440,operation)
                self.assertEqual(state.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],0)
                self.assertEqual(state.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],0)
                self.assertEqual(adapter.http.timeout,123)
                self.assertEqual(state.check_budget,original)
            finally:state.close()

if __name__=='__main__':unittest.main()
