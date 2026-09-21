"""Final comparison/handoff integration; entirely mock or inert temp files."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from localbench.config import file_hash,load
from localbench.state import State
from localbench.report import Report
from localbench.handoff import _checkpoint_pins

class FinalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.cfg=load();self.root=Path(self.temp.name)
        for key in ('state','artifacts','logs'):
            self.cfg['paths'][key]=str(self.root/key);Path(self.cfg['paths'][key]).mkdir()
        self.state=State(self.cfg,clock=lambda:1000)
        self.base='b'*64;self.spec='c'*64;self.actual=10.5;self.control_complete=True;self.drafting=True
        self.state.entity('configuration',self.base,{'requested_settings':{'speculation':'off'}})
        self.state.entity('configuration',self.spec,{'requested_settings':{'speculation':'draft-dflash',
            'speculation_baseline_configuration_id':self.base,'draft_model':{'bytes':100}}})
    def tearDown(self):self.state.close();self.temp.cleanup()
    def score(self,identifier,data,rows):
        active=identifier==self.spec
        gates={'regular_quality':True,'long_quality':True,'capacity_all_nine_values':True,
            'twenty_valid_fresh':True if active else self.control_complete,'twenty_valid_cached':True,
            'runtime_stability':True,'isolated_agent_demo':True}
        if active:gates.update(speculation_real_drafting=self.drafting,speculation_cold_latency_regression=False)
        return dict(configuration_id=identifier,configuration=data,qualified=all(gates.values()),gates=gates,
            failed_gates=[],regular={'passed':28,'total':36},long={'passed':9,'total':12},
            latency={'fresh':{'first_action':{'p95':self.actual if active else 10}},'cached':{'first_action':{'p95':.2}}},
            throughput={'exact_sustained_64k':True,'median':50 if active else 20},peak_dedicated_gpu_bytes=1000,
            resource_attribution_uncertainty=[])
    def summary(self):
        with patch.object(Report,'_configuration',side_effect=self.score):return Report(self.cfg,self.state).summarize([])
    def test_speculation_qualified_only_with_twenty_matched_fresh_controls(self):
        self.assertIn(self.spec,self.summary()['ranking'])
        self.control_complete=False;summary=self.summary()
        self.assertNotIn(self.spec,summary['ranking'])
        spec=next(s for s in summary['configurations'] if s['configuration_id']==self.spec)
        self.assertIsNone(spec['speculation_comparison']['fresh_p95_ratio'])
    def test_over_ten_percent_regression_disqualifies_fast_decoder(self):
        self.actual=11.01;summary=self.summary();self.assertNotIn(self.spec,summary['ranking'])
        self.assertEqual(summary['winner_id'],self.base)
    def test_exact_boundary_is_eligible_but_missing_real_drafting_is_not(self):
        self.actual=11;self.assertIn(self.spec,self.summary()['ranking'])
        self.drafting=False;self.assertNotIn(self.spec,self.summary()['ranking'])
    def test_missing_baseline_configuration_is_fail_closed(self):
        with self.state.db:self.state.db.execute("DELETE FROM entities WHERE kind='configuration' AND id=?",(self.base,))
        self.assertNotIn(self.spec,self.summary()['ranking'])
    def test_handoff_pins_and_rechecks_the_real_draft_file(self):
        model=self.root/'model.gguf';draft=self.root/'draft.gguf'
        model.write_bytes(b'inert target');draft.write_bytes(b'inert draft')
        data=dict(model_sha256=file_hash(model),requested_settings={'draft_model':{
            'path':str(draft),'bytes':draft.stat().st_size,'sha256':file_hash(draft)}})
        pins=_checkpoint_pins({'path':str(model),'bytes':model.stat().st_size},data,None)
        self.assertEqual({p['role'] for p in pins},{'model','speculative-draft'})
        draft.write_bytes(b'corrupted draft')
        with self.assertRaisesRegex(RuntimeError,'hash_mismatch'):
            _checkpoint_pins({'path':str(model),'bytes':model.stat().st_size},data,None)
    def test_control_must_use_identical_binary_libraries_and_sampler(self):
        from localbench.native_speculation import verify_baseline_engine
        metadata=dict(binary_sha256='1'*64,adjacent_dlls={'ggml-cuda.dll':'2'*64},version_output='inert-version')
        data={**metadata,'identity':{'sampler':copy.deepcopy(self.cfg['default_sampling'])}}
        self.state.entity('configuration',self.base,data)
        settings={'speculation_baseline_configuration_id':self.base}
        verify_baseline_engine(self.cfg,self.state,settings,metadata)
        changed={**metadata,'adjacent_dlls':{'ggml-cuda.dll':'3'*64}}
        with self.assertRaisesRegex(ValueError,'engine_or_sampler_mismatch'):
            verify_baseline_engine(self.cfg,self.state,settings,changed)
        data['identity']['sampler']['temperature']+=.01
        self.state.entity('configuration',self.base,data)
        with self.assertRaisesRegex(ValueError,'engine_or_sampler_mismatch'):
            verify_baseline_engine(self.cfg,self.state,settings,metadata)

if __name__=='__main__':unittest.main()
