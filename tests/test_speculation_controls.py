"""Controls retain failed coding evidence while speculative arms get measured."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from localbench.config import load
from localbench.state import State
from localbench.speculation_search import control_timings_complete,matched_control,comparison_due

class MatchedControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.cfg=load()
        for name in ('state','artifacts','logs'):
            self.cfg['paths'][name]=str(Path(self.temp.name)/name)
        self.state=State(self.cfg,clock=lambda:1000);self.identifier='c'*64
        self.control=dict(configuration_id=self.identifier,engine='inert',model='inert',
            runtime={'engine':'inert'},settings={'speculation':'off','cache_k':'q8_0','cache_v':'q8_0'},
            capacity_qualified=True,eligible=False,exploration_smoke_passed=False,timing=[],
            smoke=[{'fixture':'ts01','passed':False,'reason':'task_timeout'}],
            planned_screen_key='planned_screen:inert')
        self.adapter=SimpleNamespace();self.calls=[]
    def tearDown(self):self.state.close();self.temp.cleanup()
    def rows(self):
        return [dict(kind='timing',mode='fresh',target_tokens=61440,replicate=r,valid=True,passed=True,
            token_evidence={'expected_prompt_tokens':61440,'effective_context_tokens':65536,
                'context_shift':False,'truncated':False,'tokenization_verified':True,'cache_isolation_verified':True},
            metrics={'first_action_valid':True,'first_action_seconds':100})
            for r in range(self.cfg['measurement']['fresh_repetitions_screen'])]
    def measure(self,cfg,state,adapter,collector,identifier,kind,seed,mode,replicate,target,operation):
        self.calls.append((identifier,kind,seed,mode,replicate,target))
        return self.rows()[replicate]
    def run_control(self,actual=None):
        with patch('localbench.speculation_search.checkpoint_model',return_value={'id':'inert'}),\
                patch('localbench.search_core.make_adapter',return_value=(self.adapter,actual or self.identifier)),\
                patch('localbench.search_core.close_adapter') as close,\
                patch('localbench.scheduler.measured_job',side_effect=self.measure):
            result=matched_control(self.cfg,self.state,None,self.control)
            close.assert_called_once_with(self.adapter)
        return result
    def test_failed_coding_control_receives_exact_fresh_comparator_measurements(self):
        original=copy.deepcopy(self.control['smoke']);result=self.run_control()
        self.assertTrue(result['comparison_control_ready']);self.assertFalse(result['eligible'])
        self.assertEqual(result['smoke'],original);self.assertFalse(result['exploration_smoke_passed'])
        self.assertEqual(self.calls,[(self.identifier,'timing',42+r,'fresh',r,61440) for r in range(3)])
        self.assertEqual(self.state.get_control(self.control['planned_screen_key'])['smoke'],original)
    def test_complete_diagnostic_control_is_reused_without_loading_an_engine(self):
        self.control['timing']=self.rows()
        with patch('localbench.search_core.make_adapter') as launch:
            result=matched_control(self.cfg,self.state,None,self.control)
        launch.assert_not_called();self.assertTrue(result['comparison_control_ready'])
        self.assertFalse(result['eligible'])
    def test_failed_capacity_is_never_accepted_as_a_matched_control(self):
        self.control['capacity_qualified']=False
        with patch('localbench.search_core.make_adapter') as launch:
            result=matched_control(self.cfg,self.state,None,self.control)
        launch.assert_not_called();self.assertFalse(result.get('comparison_control_ready',False))
    def test_cached_or_short_or_censored_or_duplicate_samples_cannot_unlock_arms(self):
        for field,value in (('mode','cached'),('valid',False),('passed',False),('replicate',1)):
            rows=self.rows();rows[0][field]=value
            self.assertFalse(control_timings_complete(self.cfg,{'timing':rows}),(field,value))
        rows=self.rows();rows[0]['metrics']['first_action_valid']=False
        self.assertFalse(control_timings_complete(self.cfg,{'timing':rows}))
        for field,value in (('expected_prompt_tokens',16384),('expected_prompt_tokens',61441),
                ('effective_context_tokens',32768),('context_shift',True),('truncated',True),
                ('tokenization_verified',False),('cache_isolation_verified',False)):
            rows=self.rows();rows[0]['token_evidence'][field]=value
            self.assertFalse(control_timings_complete(self.cfg,{'timing':rows}),(field,value))
    def test_archived_real_measurement_shape_does_not_require_a_top_level_target(self):
        record=json.loads((Path(__file__).parent/'data/native-fresh-64k-timing.json').read_text())
        self.assertNotIn('target_tokens',record)
        rows=[{**copy.deepcopy(record),'replicate':r} for r in range(3)]
        self.assertTrue(control_timings_complete(self.cfg,{'timing':rows}))
        rows[0]['token_evidence']['expected_prompt_tokens']=61376
        self.assertTrue(control_timings_complete(self.cfg,{'timing':rows}))
        rows[0]['token_evidence']['expected_prompt_tokens']=61375
        self.assertFalse(control_timings_complete(self.cfg,{'timing':rows}))
    def test_changed_control_identity_closes_the_owned_adapter_and_fails(self):
        with patch('localbench.speculation_search.checkpoint_model',return_value={'id':'inert'}),\
                patch('localbench.search_core.make_adapter',return_value=(self.adapter,'d'*64)),\
                patch('localbench.search_core.close_adapter') as close:
            with self.assertRaisesRegex(RuntimeError,'identity_changed'):
                matched_control(self.cfg,self.state,None,self.control)
            close.assert_called_once_with(self.adapter)
        self.assertEqual(self.state.db.execute('SELECT count(*) FROM attempts').fetchone()[0],0)
    def test_enabled_incomplete_comparison_has_priority_but_cannot_extend_tuning(self):
        self.state.control('tuning_phase',{'deadline':2000})
        with patch('localbench.speculation_search.enabled_plan',return_value={'enabled':True}):
            self.state.control('speculation_comparison_phase',{'status':'screening_and_quality_complete'})
            self.assertTrue(comparison_due(self.cfg,self.state))
            self.state.clock=lambda:2000
            self.assertFalse(comparison_due(self.cfg,self.state))
        with patch('localbench.speculation_search.enabled_plan',return_value=None):
            self.assertFalse(comparison_due(self.cfg,self.state))

if __name__=='__main__':unittest.main()
