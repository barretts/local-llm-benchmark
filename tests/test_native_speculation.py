"""Mock/temp-file tests; no engines, downloads, generated code or GPU work."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from localbench.config import atomic_json,digest,file_hash,load
from localbench.state import State
from localbench import native_speculation as ns
from localbench import speculation_search as ss

HELP = '''--spec-type none,draft-mtp,draft-dflash,ngram-simple
--spec-draft-model FNAME
--spec-draft-n-max N
--spec-draft-n-min N
--spec-draft-p-min P
--spec-draft-ngl N
--spec-draft-device DEVICES
--spec-draft-type-k TYPE
 allowed values: f16, q8_0, q4_0
--spec-draft-type-v TYPE
 allowed values: f16, q8_0, q4_0
'''

def counters(drafted=40,accepted=30,steps=5):
    return '\n'.join(ns.COUNTERS[k]+' '+str(v) for k,v in
        dict(drafted_tokens=drafted,accepted_tokens=accepted,verification_steps=steps).items())+'\n'

class DraftIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.cfg=load()
        for key in ('state','logs','artifacts','new_model_root','installed_model_root'):
            self.cfg['paths'][key]=str(self.root/key);Path(self.cfg['paths'][key]).mkdir()
        self.state=State(self.cfg,clock=lambda:1000)
        self.path=Path(self.cfg['paths']['new_model_root'])/ns.DRAFT_PINS['draft-dflash']['filename']
        self.path.write_bytes(b'fixed inert weight test file')
        self.pin={**ns.DRAFT_PINS['draft-dflash'],'bytes':self.path.stat().st_size,'sha256':file_hash(self.path)}
        self.pins=patch.dict(ns.DRAFT_PINS,{'draft-dflash':self.pin});self.pins.start();self.addCleanup(self.pins.stop)
        self.base={'context':65536,'slots':1,'flash_attention':'on','cache_k':'q8_0','cache_v':'q8_0','speculation':'off'}
        self.id='b'*64
        self.state.entity('configuration',self.id,{'model_id':ns.TARGET,'requested_settings':self.base})
        self.settings={**self.base,'speculation':'draft-dflash','draft_model':{**self.pin,'path':str(self.path)},
            'draft_n_max':7,'draft_gpu_layers':99,'draft_cache_k':'q8_0','draft_cache_v':'q8_0',
            'speculation_contract_sha256':ns.contract_hash(),'speculation_baseline_configuration_id':self.id}
    def tearDown(self):self.state.close();self.temp.cleanup()
    def args(self):return ns.arguments(self.cfg,self.state,{'id':ns.TARGET},self.settings,HELP)
    def test_verified_file_launches_correct_type_and_draft(self):
        argv=self.args();self.assertEqual(argv[argv.index('--spec-type')+1],'draft-dflash')
        self.assertEqual(argv[argv.index('--spec-draft-model')+1],str(self.path))
        self.assertNotIn('--spec-synth-len',argv)
    def test_changed_draft_rejected_even_after_successful_cached_hash(self):
        self.args();self.path.write_bytes(b'corrupted inert weight file!')
        with self.assertRaisesRegex(RuntimeError,'draft_weight'):self.args()
    def test_other_targets_and_cross_target_draft_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'target'):
            ns.arguments(self.cfg,self.state,{'id':'qwen38-27b-iq3'},self.settings,HELP)
        self.settings['draft_model']['target_model_id']='qwen38-27b-iq3'
        with self.assertRaisesRegex(ValueError,'target_pin'):self.args()
    def test_matched_control_must_have_same_cache_and_settings(self):
        self.settings['cache_k']='q4_0'
        with self.assertRaisesRegex(ValueError,'matched_control_settings'):self.args()
    def test_changed_parser_contract_requires_reconciliation(self):
        self.settings['speculation_contract_sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'contract_changed'):self.args()
    def test_unavailable_flags_and_silently_clamped_width_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'unsupported_native_draft_flag'):
            ns.arguments(self.cfg,self.state,{'id':ns.TARGET},self.settings,HELP.replace('--spec-draft-device','--different-flag'))
        self.settings['draft_n_max']=16
        with self.assertRaisesRegex(ValueError,'clamped'):self.args()
    def test_off_cannot_silently_ignore_draft_options(self):
        self.settings['speculation']='off'
        with self.assertRaisesRegex(ValueError,'require_active'):self.args()
    def test_inherited_synthetic_controls_removed_without_changing_parent(self):
        env={'PATH':'original','LLAMA_ARG_SPEC_SYNTH_LEN':'8','LLAMA_ARG_SPEC_TYPE':'draft-simple','ordinary':'keep'}
        clean=ns.clean_environment(env);self.assertEqual(clean,{'PATH':'original','ordinary':'keep'})
        self.assertIn('LLAMA_ARG_SPEC_SYNTH_LEN',env)
    def test_startup_rejects_synthetic_acceptance_and_missing_draft_load(self):
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            ns.startup_evidence('synthetic speculative acceptance is enabled',self.settings)
        with self.assertRaisesRegex(RuntimeError,'draft_load'):
            ns.startup_evidence('server ready',self.settings)
        good=ns.startup_evidence('loading draft model '+self.path.name,self.settings)
        self.assertEqual(good['status'],'draft_loaded_counters_required')

class CounterEvidenceTests(unittest.TestCase):
    def test_counter_delta_uses_request_tokens_not_lifetime_acceptance(self):
        result=ns.counter_delta(ns.parse_counters(counters()),ns.parse_counters(counters(60,35,10)))
        self.assertEqual(result['drafted_tokens'],20);self.assertEqual(result['accepted_tokens'],5)
        self.assertEqual(result['acceptance_fraction'],.25)
    def test_counter_reset_impossible_acceptance_and_nonfinite_values_fail(self):
        for raw in (counters(10,20,1),counters('NaN',1,1),counters(1.5,1,1)):
            with self.assertRaises(RuntimeError):ns.parse_counters(raw)
        with self.assertRaisesRegex(RuntimeError,'reset'):
            ns.counter_delta(ns.parse_counters(counters()),ns.parse_counters(counters(1,1,1)))
    def test_missing_duplicate_or_unverified_counters_cannot_prove_drafting(self):
        for raw in ('',counters()+counters()):
            with self.assertRaises(RuntimeError):ns.parse_counters(raw)
        screen={'capacity':[{'valid':True,'metrics':{'speculation_drafted_tokens':30}}]}
        self.assertFalse(ns.real_drafting(screen))
        screen['capacity'][0]['metrics']['speculation_counters_verified']=True
        self.assertTrue(ns.real_drafting(screen))
    def test_zero_drafts_is_reported_as_no_drafting(self):
        result=ns.counter_delta(ns.parse_counters(counters()),ns.parse_counters(counters()))
        self.assertEqual(result['status'],'no_drafting_for_request');self.assertIsNone(result['acceptance_fraction'])
    def test_tokens_require_real_verification_steps(self):
        with self.assertRaisesRegex(RuntimeError,'without_target_verification'):
            ns.counter_delta(ns.parse_counters(counters()),ns.parse_counters(counters(50,35,5)))
    def test_stream_evidence_preserves_original_latency_and_token_definitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            response={'metrics':{'first_action_seconds':12.3,'generated_tokens':100},'valid_stream':True}
            adapter=SimpleNamespace(requested={'speculation':'draft-dflash','draft_model':{'sha256':'a'*64}},
                process=SimpleNamespace(pid=100),http=SimpleNamespace(measure=lambda *a,**k:copy.deepcopy(response)))
            values=[(ns.parse_counters(counters()),counters()),(ns.parse_counters(counters(60,40,10)),counters(60,40,10))]
            prefix=Path(tmp)/'stream'
            with patch.object(ns,'read_counters',side_effect=values):
                actual=ns.measured_stream(adapter,{},dict(log_prefix=prefix,validator=None))
            self.assertEqual(actual['metrics']['first_action_seconds'],12.3)
            self.assertEqual(actual['metrics']['generated_tokens'],100)
            self.assertEqual(actual['metrics']['speculation_accepted_tokens'],10)
            self.assertTrue(Path(str(prefix)+'-speculation.json').is_file())

class ComparisonResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.cfg=load()
        for key in self.cfg['paths']:
            self.cfg['paths'][key]=str(self.root/key);Path(self.cfg['paths'][key]).mkdir(exist_ok=True)
        self.state=State(self.cfg,clock=lambda:1000)
        self.help=self.root/'help.log';self.help.write_text(HELP)
        self.model={'id':ns.TARGET,'path':str(self.root/'inert.gguf'),'sha256':'1'*64}
        self.cfg['installed_model_candidates']=[self.model]
        self.state.entity('model',ns.TARGET,self.model)
        self.settings=dict(context=65536,slots=1,gpu_layers=99,flash_attention='on',cache_k='f16',cache_v='f16',
            batch=2048,ubatch=512,threads=16,threads_batch=16,jinja=True,context_shift=False,speculation='off',reasoning='default')
        self.runtime=dict(engine='upstream-llama-nightly',kind='native',binary_path='inert')
        self.anchor=dict(model=ns.TARGET,engine='upstream-llama-nightly',configuration_id='a'*64,
            capacity_qualified=True,settings=self.settings,runtime=self.runtime,timing=[{'passed':True,'metrics':{'first_action_seconds':100}}])
        self.state.entity('configuration','a'*64,{'help_log':str(self.help)})
        self.state.control('tuning_phase',dict(started=900,deadline=100000,selected_families=[ns.TARGET],status='active'))
        self.state.control('discovered_weight_variants',['old1','old2','old3'])
        self.plan=Path(self.cfg['paths']['artifacts'])/ss.PLAN_NAME;atomic_json(self.plan,ss.make_plan(self.cfg))
        self.state.control('speculation_comparison_enabled',dict(enabled=True,plan_sha256=file_hash(self.plan)))
        self.real=True;self.regress=False;self.smoke=True;self.control_smoke=True;self.completed={};self.launched=[];self.qualifications=[]
    def tearDown(self):self.state.close();self.temp.cleanup()
    def fake_screen(self,cfg,state,collector,runtime,model,settings=None,tuning=False):
        identifier=digest(settings)
        if identifier in self.completed:return copy.deepcopy(self.completed[identifier])
        self.launched.append(identifier);mode=settings['speculation']
        metrics={'first_action_seconds':120 if self.regress and mode!='off' else 100}
        if mode!='off':metrics.update(speculation_counters_verified=True,speculation_drafted_tokens=10 if self.real else 0)
        smoke=self.control_smoke if mode=='off' else self.smoke
        result=dict(engine=runtime['engine'],model=model['id'],configuration_id=identifier,settings=settings,runtime=runtime,
            eligible=smoke,capacity_qualified=True,exploration_smoke_passed=smoke,
            timing=[{'kind':'timing','mode':'fresh','target_tokens':61440,'replicate':r,
                'token_evidence':{'expected_prompt_tokens':61440,'effective_context_tokens':65536,
                    'context_shift':False,'truncated':False,'tokenization_verified':True,'cache_isolation_verified':True},
                'passed':True,'valid':True,'metrics':{**metrics,'first_action_valid':True}}
                for r in range(cfg['measurement']['fresh_repetitions_screen'])],
            capacity=[],throughput={'valid':True,'metrics':metrics},
            planned_screen_key='planned_screen:'+identifier)
        state.control(result['planned_screen_key'],result);self.completed[identifier]=result
        return copy.deepcopy(result)
    def run_comparison(self):
        def acquire(cfg,state,pin):return {'path':str(self.root/pin['filename'])}
        def qualify(cfg,state,collector,chosen,**kwargs):self.qualifications.append([s['settings']['speculation'] for s in chosen])
        with patch('localbench.search.run_screen',side_effect=self.fake_screen),patch('localbench.acquisition.acquire_gguf',side_effect=acquire),\
                patch.object(ss,'regular_and_long',side_effect=qualify),patch.object(ss.Report,'write',return_value={}):
            return ss.comparison(self.cfg,self.state,None,[self.anchor])
    def test_resume_reuses_exact_screens_and_preserves_original_phase_deadline(self):
        first=self.run_comparison();self.assertEqual(len(first),6);self.assertEqual(len(self.launched),6)
        self.run_comparison();self.assertEqual(len(self.launched),6)
        self.assertEqual(self.state.get_control('tuning_phase')['deadline'],100000)
        self.assertEqual(self.qualifications[0],['off','draft-mtp','draft-dflash'])
        self.assertEqual(len(self.state.get_control('discovered_weight_variants')),4)
    def test_no_real_drafting_and_cold_regression_cannot_advance_to_quality(self):
        self.real=False;out=self.run_comparison();self.assertEqual(self.qualifications[0],['off'])
        self.assertTrue(all(s['eligible'] is False for s in out if s['settings']['speculation']!='off'))
    def test_regressed_action_latency_cannot_advance(self):
        self.regress=True;self.run_comparison();self.assertEqual(self.qualifications[0],['off'])
    def test_failed_control_smokes_do_not_prejudge_speculative_arms(self):
        self.control_smoke=False;out=self.run_comparison();self.assertEqual(len(out),6)
        self.assertEqual(self.qualifications[0],['draft-mtp','draft-dflash'])
        self.assertTrue(all(s['eligible'] is False for s in out if s['settings']['speculation']=='off'))
        self.assertEqual(len(self.state.get_control('speculation_comparison_phase')['arm_outcomes']),4)
    def test_speculative_arms_still_require_their_own_six_smokes(self):
        self.smoke=False;out=self.run_comparison()
        self.assertEqual(self.qualifications[0],['off'])
        self.assertTrue(all(s['eligible'] is False for s in out if s['settings']['speculation']!='off'))
    def test_discovery_cap_and_three_family_limit_are_preserved(self):
        self.state.control('discovered_weight_variants',['old1','old2','old3','old4'])
        out=self.run_comparison();self.assertFalse(any(s['settings']['speculation']=='draft-dflash' for s in out))
        self.state.control('tuning_phase',dict(started=900,deadline=100000,selected_families=['other1','other2','other3']))
        with self.assertRaisesRegex(RuntimeError,'three_family'):self.run_comparison()
    def test_original_phase_budget_stops_new_launches(self):
        self.state.control('tuning_phase',dict(started=900,deadline=1100,selected_families=[ns.TARGET]))
        out=self.run_comparison();self.assertEqual(out,[]);self.assertEqual(self.launched,[])
    def test_changed_plan_is_rejected_and_baseline_resume_does_not_expand_search(self):
        self.run_comparison();phase=self.state.get_control('speculation_comparison_phase')
        source=[self.anchor]+list(self.completed.values())
        self.assertEqual(ss.baseline_quality_screens(self.state,source),[self.anchor])
        self.plan.write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'changed_requires_review'):self.run_comparison()
    def test_expired_phase_retains_completed_screens_on_resume(self):
        first=self.run_comparison()
        for result in first:self.state.control('planned_screen:'+result['configuration_id'],result)
        self.state.clock=lambda:100001
        resumed=self.run_comparison()
        self.assertEqual({s['configuration_id'] for s in resumed},{s['configuration_id'] for s in first})
        self.assertEqual(len(self.launched),6)
        self.assertEqual(len(self.state.get_control('speculation_comparison_phase')['screens']),6)
    def test_every_child_budget_check_obeys_phase_and_is_restored_afterwards(self):
        original=self.state.check_budget
        with ss.phase_budget(self.state,1500):
            self.state.check_budget(400)
            with self.assertRaises(ss.PhaseDeadline):self.state.check_budget(600)
        self.assertEqual(self.state.check_budget,original)
        self.state.check_budget(600)

if __name__=='__main__':unittest.main()
