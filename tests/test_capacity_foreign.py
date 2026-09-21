"""Functional capacity survives graphics overlap; speed samples do not."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from localbench.config import load
from localbench.report import Report
from localbench.scheduler import measured_job
from localbench.state import State
from tests import test_report_resources as report_fixtures


class CapacityForeignTests(unittest.TestCase):
    evidence=report_fixtures.ReportResourceTests.evidence
    add=report_fixtures.ReportResourceTests.add
    full_candidate=report_fixtures.ReportResourceTests.full_candidate

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.config=load()
        for name in ('project','state','logs','artifacts'):
            self.config['paths'][name]=str(Path(self.temp.name)/name)
        self.state=State(self.config)
        self.counter=0
        self.adapter=SimpleNamespace(engine='inert-test-engine',model={'id':'inert-test-model'},
            http=SimpleNamespace(timeout=self.config['limits']['per_64k_request_timeout_seconds']))
        self.collector=SimpleNamespace(snapshot=lambda *args:{'foreign_workload_overlap':True,'active_foreign_gpu_pids':[17596]})

    def tearDown(self):self.state.close();self.temp.cleanup()

    def run_job(self,kind,result):
        self.counter+=1
        key={'engine':'inert-test-engine','model':'inert-test-model','effective_settings':{'context':65536},
            'profile':{},'tokenizer':'test-only','seed':self.counter,'mode':'fresh','replicate':0,
            'configuration_id':'candidate','kind':kind}
        def operation(callback):
            callback({'prompt_hash':'test-prompt-'+str(self.counter)})
            return copy.deepcopy(result)
        with patch('localbench.scheduler.key_base',return_value=key):
            return measured_job(self.config,self.state,self.adapter,self.collector,'candidate',kind,self.counter,'fresh',0,61440,operation)

    def capacity(self,markers=3):
        return {'kind':'capacity','configuration_id':'candidate','seed':17,'passed':markers==3,'valid':True,
            'reason':'passed' if markers==3 else 'capacity_marker_or_evidence_failure',
            'markers_passed':markers,'retrieved_values':['a','b','c'],'token_evidence':self.evidence()}

    def test_successful_capacity_is_durably_passed_with_diagnostic_latency_and_no_retry(self):
        result=self.run_job('capacity',self.capacity())
        self.assertTrue(result['passed']);self.assertTrue(result['valid'])
        self.assertFalse(result['timing_valid']);self.assertTrue(result['latency_diagnostic_only'])
        self.assertEqual(result['timing_invalid_reason'],'foreign_workload_overlap')
        self.assertEqual(result['markers_passed'],3)
        self.assertEqual(self.state.db.execute('SELECT status FROM attempts').fetchone()[0],'passed')
        self.assertEqual(self.state.db.execute('SELECT status FROM jobs').fetchone()[0],'passed')

    def test_wrong_marker_and_bad_context_preserve_failure_instead_of_creating_a_pass(self):
        wrong=self.run_job('capacity',self.capacity(markers=2))
        self.assertFalse(wrong['passed']);self.assertTrue(wrong['valid'])
        self.assertEqual(wrong['reason'],'capacity_marker_or_evidence_failure')
        invalid=self.capacity();invalid.update(passed=False,valid=False,reason='actual_prompt_count_mismatch')
        invalid['token_evidence']['prompt_tokens']=1000
        context=self.run_job('capacity',invalid)
        self.assertFalse(context['passed']);self.assertFalse(context['valid'])
        statuses=[row[0] for row in self.state.db.execute('SELECT status FROM attempts ORDER BY id')]
        self.assertEqual(statuses,['failed','invalid'])

    def test_timing_throughput_and_common_byte_remain_invalid_transient_samples(self):
        for kind in ('timing','throughput','common_byte'):
            with self.subTest(kind=kind):
                result=self.run_job(kind,{'kind':kind,'passed':True,'valid':True,'reason':'passed'})
                self.assertFalse(result['passed']);self.assertFalse(result['valid'])
                self.assertEqual(result['reason'],'foreign_workload_overlap')
                self.assertFalse(result['timing_valid'])
        self.assertEqual([row[0] for row in self.state.db.execute('SELECT status FROM attempts')],['invalid']*3)
        self.assertEqual([row[0] for row in self.state.db.execute('SELECT status FROM jobs')],['pending']*3)

    def test_report_keeps_foreign_capacity_gate_and_never_ranks_its_speed(self):
        self.full_candidate('candidate',fresh_seconds=10,speed=64)
        for seed in self.config['grading']['capacity_marker_seeds']:
            self.add('candidate',{'kind':'capacity','seed':seed,'markers_passed':3,'retrieved_values':['a','b','c'],
                'token_evidence':self.evidence(),'resources':{'foreign_workload_overlap':True},
                'timing_valid':False,'latency_diagnostic_only':True,
                'metrics':{'first_action_seconds':.001,'generated_tokens':1000000,'decode_seconds':1,
                    'sustained_throughput_qualified':True}})
        report=Report(self.config,self.state);score=report.summarize()['configurations'][0]
        self.assertTrue(score['gates']['capacity_all_nine_values']);self.assertTrue(score['qualified'])
        self.assertEqual(score['latency']['fresh']['first_action']['p95'],10)
        self.assertEqual(score['throughput']['median'],64)
        before=[tuple(row) for row in self.state.db.execute('SELECT id,status FROM attempts')]
        report.write()
        self.assertEqual([tuple(row) for row in self.state.db.execute('SELECT id,status FROM attempts')],before)

    def test_report_still_rejects_marker_or_context_failure_and_keeps_old_invalid_history(self):
        for change in ('wrong_marker','bad_context','historical_invalid'):
            with self.subTest(change=change):
                result=self.capacity();result['configuration_id']=change;result['resources']={'foreign_workload_overlap':True}
                if change=='wrong_marker':result['markers_passed']=2
                if change=='bad_context':result['token_evidence']['truncated']=True
                status='invalid' if change=='historical_invalid' else 'passed'
                self.add(change,result,status=status,reason='foreign_workload_overlap' if status=='invalid' else None)
                rows=Report(self.config,self.state).attempts()
                selected=[row for row in rows if row['configuration_id']==change]
                self.assertEqual(len(selected),1)
                score=Report(self.config,self.state)._configuration(change,{},selected)
                self.assertFalse(score['gates']['capacity_all_nine_values'])
                if change=='historical_invalid':self.assertEqual(selected[0]['status'],'invalid')


if __name__=='__main__':unittest.main()
