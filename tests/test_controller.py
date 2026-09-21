import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from localbench.config import load
from localbench.state import State
from localbench.regional_prompt import regional_prompt
from localbench.scheduler import measured_job, key_base, make_native, register_configuration


class Adapter:
    def __init__(self):
        self.http = SimpleNamespace(timeout=900)
    engine='bundled-llama'
    model={'id':'test'}
    requested={'reasoning':'default','context':65536,'slots':1}
    effective={'effective_context':65536}
    def metadata(self):
        return {'binary_sha256':'binary','adjacent_dlls':{},'version_output':'version', 'model_sha256':'weights'}
    def tokenize(self,messages,tools):
        return {'count':30+sum(len(m['content']) for m in messages),'provenance':'exact mock'}


class Collector:
    def snapshot(self,*args):return {'foreign_workload_overlap':False}


class ControllerTests(unittest.TestCase):
    def state(self,root):
        c=load()
        for name in ('state','logs','artifacts'):c['paths'][name]=str(Path(root)/name)
        s=State(c);s.control('harness_verified',True)
        return c,s

    def test_all_controller_modules_import_without_probes(self):
        for module in ('quality','measurement','scheduler','regional_prompt','acquisition'):
            importlib.import_module('localbench.'+module)

    def test_regions_use_exact_template_counts_and_preserve_authorities(self):
        adapter=Adapter()
        p=regional_prompt(adapter,[], 'system', ['ACTIVE early','ACTIVE middle','ACTIVE late'], 'final query',42,4096,0)
        self.assertEqual(p['expected_tokens'],4096)
        self.assertEqual(adapter.tokenize(p['messages'],[])['count'],4096)
        for marker in ('ACTIVE early','ACTIVE middle','ACTIVE late','final query'):
            self.assertIn(marker,p['messages'][1]['content'])
        for position,fraction in zip(p['region_positions'],(.05,.5,.95)):
            self.assertLess(abs(position['fraction']-fraction),.001)

    def test_packed_job_resume_avoids_remeasurement_and_preserves_clock(self):
        with tempfile.TemporaryDirectory() as root:
            c,s=self.state(root);a=Adapter();a.state=s;collector=Collector();calls=[]
            def operation(cb):
                calls.append(True);cb({'prompt_hash':'actual-template-hash'})
                return {'kind':'capacity','passed':True,'valid':True}
            first=measured_job(c,s,a,collector,'cfg','capacity',17,'fresh',0,61440,operation)
            second=measured_job(c,s,a,collector,'cfg','capacity',17,'fresh',0,61440,operation)
            self.assertEqual(first,second);self.assertEqual(len(calls),1)
            self.assertIsNone(s.run['started'])
            self.assertEqual(s.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],1)
            s.close()

    def test_packing_error_is_durable_without_a_fabricated_prompt(self):
        with tempfile.TemporaryDirectory() as root:
            c,s=self.state(root);a=Adapter();a.state=s
            def operation(cb):raise RuntimeError('missing_exact_template_endpoint')
            result=measured_job(c,s,a,Collector(),'cfg','capacity',17,'fresh',0,61440,operation)
            self.assertFalse(result['valid']);self.assertEqual(result['reason'],'missing_exact_template_endpoint')
            self.assertEqual(s.db.execute('SELECT status FROM jobs').fetchone()[0],'invalid')
            s.close()

    def test_failed_launch_cleans_owned_handle(self):
        with tempfile.TemporaryDirectory() as root:
            c,s=self.state(root);a=Adapter();a.discover_capabilities=lambda:None
            a.launch=lambda:(_ for _ in ()).throw(RuntimeError('model_oom'))
            cleanup=[];a.unload_owned=lambda:cleanup.append(True)
            with patch('localbench.scheduler.model_identity',return_value={}),patch('localbench.scheduler.NativeAdapter',return_value=a):
                with self.assertRaisesRegex(RuntimeError,'model_oom'):make_native(c,s,Collector(),'bundled-llama',{})
            self.assertEqual(cleanup,[True]);s.close()

    def test_stable_configuration_identity_ignores_new_launch_logs(self):
        with tempfile.TemporaryDirectory() as root:
            c,s=self.state(root);a=Adapter()
            first=register_configuration(c,s,a)
            original=a.metadata
            a.metadata=lambda:{**original(),'effective_settings':{'startup_seconds':9,'startup_logs':['new.log']}}
            second=register_configuration(c,s,a)
            self.assertEqual(first,second)
            self.assertEqual(s.db.execute("SELECT COUNT(*) FROM entities WHERE kind='configuration'").fetchone()[0],1)
            s.close()
