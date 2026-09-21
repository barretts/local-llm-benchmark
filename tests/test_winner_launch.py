"""Winner launcher exports use deterministic grade rows and inert pinned files."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from localbench.config import atomic_json,digest,file_hash,load
from localbench.handoff import _config_identity,_demo_key,_pin,_source_pins
from localbench.report import Report
from localbench.scheduler import engine_identity,tokenizer_identity
from localbench.state import State
from localbench.winner_launch import generate_winner_launcher
import test_report_resources as report_fixtures


class WinnerLaunchTests(unittest.TestCase):
    evidence=report_fixtures.ReportResourceTests.evidence
    add=report_fixtures.ReportResourceTests.add
    full_candidate=report_fixtures.ReportResourceTests.full_candidate

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="winner test's ")
        self.config=load()
        for name in ('project','state','logs','artifacts'):
            self.config['paths'][name]=str(Path(self.temp.name)/name)
        self.config['paths']['project']=self.temp.name
        self.state=State(self.config)
        self.state.control('harness_verified',True)
        self.state.start_execution('runtime_probe:unit-test-only-no-engine')
        self.counter=0
        self.identifier='c'*64
        self.full_candidate(self.identifier)
        self.model_file=Path(self.temp.name)/'fixture-model.gguf'
        self.binary=Path(self.temp.name)/'inert-server.exe'
        self.model_file.write_bytes(b'not a model: static unit-test pin')
        self.binary.write_bytes(b'not an executable: static unit-test pin')
        self.script=Path(self.config['paths']['project'])/'scripts'/'launch-winner.ps1'

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def prepare_plan(self,kind='native'):
        engine='lm-studio' if kind=='lm-studio' else 'upstream-llama-nightly'
        metadata={'engine_id':engine,'binary_sha256':file_hash(self.binary),'adjacent_dlls':{},
            'version_output':'unit-test pinned version','model_sha256':file_hash(self.model_file),
            'requested_settings':{'context':65536,'slots':1,'reasoning':'default'},
            'effective_settings':{'effective_context':65536,'effective_slots':1}}
        if kind=='lm-studio':metadata['tokenizer_helper_sha256']='e'*64
        self.state.entity('configuration',self.identifier,metadata)
        model={'id':'fixture-model','path':str(self.model_file),'sha256':metadata['model_sha256']}
        self.state.entity('model',model['id'],model)
        plan={'run_id':self.state.run['id'],'configuration_id':self.identifier,'engine':engine,
            'model':model,'runtime':{'engine':engine,'kind':kind,'binary_path':str(self.binary)},
            'requested_settings':metadata['requested_settings'],'config_identity':_config_identity(self.config,self.state),
            'engine_identity':engine_identity(metadata),'model_identity':metadata['model_sha256'],
            'tokenizer_identity':tokenizer_identity(metadata),'file_pins':_source_pins()+[
                _pin(self.binary,'runtime'),_pin(self.model_file,'model')],
            'interface':{'endpoint':'http://127.0.0.1:1234' if kind=='lm-studio' else 'http://127.0.0.1:38201'},
            'observed_launch':{'owned_unit_test_only':True}}
        self.envelope={'schema_version':1,'plan_id':digest(plan),'plan':plan}
        self.plan_file=Path(self.config['paths']['artifacts'])/'launch-plans'/(self.envelope['plan_id']+'.json')
        atomic_json(self.plan_file,self.envelope)
        self.state.entity('handoff_plan',self.identifier,{'path':str(self.plan_file),
            'plan_id':self.envelope['plan_id'],'original_instance':{'stopped':True}})
        demo={'kind':'demo','configuration_id':self.identifier,'passed':True,'valid':True,'isolated':True,
            'plan_id':self.envelope['plan_id'],'demo_instance':{'stopped':True},
            'transcript':{'files':[_pin(self.model_file,'unit_test_transcript')],
                'manifest':_pin(self.binary,'unit_test_transcript_manifest')}}
        self.demo_job=self.state.enqueue(_demo_key(self.config,plan,self.envelope['plan_id']))
        self.state.finish(self.state.begin(self.demo_job),'passed',demo)
        self.state.entity('demo',self.identifier,demo)
        return self.envelope

    def snapshot(self):
        return {table:[tuple(row) for row in self.state.db.execute('SELECT * FROM '+table)]
            for table in ('runs','jobs','attempts','weights','entities','controls')}

    def generate(self):
        return generate_winner_launcher(self.config,self.state,Report(self.config,self.state).summarize())

    def test_full_qualified_plan_exports_literal_serving_script_without_db_or_runtime_changes(self):
        self.prepare_plan()
        before=self.snapshot()
        with patch('localbench.handoff.make_adapter') as runtime:
            result=self.generate()
        runtime.assert_not_called()
        self.assertEqual(result['status'],'generated')
        self.assertEqual(result['plan_file'],str(self.plan_file))
        self.assertFalse(result['runtime_probe_executed'])
        self.assertEqual(self.snapshot(),before)
        text=self.script.read_text(encoding='utf-8')
        self.assertIn("'agent-endpoint', '--plan', $benchmarkPlanFile",text)
        self.assertIn(str(self.plan_file).replace("'","''"),text)
        self.assertIn(self.envelope['plan']['model_identity'],text)
        self.assertIn('Exact requested settings',text)
        self.assertIn('-WindowStyle Hidden',text)
        self.assertNotIn('--engine',text)
        self.assertEqual(result['script_sha256'],file_hash(self.script))

    def test_report_hook_exports_only_with_current_registered_qualified_plan(self):
        self.prepare_plan()
        result=Report(self.config,self.state).write()
        self.assertEqual(result['winner_launcher']['status'],'generated')
        exported=json.loads((Path(self.config['paths']['artifacts'])/'ranking.json').read_text())
        self.assertEqual(exported['winner_launcher']['plan_id'],self.envelope['plan_id'])
        self.assertTrue(self.script.is_file())

    def test_quality_failure_or_missing_handoff_never_creates_successful_launcher(self):
        self.assertEqual(self.generate()['reason'],'registered_handoff_plan_missing')
        self.assertFalse(self.script.exists())
        self.prepare_plan()
        self.assertEqual(self.generate()['status'],'generated')
        with self.state.db:
            self.state.db.execute("UPDATE jobs SET status='invalid' WHERE id=?",(self.demo_job,))
        result=self.generate()
        self.assertEqual(result['status'],'unavailable')
        self.assertFalse(self.script.exists())

    def test_registered_plan_config_or_source_change_refuses_export(self):
        self.prepare_plan()
        original_execution_hash=self.config.get('_execution_hash')
        self.config['_execution_hash']='different-user-authorized-effective-path'
        self.assertEqual(self.generate()['reason'],'winner_plan_or_pinned_proof_invalid')
        self.config['_execution_hash']=original_execution_hash
        self.binary.write_bytes(b'changed pinned executable bytes')
        self.assertEqual(self.generate()['reason'],'winner_plan_or_pinned_proof_invalid')
        self.assertFalse(self.script.exists())

    def test_demo_transcript_missing_refuses_export_even_when_boolean_grades_qualify(self):
        self.prepare_plan()
        row=self.state.db.execute('SELECT result FROM jobs WHERE id=?',(self.demo_job,)).fetchone()
        demo=json.loads(row['result']);demo['transcript']={}
        with self.state.db:self.state.db.execute('UPDATE jobs SET result=? WHERE id=?',(json.dumps(demo),self.demo_job))
        self.assertEqual(self.generate()['reason'],'winner_plan_or_pinned_proof_invalid')
        self.assertFalse(self.script.exists())

    def test_manual_launcher_edits_are_preserved_and_never_overwritten(self):
        self.prepare_plan();self.assertEqual(self.generate()['status'],'generated')
        self.script.write_text(self.script.read_text()+"# user's local note\n",encoding='utf-8')
        edited=self.script.read_bytes()
        result=self.generate()
        self.assertEqual(result['reason'],'launcher_path_occupied_or_edited')
        self.assertTrue(result['existing_script_preserved'])
        self.assertEqual(self.script.read_bytes(),edited)

    def test_expired_stopped_benchmark_exports_without_resetting_clock_or_stop(self):
        self.prepare_plan()
        with self.state.db:self.state.db.execute('UPDATE runs SET started=100,deadline=200,stop_requested=1')
        before=self.snapshot()
        self.assertEqual(self.generate()['status'],'generated')
        self.assertEqual(self.snapshot(),before)

    def test_lm_script_uses_masked_ephemeral_env_without_reading_or_serializing_a_key(self):
        self.prepare_plan('lm-studio')
        with patch.dict(os.environ,{'LM_BENCH_TOKEN':'fixture-secret-never-exported'}):
            result=self.generate()
        self.assertEqual(result['status'],'generated')
        text=self.script.read_text()
        self.assertIn('$benchmarkRequiresLMToken = $true',text)
        self.assertIn('-AsSecureString',text)
        self.assertIn('ZeroFreeBSTR',text)
        self.assertIn('finally {',text)
        self.assertNotIn('fixture-secret-never-exported',text)
        self.assertNotIn('fixture-secret-never-exported',json.dumps(result))
        argument_line=next(line for line in text.splitlines() if '$benchmarkArguments = @' in line)
        self.assertNotIn('TOKEN',argument_line)
        self.assertNotIn('credential',argument_line.lower())

    def test_generated_powershell_parses_without_invoking_endpoint_or_prompt(self):
        powershell=Path(shutil.which('pwsh') or 'C:/Program Files/PowerShell/7/pwsh.exe')
        if not powershell.is_file():self.skipTest('PowerShell7 unavailable for AST verification')
        self.prepare_plan('lm-studio');self.assertEqual(self.generate()['status'],'generated')
        script_path=str(self.script).replace("'","''")
        command="$tokens=$null; $errors=$null; [void][System.Management.Automation.Language.Parser]::ParseFile('"+script_path+"',[ref]$tokens,[ref]$errors); if ($errors.Count) { exit 1 }"
        result=subprocess.run([str(powershell),'-NoProfile','-NonInteractive','-Command',command],
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))


if __name__=='__main__':unittest.main()
