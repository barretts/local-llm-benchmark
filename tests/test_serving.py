"""Serving authority/lifecycle tests: temporary state and mocked I/O only."""
import copy
import http.client
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from localbench.config import atomic_json,digest,file_hash,load
from localbench.handoff import _config_identity
from localbench.runner import dispatch
from localbench.serving import ServingState,_StartupResponse,run_endpoint
from localbench.state import State


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.config=load()
        for name in ('project','state','logs','artifacts'):
            self.config['paths'][name]=str(Path(self.temp.name)/name)
        self.state=State(self.config,clock=lambda:1000.)
        for name in ('doctor_verified','harness_verified','fixtures_verified'):
            self.state.control(name,True)
        self.state.control('fixture_version','fixed-fixtures')
        self.state.start_execution('runtime_probe:upstream-llama-nightly')
        key={'engine':'engine','model':'model','effective_settings':{},'profile':{},
            'fixture_prompt_hash':'fixture','tokenizer':'tokenizer','seed':42,'mode':'fresh','replicate':0}
        job=self.state.enqueue(key)
        self.state.finish(self.state.begin(job),'passed',{'passed':True,'valid':True})
        with self.state.db:self.state.db.execute('UPDATE runs SET started=100,deadline=200,stop_requested=1')
        self.source=Path(self.temp.name)/'model.gguf'
        self.source.write_bytes(b'fixed model fixture')
        self.sha=file_hash(self.source)
        self.identifier='c'*64
        self.model={'id':'fixture-model','path':str(self.source),'sha256':self.sha}
        self.configuration={'identity':{'pinned':'exact'},'configuration_id':self.identifier}
        self.state.entity('model',self.model['id'],self.model)
        self.state.entity('configuration',self.identifier,self.configuration)
        self.plan={'run_id':self.state.run['id'],'configuration_id':self.identifier,
            'engine':'upstream-llama-nightly','model':self.model,
            'runtime':{'engine':'upstream-llama-nightly','kind':'native','binary_path':str(self.source)},
            'requested_settings':{'context':65536},'config_identity':_config_identity(self.config,self.state),
            'engine_identity':{'binary_sha256':'a'*64},'model_identity':self.sha,'tokenizer_identity':'a'*64,
            'file_pins':[],'interface':{},'observed_launch':{}}
        self.plan_id=digest(self.plan)
        self.plan_path=Path(self.temp.name)/'immutable-plan.json'
        atomic_json(self.plan_path,{'schema_version':1,'plan_id':self.plan_id,'plan':self.plan})
        self.state.entity('handoff_plan',self.identifier,{'plan_id':self.plan_id,'path':str(self.plan_path),
            'original_instance':{'stopped':True}})
        pin={'path':str(self.source),'role':'demo_fixture','sha256':self.sha,'bytes':self.source.stat().st_size}
        self.demo={'passed':True,'valid':True,'isolated':True,'plan_id':self.plan_id,
            'demo_instance':{'stopped':True},'transcript':{'files':[pin],'manifest':pin}}
        self.state.entity('demo',self.identifier,self.demo)
        self.mono=100.
        self.facades=[]
        self.qualified=patch('localbench.report.Report._configuration',return_value={'qualified':True})
        self.qualified.start()

    def tearDown(self):
        self.qualified.stop()
        for facade in self.facades:
            try:facade.close()
            except sqlite3.ProgrammingError:pass
        self.state.close()
        self.temp.cleanup()

    def facade(self):
        value=ServingState(self.config,self.state,self.plan_path,monotonic=lambda:self.mono)
        self.facades.append(value)
        return value

    def benchmark_tables(self):
        return {name:[tuple(row) for row in self.state.db.execute('SELECT * FROM '+name)]
            for name in ('jobs','attempts','weights')}

    def test_expired_pinned_serving_restart_preserves_original_clock_and_all_jobs(self):
        tables=self.benchmark_tables()
        original=self.state.run
        facade=self.facade()
        facade.activate_restart()
        facade.verify_demo_pins()
        facade.start_execution('runtime_probe:upstream-llama-nightly')
        facade.check_budget(300)
        facade.finish_startup()
        self.assertEqual(facade.run['started'],original['started'])
        self.assertEqual(facade.run['deadline'],original['deadline'])
        self.assertEqual(facade.run['stop_requested'],0)
        self.assertEqual(self.benchmark_tables(),tables)
        self.assertEqual(self.state.get_control('clock_start_reason'),'runtime_probe:upstream-llama-nightly')
        self.assertEqual(self.state.get_control('serving_session')['status'],'serving')
        self.assertEqual(self.config['limits']['server_start_timeout_seconds'],300)

    def test_live_future_stop_is_not_masked_by_restart(self):
        facade=self.facade();facade.activate_restart()
        self.state.stop()
        self.assertEqual(facade.run['stop_requested'],1)
        with self.assertRaisesRegex(RuntimeError,'stop_after_current'):facade.check_budget()
        with self.assertRaisesRegex(RuntimeError,'already_activated'):facade.activate_restart()
        self.assertEqual(facade.run['stop_requested'],1)

    def test_unstarted_unqualified_or_unregistered_plan_cannot_clear_old_stop(self):
        cases=('unstarted','unqualified','unregistered','demo_invalid')
        for case in cases:
            with self.subTest(case=case):
                with self.state.db:self.state.db.execute('UPDATE runs SET started=100,deadline=200,stop_requested=1')
                self.state.entity('handoff_plan',self.identifier,{'plan_id':self.plan_id,'path':str(self.plan_path),'original_instance':{'stopped':True}})
                self.state.entity('demo',self.identifier,self.demo)
                if case=='unstarted':
                    with self.state.db:self.state.db.execute('UPDATE runs SET started=NULL,deadline=NULL')
                elif case=='unregistered':
                    with self.state.db:self.state.db.execute("DELETE FROM entities WHERE kind='handoff_plan'")
                elif case=='demo_invalid':self.state.entity('demo',self.identifier,{**self.demo,'valid':False})
                with patch('localbench.report.Report._configuration',return_value={'qualified':case!='unqualified'}):
                    with self.assertRaises(RuntimeError):self.facade()
                self.assertEqual(self.state.run['stop_requested'],1)

    def test_new_configuration_weight_reservation_jobs_and_direct_sql_are_forbidden(self):
        facade=self.facade();facade.activate_restart()
        tables=self.benchmark_tables()
        for action in (lambda:facade.reserve_weight('new',1,'copy','new'),lambda:facade.enqueue({}),
            lambda:facade.begin('job'),lambda:facade.finish(1,'passed'),facade.recover,
            lambda:facade.entity('configuration','d'*64,self.configuration),
            lambda:facade.entity('configuration',self.identifier,{'identity':{'changed':True}}),
            lambda:facade.entity('runtime_catalog','new',{}),
            lambda:facade.start_execution('runtime_image_probe_setup:new'),
            lambda:facade.start_execution('runtime_probe:unpinned-engine'),
            lambda:facade.entity('engine','unpinned',{'engine_id':'unpinned-engine'}),
            lambda:facade.control('engine_probe_budget:new',{})):
            with self.assertRaises(RuntimeError):action()
        for sql in ("UPDATE jobs SET status='invalid'","DELETE FROM weights","PRAGMA user_version=99",
                    "CREATE TEMP TABLE unauthorized(x)","WITH x AS (SELECT 1) DELETE FROM attempts"):
            with self.assertRaises(sqlite3.DatabaseError):facade.db.execute(sql)
        self.assertEqual(facade.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],1)
        self.assertEqual(self.benchmark_tables(),tables)

    def test_existing_metadata_and_ownership_evidence_allowed_only_at_exact_identity(self):
        facade=self.facade();facade.activate_restart()
        facade.entity('configuration',self.identifier,{**self.configuration,'effective_settings':{'effective_context':65536}})
        facade.entity('model',self.model['id'],self.model)
        facade.entity('handoff_launch','launch-proof',{'plan_id':self.plan_id})
        facade.control('live_agent_endpoint',{'status':'serving'})
        facade.control('model_hash:'+str(self.source),{'sha256':self.sha})
        facade.control('runtime_first_probe:new-helper',1000)
        self.assertIsNone(self.state.get_control('runtime_first_probe:new-helper'))
        with self.assertRaises(RuntimeError):facade.entity('model',self.model['id'],{**self.model,'sha256':'d'*64})
        with self.assertRaises(RuntimeError):facade.entity('ollama_weight_import',self.sha,{'additional_physical_weight_bytes':1})

    def test_one_startup_deadline_cannot_be_extended_and_copied_timeouts_shrink(self):
        facade=self.facade();facade.activate_restart()
        self.mono+=250
        facade.check_budget(300)
        self.assertEqual(facade.config['limits']['server_start_timeout_seconds'],50)
        self.assertLessEqual(facade.config['limits']['per_64k_request_timeout_seconds'],50)
        with self.assertRaisesRegex(RuntimeError,'exceeds_existing_timeout'):facade.check_budget(301)
        self.mono+=50
        with self.assertRaisesRegex(RuntimeError,'serving_startup_timeout'):facade.finish_startup()
        self.assertEqual(self.state.run['deadline'],200)

    def test_startup_command_timeout_is_capped_acquisition_refused_and_hooks_restored(self):
        facade=self.facade();facade.activate_restart();self.mono+=280
        fake_run=Mock(return_value=SimpleNamespace(returncode=0))
        with patch('subprocess.run',fake_run):
            original=subprocess.run
            with facade.startup_io():
                subprocess.run(['engine.exe','--help'],timeout=90)
                with self.assertRaisesRegex(RuntimeError,'acquisition_or_build'):
                    subprocess.run(['docker.exe','pull','new-image'],timeout=10)
                worker=threading.Thread(target=lambda:subprocess.run(['collector.exe'],timeout=90))
                worker.start();worker.join()
            self.assertIs(subprocess.run,original)
        self.assertEqual(fake_run.call_args_list[0].kwargs['timeout'],20)
        self.assertEqual(fake_run.call_args_list[1].kwargs['timeout'],90)
        self.assertFalse(facade._io_active)

    def test_slow_http_stream_cannot_run_past_total_startup_deadline(self):
        facade=self.facade();facade.activate_restart()
        def chunk(amount):self.mono+=160;return b'chunk'
        response=SimpleNamespace(read1=Mock(side_effect=chunk),fp=None,status=200)
        with self.assertRaisesRegex(RuntimeError,'serving_startup_timeout'):
            _StartupResponse(response,facade).read()
        self.assertEqual(response.read1.call_count,2)

    def test_http_connection_and_reads_keep_the_smaller_existing_timeout(self):
        facade=self.facade();facade.activate_restart();self.mono+=290
        socket=SimpleNamespace(settimeout=Mock(),gettimeout=Mock(return_value=2))
        connection=SimpleNamespace(timeout=900,sock=socket)
        response=SimpleNamespace(status=200,fp=SimpleNamespace(raw=SimpleNamespace(_sock=socket)),
            read1=Mock(side_effect=[b'payload',b'']))
        with patch.object(http.client.HTTPConnection,'request') as request,patch.object(http.client.HTTPConnection,'getresponse',return_value=response):
            with facade.startup_io():
                http.client.HTTPConnection.request(connection,'GET','/v1/models')
                wrapped=http.client.HTTPConnection.getresponse(connection)
                self.assertEqual(wrapped.read(),b'payload')
        self.assertEqual(connection.timeout,10)
        self.assertEqual(socket.settimeout.call_args_list[-1].args,(2,))
        request.assert_called_once_with(connection,'GET','/v1/models')

    def test_startup_failure_preserves_old_endpoint_descriptor_and_benchmark_history(self):
        tables=self.benchmark_tables()
        prior={'status':'stopped','configuration_id':'prior','serving_session_id':'prior-session'}
        self.state.control('live_agent_endpoint',prior)
        with patch('localbench.recovery.recover_processes',return_value=[{'status':'refused'}]),patch('localbench.doctor.doctor') as doctor,patch('localbench.resources.ResourceCollector') as collector,patch('localbench.endpoint.serve') as launch:
            with self.assertRaisesRegex(RuntimeError,'owned_recovery_unresolved'):
                run_endpoint(self.config,self.state,self.plan_path)
        doctor.assert_not_called();collector.assert_not_called();launch.assert_not_called()
        self.assertEqual(self.state.get_control('live_agent_endpoint'),prior)
        self.assertEqual(self.state.get_control('serving_session')['status'],'startup_failed')
        self.assertEqual(self.benchmark_tables(),tables)
        self.assertEqual(self.state.run['started'],100)
        self.assertEqual(self.state.run['deadline'],200)

    def test_vanished_demo_pin_refuses_startup_without_engine_or_acquisition(self):
        self.source.unlink()
        with patch('localbench.recovery.recover_processes') as recovery,patch('localbench.endpoint.serve') as launch:
            with self.assertRaises(FileNotFoundError):run_endpoint(self.config,self.state,self.plan_path)
        recovery.assert_not_called();launch.assert_not_called()
        self.assertEqual(self.state.get_control('serving_session')['status'],'startup_failed')

    def test_pending_lm_studio_load_intent_blocks_another_endpoint_before_doctor_or_gpu(self):
        tables=self.benchmark_tables()
        with patch('localbench.recovery.recover_processes',return_value=[{'status':'pending_load','reason':'owned_load_may_still_complete'}]),patch('localbench.doctor.doctor') as doctor,patch('localbench.resources.ResourceCollector') as collector,patch('localbench.endpoint.serve') as launch:
            with self.assertRaisesRegex(RuntimeError,'owned_recovery_unresolved'):
                run_endpoint(self.config,self.state,self.plan_path)
        doctor.assert_not_called();collector.assert_not_called();launch.assert_not_called()
        self.assertEqual(self.benchmark_tables(),tables)
        self.assertEqual(self.state.run['started'],100)
        self.assertEqual(self.state.run['deadline'],200)
        self.assertEqual(self.state.get_control('serving_ownership_recovery')[0]['status'],'pending_load')

    def test_runner_endpoint_uses_serving_branch_inside_gpu_lease_without_recover_or_scheduler(self):
        lease=Mock()
        lease.__enter__=Mock();lease.__exit__=Mock(return_value=False)
        with patch('localbench.runner.gpu_lock',return_value=lease),patch('localbench.serving.run_endpoint',return_value={'status':'fake_served'}) as endpoint,patch('localbench.scheduler.run') as scheduler,patch.object(self.state,'recover') as recover:
            result=dispatch(self.config,self.state,SimpleNamespace(command='agent-endpoint',plan=self.plan_path))
        self.assertEqual(result,{'status':'fake_served'})
        lease.__enter__.assert_called_once()
        endpoint.assert_called_once_with(self.config,self.state,self.plan_path)
        recover.assert_not_called();scheduler.assert_not_called()

    def test_operational_endpoint_runs_reinspection_then_observes_later_stop(self):
        tables=self.benchmark_tables()
        process=SimpleNamespace(pid=12345,poll=Mock(return_value=None))
        adapter=SimpleNamespace(engine='upstream-llama-nightly',http=SimpleNamespace(port=39999),
            effective={'effective_context':65536,'effective_slots':1},saved={'owner':'localbench'},process=process,
            instance_id=None,health=Mock(return_value=True),handoff_launch_evidence={'interface':{
                'endpoint':'http://127.0.0.1:39999','chat_path':'/v1/chat/completions','models_path':'/v1/models',
                'wire_protocol':'OpenAI-compatible streaming SSE','model':'local-model'}})
        adapter.unload_owned=Mock(side_effect=lambda:process.poll.configure_mock(return_value=0))
        collector=SimpleNamespace(start=Mock(),stop=Mock());collector.start.return_value=collector
        def doctor(config,hash_weights):
            self.assertFalse(hash_weights)
            atomic_json(Path(config['paths']['artifacts'])/'host.json',{'commands':{'lms_loaded':{'output':'[]'}}})
            return {'disk_headroom_ok':True}
        def launched(config,state,collector,path):
            self.assertIsInstance(state,ServingState)
            state.check_budget()
            return adapter,self.identifier
        with patch('localbench.doctor.doctor',side_effect=doctor) as inspect_host,patch('localbench.recovery.recover_processes',return_value=[]) as recovery,patch('localbench.resources.ResourceCollector',return_value=collector),patch('localbench.handoff.launch_plan',side_effect=launched),patch('localbench.endpoint.time.sleep',side_effect=lambda _:self.state.stop()):
            # serve's default sleep binds at definition; replace it explicitly.
            from localbench.endpoint import serve
            with patch('localbench.endpoint.serve',side_effect=lambda c,s,r,p:serve(c,s,r,p,sleep=lambda _:self.state.stop())):
                result=run_endpoint(self.config,self.state,self.plan_path)
        self.assertEqual(result['configuration_id'],self.identifier)
        inspect_host.assert_called_once();recovery.assert_called_once()
        collector.stop.assert_called_once();adapter.unload_owned.assert_called_once()
        self.assertEqual(self.benchmark_tables(),tables)
        self.assertEqual(self.state.run['started'],100)
        self.assertEqual(self.state.run['deadline'],200)
        self.assertEqual(self.state.get_control('serving_session')['status'],'stopped')


if __name__=='__main__':unittest.main()
