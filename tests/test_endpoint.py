"""Endpoint lifecycle tests use fake adapters: no GPU, sockets, or engines."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from localbench.config import atomic_json, load
from localbench.endpoint import serve
from localbench.state import State


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ('project','state','logs','artifacts'):
            self.config['paths'][name] = str(Path(self.temp.name)/name)
        self.state = State(self.config,clock=lambda:1000.)
        self.collector=object()
        self.plan=Path(self.temp.name)/'immutable-plan.json'
        self.plan.write_text('{"fixture":"read-only pinned plan"}',encoding='utf-8')
        self.original_plan=self.plan.read_bytes()
        self.identifier='c'*64

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def adapter(self,engine='upstream-llama-nightly',port=38201,model='local-model'):
        process=SimpleNamespace(pid=12345,poll=Mock(return_value=None))
        adapter=SimpleNamespace(engine=engine,http=SimpleNamespace(port=port),model={'id':'benchmark-metadata-id'},
            requested={'context':65536},effective={'effective_context':65536,'effective_slots':1},
            saved={'owner':'localbench','run_id':self.state.run['id'],'pid':process.pid},
            process=process,health=Mock(return_value=True),instance_id=None)
        interface={'endpoint':'http://127.0.0.1:'+str(port),'model':model,
            'chat_path':'/api/chat' if engine=='ollama' else '/v1/chat/completions',
            'models_path':'/api/tags' if engine=='ollama' else '/v1/models',
            'wire_protocol':'Ollama native streaming NDJSON' if engine=='ollama' else 'OpenAI-compatible streaming SSE'}
        adapter.handoff_launch_evidence={'interface':interface,'source_and_model_pins_verified':True}
        def unload():
            process.poll.return_value=0
            adapter.instance_id=None
            adapter.process=None
        adapter.unload_owned=Mock(side_effect=unload)
        return adapter

    def artifact(self):
        return json.loads((Path(self.config['paths']['artifacts'])/'agent-endpoint.json').read_text())

    def stop_after_inspection(self,inspect=None):
        def sleep(seconds):
            self.assertEqual(seconds,1)
            if inspect:inspect(self.artifact())
            self.state.stop()
        return sleep

    def test_launches_only_pinned_plan_reports_actual_native_alias_and_stops_owned_adapter(self):
        adapter=self.adapter(port=39991)
        observed=[]
        with patch('localbench.handoff.launch_plan',return_value=(adapter,self.identifier)) as launch:
            result=serve(self.config,self.state,self.collector,self.plan,
                sleep=self.stop_after_inspection(lambda status:observed.append(status)))
        launch.assert_called_once_with(self.config,self.state,self.collector,self.plan)
        self.assertEqual(result,{'status':'endpoint_stopped','configuration_id':self.identifier})
        self.assertEqual(observed[0]['status'],'serving')
        self.assertEqual(observed[0]['endpoint'],'http://127.0.0.1:39991')
        self.assertEqual(observed[0]['served_model'],'local-model')
        self.assertEqual(observed[0]['chat_path'],'/v1/chat/completions')
        self.assertEqual(observed[0]['models_path'],'/v1/models')
        self.assertEqual(observed[0]['plan'],str(self.plan.resolve()))
        self.assertEqual(self.plan.read_bytes(),self.original_plan)
        adapter.unload_owned.assert_called_once()
        self.assertEqual(adapter.health.call_count,1) # Stop avoids another health request.
        self.assertEqual(self.artifact()['status'],'stopped')
        self.assertTrue(self.artifact()['cleanup_verified'])
        self.assertIsNone(self.state.run['started'])

    def test_actual_ollama_route_and_scoped_model_and_tabby_alias_are_preserved(self):
        for engine,port,model in (('ollama',38207,'localbench-owned-digest'),('exllamav3-tabby',38206,'checkpoint-directory')):
            with self.subTest(engine=engine):
                with self.state.db:self.state.db.execute('UPDATE runs SET stop_requested=0')
                adapter=self.adapter(engine,port,model)
                observed=[]
                with patch('localbench.handoff.launch_plan',return_value=(adapter,self.identifier)):
                    serve(self.config,self.state,self.collector,self.plan,sleep=self.stop_after_inspection(observed.append))
                self.assertEqual(observed[0]['served_model'],model)
                self.assertEqual(observed[0]['chat_path'],'/api/chat' if engine=='ollama' else '/v1/chat/completions')
                self.assertEqual(observed[0]['endpoint'],'http://127.0.0.1:'+str(port))

    def test_lm_studio_records_regenerated_instance_name_and_no_credential_value(self):
        adapter=self.adapter('lm-studio',1234,'localbench-new-instance')
        adapter.instance_id='localbench-new-instance'
        with patch.dict('os.environ',{'LM_BENCH_TOKEN':'secret-fixture-never-persist'}), \
             patch('localbench.handoff.launch_plan',return_value=(adapter,self.identifier)):
            serve(self.config,self.state,self.collector,self.plan,sleep=self.stop_after_inspection())
        status=self.artifact()
        self.assertEqual(status['served_model'],'localbench-new-instance')
        self.assertIn('LM_BENCH_TOKEN',status['credential'])
        self.assertNotIn('secret-fixture-never-persist',json.dumps(status))
        self.assertIsNone(adapter.instance_id)

    def test_health_monitor_failure_cleans_adapter_and_offline_tokenizer_helper_and_records_failure(self):
        adapter=self.adapter()
        adapter.health.side_effect=[True,True,False]
        adapter.owned_tokenizer_helper=SimpleNamespace(unload_owned=Mock())
        sleep=Mock()
        with patch('localbench.handoff.launch_plan',return_value=(adapter,self.identifier)):
            with self.assertRaisesRegex(RuntimeError,'endpoint_health_failed'):
                serve(self.config,self.state,self.collector,self.plan,sleep=sleep)
        self.assertEqual(sleep.call_count,2)
        adapter.unload_owned.assert_called_once()
        adapter.owned_tokenizer_helper.unload_owned.assert_called_once()
        status=self.artifact()
        self.assertEqual(status['status'],'failed')
        self.assertEqual(status['failure_reason'],'endpoint_health_monitor_failed')
        self.assertTrue(status['cleanup_verified'])

    def test_initial_health_failure_cleans_launch_without_claiming_service(self):
        adapter=self.adapter()
        adapter.health.return_value=False
        sleep=Mock()
        with patch('localbench.handoff.launch_plan',return_value=(adapter,self.identifier)):
            with self.assertRaisesRegex(RuntimeError,'endpoint_not_ready'):
                serve(self.config,self.state,self.collector,self.plan,sleep=sleep)
        sleep.assert_not_called()
        adapter.unload_owned.assert_called_once()
        self.assertEqual(self.artifact()['status'],'failed')
        self.assertEqual(self.artifact()['failure_reason'],'endpoint_initial_health_failed')

    def test_rejected_immutable_launch_never_cleans_or_relabels_previous_endpoint(self):
        previous={'status':'serving','configuration_id':'previous-handle','endpoint':'http://127.0.0.1:39000'}
        self.state.control('live_agent_endpoint',previous)
        atomic_json(Path(self.config['paths']['artifacts'])/'agent-endpoint.json',previous)
        with patch('localbench.handoff.launch_plan',side_effect=ValueError('handoff_plan_hash_mismatch')), \
             patch('localbench.endpoint.close_adapter') as cleanup:
            with self.assertRaisesRegex(ValueError,'plan_hash_mismatch'):
                serve(self.config,self.state,self.collector,self.plan,sleep=Mock())
        cleanup.assert_not_called()
        self.assertEqual(self.artifact(),previous)
        self.assertEqual(self.state.get_control('live_agent_endpoint'),previous)

    def test_cleanup_failure_is_recorded_without_secret_exception_text_and_helper_is_attempted(self):
        adapter=self.adapter()
        adapter.unload_owned.side_effect=RuntimeError('stale handle secret-fixture-never-persist')
        adapter.owned_tokenizer_helper=SimpleNamespace(unload_owned=Mock())
        with patch('localbench.handoff.launch_plan',return_value=(adapter,self.identifier)):
            with self.assertRaisesRegex(RuntimeError,'stale handle'):
                serve(self.config,self.state,self.collector,self.plan,sleep=self.stop_after_inspection())
        adapter.owned_tokenizer_helper.unload_owned.assert_called_once()
        status=self.artifact()
        self.assertEqual(status['status'],'cleanup_failed')
        self.assertFalse(status['cleanup_verified'])
        self.assertIsNone(status['stopped_at'])
        self.assertEqual(status['failure_reason'],'owned_endpoint_cleanup_failed')
        self.assertNotIn('secret-fixture-never-persist',json.dumps(status))

    def test_silent_process_stop_failure_cannot_be_reported_as_verified_cleanup(self):
        adapter=self.adapter()
        adapter.unload_owned.side_effect=None
        with patch('localbench.handoff.launch_plan',return_value=(adapter,self.identifier)):
            with self.assertRaisesRegex(RuntimeError,'stop_unverified'):
                serve(self.config,self.state,self.collector,self.plan,sleep=self.stop_after_inspection())
        self.assertEqual(self.artifact()['status'],'cleanup_failed')
        self.assertFalse(self.artifact()['cleanup_verified'])

    def test_controller_interrupt_stops_only_owned_endpoint_and_records_shutdown(self):
        adapter=self.adapter()
        with patch('localbench.handoff.launch_plan',return_value=(adapter,self.identifier)):
            with self.assertRaises(KeyboardInterrupt):
                serve(self.config,self.state,self.collector,self.plan,sleep=Mock(side_effect=KeyboardInterrupt()))
        adapter.unload_owned.assert_called_once()
        self.assertEqual(self.artifact()['status'],'stopped')
        self.assertEqual(self.artifact()['failure_reason'],'controller_shutdown')
        self.assertTrue(self.artifact()['cleanup_verified'])


if __name__=='__main__':
    unittest.main()
