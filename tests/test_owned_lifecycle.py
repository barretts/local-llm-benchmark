"""Mock ownership checks: no actual processes, containers or GPU calls."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from localbench.config import digest
from localbench.recovery import recover_process, recover_processes
from localbench.tokenizer_helper import cpu_launch_plan
from localbench.search_core import make_adapter
from tests.test_recovery import FakeProcess


class OwnedLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'state').mkdir();(self.root/'runtime').mkdir()
        self.binary=self.root/'runtime'/'llama-server.exe';self.binary.write_bytes(b'inert fixture')
        self.config={'paths':{'state':str(self.root/'state'),'runtime_root':str(self.root/'runtime')},
            'private_ports':{'native':38201},'installed_tools':{'bundled_llama_server':str(self.binary)}}
        self.state=SimpleNamespace(run={'id':'mock-run'},control=Mock())
        flags='--model --ctx-size --parallel --n-gpu-layers --device --batch-size --ubatch-size --threads --threads-batch --host --port --cors-origins --no-kv-offload --no-op-offload --no-context-shift --jinja --no-warmup --no-webui'
        self.argv=cpu_launch_plan(self.binary,self.root/'inert.gguf',49123,flags)
        self.path=self.root/'state'/'owned-tokenizer-49123.json'

    def tearDown(self):self.temp.cleanup()

    def save(self,argv=None):
        argv=argv or self.argv
        self.path.write_text(json.dumps({'owner':'localbench','run_id':'mock-run','pid':123456,'created':100,
            'argv':argv,'command_hash':digest(argv),'endpoint':'http://127.0.0.1:49123'}),encoding='utf-8')

    def test_private_cpu_helper_original_identity_is_recovered(self):
        self.save();process=FakeProcess(self.argv)
        result=recover_processes(self.config,self.state,lambda pid:process,include_lm_studio=False)
        self.assertEqual(result[0]['status'],'recovered');self.assertEqual(process.stops,[15])
        self.assertTrue(json.loads(self.path.read_text())['stopped'])

    def test_helper_gpu_or_extra_argument_refuses_before_opening_process(self):
        for argv in (self.argv+['--arbitrary-control'],[('1' if n and self.argv[n-1]=='--n-gpu-layers' else arg) for n,arg in enumerate(self.argv)]):
            with self.subTest(argv=argv):
                self.save(argv);factory=Mock()
                with self.assertRaisesRegex(RuntimeError,'tokenizer_cpu_controls'):
                    recover_process(self.config,self.state,self.path,factory)
                factory.assert_not_called()

    def test_recycled_helper_pid_is_never_terminated(self):
        self.save();process=FakeProcess(self.argv,created=101)
        result=recover_process(self.config,self.state,self.path,lambda pid:process)
        self.assertEqual(result['status'],'original_process_replaced')
        self.assertIs(result['termination_performed'],False)
        self.assertEqual(process.stops,[])

    def test_container_recovery_precedes_discovery_and_launch(self):
        adapter=Mock();events=[]
        for name in ('recover_owned','discover_capabilities','launch'):
            getattr(adapter,name).side_effect=lambda n=name:events.append(n)
        with patch('localbench.adapters.container.ContainerAdapter',return_value=adapter),patch('localbench.search_core.register_configuration',return_value='cfg'):
            actual,identifier=make_adapter(self.config,self.state,object(),{'kind':'container','engine':'vllm','image_digest':'inert-pin'},{'id':'mock'})
        self.assertIs(actual,adapter);self.assertEqual(identifier,'cfg')
        self.assertEqual(events,['recover_owned','discover_capabilities','launch'])

    def test_container_recovery_refusal_prevents_new_launch(self):
        adapter=Mock();adapter.recover_owned.side_effect=RuntimeError('stale identity')
        with patch('localbench.adapters.container.ContainerAdapter',return_value=adapter):
            with self.assertRaisesRegex(RuntimeError,'stale identity'):
                make_adapter(self.config,self.state,object(),{'kind':'container','engine':'vllm','image_digest':'inert-pin'},{'id':'mock'})
        adapter.launch.assert_not_called();adapter.discover_capabilities.assert_not_called()
        adapter.unload_owned.assert_called_once()


if __name__=='__main__':unittest.main()
