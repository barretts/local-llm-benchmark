"""Partial managed load ownership must survive local inventory failures."""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from tests import test_managed_adapters as fixtures
from localbench.recovery import recover_lm_studio


class LMStudioLoadIntentTests(unittest.TestCase):
    setUp=fixtures.ManagedAdapterTests.setUp
    tearDown=fixtures.ManagedAdapterTests.tearDown
    lms=fixtures.ManagedAdapterTests.lms
    inventory=fixtures.ManagedAdapterTests.inventory
    load_lms=fixtures.ManagedAdapterTests.load_lms

    def test_intent_before_cli_survives_inventory_failure_and_cleans_owned_id(self):
        adapter=self.lms();inventory=self.inventory();fail=[False]
        def command(config,label,argv,**kwargs):
            pending=json.loads((Path(config['paths']['state'])/'owned-lm-studio-instance.json').read_text())
            self.assertEqual(pending['phase'],'pending_instance_config')
            self.assertIs(pending['load_completed'],False)
            inventory['models'][0]['loaded_instances']=[{'id':adapter.instance_id,'config':{'context_length':65536}}]
            fail[0]=True;return {'exit_code':0,'log':'mock-load.log','output':'loaded'}
        def api(path,body=None):
            if path=='/api/v1/models':
                if fail[0]:fail[0]=False;raise RuntimeError('temporary local inventory failure')
                return copy.deepcopy(inventory)
            if path=='/api/v1/models/unload':
                inventory['models'][0]['loaded_instances']=[];return {'instance_id':body['instance_id']}
            raise AssertionError(path)
        adapter.http=Mock();adapter.http.json.side_effect=api
        with patch.dict(os.environ,{'LM_BENCH_TOKEN':'memory-only-mock-key'}),patch('localbench.adapters.lms._logged_command',side_effect=command):
            with self.assertRaisesRegex(RuntimeError,'temporary local inventory failure'):adapter.launch()
            self.assertIsNotNone(adapter.saved);self.assertIsNotNone(adapter.instance_id)
            adapter.unload_owned()
        self.assertIsNone(adapter.instance_id);self.assertEqual(inventory['models'][0]['loaded_instances'],[])
        self.assertTrue(json.loads((Path(self.config['paths']['state'])/'owned-lm-studio-instance.json').read_text())['unloaded'])

    def test_unload_acknowledgment_without_actual_absence_retains_ownership(self):
        adapter=self.lms();inventory,_=self.load_lms(adapter);identifier=adapter.instance_id
        adapter.http.json.side_effect=lambda path,body=None:copy.deepcopy(inventory) if path=='/api/v1/models' else {'instance_id':identifier}
        with patch.dict(os.environ,{'LM_BENCH_TOKEN':'memory-only-mock-key'}):
            with self.assertRaisesRegex(RuntimeError,'owned_unload_unverified'):adapter.unload_owned()
        self.assertEqual(adapter.instance_id,identifier)

    def test_unfinished_cli_without_visible_id_cannot_claim_unloaded(self):
        adapter=self.lms();adapter.instance_id='localbench-'+'a'*32
        adapter.saved={'phase':'pending_instance_config','load_completed':False,'load_config':{},'model_key':'test-model'}
        adapter.http=Mock();adapter.http.json.return_value=self.inventory()
        with patch.dict(os.environ,{'LM_BENCH_TOKEN':'memory-only-mock-key'}):
            with self.assertRaisesRegex(RuntimeError,'pending_load_completion_unknown'):adapter.unload_owned()
        self.assertIsNotNone(adapter.instance_id)

    def test_failed_cli_exit_without_visible_id_retains_pending_load(self):
        adapter=self.lms();adapter.http=Mock();adapter.http.json.return_value=self.inventory()
        command={'exit_code':137,'log':'mock-crashed-client.log','output':''}
        with patch.dict(os.environ,{'LM_BENCH_TOKEN':'memory-only-mock-key'}),patch('localbench.adapters.lms._logged_command',return_value=command):
            with self.assertRaisesRegex(RuntimeError,'pending_load_completion_unknown'):adapter.launch()
        pending=json.loads((Path(self.config['paths']['state'])/'owned-lm-studio-instance.json').read_text())
        self.assertIs(pending['load_completed'],False)
        self.assertIsNotNone(adapter.instance_id)
        self.assertNotIn('unloaded',pending)


if __name__=='__main__':unittest.main()
