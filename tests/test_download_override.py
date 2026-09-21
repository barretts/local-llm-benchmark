"""Operational disk redirect preserves run, fixture and grading identities."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from localbench.config import load, atomic_json
from localbench.state import State
from localbench.acquisition import _disk_headroom


class DownloadOverrideTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        original=Path(__file__).resolve().parents[1]/'benchmark-spec.json'
        self.spec=self.root/'benchmark-spec.json';self.spec.write_bytes(original.read_bytes())
        self.override={'schema_version':1,'user_instruction':'use E:\\modelmadness for downloads',
            'paths':{'new_model_root':'E:\\modelmadness'},'limits':{'minimum_free_e_bytes':68719476736}}

    def tearDown(self):self.temp.cleanup()

    def test_destination_change_keeps_base_contract_and_original_run(self):
        first=load(self.spec);first['paths']['state']=str(self.root/'state')
        state=State(first,clock=lambda:1000);state.control('harness_verified',True)
        state.control('fixture_version','frozen');state.start_execution('mock runtime');before=state.run;state.close()
        atomic_json(self.root/'execution-overrides.json',self.override)
        second=load(self.spec);second['paths']['state']=str(self.root/'state')
        self.assertEqual(first['_spec_hash'],second['_spec_hash'])
        self.assertNotEqual(first['_execution_hash'],second['_execution_hash'])
        self.assertEqual(second['paths']['new_model_root'],'E:\\modelmadness')
        self.assertEqual(first['grading'],second['grading']);self.assertEqual(first['measurement'],second['measurement'])
        state=State(second,clock=lambda:2000)
        try:self.assertEqual(state.run,before);self.assertEqual(state.get_control('fixture_version'),'frozen')
        finally:state.close()

    def test_unknown_contract_overrides_and_other_destinations_are_rejected(self):
        for changes in ({'paths':{'new_model_root':'D:\\elsewhere'}},{'limits':{'new_model_weight_bytes':999}},
                        {'limits':{'minimum_free_e_bytes':0}}):
            with self.subTest(changes=changes):
                altered=copy.deepcopy(self.override);altered.update(changes)
                atomic_json(self.root/'execution-overrides.json',altered)
                with self.assertRaises(ValueError):load(self.spec)

    def test_transfer_space_is_debited_from_e_with_c_and_f_floors_preserved(self):
        atomic_json(self.root/'execution-overrides.json',self.override);config=load(self.spec)
        floor=config['limits']['minimum_free_e_bytes']
        free={'C:\\':config['limits']['minimum_free_c_bytes']+100,'F:\\':config['limits']['minimum_free_f_bytes']+100,'E:\\':floor+10}
        usage=lambda path:SimpleNamespace(free=free[path])
        _disk_headroom(config,10,usage,destination='E:\\modelmadness\\fixture.gguf')
        with self.assertRaisesRegex(RuntimeError,'disk_headroom_e'):
            _disk_headroom(config,11,usage,destination='E:\\modelmadness\\fixture.gguf')
        free['F:\\']-=101
        with self.assertRaisesRegex(RuntimeError,'disk_headroom_f'):
            _disk_headroom(config,0,usage,destination='E:\\modelmadness\\fixture.gguf')


if __name__=='__main__':unittest.main()
