"""Marked OOMs get bounded real-context followups, never synthetic victories."""
import unittest
from unittest.mock import patch
from tests import test_search as fixtures
from localbench.search import memory_screen


class MemoryFollowupTests(unittest.TestCase):
    setUp=fixtures.SearchTests.setUp
    tearDown=fixtures.SearchTests.tearDown

    def run_memory(self,outcomes,metadata=None):
        calls=[]
        def work(config,state,collector,rt,model,settings=None,**kwargs):
            calls.append(settings)
            item=outcomes[min(len(calls)-1,len(outcomes)-1)]
            return {**item,'settings':settings}
        runtime={'engine':'upstream-llama-nightly','kind':'native','binary_path':str(self.binary)}
        with patch('localbench.search.run_screen',side_effect=work),patch('localbench.doctor.gguf_metadata',return_value=metadata or {}):
            result=memory_screen(self.config,self.state,None,runtime,self.installed[0])
        return result,calls

    def test_lower_scratch_success_preserves_all_ooms_and_stops_before_offload(self):
        oom={'eligible':False,'reason':'model_oom'}
        result,calls=self.run_memory([oom,oom,oom,{'eligible':True,'capacity_qualified':True,'reason':'screen_passed'}])
        self.assertEqual(len(result),4)
        self.assertEqual(calls[-1],{'cache_k':'q4_0','cache_v':'q4_0','batch':512,'ubatch':128})
        self.assertNotIn('gpu_layers',calls[-1])

    def test_still_oom_uses_one_measured_partial_layer_setting(self):
        oom={'eligible':False,'reason':'model_oom'}
        result,calls=self.run_memory([oom]*4+[{'eligible':True,'capacity_qualified':True}],{'general.architecture':'qwen35','qwen35.block_count':32})
        self.assertEqual(len(result),5)
        self.assertEqual(calls[-1]['gpu_layers'],24)
        self.assertEqual(calls[-1]['batch'],512)
        self.assertEqual(calls[-1]['cache_k'],'q4_0')

    def test_quality_or_capacity_failure_does_not_trigger_memory_work(self):
        result,calls=self.run_memory([{'eligible':False,'reason':'capacity_marker_failure'}])
        self.assertEqual(len(calls),1)
        self.assertFalse(result[0]['eligible'])

    def test_unknown_layer_count_records_explicit_limit_without_guessing(self):
        result,calls=self.run_memory([{'eligible':False,'reason':'model_oom'}])
        self.assertEqual(len(calls),4)
        self.assertFalse(any(item and 'gpu_layers' in item for item in calls))
        rows=self.state.db.execute("SELECT data FROM entities WHERE kind='selection'").fetchall()
        self.assertIn('exact_layer_count_unavailable',rows[0]['data'])

    def test_exhausted_configuration_budget_blocks_new_followups(self):
        self.config['limits']['maximum_unique_runtime_configurations']=1
        self.state.entity('configuration','already-measured',{})
        result,calls=self.run_memory([{'eligible':False,'reason':'model_oom'}])
        self.assertEqual(len(calls),3)
        self.assertTrue(all(not item or 'batch' not in item for item in calls))


if __name__=='__main__':unittest.main()
