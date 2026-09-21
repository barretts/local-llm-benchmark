"""CPU-only checks for the new context contract, evidence gates and ownership."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from localbench.adapters.native import NativeAdapter
from localbench import focused_qwen38 as fq


class FocusedQwenTests(unittest.TestCase):
    def test_primary_64k_gate_is_preserved_and_16k_is_explicit(self):
        self.assertEqual(NativeAdapter.minimum_context_tokens(None), 65536)
        self.assertEqual(fq.FocusedAdapter.minimum_context_tokens(None), 16384)
        cfg = fq.focused_config()
        self.assertEqual(cfg['measurement']['context_window_tokens'], 16384)
        self.assertEqual(cfg['measurement']['full_prompt_tokens'] + cfg['measurement']['reserved_output_tokens'], 16384)
        self.assertEqual(cfg['measurement']['cached_stable_prefix_tokens_target'] + cfg['measurement']['cached_variable_suffix_tokens_target'], 12288)
        self.assertEqual(cfg['limits']['new_model_weight_bytes'], 300000000000)
        self.assertEqual(cfg['grading']['regular_total'], 36)
        self.assertEqual(cfg['grading']['long_total'], 12)

    def test_primary_launch_rejects_16k_before_spawning(self):
        cfg = fq.focused_config()
        adapter = NativeAdapter(cfg, Mock(), fq.ENGINE, {'id': 'qwen38-27b-inert', 'path': 'inert.gguf'},
            binary='inert.exe', settings={'context': 16384})
        adapter.help = ''
        with patch('localbench.adapters.native.check_port'), patch('localbench.adapters.native.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'primary native configuration'):
                adapter.launch()
            spawn.assert_not_called()

    def test_focused_adapter_cannot_silently_switch_model_or_context(self):
        for model, context in (('different-model', 16384), ('qwen38-27b-inert', 65536)):
            a = fq.FocusedAdapter(fq.focused_config(), Mock(), fq.ENGINE, {'id': model, 'path': 'inert.gguf'},
                binary='inert.exe', settings={'context': context})
            with patch('localbench.adapters.native.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'scope_required'):
                    a.launch()
                spawn.assert_not_called()

    def test_original_deadline_cannot_be_extended(self):
        r = dict(zip(('id', 'started', 'deadline'), fq.ORIGINAL))
        fq.assert_original_clock(SimpleNamespace(run=r))
        r['deadline'] += 1
        with self.assertRaisesRegex(RuntimeError, 'original_execution_clock'):
            fq.assert_original_clock(SimpleNamespace(run=r))

    def test_quant_selection_requires_full_offload_and_peak_headroom(self):
        adapter = SimpleNamespace(effective={'offload_layers': [66, 66]})
        gpu = dict(total_mib=16380, used_mib=15000, free_mib=1380)
        self.assertTrue(fq.memory_verdict(adapter, gpu, gpu, {'peak_gpu_used_mib': 15050})['fits'])
        self.assertFalse(fq.memory_verdict(adapter, gpu, gpu, {'peak_gpu_used_mib': 16000})['fits'])
        adapter.effective['offload_layers'] = [60, 66]
        self.assertFalse(fq.memory_verdict(adapter, gpu, gpu, {})['fits'])

    def test_reserved_gpu_memory_is_not_available_headroom(self):
        adapter = SimpleNamespace(effective={'offload_layers': [66, 66]})
        gpu = dict(total_mib=16380, used_mib=15820, free_mib=290)
        verdict = fq.memory_verdict(adapter, gpu, gpu, {'peak_gpu_used_mib': 15820})
        self.assertFalse(verdict['fits'])
        self.assertEqual(verdict['minimum_free_mib'], 290)
        self.assertEqual(verdict['reserved_mib'], 270)
        self.assertEqual(verdict['available_mib'], 16110)
        self.assertEqual(fq.memory_verdict(adapter, gpu, gpu, {'peak_gpu_used_mib': 15900})['minimum_free_mib'], 210)

    def test_16k_measurements_reject_wrong_counts_and_context_shift(self):
        cfg = fq.focused_config()
        def result():
            return {'passed': True, 'valid': True, 'metrics': {'prompt_tokens': 12288},
                'token_evidence': {'expected_prompt_tokens': 12288, 'effective_context_tokens': 16384,
                    'truncated': False, 'context_shift': False, 'tokenization_verified': True}}
        self.assertTrue(fq.validated_measurement(cfg, result())['passed'])
        for key, value in (('expected_prompt_tokens', 61440), ('effective_context_tokens', 8192),
                ('truncated', True), ('context_shift', True), ('tokenization_verified', False)):
            r = result();r['token_evidence'][key] = value
            self.assertFalse(fq.validated_measurement(cfg, r)['valid'])
        r = result();r['metrics']['prompt_tokens'] = 12289
        self.assertFalse(fq.validated_measurement(cfg, r)['valid'])

    def test_quality_coverage_does_not_count_diagnostics_invalid_or_demo(self):
        cfg = fq.focused_config()
        def row(valid=True, mode='fresh', replicate=0):
            return {'result': {'kind': 'quality', 'group': 'regular', 'fixture': 'py01', 'seed': 42,
                'valid': valid, 'passed': True}, 'job_key': {'mode': mode, 'replicate': replicate}}
        stats = fq.quality_stats(cfg, [row(), row(), row(False), row(mode='isolated-demo', replicate=1)], 'regular')
        self.assertEqual(stats['attempted'], 1)
        self.assertEqual(stats['passed'], 1)
        self.assertEqual(stats['maximum_possible'], 36)

    def test_gpu_lock_remains_held_until_owned_cleanup_after_failure(self):
        events = []
        @contextmanager
        def lock(cfg):
            events.append('enter')
            try:yield
            finally:events.append('unlock')
        with patch.object(fq, 'gpu_lock', lock):
            with self.assertRaisesRegex(RuntimeError, 'test_failure'):
                with fq.owned_gpu({}, lambda: events.append('cleanup')):
                    raise RuntimeError('test_failure')
        self.assertEqual(events, ['enter', 'cleanup', 'unlock'])


if __name__ == '__main__':
    unittest.main()
