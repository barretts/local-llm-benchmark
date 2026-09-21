"""Inert discovery queue tests: temporary State, no models/network/runtime/GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from localbench.config import load
from localbench.extra_discovery import extra_gguf_screens, select_variant, validate_manifest
from localbench.state import State


def candidate(identity='new-coder'):
    return {'id': identity, 'repo': 'author/Model-GGUF', 'revision': 'a'*40,
            'filename': 'Model.Q4_K_M.gguf', 'bytes': 10, 'sha256': 'b'*64,
            'native_context_tokens': 262144, 'context_extension': False, 'format': 'gguf',
            'sources': ['https://huggingface.co/author/Model-GGUF',
                        'https://huggingface.co/api/models/author/Model-GGUF?blobs=true']}


class ExtraDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load()
        for key in ('state', 'artifacts', 'new_model_root'):
            directory = self.root/key
            directory.mkdir()
            self.config['paths'][key] = str(directory)
        self.now = 1000.0
        self.state = State(self.config, clock=lambda: self.now)
        self.state.control('harness_verified', True)
        self.state.start_execution('mock_only')
        self.original_run = self.state.run.copy()
        self.state.control('installed_baselines_completed', True)
        self.baselines = [{'planned_screen_key': 'planned_screen:current',
                           'configuration_id': 'installed-current'}]
        self.state.control('planned_screen:current', self.baselines[0])
        self.runtimes = {name: {'engine': name, 'kind': 'native'}
                         for name in ('bundled-llama', 'upstream-llama-nightly')}
        self.runtimes['vllm'] = {'engine': 'vllm', 'kind': 'container'}
        self.unavailable = Mock()
        self.acquired = []
        self.screens = []

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def write_queue(self, candidates):
        (self.root/'artifacts'/'extra-discovered-gguf.json').write_text(
            json.dumps({'schema_version': 1, 'candidates': candidates}), encoding='utf-8')

    def acquire(self, config, state, lead):
        self.acquired.append(lead['id'])
        path = self.root/'new_model_root'/lead['filename']
        # A 10-byte temporary ledger reservation models the existing acquisition
        # contract without creating/downloading any weight file.
        state.reserve_weight(lead['id'], lead['bytes'], 'mock-only', path)
        return {key: lead[key] for key in ('id', 'revision', 'sha256', 'bytes')} | {'path': str(path)}

    def screen(self, config, state, collector, runtime, model):
        self.screens.append((runtime['engine'], copy.deepcopy(model)))
        return [{'configuration_id': runtime['engine']+'-'+model['id'],
                 'model': model['id'], 'eligible': False, 'capacity_qualified': False}]

    def run_queue(self, pending=(), header=None, acquire=None):
        header = header if header is not None else {
            'general.architecture': 'qwen35', 'qwen35.context_length': 262144,
            'tokenizer.chat_template': 'verified embedded template'}
        with patch('localbench.search.pending_for_stage', return_value=list(pending)) as unfinished, \
                patch('localbench.extra_discovery.gguf_metadata', return_value=header):
            result = extra_gguf_screens(self.config, self.state, object(), self.runtimes, self.baselines,
                acquire_model=acquire or self.acquire, memory_screen=self.screen,
                unavailable=self.unavailable)
        return result, unfinished

    def test_no_manifest_is_optional_and_does_not_acquire(self):
        result, unfinished = self.run_queue()
        self.assertEqual(result, [])
        self.assertFalse(unfinished.called)
        self.assertEqual(self.acquired, [])

    def test_strict_schema_and_public_pins_reject_invalid_claims(self):
        invalid = [dict(candidate(), extra='ignored'), dict(candidate(), revision='main'),
                   dict(candidate(), sha256='b'*63), dict(candidate(), filename='../model.gguf'),
                   dict(candidate(), repo='../model'), dict(candidate(), id=4),
                   dict(candidate(), bytes=True), dict(candidate(), bytes=0),
                   dict(candidate(), native_context_tokens=32768),
                   dict(candidate(), context_extension=True), dict(candidate(), format='safetensors'),
                   dict(candidate(), sources=['https://user:password@huggingface.co/author/model']),
                   dict(candidate(), sources=['https://huggingface.co/author/model?token=secret']),
                   dict(candidate(), sources=['https://huggingface.co/author/%252e%252e/model']),
                   dict(candidate(), provenance_artifact='../private.json')]
        for lead in invalid:
            with self.subTest(lead=lead), self.assertRaises(ValueError):
                validate_manifest({'schema_version': 1, 'candidates': [lead]})
        with self.assertRaises(ValueError):
            validate_manifest({'schema_version': 1, 'candidates': [candidate(), candidate()]})
        with self.assertRaises(ValueError):
            validate_manifest({'schema_version': 1, 'candidates': [], 'unexpected': True})

    def test_baseline_pending_or_stale_blocks_before_selection_and_acquisition(self):
        self.write_queue([candidate()])
        with self.assertRaisesRegex(RuntimeError, 'current_installed_baselines'):
            self.run_queue(pending=['unfinished-current-job'])
        self.state.control('planned_screen:current', None)
        with self.assertRaisesRegex(RuntimeError, 'current_installed_baselines'):
            self.run_queue()
        self.assertEqual(self.acquired, [])
        self.assertIsNone(self.state.get_control('discovered_weight_variants'))
        self.assertEqual(self.state.db.execute('SELECT COUNT(*) FROM weights').fetchone()[0], 0)

    def test_new_leads_use_both_native_engines_preserve_gates_and_resume_ledgers(self):
        self.state.control('discovered_weight_variants', ['ornith15-9b-exl3-hq4'])
        self.write_queue([candidate('coder-one'), candidate('coder-two')])
        first, unfinished = self.run_queue()
        second, _ = self.run_queue()
        self.assertEqual(len(first), 4)
        self.assertEqual(second, first)
        self.assertTrue(all(not item['eligible'] and not item['capacity_qualified'] for item in first))
        self.assertEqual(set(engine for engine, _ in self.screens), set(self.runtimes)-{'vllm'})
        self.assertTrue(all(model['actual_native_context_tokens'] == 262144 and
                            model['embedded_chat_template_sha256'] for _, model in self.screens))
        self.assertEqual(self.state.get_control('discovered_weight_variants'),
                         ['ornith15-9b-exl3-hq4', 'coder-one', 'coder-two'])
        self.assertEqual(self.state.db.execute('SELECT SUM(bytes) FROM weights').fetchone()[0], 20)
        self.assertEqual(self.state.run, self.original_run)
        unfinished.assert_called_once_with(self.state, 'installed-current')

    def test_maximum_four_variants_skips_new_but_reuses_existing_pin(self):
        self.assertTrue(select_variant(self.config, self.state, candidate()))
        self.state.control('discovered_weight_variants', ['new-coder', 'ornith', 'third', 'fourth'])
        self.write_queue([candidate(), candidate('fifth')])
        result, _ = self.run_queue()
        self.assertEqual(self.acquired, ['new-coder'])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(self.state.get_control('discovered_weight_variants')), 4)
        self.assertEqual(self.unavailable.call_args.args[4], 'discovery_variant_budget_exhausted')

    def test_changed_second_pin_rejects_entire_queue_before_acquiring_first(self):
        original = candidate('second')
        select_variant(self.config, self.state, original)
        self.write_queue([candidate('first'), dict(original, revision='c'*40)])
        with self.assertRaisesRegex(ValueError, 'identity_changed:second'):
            self.run_queue()
        self.assertEqual(self.acquired, [])
        self.assertEqual(self.state.get_control('discovered_weight_variants'), ['second'])
        self.assertEqual(json.loads(self.state.db.execute(
            "SELECT data FROM entities WHERE kind='extra_discovered_weight_pin'").fetchone()[0]), original)

    def test_removing_previously_selected_candidate_rejects_resume_drift(self):
        select_variant(self.config, self.state, candidate())
        self.write_queue([])
        with self.assertRaisesRegex(ValueError, 'identity_changed:new-coder'):
            self.run_queue()
        self.assertEqual(self.acquired, [])

    def test_actual_header_short_context_or_missing_template_never_screens(self):
        self.write_queue([candidate()])
        for header in ({'general.architecture': 'qwen35', 'qwen35.context_length': 32768,
                        'tokenizer.chat_template': 'template'},
                       {'general.architecture': 'qwen35', 'qwen35.context_length': 262144}):
            with self.subTest(header=header):
                result, _ = self.run_queue(header=header)
                self.assertEqual(result, [])
        self.assertEqual(self.screens, [])
        self.assertEqual(self.unavailable.call_count, 2)

    def test_wrong_acquired_hash_blocks_screen_and_verified_registry(self):
        self.write_queue([candidate()])
        bad_acquire = lambda *args: {'id': 'new-coder', 'revision': 'a'*40,
                                   'sha256': 'c'*64, 'bytes': 10, 'path': 'not-read'}
        with self.assertRaisesRegex(RuntimeError, 'acquisition_identity_mismatch'):
            self.run_queue(acquire=bad_acquire)
        self.assertEqual(self.screens, [])
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM entities WHERE kind='model'").fetchone()[0], 0)

    def test_no_native_runtime_does_not_count_or_download_variant(self):
        self.write_queue([candidate()])
        self.runtimes = {'vllm': self.runtimes['vllm']}
        result, _ = self.run_queue()
        self.assertEqual(result, [])
        self.assertEqual(self.acquired, [])
        self.assertIsNone(self.state.get_control('discovered_weight_variants'))
        self.assertEqual(self.unavailable.call_args.args[4], 'no_existing_native_runtime_for_discovered_gguf')

    def test_stop_budget_and_weight_cap_remain_existing_state_contracts(self):
        self.write_queue([candidate()])
        self.state.stop()
        with self.assertRaisesRegex(RuntimeError, 'stop_after_current'):
            self.run_queue()
        self.assertEqual(self.acquired, [])
        self.assertIsNone(self.state.get_control('discovered_weight_variants'))
        self.state.recover()
        self.config['limits']['new_model_weight_bytes'] = 9
        with self.assertRaisesRegex(RuntimeError, 'weight_budget_exhausted'):
            self.run_queue()
        self.assertEqual(self.screens, [])
        self.assertEqual(self.state.db.execute('SELECT COUNT(*) FROM weights').fetchone()[0], 0)
        self.assertEqual(self.state.run['started'], self.original_run['started'])
        self.assertEqual(self.state.run['deadline'], self.original_run['deadline'])


if __name__ == '__main__':
    unittest.main()
