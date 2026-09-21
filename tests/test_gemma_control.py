"""Gemma diagnostic/rescue contracts with inert artifacts and mocked CPU orchestration."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from localbench.config import file_hash, load
from localbench.scheduler import key_base
from localbench.search import memory_screen, run_screen, stage_key
from localbench.search_core import screen
from localbench.state import State


class MockReport:
    def __init__(self, *args):
        pass
    def write(self):
        return {}


class GemmaControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load()
        for key in self.config['paths']:
            directory = self.root / key
            directory.mkdir()
            self.config['paths'][key] = str(directory)
        self.config['paths']['project'] = str(self.root)
        self.binary = self.root / 'runtime_root' / 'inert.exe'
        self.binary.write_bytes(b'never executed')
        model_path = self.root / 'installed_model_root' / 'inert.gguf'
        model_path.write_bytes(b'never loaded')
        self.model = {'id': 'gemma4-12b-qat-q4', 'path': str(model_path), 'sha256': file_hash(model_path)}
        self.config['installed_model_candidates'] = [self.model]
        self.state = State(self.config, clock=lambda: 1000.)
        self.rt = {'engine': 'bundled-llama', 'kind': 'native', 'binary_path': str(self.binary)}
        self.settings = {'context': 65536, 'slots': 1, 'cache_k': 'f16', 'cache_v': 'f16', 'reasoning': 'default'}
        self.small = {'kind': 'capacity', 'passed': True, 'valid': True, 'target_tokens': 16384, 'seed': 17}
        self.caps = [self.capacity(seed, seed != 1009) for seed in self.config['grading']['capacity_marker_seeds']]
        self.identity = patch('localbench.search.model_identity', side_effect=lambda config, state, model: model)
        self.identity.start()
        self.addCleanup(self.identity.stop)

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def evidence(self):
        return {'effective_context_tokens': 65536, 'prompt_tokens': 61440, 'expected_prompt_tokens': 61440,
            'tokenization_verified': True, 'truncated': False, 'context_shift': False,
            'cached_tokens': 0, 'cache_isolation_verified': True}

    def capacity(self, seed, passed):
        return {'kind': 'capacity', 'seed': seed, 'target_tokens': 61440, 'valid': True, 'passed': passed,
            'reason': 'passed' if passed else 'capacity_marker_or_evidence_failure',
            'markers_passed': 3 if passed else 0, 'token_evidence': self.evidence(),
            'metrics': {'stream_complete': True, 'invalid_tool_calls': 0,
                'completion_reason': 'stop' if passed else 'length', 'generated_tokens': 32 if passed else 4096}}

    def timings(self):
        return [{'kind': 'timing', 'mode': 'fresh', 'replicate': replicate, 'final_validation': False,
            'passed': False, 'valid': True, 'reason': 'missing_or_incorrect_first_action',
            'metrics': {'first_action_seconds': None, 'first_action_valid': False}}
            for replicate in range(self.config['measurement']['fresh_repetitions_screen'])]

    def baseline(self, engine=None, complete=True):
        engine = engine or self.rt['engine']
        return {'engine': engine, 'model': self.model['id'], 'configuration_id': engine + '-default',
            'eligible': False, 'capacity_qualified': False, 'settings': copy.deepcopy(self.settings),
            'small_capacity': copy.deepcopy(self.small), 'capacity': copy.deepcopy(self.caps),
            'timing': self.timings() if complete else [], 'diagnostic_gemma_control': complete}

    def core(self, settings=None, tuning=False, caps=None, smoke_pass=True, runtime=None):
        adapter = SimpleNamespace(requested={**self.settings, **(settings or {})},
            effective={'effective_context': 65536, 'effective_slots': 1}, unload_owned=Mock())
        probes = []
        capacities = caps if caps is not None else self.caps
        def measured(*args, **kwargs):
            kind, seed, replicate, target = args[5], args[6], args[8], args[9]
            probes.append((kind, target))
            if kind == 'capacity':
                return copy.deepcopy(self.small if target == 16384 else next(row for row in capacities if row['seed'] == seed))
            if kind == 'timing':
                return {**self.timings()[replicate], 'passed': True,
                    'metrics': {'first_action_seconds': .5, 'first_action_valid': True}}
            return {'passed': True, 'valid': True, 'metrics': {'generated_tokens': 128}}
        with patch('localbench.search_core.make_adapter', return_value=(adapter, 'cfg')), \
                patch('localbench.search_core.setup_budget'), patch('localbench.search_core.compatible'), \
                patch('localbench.search_core.measured_job', side_effect=measured), \
                patch('localbench.search_core.coding_job', return_value={'passed': smoke_pass, 'valid': True}) as coding, \
                patch('localbench.search_core.Report', MockReport):
            result = screen(self.config, self.state, None, runtime or self.rt, self.model, settings, tuning)
        adapter.unload_owned.assert_called_once()
        return result, probes, coding.call_count

    def test_failed_default_retains_three_diagnostic_fresh_samples_and_no_expansion(self):
        result, probes, coding = self.core()
        self.assertFalse(result['eligible'])
        self.assertFalse(result['capacity_qualified'])
        self.assertTrue(result['diagnostic_gemma_control'])
        self.assertEqual(probes, [('capacity', 16384), ('capacity', 61440), ('capacity', 61440),
            ('capacity', 61440), ('timing', 61440), ('timing', 61440), ('timing', 61440)])
        self.assertEqual(coding, 0)
        self.assertIsNone(result['throughput'])
        self.assertEqual(result['capacity'][2]['metrics']['generated_tokens'], 4096)
        self.assertIsNone(self.state.run['started'])

    def test_invalid_context_stream_or_nonmarker_failures_never_enable_diagnostics(self):
        for label, section, field, value in (
                ('count', 'token_evidence', 'prompt_tokens', 60000),
                ('truncated', 'token_evidence', 'truncated', True),
                ('shift', 'token_evidence', 'context_shift', True),
                ('cache', 'token_evidence', 'cached_tokens', 65),
                ('unknown_tokenizer', 'token_evidence', 'tokenization_verified', False),
                ('incomplete', 'metrics', 'stream_complete', False),
                ('invalid_tool', 'metrics', 'invalid_tool_calls', 1),
                ('error', 'metrics', 'completion_reason', 'error'),
                ('invalid', None, 'valid', False),
                ('infra', None, 'reason', 'inference_infrastructure_error')):
            with self.subTest(label=label):
                caps = copy.deepcopy(self.caps)
                target = caps[-1][section] if section else caps[-1]
                target[field] = value
                result, probes, _ = self.core(caps=caps)
                self.assertFalse(result['diagnostic_gemma_control'])
                self.assertFalse(any(kind == 'timing' for kind, _ in probes))

    def test_only_installed_default_native_control_has_exception(self):
        for settings, tuning, rt in (({'reasoning': 'reduced'}, True, self.rt),
                ({'cache_k': 'q8_0'}, False, self.rt),
                (None, False, {**self.rt, 'engine': 'turboquant-cuda'})):
            with self.subTest(settings=settings, tuning=tuning, engine=rt['engine']):
                result, probes, _ = self.core(settings, tuning, runtime=rt)
                self.assertFalse(result['diagnostic_gemma_control'])
                self.assertFalse(any(kind == 'timing' for kind, _ in probes))

    def add_old_small(self, identifier):
        adapter = SimpleNamespace(state=self.state, requested=self.settings,
            metadata=lambda: {'binary_sha256': file_hash(self.binary), 'adjacent_dlls': {},
                'version_output': 'inert', 'model_sha256': self.model['sha256']})
        key = {**key_base(self.config, adapter, identifier, 'capacity', 17, 'fresh', 0, 16384),
            'fixture_prompt_hash': 'inert-small-prompt'}
        self.state.finish(self.state.begin(self.state.enqueue(key)), 'passed', self.small)

    def cache(self, rt, result):
        self.state.control('planned_screen:' + stage_key(rt, self.model, None), result)

    def test_only_missing_cached_gemma_timings_reopen_without_restarting_existing_jobs(self):
        old = self.baseline(complete=False)
        del old['small_capacity']
        self.add_old_small(old['configuration_id'])
        self.cache(self.rt, old)
        jobs = self.state.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
        completed = self.baseline()
        with patch('localbench.search.screen', return_value=completed) as work:
            self.assertEqual(len(run_screen(self.config, self.state, None, self.rt, self.model)['timing']), 3)
            run_screen(self.config, self.state, None, self.rt, self.model)
        work.assert_called_once()
        self.assertEqual(self.state.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], jobs)
        self.assertIsNone(self.state.run['started'])

    def test_cached_invalid_capacity_or_unproven_16k_pass_does_not_reopen(self):
        for missing_small, invalid_context in ((True, False), (False, True)):
            with self.subTest(missing_small=missing_small):
                old = self.baseline(complete=False)
                if missing_small:
                    del old['small_capacity']
                if invalid_context:
                    old['capacity'][-1]['token_evidence']['truncated'] = True
                self.cache(self.rt, old)
                with patch('localbench.search.screen') as work:
                    run_screen(self.config, self.state, None, self.rt, self.model)
                work.assert_not_called()

    def add_help(self, baseline, supported=True):
        path = self.root / 'logs' / (baseline['configuration_id'] + '-help.log')
        path.write_text('--reasoning-budget N  token budget\n' if supported else '--context N\n', encoding='utf-8')
        self.state.entity('configuration', baseline['configuration_id'], {'help_log': str(path)})

    def test_one_reduced_profile_per_native_engine_resumes_and_changes_only_reasoning(self):
        for engine in ('bundled-llama', 'upstream-llama-nightly'):
            rt = {**self.rt, 'engine': engine}
            baseline = self.baseline(engine)
            self.cache(rt, baseline)
            self.add_help(baseline)
            def screened(config, state, collector, runtime, model, settings, tuning):
                self.assertEqual(settings, {**baseline['settings'], 'reasoning': 'reduced'})
                self.assertTrue(tuning)
                return {'configuration_id': engine + '-reduced', 'eligible': True, 'capacity_qualified': True,
                    'settings': settings, 'timing': [], 'smoke': []}
            with patch('localbench.search.screen', side_effect=screened) as work:
                first = memory_screen(self.config, self.state, None, rt, self.model)
                second = memory_screen(self.config, self.state, None, rt, self.model)
            self.assertEqual(len(first), 2)
            self.assertEqual(first, second)
            work.assert_called_once()
        decisions = [json.loads(row['data']) for row in self.state.db.execute("SELECT data FROM entities WHERE kind='selection'")]
        self.assertEqual(len(decisions), 2)
        self.assertTrue(all(row['requested_reasoning_budget_tokens'] == 256 for row in decisions))
        self.assertTrue(all(row['effective_budget_enforcement'] == 'unverified' for row in decisions))

    def test_reduced_followup_waits_for_default_diagnostics_and_supported_help(self):
        for complete, supported in ((False, True), (True, False)):
            with self.subTest(complete=complete):
                baseline = self.baseline(complete=complete)
                self.add_help(baseline, supported)
                with patch('localbench.search.run_screen', return_value=baseline) as work:
                    results = memory_screen(self.config, self.state, None, self.rt, self.model)
                self.assertEqual(len(results), 1)
                work.assert_called_once()

    def test_reduced_profile_retains_capacity_and_six_smoke_timing_gates(self):
        settings = {**self.settings, 'reasoning': 'reduced'}
        result, probes, _ = self.core(settings, True)
        self.assertFalse(result['eligible'])
        self.assertFalse(any(kind == 'timing' for kind, _ in probes))
        passing = [self.capacity(seed, True) for seed in self.config['grading']['capacity_marker_seeds']]
        result, probes, coding = self.core(settings, True, passing, False)
        self.assertFalse(result['eligible'])
        self.assertEqual(coding, 6)
        self.assertFalse(any(kind == 'timing' for kind, _ in probes))
        result, probes, coding = self.core(settings, True, passing, True)
        self.assertTrue(result['eligible'])
        self.assertEqual(coding, 6)
        self.assertEqual(sum(kind == 'timing' for kind, _ in probes), 3)

    def test_configuration_limit_prevents_new_reduced_profile(self):
        baseline = self.baseline()
        self.cache(self.rt, baseline)
        self.add_help(baseline)
        self.config['limits']['maximum_unique_runtime_configurations'] = 1
        with patch('localbench.search.screen') as work:
            results = memory_screen(self.config, self.state, None, self.rt, self.model)
        work.assert_not_called()
        self.assertEqual(len(results), 1)
        decision = json.loads(self.state.db.execute("SELECT data FROM entities WHERE kind='selection'").fetchone()['data'])
        self.assertEqual(decision['reason'], 'configuration_budget_exhausted')


if __name__ == '__main__':
    unittest.main()
