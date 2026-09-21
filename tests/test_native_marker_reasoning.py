"""Temporary-state reasoning intake; mocked streams, no runtime/network/GPU."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from localbench.config import digest
from localbench.native_marker_reasoning import marker_reasoning_eligible
from localbench.scheduler import key_base
from localbench.search import stage_key
import localbench.tuning as tuning_module
import tests.test_tuning as tuning_tests


class MarkerReasoningTests(unittest.TestCase):
    setUp = tuning_tests.TuningTests.setUp
    tearDown = tuning_tests.TuningTests.tearDown
    candidate = tuning_tests.TuningTests.candidate
    settings = tuning_tests.TuningTests.settings
    smoke = tuning_tests.TuningTests.smoke
    latency = tuning_tests.TuningTests.latency

    def marker(self, model=None, engine='upstream-llama-nightly'):
        model = model or self.models[0]
        baseline = self.candidate(model, engine=engine, checkpoint=False)
        baseline.update(eligible=False, capacity_qualified=False, smoke=[], timing=[],
                        reason='capacity_or_action_or_smoke_gate_failed')
        baseline['small_capacity'] = {'kind': 'capacity', 'target_tokens': 16384,
                                     'passed': True, 'valid': True}
        evidence = {'effective_context_tokens': 65536, 'prompt_tokens': 61440,
                    'expected_prompt_tokens': 61440, 'tokenization_verified': True,
                    'truncated': False, 'context_shift': False, 'cached_tokens': 0,
                    'cache_isolation_verified': True}
        baseline['capacity'] = []
        for index, seed in enumerate(self.config['grading']['capacity_marker_seeds']):
            passed = index != 0
            baseline['capacity'].append({'kind': 'capacity', 'target_tokens': 61440,
                'seed': seed, 'passed': passed, 'valid': True,
                'reason': 'passed' if passed else 'capacity_marker_or_evidence_failure',
                'token_evidence': copy.deepcopy(evidence), 'metrics': {'prompt_tokens': 61440,
                    'stream_complete': True, 'invalid_tool_calls': 0,
                    'completion_reason': 'stop' if passed else 'length',
                    'generated_tokens': 100 if passed else 4096,
                    'first_action_valid': passed, 'first_action_seconds': .3 if passed else None}})
        baseline['planned_screen_key'] = 'planned_screen:'+stage_key(baseline['runtime'], model, None)
        self.state.control(baseline['planned_screen_key'], baseline)
        return baseline

    def eligible(self, baseline, model=None):
        return marker_reasoning_eligible(self.config, self.state, baseline, model or self.models[0])

    def test_clean_default_considered_across_native_engines_without_claiming_capacity(self):
        for engine in ('bundled-llama', 'upstream-llama-nightly', 'turboquant-cuda'):
            with self.subTest(engine=engine):
                baseline = self.marker(engine=engine)
                self.assertTrue(self.eligible(baseline))
                self.assertFalse(baseline['eligible'] or baseline['capacity_qualified'])
                self.assertEqual(baseline['settings']['reasoning'], 'default')

    def test_bad_context_stream_failure_or_uncensored_output_never_enters(self):
        changes = [('valid', None, 'valid', False),
            ('oom', None, 'reason', 'model_oom'),
            ('infrastructure', None, 'reason', 'inference_infrastructure_error'),
            ('count', 'token_evidence', 'prompt_tokens', 60000),
            ('shift', 'token_evidence', 'context_shift', True),
            ('truncation', 'token_evidence', 'truncated', True),
            ('tokenizer', 'token_evidence', 'tokenization_verified', False),
            ('cache', 'token_evidence', 'cached_tokens', 999),
            ('stream', 'metrics', 'stream_complete', False),
            ('invalid_tool', 'metrics', 'invalid_tool_calls', 1),
            ('native_count', 'metrics', 'prompt_tokens', 60000),
            ('stop', 'metrics', 'completion_reason', 'stop'),
            ('unknown_generation', 'metrics', 'generated_tokens', None),
            ('not_exhausted', 'metrics', 'generated_tokens', 4095),
            ('existing_action', 'metrics', 'first_action_valid', True)]
        for label, section, field, value in changes:
            with self.subTest(label=label):
                baseline = self.marker()
                target = baseline['capacity'][0][section] if section else baseline['capacity'][0]
                target[field] = value
                self.assertFalse(self.eligible(baseline))

    def test_requires_all_seeds_valid16k_default_and_supported_pinned_help(self):
        baseline = self.marker()
        baseline['capacity'].pop()
        self.assertFalse(self.eligible(baseline))
        baseline = self.marker()
        baseline['small_capacity']['valid'] = False
        self.assertFalse(self.eligible(baseline))
        baseline = self.marker()
        baseline['settings']['reasoning'] = 'reduced'
        self.assertFalse(self.eligible(baseline))
        baseline = self.marker()
        baseline['runtime']['kind'] = 'container'
        self.assertFalse(self.eligible(baseline))
        baseline = self.marker()
        self.native_help.write_text('No supported reasoning control', encoding='utf-8')
        self.assertFalse(self.eligible(baseline))

    def test_terminal_current_stage_required_and_pending_capacity_blocks(self):
        baseline = self.marker()
        stale = dict(baseline, planned_screen_key='planned_screen:obsolete')
        self.assertFalse(self.eligible(stale))
        adapter = SimpleNamespace(state=self.state, requested=baseline['settings'],
            metadata=lambda: {'model_sha256': self.models[0]['sha256'], 'version_output': 'mock'})
        key = key_base(self.config, adapter, baseline['configuration_id'], 'capacity', 17, 'fresh', 0, 61440)
        self.state.enqueue(dict(key, fixture_prompt_hash=digest('mock-packed')))
        self.assertFalse(self.eligible(baseline))


@unittest.skipUnless(hasattr(tuning_module, 'marker_reasoning_eligible'), 'shared tuning integration is staged')
class MarkerReasoningIntegrationTests(unittest.TestCase):
    setUp = MarkerReasoningTests.setUp
    tearDown = MarkerReasoningTests.tearDown
    candidate = MarkerReasoningTests.candidate
    settings = MarkerReasoningTests.settings
    smoke = MarkerReasoningTests.smoke
    latency = MarkerReasoningTests.latency
    marker = MarkerReasoningTests.marker
    screened = tuning_tests.TuningTests.screened
    quality_qualified = tuning_tests.TuningTests.quality_qualified
    qualify = tuning_tests.TuningTests.qualify
    mock_runtime = tuning_tests.TuningTests.mock_runtime
    proposals = tuning_tests.TuningTests.proposals
    decisions = tuning_tests.TuningTests.decisions

    def test_marker_failed_default_profiles_run_first_and_each_full_quality_is_separate(self):
        baseline = self.marker()
        original = copy.deepcopy(baseline)
        self.quality_fail_profiles = {'reduced'}
        with self.mock_runtime(), patch.object(tuning_module._Tuner, 'cache_and_prefill',
                side_effect=AssertionError('censored default must probe reasoning first')):
            tuning_module.tune(self.config, self.state, object(), [baseline])
        self.assertEqual([call['settings']['reasoning'] for call in self.calls], ['reduced', 'disabled'])
        self.assertEqual({c['settings']['reasoning'] for c in self.qualifications}, {'reduced', 'disabled'})
        self.assertEqual(len({c['configuration_id'] for c in self.qualifications}), 2)
        self.assertTrue(all(c['configuration_id'] != baseline['configuration_id'] for c in self.qualifications))
        self.assertNotIn(baseline['configuration_id'], self.qualified)
        self.assertEqual(baseline, original)
        self.assertIsNone(self.state.run['started'])
        count = len(self.calls)
        with self.mock_runtime():
            tuning_module.tune(self.config, self.state, object(), [baseline])
        self.assertEqual(len(self.calls), count, 'current completed trials reuse their exact checkpoints')

    def test_disabled_smoke_failure_is_preserved_and_never_enters_full_qualification(self):
        baseline = self.marker()
        self.fail_smoke = lambda settings: settings.get('reasoning') == 'disabled'
        with self.mock_runtime():
            tuning_module.tune(self.config, self.state, object(), [baseline])
        self.assertEqual({c['settings']['reasoning'] for c in self.qualifications}, {'reduced'})
        self.assertEqual(len(self.calls), 2)

    def test_three_family_and_combined_three_rescue_anchor_limits(self):
        baselines = [self.marker(model) for model in self.models[:4]]
        with self.mock_runtime():
            tuning_module.tune(self.config, self.state, object(), baselines)
        phase = self.state.get_control('tuning_phase')
        self.assertEqual(len(phase['selected_families']), 3)
        self.assertEqual(len(self.calls), 6)
        self.assertLessEqual(len({call['model'] for call in self.calls}), 3)
        self.assertLessEqual(len(self.qualifications), self.config['limits']['maximum_final_configurations'])

    def test_existing_reduced_gemma_settings_reuse_and_disabled_remains_considered(self):
        model = dict(self.models[0], id='gemma4-12b-qat-q4')
        self.models[0] = model
        self.config['installed_model_candidates'][0] = model
        baseline = self.marker(model)
        reduced = self.candidate(model, settings={**baseline['settings'], 'reasoning': 'reduced'})
        with self.mock_runtime():
            tuning_module.tune(self.config, self.state, object(), [baseline])
        self.assertEqual([call['settings']['reasoning'] for call in self.calls], ['disabled'])
        self.assertEqual({c['settings']['reasoning'] for c in self.qualifications}, {'reduced', 'disabled'})

    def test_existing_configuration_cap_blocks_new_profiles_before_streams(self):
        baseline = self.marker()
        self.config['limits']['maximum_unique_runtime_configurations'] = 1
        with self.mock_runtime():
            tuning_module.tune(self.config, self.state, object(), [baseline])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.qualifications, [])
        self.assertTrue(any(d['reason'] == 'configuration_budget_exhausted' for d in self.decisions()))


if __name__ == '__main__':
    unittest.main()
