"""Mock-only full search integration, enabled when the staged import is applied."""
import json
import unittest
from unittest.mock import patch

import localbench.search as search_module
import tests.test_search as search_tests
from tests.test_extra_discovery import candidate


@unittest.skipUnless(hasattr(search_module, 'extra_gguf_screens'), 'search patch is staged until GPU is free')
class ExtraDiscoveryIntegrationTests(unittest.TestCase):
    setUp = search_tests.SearchTests.setUp
    tearDown = search_tests.SearchTests.tearDown
    runtime = search_tests.SearchTests.runtime
    memory = search_tests.SearchTests.memory
    pending_job = search_tests.SearchTests.pending_job
    run_full = search_tests.SearchTests.run_full

    def model(self, config, state, lead, format='gguf'):
        value = search_tests.SearchTests.model(self, config, state, lead, format)
        if lead['id'].startswith('extra-'):
            value.update(sha256=lead['sha256'], bytes=lead['bytes'])
        return value

    def queue(self):
        (self.root/'artifacts'/'extra-discovered-gguf.json').write_text(json.dumps({
            'schema_version': 1, 'candidates': [candidate('extra-one'),
                dict(candidate('extra-two'), sha256='c'*64)]}), encoding='utf-8')

    def test_discovered_candidates_reach_normal_quality_tuning_final_and_demo_in_order(self):
        self.queue()
        with patch('localbench.extra_discovery.gguf_metadata', return_value={
                'general.architecture': 'qwen35', 'qwen35.context_length': 262144,
                'tokenizer.chat_template': 'embedded test template'}):
            result, final, demo = self.run_full(
                tuning=lambda *args: self.events.append(('tuning', len(args[-1]))) or [],
                qualified=lambda *args: [s for s in args[-1] if s['model'].startswith('extra-')])
        self.assertEqual(result['status'], 'covered_search_complete')
        acquisitions = [event for event in self.events if event[0] == 'acquire']
        self.assertEqual([event[1] for event in acquisitions if event[1].startswith('extra-')],
                         ['extra-one', 'extra-two'])
        extra_screens = [event for event in self.events if event[0] == 'screen' and event[2].startswith('extra-')]
        self.assertEqual(set(extra_screens), {('screen', engine, model) for engine in
            ('bundled-llama', 'upstream-llama-nightly') for model in ('extra-one', 'extra-two')})
        first_extra = next(index for index, event in enumerate(self.events)
                           if event[0] == 'acquire' and event[1].startswith('extra-'))
        installed = [index for index, event in enumerate(self.events)
                     if event[0] == 'screen' and event[2].startswith('installed-')
                     and event[1] in ('bundled-llama', 'upstream-llama-nightly')]
        self.assertEqual(len(installed), 4)
        self.assertLess(max(installed), first_extra)
        quality = next(index for index, event in enumerate(self.events) if event[0] == 'quality')
        tuning = next(index for index, event in enumerate(self.events) if event[0] == 'tuning')
        first_final = next(index for index, event in enumerate(self.events) if event[0] == 'final')
        first_demo = next(index for index, event in enumerate(self.events) if event[0] == 'demo')
        self.assertLess(quality, tuning)
        self.assertLess(tuning, first_final)
        self.assertLess(first_final, first_demo)
        self.assertEqual(final.call_count, min(4, self.config['limits']['maximum_final_configurations']))
        self.assertEqual(demo.call_count, final.call_count)
        self.assertEqual(self.state.get_control('discovered_weight_variants'),
                         ['ornith15-9b-exl3-hq4', 'extra-one', 'extra-two'])
        discovery = json.loads((self.root/'artifacts'/'engine-discovery.json').read_text(encoding='utf-8'))
        self.assertEqual(discovery['new_weight_leads'][0]['id'], 'ornith15-9b-exl3-hq4')

    def test_current_installed_pending_job_returns_before_any_discovery_download(self):
        self.queue()
        original = self.memory
        def memory(config, state, collector, runtime, model, settings=None):
            result = original(config, state, collector, runtime, model, settings)
            if model['id'] == 'installed-0' and runtime['engine'] == 'bundled-llama':
                self.pending_job(result[0]['configuration_id'], model['sha256'])
            return result
        self.memory = memory
        result, final, demo = self.run_full()
        self.assertEqual(result['status'], 'baseline_measurements_pending')
        self.assertFalse(any(event[0] == 'acquire' for event in self.events))
        self.assertIsNone(self.state.get_control('discovered_weight_variants'))
        self.assertFalse(final.called or demo.called)


if __name__ == '__main__':
    unittest.main()
