"""Trusted builder tests only; emitted TypeScript/JavaScript is never executed here."""

import json
import re
import unittest

from localbench.fixture_typescript import build


FIXTURES = ("ts01", "ts02", "ts03", "ts04", "ts05", "ts06", "long04")
SEEDS = (42, 314159, 271828)


class TypeScriptFixtureBuilderTests(unittest.TestCase):
    def test_all_seeded_fixtures_are_deterministic_and_json_serializable(self):
        for fixture_id in FIXTURES:
            for seed in SEEDS:
                with self.subTest(fixture=fixture_id, seed=seed):
                    first = build(fixture_id, seed)
                    self.assertEqual(first, build(fixture_id, seed))
                    self.assertEqual(first, json.loads(json.dumps(first)))
                    self.assertNotEqual(first["cases"], build(fixture_id, seed + 1)["cases"])

    def test_builder_interface_source_paths_and_expected_counts(self):
        expected_paths = {
            "ts01": {"src/cache.ts"}, "ts02": {"src/page.ts"},
            "ts03": {"src/edit.ts"}, "ts04": {"src/graph.ts"},
            "ts05": {"src/settings.ts"},
            "ts06": {"src/domain.ts", "src/repository.ts", "src/store.ts"},
            "long04": {"src/request.ts", "src/errors.ts"},
        }
        for fixture_id in FIXTURES:
            with self.subTest(fixture=fixture_id):
                fixture = build(fixture_id, 42)
                self.assertEqual(fixture["language"], "typescript")
                self.assertEqual(set(fixture["source"]), expected_paths[fixture_id])
                self.assertEqual(set(fixture["gold"]), expected_paths[fixture_id])
                self.assertNotEqual(fixture["source"], fixture["gold"])
                self.assertEqual(set(fixture["public_tests"]), {"test_public.cjs"})
                self.assertEqual(set(fixture["hidden_tests"]), {"test_hidden.cjs"})
                self.assertGreaterEqual(fixture["public_count"], 2)
                self.assertGreaterEqual(fixture["hidden_count"], 6)
                self.assertGreater(len(fixture["contract"]), 300)
                for visibility in ("public", "hidden"):
                    emitted = fixture[f"{visibility}_tests"][f"test_{visibility}.cjs"]
                    records = re.findall(r"^test\('", emitted, re.MULTILINE)
                    self.assertEqual(len(records), fixture[f"{visibility}_count"])
                    self.assertIn("require('node:test')", emitted)
                    self.assertIn("require('node:assert/strict')", emitted)
                    self.assertIn("require('/workspace/.build/", emitted)
                    self.assertNotIn("Math.random", emitted)
                    self.assertNotIn("setTimeout", emitted)

    def test_long_retry_dependency_is_identical_in_bug_and_gold(self):
        fixture = build("long04", 314159)
        self.assertEqual(fixture["source"]["src/errors.ts"], fixture["gold"]["src/errors.ts"])
        self.assertIn("READ-ONLY dependency", fixture["contract"])
        self.assertIn('import {TimeoutError} from "./errors"', fixture["source"]["src/request.ts"])
        self.assertIn("' \\t\\n '", fixture["hidden_tests"]["test_hidden.cjs"])

    def test_seeded_text_cases_preserve_utf16_surrogate_boundaries(self):
        for seed in SEEDS:
            for case in build("ts03", seed)["cases"]:
                self.assertIn("\r\n😀", case["text"])
                first, second = case["edits"]
                self.assertEqual((first["start"], first["end"]), (0, 4))
                self.assertEqual((second["start"], second["end"]), (9, 11))
                self.assertNotIn("😀", case["expected"])
                self.assertIn("𝄞tail", case["expected"])
                self.assertIn("\r\n", case["expected"])

    def test_invalid_fixture_and_seed_are_rejected(self):
        with self.assertRaises(ValueError):
            build("ts07", 42)
        for seed in (True, False, "42", 42.0, None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                build("ts01", seed)


if __name__ == "__main__":
    unittest.main()
