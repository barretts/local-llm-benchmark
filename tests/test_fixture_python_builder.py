"""Host-safe builder verification; emitted code is parsed, never executed."""

import ast
import json
import unittest

from localbench.fixture_python import build, SEEDS


FIXTURES = ("py01", "py02", "py03", "py04", "py05", "py06", "long03")


class PythonFixtureBuilderTests(unittest.TestCase):
    def test_deterministic_for_all_required_seeds(self):
        for fixture_id in FIXTURES:
            for seed in SEEDS:
                with self.subTest(fixture=fixture_id, seed=seed):
                    first = build(fixture_id, seed)
                    self.assertEqual(first, build(fixture_id, seed))
                    json.dumps(first, allow_nan=False, sort_keys=True)

    def test_seeded_cases_differ(self):
        for fixture_id in FIXTURES:
            with self.subTest(fixture=fixture_id):
                self.assertNotEqual(build(fixture_id, 42)["cases"], build(fixture_id, 314159)["cases"])

    def test_all_emitted_python_is_syntactically_valid(self):
        for fixture_id in FIXTURES:
            for seed in SEEDS:
                fixture = build(fixture_id, seed)
                for section in ("source", "gold", "public_tests", "hidden_tests"):
                    for path, contents in fixture[section].items():
                        with self.subTest(fixture=fixture_id, seed=seed, section=section, path=path):
                            ast.parse(contents, filename=path)
                            self.assertTrue(contents.endswith("\n"))

    def test_complete_counted_test_methods(self):
        for fixture_id in FIXTURES:
            fixture = build(fixture_id, 42)
            self.assertEqual(fixture["language"], "python")
            self.assertTrue(fixture["contract"])
            self.assertEqual(set(fixture["public_tests"]), {"test_public.py"})
            self.assertEqual(set(fixture["hidden_tests"]), {"test_hidden.py"})
            for section, count_key, minimum in (("public_tests", "public_count", 2), ("hidden_tests", "hidden_count", 6)):
                methods = []
                for contents in fixture[section].values():
                    tree = ast.parse(contents)
                    methods.extend(node.name for node in ast.walk(tree)
                                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                   and node.name.startswith("test_"))
                with self.subTest(fixture=fixture_id, section=section):
                    self.assertEqual(len(methods), fixture[count_key])
                    self.assertGreaterEqual(len(methods), minimum)
                    self.assertEqual(len(methods), len(set(methods)))

    def test_source_only_paths_and_real_bug_variants(self):
        for fixture_id in FIXTURES:
            fixture = build(fixture_id, 42)
            with self.subTest(fixture=fixture_id):
                self.assertEqual(set(fixture["source"]), set(fixture["gold"]))
                self.assertTrue(all(path.startswith("src/") and ".." not in path for path in fixture["source"]))
                self.assertNotEqual(fixture["source"], fixture["gold"])
                for contents in fixture["source"].values():
                    self.assertNotIn("TODO", contents)

    def test_settlement_only_one_writable_source(self):
        fixture = build("long03", 42)
        self.assertEqual(set(fixture["source"]), {"src/settlement.py"})
        tree = ast.parse(fixture["gold"]["src/settlement.py"])
        settle = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "settle")
        self.assertEqual([argument.arg for argument in settle.args.args], ["lines", "tax_rate"])

    def test_rejects_unknown_fixture_and_noninteger_seed(self):
        with self.assertRaises(ValueError):
            build("missing", 42)
        for seed in (True, "42", 1.5, None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                build("py01", seed)


if __name__ == "__main__":
    unittest.main()
