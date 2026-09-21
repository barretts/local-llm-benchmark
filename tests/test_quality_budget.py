"""Inert-code controller tests of the literal regular and long output budgets."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_quality_controller as controller
from tests.test_quality_controller import FakeAdapter, FakeClock, call
from localbench.tools import ToolExecutor


class QualityBudgetTests(unittest.TestCase):
    # Reuse inert fixture setup without inheriting or duplicating its tests.
    setUp = controller.QualityControllerTests.setUp
    tearDown = controller.QualityControllerTests.tearDown
    run_quality = controller.QualityControllerTests.run_quality

    def test_regular_actions_after_more_than_4096_generated_tokens_are_allowed(self):
        adapter = FakeAdapter([
            {"calls": [call("read_file", {"path": "src/example.py"})],
             "events": [{"type": "reasoning", "text": "mock reasoning before source inspection"}],
             "metrics": {"generated_tokens": 10000}},
            {"calls": [call("write_file", {"path": "src/example.py", "content": "INERT REPAIRED SOURCE\n"})],
             "metrics": {"generated_tokens": 5500}},
            {"content": "complete", "metrics": {"generated_tokens": 0}},
        ])
        result = self.run_quality(adapter)
        self.assertTrue(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual([entry["request"]["max_tokens"] for entry in adapter.requests], [32768, 22768, 17268])
        self.assertEqual(result["metrics"]["generated_tokens"], 15500)
        self.assertTrue(result["metrics"]["generated_tokens_complete"])
        self.assertEqual(result["metrics"]["generated_tokens_known"], 15500)
        self.assertIn("REPAIRED", Path(result["workspace"], "src/example.py").read_text(encoding="utf-8"))

    def test_regular_remaining_total_budget_cannot_be_exceeded_by_later_action(self):
        adapter = FakeAdapter([
            {"calls": [call("read_file", {"path": "src/example.py"})], "metrics": {"generated_tokens": 32760}},
            {"calls": [call("write_file", {"path": "src/example.py", "content": "REPAIRED"})],
             "metrics": {"generated_tokens": 9}},
        ])
        result = self.run_quality(adapter)
        self.assertEqual([entry["request"]["max_tokens"] for entry in adapter.requests], [32768, 8])
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "task_token_budget_violation")
        self.assertNotIn("REPAIRED", Path(result["workspace"], "src/example.py").read_text(encoding="utf-8"))

    def test_regular_output_budget_preserves_actual_remaining_context_limit(self):
        adapter = FakeAdapter([{"metrics": {"generated_tokens": 0}}], count=60000)
        self.run_quality(adapter)
        self.assertEqual(adapter.requests[0]["request"]["max_tokens"], 5536)

    def test_long_still_has_exactly_one_4096_token_response(self):
        fixture = copy.deepcopy(self.fixture)
        fixture.update(id="long-unit", permitted_final_action="write_file", allowed_tools=["write_file"],
            documents=[{"text": "ACTIVE EARLY"}, {"text": "ACTIVE MIDDLE"}, {"text": "ACTIVE LATE"}])
        packed = {"messages": [], "expected_tokens": 61440, "tokenization": {"count": 61440}, "region_positions": []}
        adapter = FakeAdapter([
            {"calls": [call("write_file", {"path": "src/example.py", "content": "REPAIRED"})],
             "metrics": {"generated_tokens": 4096}},
            {"metrics": {"generated_tokens": 0}},
        ], count=61440)
        with patch("localbench.quality.regional_prompt", return_value=packed):
            result = self.run_quality(adapter, fixture)
        self.assertTrue(result["passed"])
        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(adapter.requests[0]["request"]["max_tokens"], 4096)
        self.assertEqual(result["metrics"]["generated_tokens"], 4096)

    def test_stream_timeout_at_overall_task_deadline_is_functional_timeout(self):
        clock = FakeClock()
        adapter = FakeAdapter([{"valid_stream": False, "error": "TimeoutError: total request deadline elapsed",
            "after": lambda: setattr(clock, "now", 900.0)}])
        result = self.run_quality(adapter, clock=clock)
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "task_timeout")

    def test_early_stream_timeout_remains_infrastructure_invalid(self):
        clock = FakeClock()
        adapter = FakeAdapter([{"valid_stream": False, "error": "TimeoutError: socket receive timed out",
            "after": lambda: setattr(clock, "now", 20.0)}])
        result = self.run_quality(adapter, clock=clock)
        self.assertFalse(result["passed"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "inference_infrastructure_error")

    def test_tokenization_consuming_deadline_prevents_new_inference_request(self):
        clock = FakeClock()
        adapter = FakeAdapter([])
        tokenize = adapter.tokenize
        def delayed_tokenize(*args):
            value = tokenize(*args)
            clock.now = 901.0
            return value
        adapter.tokenize = delayed_tokenize
        result = self.run_quality(adapter, clock=clock)
        self.assertEqual(adapter.requests, [])
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "task_timeout")
        self.assertEqual([entry["phase"] for entry in self.grader.calls], ["public", "hidden"])

    def test_valid_stream_finishing_after_deadline_cannot_execute_late_action(self):
        clock = FakeClock()
        adapter = FakeAdapter([{"calls": [call("write_file", {"path": "src/example.py", "content": "REPAIRED"})],
            "after": lambda: setattr(clock, "now", 901.0)}])
        result = self.run_quality(adapter, clock=clock)
        self.assertEqual(result["metrics"]["generated_tokens"], 10)
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "task_timeout")
        self.assertNotIn("REPAIRED", Path(result["workspace"], "src/example.py").read_text(encoding="utf-8"))

    def test_last_regular_turn_tool_overrun_cannot_pass_or_execute_next_tool(self):
        self.config["limits"]["maximum_fixture_turns"] = 1
        clock = FakeClock()
        adapter = FakeAdapter([{"calls": [
            call("write_file", {"path": "src/example.py", "content": "REPAIRED"}), call("run_tests")]}])
        execute = ToolExecutor.execute
        calls = []
        def delayed_execute(tool, name, arguments):
            calls.append(name)
            value = execute(tool, name, arguments)
            clock.now = 901.0
            return value
        with patch("localbench.quality.ToolExecutor.execute", delayed_execute):
            result = self.run_quality(adapter, clock=clock)
        self.assertEqual(calls, ["write_file"])
        self.assertTrue(result["public_grading"]["passed"])
        self.assertTrue(result["hidden_grading"]["passed"])
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "task_timeout")

    def test_long_final_action_overrun_is_scored_but_cannot_pass(self):
        fixture = copy.deepcopy(self.fixture)
        fixture.update(id="long-unit", permitted_final_action="write_file", allowed_tools=["write_file"],
            documents=[{"text": "ACTIVE EARLY"}, {"text": "ACTIVE MIDDLE"}, {"text": "ACTIVE LATE"}])
        packed = {"messages": [], "expected_tokens": 61440, "tokenization": {"count": 61440}, "region_positions": []}
        adapter = FakeAdapter([{"calls": [call("write_file", {"path": "src/example.py", "content": "REPAIRED"})]}], count=61440)
        clock = FakeClock()
        execute = ToolExecutor.execute
        def delayed_execute(*args):
            value = execute(*args)
            clock.now = 901.0
            return value
        with patch("localbench.quality.regional_prompt", return_value=packed), patch(
                "localbench.quality.ToolExecutor.execute", delayed_execute):
            result = self.run_quality(adapter, fixture, clock)
        self.assertEqual(len(adapter.requests), 1)
        self.assertTrue(result["public_grading"]["passed"])
        self.assertTrue(result["hidden_grading"]["passed"])
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "task_timeout")

    def test_final_scoring_overrun_keeps_both_diagnostics_without_awarding_pass(self):
        self.config["limits"]["maximum_fixture_turns"] = 1
        clock = FakeClock()
        adapter = FakeAdapter([{"calls": [call("write_file", {"path": "src/example.py", "content": "REPAIRED"})]}])
        grade = self.grader.grade
        def delayed_grade(*args, **kwargs):
            value = grade(*args, **kwargs)
            clock.now = 901.0
            return value
        self.grader.grade = delayed_grade
        result = self.run_quality(adapter, clock=clock)
        self.assertEqual([entry["phase"] for entry in self.grader.calls], ["public", "hidden"])
        self.assertTrue(result["public_grading"]["passed"])
        self.assertTrue(result["hidden_grading"]["passed"])
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "task_timeout")

    def test_timeout_with_unknown_partial_generation_preserves_known_count_without_zero_total(self):
        clock = FakeClock()
        adapter = FakeAdapter([
            {"calls": [call("read_file", {"path": "src/example.py"})], "metrics": {"generated_tokens": 6000}},
            {"valid_stream": False, "error": "TimeoutError:local_request_deadline_exceeded",
             "metrics": {"generated_tokens": None}, "after": lambda: setattr(clock, "now", 900.0)},
        ])
        result = self.run_quality(adapter, clock=clock)
        self.assertEqual([entry["request"]["max_tokens"] for entry in adapter.requests], [32768, 26768])
        self.assertIsNone(result["metrics"]["generated_tokens"])
        self.assertFalse(result["metrics"]["generated_tokens_complete"])
        self.assertEqual(result["metrics"]["generated_tokens_known"], 6000)
        self.assertIn("server-reported", result["metrics"]["generation_token_count_provenance"])
        self.assertEqual(result["reason"], "task_timeout")
        self.assertTrue(result["valid"])

    def test_first_interrupted_stream_does_not_report_unknown_total_as_zero(self):
        adapter = FakeAdapter([{"valid_stream": False, "error": "ConnectionResetError",
            "metrics": {"generated_tokens": None}}])
        result = self.run_quality(adapter)
        self.assertIsNone(result["metrics"]["generated_tokens"])
        self.assertFalse(result["metrics"]["generated_tokens_complete"])
        self.assertEqual(result["metrics"]["generated_tokens_known"], 0)
        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
