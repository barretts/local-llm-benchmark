"""Inert-source, mock-grader tests of coding-agent controller contracts."""

import copy
import json
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from localbench.config import file_hash, load
from localbench.quality import quality_task, source_scope
from localbench.measurement import capacity
from localbench.regional_prompt import regional_prompt
from localbench.scheduler import coding_job, measured_job
from localbench.state import State


class FakeState:
    run = {"id": "quality-mock-run"}


class FakeGrader:
    """Records phases and checks inert text; never imports agent source."""
    def __init__(self):
        self.calls = []

    def grade(self, workspace, fixture, phase="public", log_id=None):
        self.calls.append({"phase": phase, "log_id": log_id,
                           "workspace": str(workspace)})
        path = Path(workspace)/"src/example.py"
        passed = not path.exists() or "REPAIRED" in path.read_text(encoding="utf-8")
        return {"passed": passed, "complete": True, "reason": "passed" if passed else "test_failure",
                "feedback": "public assertion requires REPAIRED" if phase == "public" else {"passed": passed},
                "records": [{"id": phase+"-mock", "status": "passed" if passed else "failed"}]}


class FakeClock:
    now = 0.0
    def monotonic(self):
        return self.now


class FakeAdapter:
    engine = "mock-engine"
    requested = {"context": 65536, "slots": 1, "reasoning": "default"}

    def __init__(self, script, count=100):
        self.script, self.count = list(script), count
        self.requests, self.validations, self.restarts = [], [], 0
        self.effective = {"effective_context": 65536, "context_shift_disabled": True}
        self.http = SimpleNamespace(timeout=900)

    def fresh_restart(self):
        self.restarts += 1
        return {"owned": True}

    def tokenize(self, messages, tools):
        return {"count": self.count, "provenance": "deterministic mock exact count"}

    def tokenize_text(self, text):
        # A fixed exact-test tokenizer keeps response metadata convergence
        # independent of arbitrary host encoding; this is never a measurement.
        return [1] * 20

    def metadata(self):
        return {"image_digest": "mock-image-pin", "image_id": "mock-image-id",
                "help_sha256": "mock-help", "model_sha256": "mock-checkpoint",
                "tokenizer_helper_sha256": "mock-tokenizer"}

    model = {"id": "mock-model"}

    def stream_chat(self, request, **kwargs):
        self.requests.append({"request": copy.deepcopy(request), "kwargs": dict(kwargs)})
        entry = self.script.pop(0) if self.script else {}
        if callable(entry):
            entry = entry(request, kwargs)
        calls = copy.deepcopy(entry.get("calls", []))
        valid = entry.get("valid_stream", True)
        error = entry.get("error", "")
        if valid:
            try:
                for call in calls:
                    kwargs["validator"](call["name"], call["arguments"])
                    self.validations.append(copy.deepcopy(call))
            except ValueError as exc:
                valid, error = False, "StreamError:" + str(exc)
        metrics = {"prompt_tokens": self.count, "generated_tokens": 10,
                   "wall_seconds": 0.01, "first_action_seconds": 0.005 if calls else None,
                   "cached_tokens": 0, "evaluated_tokens": self.count}
        metrics.update(entry.get("metrics", {}))
        if entry.get("after"):
            entry["after"]()
        return {"valid_stream": valid, "error": error, "calls": calls,
                "content": entry.get("content", ""),
                "events": copy.deepcopy(entry.get("events", [])), "metrics": metrics}


def call(name, arguments=None, identity=None):
    value = {"name": name, "arguments": arguments or {}}
    if identity:
        value["id"] = identity
    return value


class QualityControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load()
        for key in ("project", "logs", "state", "artifacts", "fixture_source",
                    "agent_workspaces", "held_out_tests"):
            directory = self.root/key
            directory.mkdir()
            self.config["paths"][key] = str(directory)
        self.config["paths"]["project"] = str(self.root)
        self.fixture = {"id": "py-unit", "seed": 42, "version": "test-version",
            "fixture_hash": "test-fixture-hash", "language": "python",
            "source": {"src/example.py": "INERT INITIAL SOURCE\n"},
            "writable": ["src/example.py"], "contract": "Replace INITIAL with REPAIRED.",
            "allowed_tools": ["list_files", "read_file", "write_file", "apply_patch", "run_tests"]}
        self.grader = FakeGrader()
        self.serial = 0

    def tearDown(self):
        self.temp.cleanup()

    def run_quality(self, adapter, fixture=None, clock=None):
        self.serial += 1
        with patch("localbench.quality.DockerGrader", return_value=self.grader):
            if clock is None:
                return quality_task(self.config, FakeState(), adapter, fixture or self.fixture,
                                    "cfg", "job-"+str(self.serial), 1)
            with patch("localbench.quality.time.monotonic", clock.monotonic):
                return quality_task(self.config, FakeState(), adapter, fixture or self.fixture,
                                    "cfg", "job-"+str(self.serial), 1)

    def test_regular_tool_loop_keeps_failed_public_feedback_all_messages_and_reasoning(self):
        adapter = FakeAdapter([
            {"calls": [call("read_file", {"path": "src/example.py"}, "read-1")],
             "content": "inspect source", "events": [{"type": "reasoning", "text": "reason one"}]},
            {"calls": [call("run_tests", {}, "test-1")]},
            {"calls": [call("write_file", {"path": "src/example.py", "content": "INERT REPAIRED SOURCE\n"}, "write-1")]},
            {"calls": [call("run_tests", {}, "test-2")]},
            {"content": "finished"},
        ])
        result = self.run_quality(adapter)
        self.assertTrue(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(len(adapter.requests), 5)
        final_history = adapter.requests[-1]["request"]["messages"]
        self.assertEqual([message["role"] for message in final_history],
                         ["system", "user", "assistant", "tool", "assistant", "tool",
                          "assistant", "tool", "assistant", "tool"])
        self.assertEqual(final_history[2]["reasoning_content"], "reason one")
        self.assertEqual(final_history[3]["tool_call_id"], "read-1")
        public_feedback = json.loads(final_history[5]["content"])
        self.assertTrue(public_feedback["ok"])
        self.assertFalse(public_feedback["result"]["passed"])
        self.assertIn("public assertion", public_feedback["result"]["feedback"])
        self.assertNotIn("hidden", final_history[5]["content"])
        self.assertEqual([entry["phase"] for entry in self.grader.calls],
                         ["public", "public", "public", "hidden"])
        self.assertEqual(result["metrics"]["generated_tokens"], 50)
        self.assertEqual([entry["kwargs"]["mode"] for entry in adapter.requests],
                         ["fresh", "fixture_cached", "fixture_cached", "fixture_cached", "fixture_cached"])
        self.assertIsNotNone(result["metrics"]["first_edit_seconds"])
        self.assertIsNotNone(result["metrics"]["first_public_pass_seconds"])

    def test_missing_actual_input_or_generation_counter_is_infrastructure_invalid(self):
        for metrics in ({"prompt_tokens": None}, {"generated_tokens": None},
                        {"prompt_tokens": 1000}, {"generated_tokens": True}):
            with self.subTest(metrics=metrics):
                adapter = FakeAdapter([{"calls": [call("write_file", {"path": "src/example.py", "content": "REPAIRED"})],
                                        "metrics": metrics}])
                result = self.run_quality(adapter)
                self.assertFalse(result["passed"])
                self.assertFalse(result["valid"])
                self.assertEqual(result["reason"], "unverified_generation_or_input_counters")
                self.assertNotIn("REPAIRED", Path(result["workspace"], "src/example.py").read_text(encoding="utf-8"))

    def test_model_stream_error_and_network_error_remain_distinct(self):
        for error, reason, valid in (
            ("StreamError: malformed JSON", "malformed_or_unpermitted_tool_stream", True),
            ("ConnectionResetError", "inference_infrastructure_error", False)):
            with self.subTest(error=error):
                result = self.run_quality(FakeAdapter([{"valid_stream": False, "error": error}]))
                self.assertFalse(result["passed"])
                self.assertEqual(result["valid"], valid)
                self.assertEqual(result["reason"], reason)

    def test_regular_budget_is_exactly_16_turns_32768_tokens_and_900_seconds(self):
        self.assertEqual(self.config["limits"]["maximum_fixture_turns"], 16)
        self.assertEqual(self.config["limits"]["maximum_fixture_generated_tokens"], 32768)
        self.assertEqual(self.config["limits"]["per_task_timeout_seconds"], 900)
        read = {"calls": [call("read_file", {"path": "src/example.py"})]}
        adapter = FakeAdapter([copy.deepcopy(read) for _ in range(20)])
        self.run_quality(adapter)
        self.assertEqual(len(adapter.requests), 16)
        self.assertEqual([request["request"]["max_tokens"] for request in adapter.requests], [32768-10*turn for turn in range(16)])
        adapter = FakeAdapter([{**copy.deepcopy(read), "metrics": {"generated_tokens": 4096}} for _ in range(20)])
        result = self.run_quality(adapter)
        self.assertEqual(len(adapter.requests), 8)
        self.assertEqual(result["metrics"]["generated_tokens"], 32768)
        self.assertEqual(result["reason"], "task_context_or_generated_budget_exhausted")
        clock = FakeClock()
        adapter = FakeAdapter([{**read, "after": lambda: setattr(clock, "now", 901.0)}])
        result = self.run_quality(adapter, clock=clock)
        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(result["reason"], "task_timeout")
        self.assertLessEqual(adapter.http.timeout, 900)

    def test_server_generated_more_than_requested_is_model_invalid(self):
        adapter = FakeAdapter([{"metrics": {"generated_tokens": 32769}}])
        result = self.run_quality(adapter)
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "task_token_budget_violation")

    def test_negative_generated_usage_cannot_reduce_budget_and_pass(self):
        adapter = FakeAdapter([
            {"calls": [call("write_file", {"path": "src/example.py", "content": "REPAIRED"})],
             "metrics": {"generated_tokens": -1}}, {}])
        result = self.run_quality(adapter)
        self.assertFalse(result["valid"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "unverified_generation_or_input_counters")

    def test_safe_failed_task_still_receives_final_public_and_heldout_diagnostics(self):
        result = self.run_quality(FakeAdapter([{"valid_stream": False, "error": "StreamError: malformed JSON"}]))
        self.assertTrue(result["valid"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "malformed_or_unpermitted_tool_stream")
        self.assertIsNotNone(result["public_grading"])
        self.assertIsNotNone(result["hidden_grading"])
        self.assertEqual([entry["phase"] for entry in self.grader.calls], ["public", "hidden"])

    def test_long_uses_one_action_one_turn_no_model_feedback_and_three_document_dicts(self):
        fixture = copy.deepcopy(self.fixture)
        fixture.update(id="long-unit", permitted_final_action="write_file",
                       allowed_tools=["write_file"],
                       documents=[{"id": "early", "text": "ACTIVE EARLY"},
                                  {"id": "middle", "text": "ACTIVE MIDDLE"},
                                  {"id": "late", "text": "ACTIVE LATE"}])
        adapter = FakeAdapter([
            {"calls": [call("write_file", {"path": "src/example.py", "content": "REPAIRED"})]},
            {"calls": [call("run_tests")]}], count=61440)
        packed = {"messages": [{"role": "system", "content": "system"},
                               {"role": "user", "content": "all three active documents and request"}],
                  "expected_tokens": 61440, "tokenization": {"count": 61440, "provenance": "mock"},
                  "region_positions": [{"fraction": value} for value in (.05, .5, .95)]}
        with patch("localbench.quality.regional_prompt", return_value=packed) as regional:
            result = self.run_quality(adapter, fixture)
        self.assertTrue(result["passed"])
        self.assertEqual(adapter.restarts, 1)
        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(adapter.requests[0]["request"]["max_tokens"], 4096)
        self.assertEqual(regional.call_args[0][3], fixture["documents"])
        self.assertEqual([entry["phase"] for entry in self.grader.calls], ["public", "hidden"])
        self.assertTrue(result["token_evidence"]["cache_isolation_verified"])

    def test_long_rejects_multiple_final_actions_before_any_source_change(self):
        fixture = copy.deepcopy(self.fixture)
        fixture.update(id="long-unit", permitted_final_action="write_file", allowed_tools=["write_file"],
                       documents=["early", "middle", "late"])
        adapter = FakeAdapter([{"calls": [
            call("write_file", {"path": "src/example.py", "content": "REPAIRED"}),
            call("write_file", {"path": "src/example.py", "content": "REPAIRED AGAIN"})]}], count=61440)
        packed = {"messages": [], "expected_tokens": 61440,
                  "tokenization": {"count": 61440}, "region_positions": []}
        with patch("localbench.quality.regional_prompt", return_value=packed):
            result = self.run_quality(adapter, fixture)
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "long_requires_exactly_one_final_action")
        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(Path(result["workspace"], "src/example.py").read_text(encoding="utf-8"),
                         fixture["source"]["src/example.py"])

    def test_scope_rejects_readonly_changes_extra_files_and_ownership_tampering(self):
        root = self.root/"scope"
        (root/"src").mkdir(parents=True)
        marker = root/".localbench-workspace.json"
        marker.write_text("{}", encoding="utf-8")
        source = root/"src"/"example.py"
        source.write_text("INITIAL", encoding="utf-8")
        fixture = {"source": {"src/example.py": "INITIAL"}, "writable": [], "language": "python"}
        marker_hash = file_hash(marker)
        self.assertIsNone(source_scope(root, fixture, marker_hash))
        source.write_text("TAMPERED", encoding="utf-8")
        self.assertEqual(source_scope(root, fixture, marker_hash), "readonly_dependency_tampering")
        source.write_text("INITIAL", encoding="utf-8")
        extra = root/"personal.txt"
        extra.write_text("extra", encoding="utf-8")
        self.assertEqual(source_scope(root, fixture, marker_hash), "unauthorized_workspace_file")
        extra.unlink()
        marker.write_text('{"tampered":true}', encoding="utf-8")
        self.assertEqual(source_scope(root, fixture, marker_hash), "workspace_ownership_tampering")

    def test_source_path_attack_is_model_invalid_without_hidden_source_access(self):
        adapter = FakeAdapter([{"calls": [call("read_file", {"path": "../hidden/tests.py"})]}])
        result = self.run_quality(adapter)
        self.assertFalse(result["passed"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "malformed_or_unpermitted_tool_stream")
        self.assertEqual(len(adapter.requests), 1)

    def test_capacity_16k_has_only_literal_submit_tool_and_correct_ordered_evidence(self):
        captured = {}
        def regional(adapter, tools, system, regions, query, seed, target, tolerance):
            captured.update(tools=copy.deepcopy(tools), regions=regions, query=query)
            return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": query}],
                    "expected_tokens": target, "tokenization": {"count": target, "provenance": "mock"},
                    "region_positions": [{"fraction": f} for f in (.05, .5, .95)], "prompt_hash": "mock-packed"}
        def response(request, kwargs):
            values = [re.search(r"exact_value=(.*)", region).group(1) for region in captured["regions"]]
            ids = [re.search(r"DOCUMENT id=(.*)", region).group(1) for region in captured["regions"]]
            return {"calls": [call("submit_answer", {"answer": {"markers": values}, "evidence_ids": ids})]}
        adapter = FakeAdapter([response], count=16384)
        with patch("localbench.measurement.warm"), patch("localbench.measurement.regional_prompt", side_effect=regional):
            result = capacity(self.config, FakeState(), adapter, "cfg", 17, target=16384)
        self.assertTrue(result["passed"])
        self.assertEqual([tool["function"]["name"] for tool in captured["tools"]], ["submit_answer"])
        self.assertIn("answer={markers:[the three exact values]}", captured["query"])
        self.assertEqual(result["evidence_ids"], ["capacity-17-0", "capacity-17-1", "capacity-17-2"])
        self.assertEqual(result["markers_passed"], 3)
        self.assertEqual(adapter.restarts, 1)

    def test_region_packer_accepts_document_dicts_and_measures_all_three_positions(self):
        class CountingAdapter:
            def tokenize(self, messages, tools):
                return {"count": 30+sum(len(message["content"]) for message in messages),
                        "provenance": "mock exact"}
        documents = [{"text": "SIGNED EARLY"}, {"text": "SIGNED MIDDLE"}, {"text": "SIGNED LATE"}]
        packed = regional_prompt(CountingAdapter(), [], "system", documents, "FINAL QUERY", 42, 4096, 0)
        self.assertEqual(packed["expected_tokens"], 4096)
        for document in documents:
            self.assertIn(document["text"], packed["messages"][1]["content"])
        for position, expected in zip(packed["region_positions"], (.05, .5, .95)):
            self.assertLess(abs(position["fraction"]-expected), .001)

    def test_scheduler_retries_only_infrastructure_and_preserves_functional_grade_under_foreign_workload(self):
        class Collector:
            def snapshot(self, *args):
                return {"foreign_workload_overlap": True}
        state = State(self.config)
        try:
            adapter = FakeAdapter([])
            adapter.state = state
            cases = (
                ("model-failure", {"passed": False, "valid": True, "reason": "deterministic_test_failure"}, "failed"),
                ("counter-invalid", {"passed": False, "valid": False, "reason": "unverified_generation_or_input_counters"}, "invalid"),
                ("infrastructure", {"passed": False, "valid": False, "reason": "inference_infrastructure_error"}, "pending"),
                ("functional-pass", {"passed": True, "valid": True, "reason": "passed"}, "passed"),
            )
            for fixture_id, outcome, status in cases:
                with self.subTest(fixture_id=fixture_id):
                    fixture = {**self.fixture, "id": fixture_id, "fixture_hash": "hash-"+fixture_id}
                    with patch("localbench.scheduler.load_fixture", return_value=fixture), patch(
                            "localbench.scheduler.quality_task", return_value=copy.deepcopy(outcome)):
                        result = coding_job(self.config, state, adapter, Collector(),
                                            "cfg", fixture_id, 42)
                    row = state.db.execute("SELECT status FROM jobs ORDER BY rowid DESC LIMIT 1").fetchone()
                    self.assertEqual(row["status"], status)
                    self.assertEqual(result["valid"], outcome["valid"])
                    self.assertEqual(result["passed"], outcome["passed"])
            self.assertIsNone(state.run["started"])
        finally:
            state.close()

    def test_scheduler_rejects_timing_overlapping_foreign_gpu_workload(self):
        class Collector:
            def snapshot(self, *args):
                return {"foreign_workload_overlap": True}
        state = State(self.config)
        adapter = FakeAdapter([])
        adapter.state = state
        try:
            def operation(on_packed):
                on_packed({"prompt_hash": "exact-mock-hash"})
                return {"passed": True, "valid": True, "reason": "passed"}
            result = measured_job(self.config, state, adapter, Collector(), "cfg",
                                  "timing", 42, "fresh", 0, 61440, operation)
            self.assertFalse(result["valid"])
            self.assertFalse(result["passed"])
            self.assertEqual(result["reason"], "foreign_workload_overlap")
            self.assertEqual(state.db.execute("SELECT status FROM jobs").fetchone()["status"], "pending")
            self.assertIsNone(state.run["started"])
        finally:
            state.close()


if __name__ == "__main__":
    unittest.main()
