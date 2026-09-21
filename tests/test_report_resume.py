import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from localbench.config import canonical, digest, load
from localbench.report import Report
from localbench.scheduler import PROTOCOL_HASHES, key_base
from localbench.state import State


class MockAdapter:
    requested = {"context": 65536, "slots": 1, "reasoning": "default"}

    def __init__(self, state):
        self.state = state

    def metadata(self):
        return {"binary_sha256": "a" * 64, "adjacent_dlls": {}, "version_output": "mock-pinned-v1",
            "model_sha256": "b" * 64}


class ReportResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ("project", "state", "logs", "artifacts"):
            self.config["paths"][name] = str(Path(self.temp.name) / name)
        self.now = 1000.
        self.state = State(self.config, clock=lambda: self.now)
        self.adapter = MockAdapter(self.state)
        self.state.entity("configuration", "candidate", {"engine_id": "mock-native"})

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def evidence(self):
        return {"effective_context_tokens": 65536, "prompt_tokens": 61440,
            "expected_prompt_tokens": 61440, "tokenization_verified": True, "truncated": False,
            "context_shift": False, "cached_tokens": 0, "cache_isolation_verified": True}

    def key(self, kind="quality", obsolete=False, mode="fresh", replicate=0, prompt_hash="fixture-py01"):
        target = None if kind in {"quality", "demo", "common_byte"} else 61440
        fixture_hash = prompt_hash if kind == "quality" else None
        if obsolete:
            with patch.dict(PROTOCOL_HASHES, {name: "obsolete-" + value for name, value in PROTOCOL_HASHES.items()}):
                base = key_base(self.config, self.adapter, "candidate", kind, 42 if kind == "quality" else 17,
                    mode, replicate, target, fixture_hash)
        else:
            base = key_base(self.config, self.adapter, "candidate", kind, 42 if kind == "quality" else 17,
                mode, replicate, target, fixture_hash)
        return {**base, "fixture_prompt_hash": prompt_hash}

    def result(self, kind="quality", **changes):
        return {"kind": kind, "configuration_id": "candidate", "passed": True, "valid": True,
            "seed": 42 if kind == "quality" else 17, "fixture": "py01", "group": "regular",
            "token_evidence": self.evidence(), "isolated": True, "markers_passed": 3,
            "retrieved_values": ["a", "b", "c"], "mode": "fresh", "replicate": 1000,
            "final_validation": kind == "timing", "stability_passed": True,
            "metrics": {"first_action_seconds": 10., "first_action_valid": True, "invalid_tool_calls": 0,
                "generated_tokens": 128, "decode_seconds": 2., "sustained_throughput_qualified": True}, **changes}

    def add(self, key, result=None, status="passed", transient=False):
        job = self.state.enqueue(key)
        self.state.finish(self.state.begin(job), status,
            result or self.result(key["kind"]), reason=None if status == "passed" else "mock_infrastructure",
            transient=transient)
        self.now += 1
        return job

    def score(self):
        return Report(self.config, self.state).summarize()["configurations"][0]

    def test_current_invalid_demo_does_not_inherit_historical_pass_and_raw_export_keeps_it(self):
        key = self.key("demo")
        key["logical_plan_hash"] = "c" * 64  # Handoff uses its immutable plan hash.
        job = self.add(key)
        self.assertTrue(self.score()["gates"]["isolated_agent_demo"])
        invalid = self.result("demo", passed=False, valid=False, checkpoint_validation_failed=True)
        with self.state.db:
            self.state.db.execute("UPDATE jobs SET status='invalid',reason=?,result=? WHERE id=?",
                ("demo_checkpoint_validation_failed", canonical(invalid), job))
        report = Report(self.config, self.state)
        score = report.write()["configurations"][0]
        self.assertFalse(score["gates"]["isolated_agent_demo"])
        raw = json.loads((Path(self.config["paths"]["artifacts"]) / "results.json").read_text())
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["status"], "passed")
        self.assertEqual(raw[0]["job_status"], "invalid")
        self.assertTrue(raw[0]["result"]["passed"])
        self.assertIsNone(self.state.run["started"])

    def test_obsolete_scheduler_protocol_never_qualifies_any_recognized_workload(self):
        for kind in ("quality", "capacity", "timing", "throughput", "common_byte"):
            self.add(self.key(kind, obsolete=True, replicate=1000 if kind == "timing" else 0))
        report = Report(self.config, self.state)
        self.assertEqual(report._latest_completed(report.attempts()), [])
        score = self.score()
        self.assertEqual(score["regular"]["passed"], 0)
        self.assertFalse(score["capacity"]["17"])
        self.assertEqual(score["latency"]["fresh"]["first_action"]["n"], 0)
        self.assertFalse(score["throughput"]["exact_sustained_64k"])
        self.assertEqual(score["attempt_count"], 5)

    def test_obsolete_quality_pass_is_missing_while_current_protocol_replacement_is_pending(self):
        self.add(self.key(obsolete=True))
        pending = self.state.enqueue(self.key())
        self.assertEqual(self.state.db.execute("SELECT status FROM jobs WHERE id=?", (pending,)).fetchone()[0], "pending")
        score = self.score()
        self.assertEqual(score["regular"]["passed"], 0)
        self.assertEqual(score["regular"]["attempted"], 0)
        self.assertIn({"fixture": "py01", "seed": 42}, score["regular"]["missing"])
        self.assertEqual(score["attempt_count"], 1)

    def test_obsolete_final_failure_stays_raw_but_does_not_block_current_sequence(self):
        obsolete = self.key("timing", obsolete=True, replicate=1000)
        self.add(obsolete, self.result("timing", passed=False, valid=False, stability_passed=False),
            status="failed")
        self.add(self.key("timing", replicate=1000), self.result("timing"))
        report = Report(self.config, self.state)
        score = report.write()["configurations"][0]
        self.assertTrue(score["gates"]["runtime_stability"])
        self.assertEqual(score["stability_failures"], [])
        self.assertEqual(score["latency"]["fresh"]["first_action"]["n"], 1)
        raw = json.loads((Path(self.config["paths"]["artifacts"]) / "results.json").read_text())
        self.assertEqual(len(raw), 2)
        self.assertEqual(raw[0]["status"], "failed")
        self.assertFalse(raw[0]["result"]["stability_passed"])

    def test_current_protocol_runtime_crash_cannot_be_erased_by_successful_retry(self):
        key = self.key("timing", replicate=1000)
        job = self.state.enqueue(key)
        failure = self.result("timing", passed=False, valid=False, stability_passed=False)
        self.state.finish(self.state.begin(job), "invalid", failure, reason="native_runtime_crash", transient=True)
        self.state.finish(self.state.begin(job), "passed", self.result("timing"))
        score = self.score()
        self.assertFalse(score["gates"]["runtime_stability"])
        self.assertEqual(len(score["stability_failures"]), 1)
        self.assertEqual(score["stability_failures"][0]["reason"], "native_runtime_crash")
        self.assertEqual(score["latency"]["fresh"]["first_action"]["n"], 1)

    def test_obsolete_throughput_count_does_not_satisfy_current_sample_gate(self):
        self.add(self.key("throughput", obsolete=True), self.result("throughput"))
        self.assertFalse(self.score()["gates"]["full_context_throughput_sample"])
        self.add(self.key("throughput"), self.result("throughput"))
        self.assertTrue(self.score()["gates"]["full_context_throughput_sample"])

    def test_new_pending_job_without_attempt_supersedes_same_protocol_quality_pass(self):
        self.add(self.key())
        replacement = {**self.key(), "tokenizer": "new-verified-tokenizer"}
        self.state.enqueue(replacement)
        self.assertEqual(self.score()["regular"]["passed"], 0)
        self.assertEqual(self.score()["regular"]["attempted"], 0)
        self.add(replacement)
        self.assertEqual(self.score()["regular"]["passed"], 1)

    def capacity_placeholder(self, packed_key, status="skipped", attempted=False, **changes):
        reason = "stop_before_execution_unpacked_placeholder"
        base = {name: value for name, value in packed_key.items() if name != "fixture_prompt_hash"}
        key = {**base, "fixture_prompt_hash": digest({"unpacked_plan": base})}
        packed_id = digest(packed_key)
        result = {"unstarted": True, "superseded_by": packed_id, "reason": reason, **changes}
        job = self.state.enqueue(key)
        if attempted:
            self.state.finish(self.state.begin(job), status, result, reason=reason)
        else:
            with self.state.db:
                self.state.db.execute("UPDATE jobs SET status=?,reason=?,result=? WHERE id=?",
                    (status, reason, canonical(result), job))
        return job

    def test_retired_proven_unstarted_placeholder_keeps_completed_packed_capacity_visible(self):
        packed = self.key("capacity", prompt_hash="actual-packed-prompt")
        packed_id = self.add(packed)
        orphan = self.capacity_placeholder(packed)
        report = Report(self.config, self.state)
        selected = report._latest_completed(report.attempts())
        self.assertEqual([row["job_id"] for row in selected], [packed_id])
        self.assertTrue(self.score()["capacity"]["17"])
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM attempts WHERE job_id=?", (orphan,)).fetchone()[0], 0)
        self.assertEqual(len(report.attempts()), 1)
        self.assertIsNone(self.state.run["started"])

    def test_genuine_pending_capacity_replacement_without_attempt_still_blocks_completed_result(self):
        packed = self.key("capacity", prompt_hash="actual-packed-prompt")
        self.add(packed)
        replacement = {**packed, "fixture_prompt_hash": "new-packed-prompt"}
        pending = self.state.enqueue(replacement)
        self.assertEqual(self.state.db.execute("SELECT status FROM jobs WHERE id=?", (pending,)).fetchone()[0], "pending")
        self.assertFalse(self.score()["capacity"]["17"])

    def test_attempted_skipped_placeholder_marker_does_not_hide_replacement(self):
        packed = self.key("capacity", prompt_hash="actual-packed-prompt")
        self.add(packed)
        self.capacity_placeholder(packed, attempted=True)
        self.assertFalse(self.score()["capacity"]["17"])
        self.assertEqual(len(Report(self.config, self.state).attempts()), 2)

    def test_retirement_marker_without_matching_replacement_proof_still_blocks_result(self):
        packed = self.key("capacity", prompt_hash="actual-packed-prompt")
        self.add(packed)
        orphan = self.capacity_placeholder(packed, superseded_by="nonexistent-job")
        self.assertFalse(self.score()["capacity"]["17"])
        with self.state.db:
            result = {"unstarted": True, "superseded_by": digest(packed),
                "reason": "stop_before_execution_unpacked_placeholder"}
            self.state.db.execute("UPDATE jobs SET reason='different_retirement',result=? WHERE id=?",
                (canonical(result), orphan))
        self.assertFalse(self.score()["capacity"]["17"])

    def test_new_invalid_logical_grade_supersedes_pass_without_counting_as_functional_failure(self):
        self.add(self.key())
        replacement = {**self.key(), "tokenizer": "new-verified-tokenizer"}
        self.add(replacement, self.result(passed=False, valid=False), status="invalid")
        score = self.score()
        self.assertEqual(score["regular"]["passed"], 0)
        self.assertEqual(score["regular"]["attempted"], 0)
        self.assertFalse(score["gates"]["regular_coverage"])
        self.assertEqual(score["attempt_count"], 2)

    def test_current_pending_retry_never_uses_earlier_completed_pass(self):
        job = self.add(self.key())
        with self.state.db:
            self.state.db.execute("UPDATE jobs SET status='pending' WHERE id=?", (job,))
        attempt = self.state.begin(job)
        self.assertEqual(self.score()["regular"]["passed"], 0)
        self.state.finish(attempt, "passed", self.result())
        self.assertEqual(self.score()["regular"]["passed"], 1)

    def test_isolated_demo_coding_run_does_not_replace_required_regular_fixture_grade(self):
        self.add(self.key(), self.result(passed=False, valid=True), status="failed")
        self.add(self.key(mode="isolated-demo", replicate=1))
        score = self.score()
        self.assertEqual(score["regular"]["passed"], 0)
        self.assertEqual(score["regular"]["attempted"], 1)

    def test_unknown_aggregate_workload_attribution_is_explicit_uncertainty_not_invented_foreign(self):
        resource = {"foreign_workload_overlap": False, "foreign_workload_attribution_uncertain": True,
            "foreign_workload_evidence_available": False, "uncertain_gpu_pids": [99]}
        self.add(self.key("capacity"), self.result("capacity", resources=resource))
        summary = Report(self.config, self.state).summarize()
        score = summary["configurations"][0]
        self.assertTrue(score["capacity"]["17"])
        self.assertTrue(score["resource_attribution_uncertainty"])
        self.assertTrue(any("aggregate WSL/Docker" in note for note in summary["ranking_uncertainty"]))
        self.assertNotIn("foreign_workload", score["failed_gates"])

    def test_quality_functional_grade_remains_valid_under_foreign_gpu_overlap(self):
        self.add(self.key(), self.result(resources={"foreign_workload_overlap": True}))
        self.assertEqual(self.score()["regular"]["passed"], 1)


if __name__ == "__main__":
    unittest.main()
