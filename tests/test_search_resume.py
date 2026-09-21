"""Resume regression tests with real durable State and mocked stage operations.

Only tiny temporary fixture bytes are read. No model/runtime, GPU, network,
package installation, compiler or external process is used by this suite.
"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import tempfile
import unittest

from localbench import search
from localbench.config import atomic_json, digest, file_hash
from localbench.scheduler import PROTOCOL_HASHES, key_base, prior_result
from localbench.state import State


class SearchResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.binary = self.root / "dummy-runtime.exe"
        self.binary.write_bytes(b"not executable; mock-only runtime identity")
        self.models = []
        for number in range(2):
            path = self.root / ("fixture-" + str(number) + ".gguf")
            path.write_bytes(("tiny temporary fixture " + str(number)).encode())
            self.models.append({"id": "fixture-" + str(number), "path": str(path), "priority": number + 2})
        self.config = {"_spec_hash": "temporary-search-resume-spec", "paths": {
            "state": str(self.root / "state"), "artifacts": str(self.root / "artifacts"), "logs": str(self.root / "logs")},
            "installed_tools": {"bundled_llama_server": str(self.binary)}, "installed_model_candidates": self.models,
            "download_weights": [], "linux_checkpoint_probes": [], "default_sampling": {"temperature": 0.6},
            "private_ports": {"native": 38201}, "limits": {"benchmark_elapsed_seconds": 604800,
                "new_model_weight_bytes": 300000000000, "retry_transient_attempts": 2,
                "maximum_final_configurations": 1, "maximum_new_discovery_weight_variants": 2},
            "grading": {"smoke_jobs": []}}
        self.state = State(self.config, clock=lambda: 1000)
        self.state.control("harness_verified", True)
        self.state.control("fixture_version", "mock-fixture-version")
        self.rt = {"engine": "mock-engine", "kind": "native", "binary_path": str(self.binary)}
        self.settings = {"context": 65536, "slots": 1, "reasoning": "default"}
        self.collector = object()
        self.calls = []
        self.failures_before_success = 0
        self.no_progress = set()
        self.raise_after_attempt = False
        self.next_setup_failure = False
        atomic_json(self.root / "artifacts" / "engine-discovery.json", {"new_weight_leads": []})
        self.stack = ExitStack()
        self.stage_mock = self.stack.enter_context(patch("localbench.search.screen", side_effect=self.stage))
        # Production stage_key hashes installed cuBLAS dependencies for the
        # bundled engine. This test uses its temporary identity, avoiding reads
        # of the user's installed runtime libraries.
        self.stack.enter_context(patch("localbench.search.stage_key", side_effect=self.stage_key))

    def tearDown(self):
        self.stack.close()
        self.state.close()
        self.tmp.cleanup()

    def stage_key(self, rt, model, settings):
        return digest({"engine": rt["engine"], "runtime": rt, "model": model.get("sha256") or file_hash(model["path"]),
                       "settings": settings, "protocol": PROTOCOL_HASHES})

    def identifier(self, rt=None, model=None, settings=None):
        return digest({"engine": (rt or self.rt)["engine"], "model": (model or self.models[0])["id"],
                       "settings": settings or self.settings})

    def adapter(self, rt=None, model=None, settings=None):
        rt, model, requested = rt or self.rt, model or self.models[0], settings or self.settings
        metadata = {"binary_sha256": digest(rt["engine"]), "adjacent_dlls": {}, "version_output": "mock runtime",
                    "model_sha256": model.get("sha256") or file_hash(model["path"])}
        return SimpleNamespace(state=self.state, engine=rt["engine"], model=model, requested=requested,
                               metadata=lambda: copy.deepcopy(metadata))

    def job_key(self, kind="capacity", rt=None, model=None, fixture_hash=None, **changes):
        adapter = self.adapter(rt, model)
        identifier = self.identifier(rt, model)
        target = None if kind in ("quality", "common_byte") else 16384
        key = key_base(self.config, adapter, identifier, kind, 42 if kind == "quality" else 17,
                       "fresh", 0, target, fixture_hash)
        key["fixture_prompt_hash"] = fixture_hash or digest({"temporary_public_prompt": kind})
        key.update(changes)
        return key

    def pending(self, identifier=None):
        return search.pending_for_stage(self.state, identifier or self.identifier())

    def stage(self, config, state, collector, rt, model, settings=None, tuning=False):
        self.calls.append((rt["engine"], model["id"]))
        identifier = self.identifier(rt, model, settings)
        if self.next_setup_failure and len(self.calls) > 1:
            return {"kind": "engine_probe", "engine": rt["engine"], "model": model["id"],
                    "eligible": False, "passed": False, "valid": False, "reason": "mock_setup_unavailable"}
        key = self.job_key(rt=rt, model=model)
        previous = prior_result(state, {name: value for name, value in key.items() if name != "fixture_prompt_hash"})
        if previous is not None:
            passed, reason = previous.get("passed", False), previous.get("reason")
        else:
            job = state.enqueue(key)
            if (rt["engine"], model["id"]) in self.no_progress:
                passed, reason = False, "mock_stage_not_progressing"
            else:
                count = state.db.execute("SELECT COUNT(*) FROM attempts WHERE job_id=?", (job,)).fetchone()[0]
                attempt = state.begin(job)
                passed = count >= self.failures_before_success
                reason = "passed" if passed else "foreign_workload_overlap"
                result = {"kind": "capacity", "configuration_id": identifier, "passed": passed, "valid": passed, "reason": reason}
                state.finish(attempt, "passed" if passed else "invalid", result, reason=reason, transient=not passed)
                if self.raise_after_attempt:
                    self.raise_after_attempt = False
                    raise RuntimeError("stop_after_current")
        return {"engine": rt["engine"], "model": model["id"], "configuration_id": identifier, "eligible": passed,
                "capacity_qualified": passed, "passed": passed, "valid": passed, "reason": reason,
                "settings": settings or copy.deepcopy(self.settings), "runtime": rt}

    def run_stage(self):
        return search.run_screen(self.config, self.state, self.collector, self.rt, self.models[0])

    def marker(self):
        model = {**self.models[0], "sha256": file_hash(self.models[0]["path"])}
        return "planned_screen:" + self.stage_key(self.rt, model, None)

    def full_search(self):
        report = self.stack.enter_context(patch("localbench.search.Report"))
        report.return_value.summarize.return_value = {"configurations": []}
        report.return_value.write.return_value = {"configurations": [], "winner_id": None}
        self.stack.enter_context(patch("localbench.search.runtime", side_effect=lambda config, state, engine:
                                      {"engine": engine, "kind": "native", "binary_path": str(self.binary)}))
        self.stack.enter_context(patch("localbench.search.regular_and_long", return_value={}))
        self.stack.enter_context(patch("localbench.search.quality_qualified", return_value=[]))
        self.stack.enter_context(patch("localbench.tuning.tune", return_value=[]))
        self.stack.enter_context(patch("localbench.search.unavailable", return_value={"eligible": False}))
        return search.full_search(self.config, self.state, self.collector)

    def test_current_pending_capacity_blocks_completion_marker(self):
        self.state.enqueue(self.job_key())
        self.assertTrue(self.pending())
        self.assertIsNone(self.state.get_control(self.marker()))

    def test_current_running_capacity_is_unfinished_until_recovered(self):
        job = self.state.enqueue(self.job_key())
        self.state.begin(job)
        self.assertTrue(self.pending())
        self.state.recover()
        self.assertTrue(self.pending())
        self.assertEqual(self.state.db.execute("SELECT status FROM jobs WHERE id=?", (job,)).fetchone()[0], "pending")

    def test_pending_different_configuration_does_not_block_this_stage(self):
        other = self.job_key(configuration_id="d" * 64)
        self.state.enqueue(other)
        self.assertFalse(self.pending())
        result = self.run_stage()
        self.assertTrue(result["eligible"])
        self.assertEqual(len(self.calls), 1)
        self.assertIsNotNone(self.state.get_control(self.marker()))

    def test_obsolete_measurement_protocol_pending_is_ignored_and_superseded(self):
        with patch.dict(PROTOCOL_HASHES, {"measurement.py": "a" * 64}):
            obsolete = self.job_key()
        old_job = self.state.enqueue(obsolete)
        self.assertFalse(self.pending())
        result = self.run_stage()
        self.assertTrue(result["eligible"])
        self.assertEqual(len(self.calls), 1)
        self.assertIsNotNone(self.state.get_control(self.marker()))
        self.assertEqual(self.state.db.execute("SELECT status FROM jobs WHERE id=?", (old_job,)).fetchone()[0], "pending")
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 2)

    def test_obsolete_recovered_controller_job_does_not_block_stage(self):
        with patch.dict(PROTOCOL_HASHES, {"measurement.py": "b" * 64}):
            job = self.state.enqueue(self.job_key())
            self.state.begin(job)
        self.state.recover()
        self.assertFalse(self.pending())
        self.assertTrue(self.run_stage()["eligible"])
        self.assertIsNotNone(self.state.get_control(self.marker()))

    def test_quality_hash_reconstruction_uses_fixture_hash(self):
        fixture = digest({"frozen_py02_variant": 42})
        current = self.job_key("quality", fixture_hash=fixture)
        self.state.enqueue(current)
        self.assertTrue(self.pending())
        with patch.dict(PROTOCOL_HASHES, {"quality.py": "c" * 64}):
            self.assertFalse(self.pending())
        self.assertTrue(self.pending())

    def test_common_byte_hash_reconstruction_uses_its_own_protocol(self):
        self.state.enqueue(self.job_key("common_byte"))
        self.assertTrue(self.pending())
        with patch.dict(PROTOCOL_HASHES, {"common_byte.py": "d" * 64}):
            self.assertFalse(self.pending())
        self.assertTrue(self.pending())

    def test_setup_job_without_current_logical_hash_does_not_block(self):
        setup = self.job_key()
        setup.pop("logical_plan_hash")
        setup["kind"] = "engine_probe"
        self.state.enqueue(setup)
        self.assertFalse(self.pending())

    def test_demo_uses_separate_launch_plan_hash_contract(self):
        demo = self.job_key()
        demo["kind"] = "demo"
        demo["logical_plan_hash"] = digest({"immutable_launch_plan": True})
        self.state.enqueue(demo)
        self.assertFalse(self.pending())

    def test_transient_retries_same_job_then_freezes_success(self):
        self.failures_before_success = 1
        result = self.run_stage()
        self.assertTrue(result["eligible"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 2)
        self.assertFalse(self.pending())
        self.assertIsNotNone(self.state.get_control(self.marker()))
        self.assertEqual(self.run_stage(), result)
        self.assertEqual(len(self.calls), 2)

    def test_three_transient_attempts_exhaust_same_job_then_freeze_terminal_invalid(self):
        self.failures_before_success = 99
        result = self.run_stage()
        self.assertFalse(result["eligible"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 3)
        self.assertEqual(self.state.db.execute("SELECT status FROM jobs").fetchone()[0], "invalid")
        self.assertFalse(self.pending())
        self.assertIsNotNone(self.state.get_control(self.marker()))
        self.assertEqual(self.run_stage(), result)
        self.assertEqual(len(self.calls), 3)

    def test_terminal_invalid_is_reused_after_retry_exhaustion(self):
        key = self.job_key()
        job = self.state.enqueue(key)
        result = {"kind": "capacity", "passed": False, "valid": False, "reason": "foreign_workload_overlap"}
        for number in range(3):
            self.state.finish(self.state.begin(job), "invalid", result, reason=result["reason"], transient=True)
            cached = prior_result(self.state, {name: value for name, value in key.items() if name != "fixture_prompt_hash"})
            self.assertEqual(cached, result if number == 2 else None)
        self.assertFalse(self.pending())

    def test_current_stage_that_never_progresses_is_bounded_and_not_frozen(self):
        self.no_progress.add((self.rt["engine"], self.models[0]["id"]))
        result = self.run_stage()
        self.assertFalse(result["eligible"])
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(self.pending())
        self.assertIsNone(self.state.get_control(self.marker()))

    def test_setup_failure_during_retry_cannot_erase_current_pending_identifier(self):
        self.failures_before_success = 99
        self.next_setup_failure = True
        result = self.run_stage()
        self.assertFalse(result.get("eligible"))
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(self.pending())
        self.assertIsNone(self.state.get_control(self.marker()))

    def test_interrupt_keeps_same_pending_job_for_resume_without_completed_marker(self):
        self.failures_before_success = 1
        self.raise_after_attempt = True
        with self.assertRaisesRegex(RuntimeError, "stop_after_current"):
            self.run_stage()
        self.assertTrue(self.pending())
        self.assertIsNone(self.state.get_control(self.marker()))
        self.state.recover()
        result = self.run_stage()
        self.assertTrue(result["eligible"])
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 2)

    def test_all_required_baselines_terminal_sets_completion_flags(self):
        self.full_search()
        required = {(engine, model["id"]) for engine in ("bundled-llama", "upstream-llama-nightly") for model in self.models}
        self.assertTrue(required.issubset(set(self.calls)))
        self.assertIs(self.state.get_control("native_screens_completed"), True)
        self.assertIs(self.state.get_control("installed_baselines_completed"), True)

    def test_exhausted_required_baselines_are_terminal_outcomes_not_unfinished(self):
        self.failures_before_success = 99
        self.full_search()
        self.assertEqual(len(self.calls), 12)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0], 0)
        self.assertIs(self.state.get_control("native_screens_completed"), True)
        self.assertIs(self.state.get_control("installed_baselines_completed"), True)

    def test_pending_required_baseline_prevents_completion_and_search_covered(self):
        self.no_progress.add(("bundled-llama", self.models[0]["id"]))
        self.full_search()
        self.assertTrue(search.pending_for_stage(self.state, self.identifier({"engine": "bundled-llama"}, self.models[0])))
        self.assertIsNot(self.state.get_control("native_screens_completed"), True)
        self.assertIsNot(self.state.get_control("installed_baselines_completed"), True)
        self.assertIsNot(self.state.get_control("search_covered"), True)

    def test_stale_legacy_baseline_flag_does_not_skip_current_unfinished_plans(self):
        self.state.control("native_screens_completed", True)
        self.state.control("installed_baselines_completed", True)
        rt = {"engine": "bundled-llama", "kind": "native", "binary_path": str(self.binary)}
        self.state.enqueue(self.job_key(rt=rt))
        self.full_search()
        required = {(engine, model["id"]) for engine in ("bundled-llama", "upstream-llama-nightly") for model in self.models}
        self.assertTrue(required.issubset(set(self.calls)))
        self.assertFalse(search.pending_for_stage(self.state, self.identifier(rt)))
        self.assertIs(self.state.get_control("native_screens_completed"), True)


if __name__ == "__main__":
    unittest.main()
