import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from localbench.config import digest, load
from localbench.state import State
from localbench.tuning import _Tuner, _smoke_passed, tune


NATIVE_HELP = """--cache-type-k TYPE
 allowed values: f16, q8_0, q4_0
--cache-type-v TYPE
 allowed values: f16, q8_0, q4_0
--batch-size N
--ubatch-size N
--threads N
--threads-batch N
--reasoning-budget N
--reasoning auto,off,on
--spec-type none,draft-mtp
"""
LINUX_HELP = """--gpu-memory-utilization N
--mem-fraction-static N
--max-num-batched-tokens N
--chunked-prefill-size N
--kv-cache-dtype {auto,fp8,fp8_e4m3}
"""


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ("project", "state", "logs", "artifacts", "runtime_root"):
            self.config["paths"][name] = str(Path(self.temp.name) / name)
        self.config["limits"].update(server_start_timeout_seconds=1, per_task_timeout_seconds=10,
            per_64k_request_timeout_seconds=2)
        self.clock = 1000.0
        self.state = State(self.config, clock=lambda: self.clock)
        self.state.control("harness_verified", True)
        self.models = [{"id": "model-" + str(index), "path": str(Path(self.temp.name) / ("model-" + str(index) + ".gguf")),
            "sha256": str(index) * 64, "weight_family": "family-" + str(index)} for index in range(1, 7)]
        self.config["installed_model_candidates"] = self.models
        self.native_help = Path(self.temp.name) / "observed-native-help.log"
        self.native_help.write_text(NATIVE_HELP, encoding="utf-8")
        self.linux_help = Path(self.temp.name) / "observed-linux-help.log"
        self.linux_help.write_text(LINUX_HELP, encoding="utf-8")
        self.calls, self.qualifications = [], []
        self.qualified = set()
        self.fail_smoke = None
        self.fail_runtime = None
        self.full_residency = False
        self.extra_metadata = {}
        self.quality_fail_profiles = set()

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def smoke(self, passed=True):
        return [{"fixture": job["fixture"], "seed": job["seed"], "passed": passed, "valid": True}
            for job in self.config["grading"]["smoke_jobs"]]

    def settings(self, kind="native"):
        if kind == "container":
            return {"context": 65536, "slots": 1, "gpu_memory_utilization": .85, "prefill_tokens": 8192,
                "kv_cache_dtype": "auto", "reasoning": "default"}
        if kind == "tabby":
            return {"context": 65536, "slots": 1, "cache_size": 65536, "cache_mode": "FP16", "chunk_size": 2048,
                "reasoning": "default", "speculation": "off"}
        if kind == "lm-studio":
            return {"context": 65536, "gpu": "max", "ttl": 1800, "reasoning": "default"}
        return {"context": 65536, "slots": 1, "gpu_layers": 99, "flash_attention": "on", "cache_k": "f16", "cache_v": "f16",
            "batch": 2048, "ubatch": 512, "threads": 16, "threads_batch": 16, "reasoning": "default", "speculation": "off"}

    def latency(self, settings):
        score = 30.0
        score -= {"f16": 0, "q8_0": 5, "q4_0": 2}.get(settings.get("cache_k"), 0)
        score += abs(settings.get("batch", 1024) - 1024) / 1000
        score += abs(settings.get("ubatch", 256) - 256) / 1000
        score += abs(settings.get("threads", 8) - 8) / 100
        score += abs(settings.get("threads_batch", 2) - 2) / 100
        score -= {"default": 0, "reduced": 2, "disabled": 4}[settings.get("reasoning", "default")]
        score += abs(settings.get("gpu_memory_utilization", .8) - .8) * 10
        score += settings.get("prefill_tokens", 4096) / 100000
        score -= 1 if settings.get("kv_cache_dtype") == "fp8" else 0
        return score

    def candidate(self, model=None, engine="upstream-llama-nightly", kind="native", settings=None, seconds=None, checkpoint=True):
        model = model or self.models[0]
        settings = settings or self.settings(kind)
        runtime = {"engine": engine, "kind": kind, "pin": "fixed-runtime-pin"}
        identifier = digest([engine, model["sha256"], settings])
        metadata = {"engine_id": engine, "model_id": model["id"], "help_log": str(self.linux_help if kind == "container" and engine != "ik-llama" else self.native_help),
            "effective_settings": {"offload_layers": [10, 10] if self.full_residency else [8, 10], "startup_seconds": .01},
            **self.extra_metadata}
        self.state.entity("configuration", identifier, metadata)
        result = {"engine": engine, "model": model["id"], "configuration_id": identifier, "runtime": runtime,
            "settings": copy.deepcopy(settings), "eligible": True, "capacity_qualified": True,
            "smoke": self.smoke(), "timing": [{"passed": True, "valid": True, "metrics": {"first_action_seconds": seconds or self.latency(settings)}}],
            "reason": "screen_passed"}
        if checkpoint:
            from localbench.search import stage_key
            result["planned_screen_key"] = 'planned_screen:' + stage_key(runtime, model, settings)
            self.state.control(result["planned_screen_key"], result)
        return result

    def screened(self, config, state, collector, runtime, model, settings=None, tuning=False):
        self.assertTrue(tuning)
        self.calls.append({"engine": runtime["engine"], "model": model["id"], "settings": copy.deepcopy(settings)})
        if self.fail_runtime and self.fail_runtime(settings):
            return {"engine": runtime["engine"], "model": model["id"], "runtime": runtime, "settings": settings,
                "eligible": False, "valid": False, "reason": "model_oom:mock-owned.log"}
        result = self.candidate(model, runtime["engine"], runtime["kind"], settings, checkpoint=False)
        if self.fail_smoke and self.fail_smoke(settings):
            result.update(eligible=False, smoke=self.smoke(False), timing=[], reason="exploration_smoke_gate_failed")
        return result

    def quality_qualified(self, config, state, candidates):
        return [candidate for candidate in candidates if candidate.get("configuration_id") in self.qualified]

    def qualify(self, config, state, collector, candidates):
        self.qualifications.extend(copy.deepcopy(candidates))
        for candidate in candidates:
            if candidate["settings"].get("reasoning") not in self.quality_fail_profiles:
                self.qualified.add(candidate["configuration_id"])
        return {"qualification_complete": True}

    def mock_runtime(self):
        stack = ExitStack()
        stack.enter_context(patch("localbench.search.screen", self.screened))
        stack.enter_context(patch.multiple("localbench.tuning",
            regular_and_long=self.qualify, quality_qualified=self.quality_qualified))
        return stack

    def proposals(self):
        return [json.loads(row[0]) for row in self.state.db.execute("SELECT data FROM entities WHERE kind='tuning_proposal'")]

    def decisions(self):
        return [json.loads(row[0]) for row in self.state.db.execute("SELECT data FROM entities WHERE kind='tuning_decision'")]

    def test_quality_first_selection_uses_three_families_and_keeps_engines_eligible(self):
        baselines = [self.candidate(model, seconds=index + 1) for index, model in enumerate(self.models[:4])]
        self.qualified.add(baselines[3]["configuration_id"])
        alternative = self.candidate(self.models[3], engine="bundled-llama", seconds=8)
        baselines.append(alternative)
        with self.mock_runtime():
            returned = tune(self.config, self.state, object(), baselines)
        phase = self.state.get_control("tuning_phase")
        self.assertEqual(phase["selected_families"][0], "family-4")
        self.assertEqual(len(phase["selected_families"]), 3)
        self.assertNotIn("model-3", {call["model"] for call in self.calls})
        self.assertIn("bundled-llama", {call["engine"] for call in self.calls if call["model"] == "model-4"})
        self.assertGreater(len(returned), len(baselines))
        self.assertIsNone(self.state.run["started"])

    def test_native_coordinate_search_smoke_gates_profiles_and_full_quality(self):
        baseline = self.candidate()
        self.qualified.add(baseline["configuration_id"])
        self.quality_fail_profiles = {"disabled"}
        with self.mock_runtime():
            returned = tune(self.config, self.state, None, [baseline])
        proposals = self.proposals()
        self.assertEqual({item["settings"]["cache_k"] for item in proposals if item["stage"] == "cache"}, {"q8_0", "q4_0"})
        batch = [item for item in proposals if item["stage"] == "batch_pair"]
        self.assertEqual({(item["settings"]["batch"], item["settings"]["ubatch"]) for item in batch},
            {(512, 128), (1024, 256), (2048, 1024), (4096, 2048)})
        # The 2048/512 pair already has a completed q8 screen from cache search.
        self.assertTrue(any(item["stage"] == "cache" and item["settings"]["batch"] == 2048 and
            item["settings"]["ubatch"] == 512 and item["settings"]["cache_k"] == "q8_0" for item in proposals))
        self.assertTrue(all(item["settings"]["cache_k"] == "q8_0" for item in batch))
        self.assertTrue(any(item["stage"] == "asymmetric_cache" for item in proposals))
        self.assertTrue(any(item["stage"] == "batch_coordinate" for item in proposals))
        self.assertTrue(any(item["stage"] == "generation_threads" for item in proposals))
        self.assertTrue(any(item["stage"] == "batch_threads" for item in proposals))
        profiles = {candidate["settings"]["reasoning"] for candidate in self.qualifications}
        self.assertEqual(profiles, {"default", "reduced", "disabled"})
        identifiers = {candidate["configuration_id"] for candidate in self.qualifications}
        self.assertEqual(len(identifiers), len(self.qualifications))
        self.assertTrue(any(item["reason"] == "full_quality_not_qualified" for item in self.decisions()))
        self.assertTrue(all(candidate["settings"].get("speculation", "off") == "off" for candidate in returned))
        self.assertLess(len(self.calls), 30)

    def test_bad_smoke_never_becomes_a_deep_or_full_quality_candidate(self):
        baseline = self.candidate()
        self.qualified.add(baseline["configuration_id"])
        self.fail_smoke = lambda settings: settings.get("cache_k") != "f16"
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        self.assertTrue(any(item.get("result", {}).get("reason") == "exploration_smoke_gate_failed" for item in self.proposals()))
        self.assertTrue(all(candidate["settings"].get("cache_k") == "f16" for candidate in self.qualifications))
        self.assertTrue(all(item["settings"].get("cache_k") == "f16" for item in self.proposals()
            if item["stage"] in {"batch_pair", "generation_threads", "reasoning_profile"}))

    def test_literal_smoke_grid_rejects_duplicate_missing_or_invalid_jobs(self):
        candidate = self.candidate()
        self.assertTrue(_smoke_passed(self.config, candidate))
        candidate["smoke"][-1] = copy.deepcopy(candidate["smoke"][0])
        self.assertFalse(_smoke_passed(self.config, candidate))
        candidate["smoke"] = self.smoke()
        candidate["smoke"][0]["valid"] = False
        self.assertFalse(_smoke_passed(self.config, candidate))

    def test_full_gpu_evidence_skips_generation_threads_but_permits_batch_threads(self):
        self.full_residency = True
        baseline = self.candidate()
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        stages = {item["stage"] for item in self.proposals()}
        self.assertNotIn("generation_threads", stages)
        self.assertIn("batch_threads", stages)
        self.assertTrue(all(call["settings"]["threads"] == 16 for call in self.calls))

    def test_linux_memory_prefill_and_dtype_are_separate_coordinates(self):
        baseline = self.candidate(engine="vllm", kind="container")
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        proposals = self.proposals()
        memory = [item for item in proposals if item["stage"] == "linux_memory"]
        self.assertEqual({item["settings"]["gpu_memory_utilization"] for item in memory}, {.8, .9})
        self.assertTrue(all(item["settings"]["prefill_tokens"] == 8192 for item in memory))
        prefill = [item for item in proposals if item["stage"] == "linux_prefill"]
        self.assertTrue(all(item["settings"]["gpu_memory_utilization"] == .8 for item in prefill))
        cache = [item for item in proposals if item["stage"] == "linux_cache"]
        self.assertEqual({item["settings"]["kv_cache_dtype"] for item in cache}, {"fp8", "fp8_e4m3"})
        self.assertTrue(all(item["settings"]["prefill_tokens"] == 4096 for item in cache))
        self.assertTrue(all(call["settings"]["reasoning"] == "default" for call in self.calls))
        self.assertLess(len(self.calls), 9)

    def test_tabby_cache_pairs_preserve_chunk_and_remove_conflicting_common_keys(self):
        settings = {**self.settings("tabby"), "cache_k": "f16", "cache_v": "f16"}
        baseline = self.candidate(engine="exllamav3-tabby", kind="tabby", settings=settings)
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        caches = [item for item in self.proposals() if item["stage"] == "tabby_cache"]
        self.assertEqual({item["settings"]["cache_mode"] for item in caches}, {"FP16", "8,8", "8,4", "4,4"})
        self.assertTrue(all(item["settings"]["chunk_size"] == 2048 for item in caches))
        self.assertTrue(all("cache_k" not in item["settings"] and "cache_v" not in item["settings"] for item in caches))
        self.assertTrue(all(call["settings"]["context"] == 65536 for call in self.calls))
        self.assertTrue(all(call["settings"]["slots"] == 1 for call in self.calls))

    def test_harmony_low_effort_is_one_smoke_gated_coordinate_with_full_quality(self):
        self.models[0]["id"] = "gptoss20b-mxfp4"
        settings = {**self.settings("container"), "reasoning_effort": "medium"}
        baseline = self.candidate(engine="vllm", kind="container", settings=settings)
        self.qualified.add(baseline["configuration_id"])
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        proposals = [item for item in self.proposals() if item["stage"] == "harmony_effort"]
        self.assertEqual(len(proposals), 1)
        low = proposals[0]
        self.assertEqual(low["settings"]["reasoning"], "default")
        self.assertEqual(low["settings"]["reasoning_effort"], "low")
        self.assertTrue(_smoke_passed(self.config, low["result"]))
        self.assertIn(low["result"]["configuration_id"], {item["configuration_id"] for item in self.qualifications})
        self.assertTrue(all(call["settings"]["reasoning"] == "default" for call in self.calls))
        self.assertTrue(all("reasoning_budget_tokens" not in call["settings"] and "enable_thinking" not in call["settings"] for call in self.calls))
        self.assertTrue(any("numeric256_budget_unsupported" in item["reason"] for item in self.decisions()))

    def test_harmony_low_effort_bad_smokes_never_advance_to_full_quality(self):
        self.models[0]["id"] = "gptoss20b-mxfp4"
        baseline = self.candidate(engine="sglang", kind="container",
            settings={**self.settings("container"), "reasoning_effort": "medium"})
        self.fail_smoke = lambda settings: settings.get("reasoning_effort") == "low"
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        low = [item for item in self.proposals() if item["stage"] == "harmony_effort"]
        self.assertEqual(len(low), 1)
        self.assertFalse(low[0]["result"]["eligible"])
        self.assertNotIn(low[0]["result"]["configuration_id"], {item["configuration_id"] for item in self.qualifications})

    def test_missing_help_and_managed_private_controls_produce_honest_skips(self):
        baseline = self.candidate(engine="lm-studio", kind="lm-studio")
        self.native_help.write_text("no publicly exposed cache/batch/reasoning controls", encoding="utf-8")
        self.qualified.add(baseline["configuration_id"])
        with self.mock_runtime():
            returned = tune(self.config, self.state, None, [baseline])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.qualifications, [])
        self.assertEqual(returned, [baseline])
        decisions = self.decisions()
        self.assertTrue(any(item["stage"] == "reasoning_profile" and item["reason"].startswith("unsupported") for item in decisions))
        self.assertTrue(any(item["stage"] == "speculation" and "no_verified" in item["reason"] for item in decisions))

    def test_global_configuration_cap_blocks_new_settings_and_still_qualifies_existing(self):
        baseline = self.candidate()
        self.config["limits"]["maximum_unique_runtime_configurations"] = 2
        with self.mock_runtime():
            returned = tune(self.config, self.state, None, [baseline])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM entities WHERE kind='configuration'").fetchone()[0], 2)
        self.assertTrue(self.qualifications)
        self.assertTrue(any(item["reason"] == "configuration_budget_exhausted" for item in self.decisions()))
        self.assertGreater(len(returned), 1)

    def test_resume_reuses_exact_proposals_freezes_anchors_and_does_not_rescreen(self):
        baseline = self.candidate()
        self.qualified.add(baseline["configuration_id"])
        with self.mock_runtime():
            first = tune(self.config, self.state, None, [baseline])
            count, qualified_count = len(self.calls), len(self.qualifications)
            second = tune(self.config, self.state, None, list(reversed(first)))
        self.assertEqual(len(self.calls), count)
        self.assertEqual(len(self.qualifications), qualified_count)
        self.assertEqual({item.get("configuration_id") for item in first}, {item.get("configuration_id") for item in second})
        self.assertEqual(self.state.get_control("tuning_phase")["anchors"], [baseline])
        self.assertEqual(len(self.proposals()), count)

    def test_interrupted_running_proposal_resumes_same_controller_job(self):
        baseline = self.candidate()
        with self.mock_runtime():
            tuner = _Tuner(self.config, self.state, None, [baseline])
            settings = {**baseline["settings"], "cache_k": "q8_0", "cache_v": "q8_0"}
            identifier = digest([self.state.run["id"], baseline["engine"], baseline["runtime"], self.models[0]["sha256"], settings,
                tuner.screen_key(baseline["runtime"], self.models[0], settings)])
            self.state.entity("tuning_proposal", identifier, {"status": "running", "settings": settings, "stage": "cache"})
            result = tuner.trial(baseline, {"cache_k": "q8_0", "cache_v": "q8_0"}, "cache")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["settings"], settings)
        self.assertTrue(result["eligible"])
        self.assertEqual(json.loads(self.state.db.execute("SELECT data FROM entities WHERE kind='tuning_proposal' AND id=?", (identifier,)).fetchone()[0])["status"], "completed")

    def test_runtime_failures_are_classified_and_not_used_as_measured_losers(self):
        baseline = self.candidate()
        self.fail_runtime = lambda settings: settings.get("cache_k") == "q4_0"
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        failed = [item for item in self.proposals() if item.get("reason_class") == "memory"]
        self.assertTrue(failed)
        self.assertTrue(all(not item["result"]["valid"] and not item["result"]["eligible"] for item in failed))
        self.assertTrue(all(candidate["settings"].get("cache_k") != "q4_0" for candidate in self.qualifications))

    def test_phase_and_global_reserve_stop_before_runtime_calls(self):
        baseline = self.candidate()
        self.state.control("tuning_phase", {"started": self.clock - 100, "deadline": self.clock - 1,
            "selected_families": None, "status": "active"})
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.qualifications, [])
        self.assertEqual(self.state.get_control("tuning_phase")["status"], "stopped_phase_budget")
        self.state.control("tuning_phase", {"started": self.clock, "deadline": self.clock + 9000,
            "selected_families": None, "status": "active"})
        with self.state.db:
            self.state.db.execute("UPDATE runs SET started=?,deadline=?", (self.clock - 100, self.clock + 7200))
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.qualifications, [])
        artifact = json.loads((Path(self.config["paths"]["artifacts"]) / "tuning-exploration.json").read_text(encoding="utf-8"))
        self.assertTrue(artifact["proposals_and_decisions"])

    def pending_capacity(self, candidate):
        from localbench.scheduler import PROTOCOL_HASHES
        key = {"engine": "mock-engine", "model": self.models[0]["sha256"],
            "effective_settings": candidate["settings"], "profile": {}, "fixture_prompt_hash": "mock-prompt",
            "tokenizer": "mock-tokenizer", "seed": 17, "mode": "fresh", "replicate": 0,
            "kind": "capacity", "target_tokens": 61440, "configuration_id": candidate["configuration_id"]}
        key["logical_plan_hash"] = digest({"protocol": {name: PROTOCOL_HASHES[name]
            for name in ("measurement.py", "prompts.py", "regional_prompt.py")}, "run": self.state.run["id"],
            "fixture_hash": None, "kind": "capacity", "seed": 17, "mode": "fresh", "replicate": 0, "target": 61440})
        job = self.state.enqueue(key)
        self.state.finish(self.state.begin(job), "invalid", {"passed": False, "valid": False},
            reason="foreign_workload_overlap", transient=True)
        return job

    def test_completed_proposal_with_pending_job_reconciles_current_screen(self):
        baseline = self.candidate()
        changes = {"cache_k": "q8_0", "cache_v": "q8_0"}
        with self.mock_runtime():
            tuner = _Tuner(self.config, self.state, None, [baseline])
            first = tuner.trial(baseline, changes, "cache")
            job = self.pending_capacity(first)
            resumed = _Tuner(self.config, self.state, None, [baseline])
            self.assertNotIn(first["configuration_id"], {item.get("configuration_id") for item in resumed.screens})
            def recovered_screen(*args, **kwargs):
                result = self.screened(*args, **kwargs)
                self.state.finish(self.state.begin(job), "passed", {"passed": True, "valid": True})
                return result
            with patch("localbench.search.screen", recovered_screen):
                second = resumed.trial(baseline, changes, "cache")
            count = len(self.calls)
            third = resumed.trial(baseline, changes, "cache")
        self.assertEqual(count, 2)
        self.assertEqual(len(self.calls), count)
        self.assertEqual(first["configuration_id"], second["configuration_id"])
        self.assertEqual(second, third)
        self.assertEqual(self.state.db.execute("SELECT status FROM jobs WHERE id=?", (job,)).fetchone()[0], "passed")
        self.assertEqual(len(self.proposals()), 1)
        self.assertEqual(self.proposals()[0]["status"], "completed")

    def test_pending_screen_retries_with_tuning_smoke_gate_before_completion(self):
        baseline = self.candidate()
        original = self.screened
        pending = []
        def interrupted_screen(*args, **kwargs):
            result = original(*args, **kwargs)
            if not pending:
                pending.append(self.pending_capacity(result))
            else:
                self.state.finish(self.state.begin(pending[0]), "passed", {"passed": True, "valid": True})
            return result
        with self.mock_runtime(), patch("localbench.search.screen", interrupted_screen):
            tuner = _Tuner(self.config, self.state, None, [baseline])
            result = tuner.trial(baseline, {"cache_k": "q8_0", "cache_v": "q8_0"}, "cache")
        self.assertTrue(result["eligible"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.proposals()[0]["status"], "completed")
        self.assertTrue(self.state.get_control(result["planned_screen_key"]))

    def test_old_protocol_completed_proposal_is_retained_and_remeasured(self):
        from localbench.scheduler import PROTOCOL_HASHES
        baseline = self.candidate()
        changes = {"cache_k": "q8_0", "cache_v": "q8_0"}
        with self.mock_runtime():
            tuner = _Tuner(self.config, self.state, None, [baseline])
            old = tuner.trial(baseline, changes, "cache")
            old_rows = {row["id"]: row["data"] for row in self.state.db.execute(
                "SELECT id,data FROM entities WHERE kind='tuning_proposal'")}
            with patch.dict(PROTOCOL_HASHES, {"quality.py": "f" * 64}):
                resumed = _Tuner(self.config, self.state, None, [baseline])
                self.assertNotIn(old["configuration_id"], {item.get("configuration_id") for item in resumed.screens})
                current = resumed.trial(baseline, changes, "cache")
                self.assertNotEqual(old["planned_screen_key"], current["planned_screen_key"])
                count = len(self.calls)
                resumed.trial(baseline, changes, "cache")
                self.assertEqual(len(self.calls), count)
                current_tuner = _Tuner(self.config, self.state, None, [baseline])
                self.assertEqual([item["planned_screen_key"] for item in current_tuner.screens
                    if item.get("configuration_id") == current["configuration_id"]],
                    [current["planned_screen_key"]])
        new_rows = {row["id"]: row["data"] for row in self.state.db.execute(
            "SELECT id,data FROM entities WHERE kind='tuning_proposal'")}
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(new_rows), 2)
        self.assertTrue(all(new_rows[key] == value for key, value in old_rows.items()))

    def test_failed_default_smokes_can_rescue_disabled_reasoning_and_full_quality(self):
        baseline = self.candidate()
        baseline.update(eligible=False, smoke=self.smoke(False), timing=[], reason="default_reasoning_smoke_failed")
        self.state.control(baseline["planned_screen_key"], baseline)
        self.fail_smoke = lambda settings: settings.get("reasoning") != "disabled"
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        self.assertTrue(any(call["settings"]["reasoning"] == "disabled" for call in self.calls))
        self.assertTrue(self.qualifications)
        self.assertTrue(all(candidate["settings"]["reasoning"] == "disabled" for candidate in self.qualifications))
        self.assertTrue(all(_smoke_passed(self.config, candidate) for candidate in self.qualifications))
        self.assertTrue(any(item["stage"] == "full_quality" and item["status"] == "qualified" for item in self.decisions()))

    def test_stale_same_settings_input_screen_cannot_bypass_reconciliation(self):
        from localbench.scheduler import PROTOCOL_HASHES
        baseline = self.candidate()
        changes = {"cache_k": "q8_0", "cache_v": "q8_0"}
        with self.mock_runtime():
            old = _Tuner(self.config, self.state, None, [baseline]).trial(baseline, changes, "cache")
            with patch.dict(PROTOCOL_HASHES, {"quality.py": "e" * 64}):
                resumed = _Tuner(self.config, self.state, None, [baseline, old])
                current = resumed.trial(baseline, changes, "cache")
                self.assertNotEqual(old["planned_screen_key"], current["planned_screen_key"])
                saved = next(item for item in resumed.screens if item.get("configuration_id") == current["configuration_id"])
                self.assertEqual(saved["planned_screen_key"], current["planned_screen_key"])
        self.assertEqual(len(self.calls), 2)

    def test_reasoning_rescue_does_not_claim_unsupported_controls(self):
        baseline = self.candidate()
        baseline.update(eligible=False, smoke=self.smoke(False), timing=[], reason="default_reasoning_smoke_failed")
        self.state.control(baseline["planned_screen_key"], baseline)
        self.native_help.write_text("no publicly supported reasoning or cache controls", encoding="utf-8")
        with self.mock_runtime():
            tune(self.config, self.state, None, [baseline])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.qualifications, [])
        self.assertTrue(any(item["stage"] == "reasoning_profile" and item["settings"]["reasoning"] == "disabled"
            and item["reason"] == "unsupported_actual_reasoning_control" for item in self.decisions()))

    def test_reasoning_rescue_is_bounded_to_three_capacity_qualified_anchors(self):
        baselines = [self.candidate(model, engine=engine) for model in self.models[:3]
            for engine in ("bundled-llama", "upstream-llama-nightly")]
        for baseline in baselines:
            baseline.update(eligible=False, smoke=self.smoke(False), timing=[], reason="default_reasoning_smoke_failed")
            self.state.control(baseline["planned_screen_key"], baseline)
        self.fail_smoke = lambda settings: settings.get("reasoning") != "disabled"
        with self.mock_runtime():
            tune(self.config, self.state, None, baselines)
        disabled = [call for call in self.calls if call["settings"]["reasoning"] == "disabled"]
        self.assertEqual(len(disabled), 3)
        self.assertTrue(all(_smoke_passed(self.config, candidate) for candidate in self.qualifications))


if __name__ == "__main__":
    unittest.main()
