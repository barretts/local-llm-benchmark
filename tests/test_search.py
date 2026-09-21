"""Mock-only finite search orchestration; no network, runtime or GPU calls."""

from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from localbench.config import digest, file_hash, load
from localbench.search import full_search, run_screen, stage_key, memory_screen
from localbench.search_core import screen, unavailable, checkpoint_model
from localbench.runtime_catalog import setup_budget, compatible, select_manifest
from localbench.state import State
from localbench.scheduler import key_base

ENGINES = ("bundled-llama", "upstream-llama-nightly", "ik-llama",
           "turboquant-cuda", "exllamav3-tabby", "vllm", "sglang", "lm-studio", "ollama")


class MockReport:
    scores = []
    def __init__(self, config, state):
        pass
    def summarize(self):
        return {"configurations": copy.deepcopy(self.scores), "winner_id": None}
    def write(self):
        return self.summarize()


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        home = patch("localbench.search.Path.home", return_value=self.root)
        home.start()
        self.addCleanup(home.stop)
        self.config = load()
        self.now = 1000.0
        for key in self.config["paths"]:
            directory = self.root/key
            directory.mkdir()
            self.config["paths"][key] = str(directory)
        self.config["paths"]["project"] = str(self.root)
        self.binary = self.root/"runtime_root"/"inert-engine.exe"
        self.binary.write_bytes(b"inert test artifact; never executed")
        self.config["installed_tools"]["bundled_llama_server"] = str(self.binary)
        self.installed = []
        for index, priority in enumerate((1, 2)):
            path = self.root/"installed_model_root"/("installed-"+str(index)+".gguf")
            path.write_bytes(b"inert weight artifact "+bytes([index]))
            self.installed.append({"id": "installed-"+str(index), "path": str(path),
                                  "priority": priority, "sha256": file_hash(path)})
        self.config["installed_model_candidates"] = self.installed
        self.config["download_weights"] = [
            {"id": "control-q4", "priority": 1, "filename": "control-Q4_K_M.gguf",
             "repo": "test/control", "revision": "a"*40, "sha256": "b"*64, "bytes": 1},
            {"id": "control-q6", "priority": 2, "filename": "control-Q6_K.gguf",
             "repo": "test/control", "revision": "c"*40, "sha256": "d"*64, "bytes": 1}]
        self.config["linux_checkpoint_probes"] = [
            {"id": "linux-snapshot", "repo": "test/snapshot", "revision": "e"*40}]
        discovery = {"new_weight_leads": [{"id": "ornith15-9b-exl3-hq4", "repo": "test/exl3",
            "revision": "f"*40, "weight_bytes": 1, "filename": "model.safetensors"}]}
        (self.root/"artifacts"/"engine-discovery.json").write_text(json.dumps(discovery), encoding="utf-8")
        self.state = State(self.config, clock=lambda: self.now)
        self.events = []
        MockReport.scores = []

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def runtime(self, config, state, engine):
        self.events.append(("runtime", engine))
        kind = ("container" if engine in ("ik-llama", "vllm", "sglang")
                else "tabby" if engine == "exllamav3-tabby"
                else engine if engine in ("lm-studio", "ollama") else "native")
        return {"engine": engine, "kind": kind, "binary_path": str(self.binary),
                "image_digest": engine+"@sha256:"+"a"*64}

    def model(self, config, state, candidate, format="gguf"):
        self.events.append(("acquire", candidate["id"], format))
        value = {"id": candidate["id"], "path": str(self.root/"new_model_root"/candidate["id"]),
                 "revision": candidate["revision"], "sha256": digest(candidate)}
        state.entity("model", value["id"], value)
        return value

    def memory(self, config, state, collector, runtime, model, settings=None):
        marker = "planned_screen:" + stage_key(runtime, model, settings)
        completed = state.get_control(marker)
        if completed is not None:
            return [completed]
        self.events.append(("screen", runtime["engine"], model["id"]))
        value = {"engine": runtime["engine"], "model": model["id"],
                 "planned_screen_key": marker,
                 "configuration_id": runtime["engine"]+"-"+model["id"]+("-"+digest(settings)[:8] if settings else ""),
                 "eligible": False, "settings": {"reasoning": "default", **(settings or {})},
                 "runtime": copy.deepcopy(runtime)}
        state.entity("screen", digest([value["engine"], value["model"]]), value)
        state.control(marker, value)
        return [value]

    def run_full(self, runtime=None, tuning=None, qualified=None, credential=True):
        handoff = ModuleType("localbench.handoff")
        handoff.demo_and_prepare = Mock(side_effect=lambda *args: self.events.append(("demo", args[-1]["configuration_id"])))
        final = Mock(side_effect=lambda *args: self.events.append(("final", args[-1]["configuration_id"])))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"localbench.handoff": handoff}))
            stack.enter_context(patch("localbench.search.runtime", side_effect=runtime or self.runtime))
            stack.enter_context(patch("localbench.search.acquire_model", side_effect=self.model))
            stack.enter_context(patch("localbench.search.memory_screen", side_effect=self.memory))
            stack.enter_context(patch("localbench.search.run_screen", side_effect=lambda config, state, collector, rt, model, **kwargs: self.memory(config, state, collector, rt, model, settings=kwargs.get("settings"))[0]))
            stack.enter_context(patch("localbench.search.regular_and_long", side_effect=lambda *args: self.events.append(("quality", len(args[-1])))))
            stack.enter_context(patch("localbench.search.Report", MockReport))
            stack.enter_context(patch("localbench.tuning.tune", side_effect=tuning or (lambda *args: [])))
            stack.enter_context(patch("localbench.search.quality_qualified", side_effect=qualified or (lambda *args: [])))
            stack.enter_context(patch("localbench.search.final_samples", final))
            stack.enter_context(patch.dict(os.environ, {"LM_BENCH_TOKEN": "memory-only-mock-key"} if credential else {}, clear=not credential))
            result = full_search(self.config, self.state, SimpleNamespace())
        return result, final, handoff.demo_and_prepare

    def pending_job(self, configuration="unrelated", model="other"):
        # Pending stage work must have the same current protocol identity as
        # real scheduler jobs; an incomplete synthetic setup key is unrelated.
        metadata = {"binary_sha256": file_hash(self.binary), "adjacent_dlls": {},
                    "version_output": "mock runtime", "model_sha256": model}
        adapter = SimpleNamespace(state=self.state, requested={}, metadata=lambda: metadata)
        key = key_base(self.config, adapter, configuration, "capacity", 17, "fresh", 0, 16384)
        key["fixture_prompt_hash"] = digest({"mock_capacity_prompt": model})
        return self.state.enqueue(key)

    def test_actual_spec_keys_cover_all_engines_smoke_targets_and_caps(self):
        original = load()
        self.assertTrue(set(ENGINES)-{"bundled-llama"} <= {item["id"] for item in original["runtime_candidates"]})
        self.assertEqual(len(original["grading"]["smoke_jobs"]), 6)
        self.assertEqual(original["measurement"]["smaller_prompt_targets"], [4096, 16384])
        self.assertTrue(all(type(item["probe_minutes"]) is int for item in original["runtime_candidates"]))

    def test_screen_plan_resume_reuses_exact_completed_stage_with_unrelated_pending_job(self):
        runtime = self.runtime(self.config, self.state, "bundled-llama")
        self.pending_job()
        result = {"configuration_id": "own-stage", "eligible": False, "reason": "deterministic_model_failure"}
        with patch("localbench.search.screen", return_value=result) as work:
            first = run_screen(self.config, self.state, None, runtime, self.installed[0])
            second = run_screen(self.config, self.state, None, runtime, self.installed[0])
        self.assertEqual(first, second)
        self.assertEqual(work.call_count, 1)
        self.assertIsNone(self.state.run["started"])

    def test_stage_with_its_own_pending_attempt_remains_resumable(self):
        runtime = self.runtime(self.config, self.state, "bundled-llama")
        self.pending_job(configuration="own-stage", model=self.installed[0]["sha256"])
        with patch("localbench.search.screen", return_value={"configuration_id": "own-stage", "eligible": False}) as work:
            run_screen(self.config, self.state, None, runtime, self.installed[0])
            run_screen(self.config, self.state, None, runtime, self.installed[0])
        self.assertEqual(work.call_count, 2 * (self.config["limits"]["retry_transient_attempts"] + 1))
        marker = "planned_screen:" + stage_key(runtime, self.installed[0], None)
        self.assertIsNone(self.state.get_control(marker))

    def test_runtime_revision_and_settings_change_have_distinct_stage_keys(self):
        runtime = {"engine": "vllm", "image_digest": "image@sha256:"+"a"*64}
        model = {"id": "m", "revision": "r"}
        self.assertNotEqual(stage_key(runtime, model, {}), stage_key({**runtime, "image_digest": "image@sha256:"+"b"*64}, model, {}))
        self.assertNotEqual(stage_key(runtime, model, {}), stage_key(runtime, model, {"kv_cache_dtype": "fp8"}))

    def test_native_launcher_stage_identity_includes_adjacent_implementation_dlls(self):
        dll = self.binary.parent/"implementation.dll"
        dll.write_bytes(b"original inert implementation")
        runtime = {"engine": "bundled-llama", "kind": "native", "binary_path": str(self.binary)}
        before = stage_key(runtime, self.installed[0], {})
        dll.write_bytes(b"changed inert implementation")
        self.assertNotEqual(stage_key(runtime, self.installed[0], {}), before)

    def test_installed_bundled_screens_precede_upstream_setup_and_all_weight_acquisition(self):
        self.run_full()
        upstream = self.events.index(("runtime", "upstream-llama-nightly"))
        bundled = [self.events.index(("screen", "bundled-llama", model["id"])) for model in self.installed]
        self.assertTrue(all(position < upstream for position in bundled))
        first_weights = next(index for index, event in enumerate(self.events) if event[0] == "acquire")
        baselines = [index for index, event in enumerate(self.events)
                     if event[0] == "screen" and event[2].startswith("installed-") and event[1] in ("bundled-llama", "upstream-llama-nightly")]
        self.assertTrue(all(position < first_weights for position in baselines))

    def test_all_engine_families_are_considered_without_real_work_or_synthetic_winner(self):
        result, final, demo = self.run_full()
        considered = {event[1] for event in self.events if event[0] == "runtime"}
        self.assertEqual(considered, set(ENGINES))
        self.assertTrue(self.state.get_control("search_covered"))
        self.assertEqual(result["summary"]["winner_id"], None)
        final.assert_not_called()
        demo.assert_not_called()
        self.assertIsNone(self.state.run["started"])
        self.assertEqual(self.state.db.execute("SELECT COALESCE(SUM(bytes),0) FROM weights").fetchone()[0], 0)

    def test_native_resume_skips_existing_installed_stages(self):
        self.run_full()
        self.events.clear()
        self.run_full()
        self.assertFalse(any(event[0] == "screen" and event[2].startswith("installed-")
                             and event[1] in ("bundled-llama", "upstream-llama-nightly") for event in self.events))

    def test_source_review_pending_preserves_unaffected_coverage_without_completed_claim(self):
        def runtime(config, state, engine):
            if engine in ("turboquant-cuda", "exllamav3-tabby"):
                self.events.append(("runtime", engine))
                raise RuntimeError("source_review_pending:mock-review-bundle")
            return self.runtime(config, state, engine)
        result, _, _ = self.run_full(runtime=runtime)
        self.assertEqual(set(result["pending_engines"]), {"turboquant-cuda", "exllamav3-tabby"})
        self.assertFalse(self.state.get_control("search_covered"))
        self.assertTrue(any(event[:2] == ("screen", "sglang") for event in self.events))
        self.assertTrue(any(event[:2] == ("screen", "ollama") for event in self.events))

    def test_missing_managed_credential_is_recorded_skip_and_other_engines_continue(self):
        self.run_full(credential=False)
        rows = [json.loads(row["data"]) for row in self.state.db.execute("SELECT data FROM entities WHERE kind='engine_outcome'")]
        self.assertTrue(any(row["engine"] == "lm-studio" and "authentication_unavailable" in row["reason"] for row in rows))
        self.assertTrue(any(event[:2] == ("screen", "ollama") for event in self.events))
        self.assertTrue(any(row[0] == "skipped" for row in self.state.db.execute("SELECT status FROM jobs")))

    def test_tuning_returns_existing_screens_without_duplicate_final_samples_or_demos(self):
        tuning = lambda config, state, collector, screens: list(screens)
        qualified = lambda config, state, screens: [item for item in screens if item["configuration_id"] == "bundled-llama-installed-0"]
        _, final, demo = self.run_full(tuning=tuning, qualified=qualified)
        self.assertEqual(final.call_count, 1)
        self.assertEqual(demo.call_count, 1)
        self.assertEqual(self.state.get_control("finalist_configuration_ids"), ["bundled-llama-installed-0"])

    def test_setup_cap_is_durable_and_compatibility_does_not_reset_it(self):
        first = setup_budget(self.config, self.state, "vllm")
        self.now = first["deadline"] + 1
        with self.assertRaisesRegex(RuntimeError, "engine_probe_budget_exhausted"):
            setup_budget(self.config, self.state, "vllm")
        self.assertEqual(self.state.get_control("engine_probe_budget:vllm")["started"], first["started"])
        compatible(self.state, "vllm", {"runtime_feasibility": True})
        self.assertTrue(setup_budget(self.config, self.state, "vllm")["compatible"])
        self.assertIsNone(self.state.run["started"])

    def test_setup_exhaustion_is_skipped_and_global_budget_stop_propagates(self):
        runtime = self.runtime(self.config, self.state, "vllm")
        with patch("localbench.search_core.setup_budget", side_effect=RuntimeError("engine_probe_budget_exhausted:vllm")), patch(
                "localbench.search_core.Report", MockReport):
            result = screen(self.config, self.state, None, runtime, self.installed[0])
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "engine_probe_budget_exhausted:vllm")
        self.assertEqual(self.state.db.execute("SELECT status FROM jobs").fetchone()[0], "skipped")
        with patch("localbench.search_core.setup_budget", side_effect=RuntimeError("budget_exhausted")), patch(
                "localbench.search_core.Report", MockReport):
            with self.assertRaisesRegex(RuntimeError, "budget_exhausted"):
                screen(self.config, self.state, None, runtime, self.installed[0])

    def test_real_spec_screen_uses_six_literal_smoke_jobs_and_all_throughput_targets(self):
        adapter = SimpleNamespace(requested={"context": 65536, "slots": 1, "reasoning": "default"},
            effective={"effective_context": 65536, "effective_slots": 1}, unload_owned=Mock())
        probes = []
        def measured(*args, **kwargs):
            kind, target = args[5], args[9]
            probes.append((kind, target))
            return {"passed": True, "valid": True, "metrics": {"generated_tokens": 256, "first_action_seconds": .5}}
        def code(*args, **kwargs):
            return {"passed": True, "valid": True, "fixture": args[5], "seed": args[6]}
        runtime = self.runtime(self.config, self.state, "bundled-llama")
        with patch("localbench.search_core.make_adapter", return_value=(adapter, "cfg")), patch(
                "localbench.search_core.measured_job", side_effect=measured), patch(
                "localbench.search_core.coding_job", side_effect=code) as coding, patch(
                "localbench.search_core.Report", MockReport):
            result = screen(self.config, self.state, None, runtime, self.installed[0])
        self.assertTrue(result["eligible"])
        self.assertEqual([(entry.args[5], entry.args[6]) for entry in coding.call_args_list],
                         [(item["fixture"], item["seed"]) for item in self.config["grading"]["smoke_jobs"]])
        self.assertEqual([target for kind, target in probes if kind == "throughput"], [4096, 16384, 61440])
        adapter.unload_owned.assert_called_once()

    def test_pinned_manifest_selects_linux_amd64_and_missing_platform_fails(self):
        pin = "sha256:"+"a"*64
        payload = {"manifests": [
            {"digest": "sha256:"+"b"*64, "platform": {"os": "linux", "architecture": "arm64"}},
            {"digest": pin, "platform": {"os": "linux", "architecture": "amd64"}}]}
        self.assertEqual(select_manifest(payload), pin)
        with self.assertRaisesRegex(RuntimeError, "linux_amd64"):
            select_manifest({"manifests": payload["manifests"][:1]})

    def test_missing_generated_counter_in_throughput_does_not_abort_other_stage_work(self):
        adapter = SimpleNamespace(requested={"context": 65536, "slots": 1, "reasoning": "default"},
            effective={"effective_context": 65536, "effective_slots": 1}, unload_owned=Mock())
        def measured(*args, **kwargs):
            metrics = {"generated_tokens": None if args[5] == "throughput" else 256,
                       "first_action_seconds": .5}
            return {"passed": args[5] != "throughput", "valid": args[5] != "throughput", "metrics": metrics}
        with patch("localbench.search_core.make_adapter", return_value=(adapter, "cfg")), patch(
                "localbench.search_core.measured_job", side_effect=measured), patch(
                "localbench.search_core.coding_job", return_value={"passed": True, "valid": True}), patch(
                "localbench.search_core.Report", MockReport):
            result = screen(self.config, self.state, None,
                self.runtime(self.config, self.state, "bundled-llama"), self.installed[0])
        self.assertIsNone(result["throughput"]["metrics"]["generated_tokens"])
        adapter.unload_owned.assert_called_once()


if __name__ == "__main__":
    unittest.main()
