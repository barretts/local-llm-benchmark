import copy
import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from localbench.config import load
from localbench.report import Report, context_reasons
from localbench.resources import ResourceCollector, GPU_FIELDS, parse_gpu_line, parse_processes
from localbench.state import State


class ReportResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ("project", "state", "logs", "artifacts"):
            self.config["paths"][name] = str(Path(self.temp.name) / name)
        self.state = State(self.config)
        self.counter = 0

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def evidence(self, mode="fresh"):
        return {"effective_context_tokens": 65536, "prompt_tokens": 61440,
            "expected_prompt_tokens": 61440, "tokenization_verified": True,
            "truncated": False, "context_shift": False,
            "cached_tokens": 59392 if mode == "cached" else 0,
            "cache_isolation_verified": mode == "fresh", "prefix_reuse_verified": mode == "cached"}

    def add(self, identifier, result, status="passed", reason=None, job=None, transient=False):
        if job is None:
            self.counter += 1
            job = self.state.enqueue({"engine": identifier, "model": "weight-sha",
                "effective_settings": {"context": 65536}, "profile": "default",
                "fixture_prompt_hash": "fixture-" + str(self.counter), "tokenizer": "exact-v1",
                "seed": result.get("seed", 42), "mode": result.get("mode", result.get("kind")),
                "replicate": result.get("replicate", self.counter), "configuration_id": identifier})
        data = {"configuration_id": identifier, "valid": status == "passed", "passed": status == "passed", **result}
        self.state.finish(self.state.begin(job), status, data, reason, transient=transient)
        return job

    def full_candidate(self, identifier="candidate", fresh_seconds=10., speed=64., regular_passes=36, long_passes=12):
        self.state.entity("configuration", identifier, {"engine": {"id": "pinned-engine", "commit": "abc"},
            "model": {"sha256": "weight-sha"}, "requested_settings": {"context": 65536},
            "effective_settings": {"context": 65536}, "launch_command": ["owned-server", "--host", "127.0.0.1"]})
        for seed in self.config["grading"]["capacity_marker_seeds"]:
            self.add(identifier, {"kind": "capacity", "seed": seed, "markers_passed": 3,
                "retrieved_values": ["exact-a", "exact-b", "exact-c"], "token_evidence": self.evidence()})
        for group, fixtures, passed_limit in (("regular", self.config["grading"]["regular_fixture_ids"], regular_passes),
                ("long", self.config["grading"]["long_fixture_ids"], long_passes)):
            index = 0
            for fixture in fixtures:
                for seed in self.config["grading"]["seeds"]:
                    self.add(identifier, {"kind": "quality", "fixture": fixture, "seed": seed, "group": group,
                        "token_evidence": self.evidence()}, "passed" if index < passed_limit else "failed", "quality_failure" if index >= passed_limit else None)
                    index += 1
        for mode in ("fresh", "cached"):
            for replicate in range(20):
                self.add(identifier, {"kind": "timing", "mode": mode, "replicate": replicate,
                    "final_validation": True, "stability_passed": True, "token_evidence": self.evidence(mode),
                    "metrics": {"mode": mode, "first_action_seconds": fresh_seconds if mode == "fresh" else .5,
                        "first_stream_token_seconds": 1., "first_action_valid": True, "invalid_tool_calls": 0}})
        self.add(identifier, {"kind": "demo", "isolated": True})
        if speed is not None:
            for _ in range(3):
                self.add(identifier, {"kind": "throughput", "token_evidence": self.evidence(),
                    "metrics": {"generated_tokens": 128, "decode_seconds": 128 / speed,
                        "sustained_throughput_qualified": True}})
        return identifier

    def test_full_gates_and_complete_exports(self):
        self.full_candidate(regular_passes=27, long_passes=9)
        report = Report(self.config, self.state)
        summary = report.write()
        self.assertEqual(summary["winner_id"], "candidate")
        self.assertTrue(all(summary["configurations"][0]["gates"].values()))
        self.assertEqual(summary["configurations"][0]["latency"]["fresh"]["first_action"]["n"], 20)
        self.assertEqual(summary["configurations"][0]["throughput"]["median"], 64.)
        artifacts = Path(self.config["paths"]["artifacts"])
        rows = json.loads((artifacts / "results.json").read_text(encoding="utf-8"))
        with (artifacts / "results.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), len(rows))
        self.assertIn("--host", (artifacts / "engine-manifest.json").read_text(encoding="utf-8"))
        self.assertIn("evidence and exact configuration", (artifacts / "report.html").read_text(encoding="utf-8"))
        self.assertIsNone(self.state.run["started"])

    def test_primary_cold_ranking_and_both_tie_thresholds(self):
        self.full_candidate("fast", 10., 50.)
        self.full_candidate("tied-faster-decode", 10.8, 100.)
        self.full_candidate("outside-one-second", 11.1, 200.)
        summary = Report(self.config, self.state).summarize()
        self.assertEqual(summary["ranking"], ["tied-faster-decode", "fast", "outside-one-second"])
        self.assertEqual(summary["winner_id"], "tied-faster-decode")

    def test_no_exact_throughput_does_not_invent_tie_break(self):
        self.full_candidate("a", 10., None)
        self.add("a", {"kind": "throughput", "token_evidence": self.evidence(),
            "metrics": {"generated_tokens": 128, "decode_seconds": None,
                "sustained_throughput_qualified": False}})
        self.full_candidate("b", 10.5, 100.)
        summary = Report(self.config, self.state).summarize()
        self.assertIsNone(summary["winner_id"])
        self.assertEqual(summary["qualification_status"], "qualified_tie_unresolved")
        self.assertTrue(summary["ranking_uncertainty"])
        first = next(score for score in summary["configurations"] if score["configuration_id"] == "a")
        self.assertTrue(first["qualified"])
        self.assertTrue(first["gates"]["full_context_throughput_sample"])
        self.assertFalse(first["throughput"]["exact_sustained_64k"])
        self.assertIsNone(first["throughput"]["median"])

    def test_missing_unknown_short_or_invalid_full_context_throughput_cannot_qualify(self):
        self.full_candidate(speed=None)
        report = Report(self.config, self.state)
        score = report.summarize()["configurations"][0]
        self.assertEqual(score["failed_gates"], ["full_context_throughput_sample"])
        self.assertFalse(score["qualified"])
        for name, generated, status, resource, prompt in (
                ("unknown", None, "passed", {}, 61440),
                ("boolean", True, "passed", {}, 61440),
                ("short", 63, "passed", {}, 61440),
                ("negative", -1, "passed", {}, 61440),
                ("failed", 128, "failed", {}, 61440),
                ("foreign", 128, "passed", {"foreign_workload_overlap": True}, 61440),
                ("smaller_context", 128, "passed", {}, 16384)):
            with self.subTest(name=name):
                evidence = self.evidence()
                evidence.update(prompt_tokens=prompt, expected_prompt_tokens=prompt)
                self.add("candidate", {"kind": "throughput", "token_evidence": evidence,
                    "resources": resource, "metrics": {"generated_tokens": generated,
                        "decode_seconds": 2., "sustained_throughput_qualified": True}}, status)
                score = report.summarize()["configurations"][0]
                self.assertFalse(score["gates"]["full_context_throughput_sample"])
                self.assertFalse(score["qualified"])
                self.assertEqual(score["throughput"]["actual_count_samples"], 0)

    def test_actual_full_context_count_qualifies_without_exact_decode_duration(self):
        self.full_candidate(speed=None)
        self.add("candidate", {"kind": "throughput", "token_evidence": self.evidence(),
            "metrics": {"generated_tokens": 64, "decode_seconds": None,
                "sustained_throughput_qualified": False}})
        score = Report(self.config, self.state).summarize()["configurations"][0]
        self.assertTrue(score["qualified"])
        self.assertEqual(score["throughput"]["actual_count_samples"], 1)
        self.assertFalse(score["throughput"]["exact_sustained_64k"])
        self.assertIsNone(score["throughput"]["median"])

    def test_same_failure_cannot_be_erased_by_successful_retry(self):
        self.full_candidate()
        result = {"kind": "timing", "mode": "fresh", "replicate": 30, "final_validation": True,
            "stability_passed": False, "token_evidence": self.evidence(),
            "metrics": {"first_action_seconds": 10., "first_action_valid": True}}
        job = self.add("candidate", result, "invalid", "runtime_crash", transient=True)
        self.add("candidate", {**result, "stability_passed": True}, job=job)
        score = Report(self.config, self.state).summarize()["configurations"][0]
        self.assertFalse(score["gates"]["runtime_stability"])
        self.assertFalse(score["qualified"])
        self.assertEqual(len(score["stability_failures"]), 1)

    def test_unsupported_context_missing_action_and_duplicates_do_not_qualify(self):
        self.state.entity("configuration", "incomplete", {"effective_settings": {"context": 65536}})
        for replicate in range(25):
            self.add("incomplete", {"kind": "timing", "mode": "fresh", "replicate": 0,
                "final_validation": True, "stability_passed": True, "token_evidence": self.evidence(),
                "metrics": {"first_action_seconds": None, "first_action_valid": False}})
        summary = Report(self.config, self.state).summarize()
        score = summary["configurations"][0]
        self.assertFalse(score["qualified"])
        self.assertEqual(score["latency"]["fresh"]["first_action"]["n"], 0)
        self.assertEqual(summary["qualification_status"], "nothing_qualifies")

    def test_raw_retries_and_skips_preserved_quality_latest_only(self):
        result = {"kind": "quality", "group": "regular", "fixture": "py01", "seed": 42}
        job = self.add("partial", result, "failed", "temporary_infrastructure", transient=True)
        self.add("partial", result, job=job)
        self.add("partial", {"kind": "engine_probe"}, "skipped", "missing_compiler")
        report = Report(self.config, self.state)
        summary = report.write()
        self.assertEqual(summary["attempt_count"], 3)
        self.assertEqual(summary["configurations"][0]["regular"]["passed"], 1)
        self.assertEqual(summary["failure_counts"]["missing_compiler"], 1)

    def test_context_evidence_strict_and_unknown_cached_allowed_with_isolation(self):
        result = {"token_evidence": self.evidence()}
        self.assertEqual(context_reasons(result, self.config, "fresh"), [])
        for name, value, expected in (("prompt_tokens", 61441, "actual_full_prompt_count_unverified"),
                ("expected_prompt_tokens", 61441, "full_prompt_target_unverified"),
                ("truncated", True, "truncation_not_excluded"),
                ("cached_tokens", 65, "fresh_prefix_reuse"),
                ("tokenization_verified", False, "exact_tokenization_unverified")):
            changed = copy.deepcopy(result)
            changed["token_evidence"][name] = value
            self.assertIn(expected, context_reasons(changed, self.config, "fresh"))
        result["token_evidence"]["cached_tokens"] = None
        self.assertEqual(context_reasons(result, self.config, "fresh"), [])
        result["token_evidence"]["cache_isolation_verified"] = False
        self.assertIn("fresh_cache_isolation_unverified", context_reasons(result, self.config, "fresh"))

    def test_reports_redact_structured_and_text_credentials(self):
        self.state.entity("configuration", "secret-safe", {"Authorization": "Bearer abc-credential",
            "api_key": "another-credential", "notes": "Authorization: Bearer third-credential"})
        Report(self.config, self.state).write()
        for path in Path(self.config["paths"]["artifacts"]).iterdir():
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("abc-credential", content)
            self.assertNotIn("another-credential", content)
            self.assertNotIn("third-credential", content)

    def test_controlled_cached_miss_is_valid_without_claiming_reuse(self):
        evidence = self.evidence("cached")
        evidence.update(cached_tokens=0, prefix_reuse_verified=False,
            controlled_prefix_experiment_verified=True, observed_prefix_cache_hit=False)
        self.assertEqual(context_reasons({"token_evidence": evidence}, self.config, "cached"), [])
        evidence["cached_tokens"] = None
        self.assertIn("controlled_cached_counter_unverified",
            context_reasons({"token_evidence": evidence}, self.config, "cached"))
        evidence["cached_tokens"] = 0
        evidence["controlled_prefix_experiment_verified"] = False
        self.assertIn("controlled_prefix_experiment_unverified",
            context_reasons({"token_evidence": evidence}, self.config, "cached"))

    def test_gpu_parse_keeps_unknown_null_and_raw_driver_timestamp(self):
        line = ", ".join("2026/09/18 12:00:00" if field == "timestamp" else "GPU-a" if field == "uuid"
            else "N/A" if field == "power.draw" else "5" for field in GPU_FIELDS)
        parsed = parse_gpu_line(line)
        self.assertIsNone(parsed["power.draw"])
        self.assertEqual(parsed["timestamp"], "2026/09/18 12:00:00")
        self.assertIsNone(parse_gpu_line("driver error"))
        self.assertEqual(parse_processes("17, desktop.exe, [N/A]\n20, owned.exe, 200"),
            [{"pid": 17, "name": "desktop.exe", "used_gpu_mib": None},
             {"pid": 20, "name": "owned.exe", "used_gpu_mib": 200.}])

    def test_resource_snapshot_baseline_delta_interval_and_foreign_flags(self):
        self.state.control("doctor_baseline_gpu_pids", [17])
        now = [100.]
        collector = ResourceCollector(self.config, self.state, clock=lambda: now[0])
        collector.mark_owned_pid(20)
        self.assertEqual(collector.baseline_pids, {17})
        collector.shared_baseline = 1000
        collector._emit({"kind": "gpu", "gpu": {"memory.used": 5000, "utilization.encoder": None}})
        now[0] = 110.
        collector._emit({"kind": "host", "host": {"ram_used_bytes": 3000, "cpu_percent": 25},
            "memory": {"dedicated_gpu_bytes": 500, "shared_gpu_delta_bytes": 200},
            "potential_foreign_gpu_pids": [30], "foreign_workload_overlap": True,
            "foreign_workload_evidence_available": True})
        summary = collector.snapshot(started=105.)
        self.assertIsNone(summary["peak_gpu_used_mib"])
        self.assertEqual(summary["peak_shared_gpu_delta_bytes"], 200)
        self.assertTrue(summary["foreign_workload_overlap"])
        self.assertEqual(summary["potential_foreign_gpu_pids"], [30])
        self.assertFalse(collector._encoder_foreign({"utilization.encoder": None}))
        self.assertTrue(collector._encoder_foreign({"utilization.encoder": 10}))
        collector.memory.close()

    def test_persistent_collector_logs_clocked_samples_without_real_runtime(self):
        self.state.control("doctor_baseline_gpu_pids", [17])
        observed = threading.Event()

        class Host:
            def sample(self, owned):
                observed.set()
                return {"cpu_percent": None, "ram_used_bytes": 1234}

        class Memory:
            def __init__(self):
                self.count = 0

            def sample(self, owned):
                self.count += 1
                return {"dedicated_gpu_bytes": 2000,
                    "shared_gpu_bytes": 1000 if self.count == 1 else 1200}

            def close(self):
                pass

        class Process:
            def __init__(self):
                self.code = None
                self.stdout = [", ".join("timestamp" if field == "timestamp" else "GPU-a" if field == "uuid"
                    else "0" if field == "utilization.encoder" else "5" for field in GPU_FIELDS) + "\n"]

            def poll(self):
                return self.code

            def terminate(self):
                self.code = 0

            def wait(self, timeout):
                return self.code

        fake = Process()
        with patch("localbench.resources.subprocess.Popen", return_value=fake) as launch, \
                patch("localbench.resources.subprocess.run", return_value=SimpleNamespace(
                    returncode=0, stdout="| 0 N/A N/A 17 C+G desktop.exe N/A |\n| 0 N/A N/A 30 C foreign.exe 200MiB |")):
            collector = ResourceCollector(self.config, self.state, host_sampler=Host(), memory_sampler=Memory())
            collector.start()
            self.assertTrue(observed.wait(timeout=2))
            # Join on the collector's own loop event rather than starting GPU work.
            collector.mark_owned_pid(20)
            summary = collector.stop()
        self.assertEqual(launch.call_count, 1)
        self.assertIn("-l", launch.call_args.args[0])
        self.assertEqual(summary["shared_gpu_baseline_bytes"], 1000)
        self.assertEqual(summary["peak_shared_gpu_delta_bytes"], 200)
        self.assertTrue(summary["foreign_workload_overlap"])
        self.assertIsNone(self.state.run["started"])
        events = [json.loads(line) for line in Path(summary["log"]).read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all("perf_counter_ns" in event and "time" in event for event in events))
        self.assertEqual(events[-1]["kind"], "collector_stop")


if __name__ == "__main__":
    unittest.main()
