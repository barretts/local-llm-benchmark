import json
from pathlib import Path
import tempfile
import unittest

from localbench.config import load
from localbench.resources import ResourceCollector
from localbench.state import State


class ClosedSampler:
    def close(self):
        pass


class ResourceRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ("project", "state", "logs", "artifacts"):
            self.config["paths"][name] = str(Path(self.temp.name) / name)
        self.state = State(self.config)
        self.now = 0.
        self.collector = ResourceCollector(self.config, self.state, clock=lambda: self.now,
            host_sampler=ClosedSampler(), memory_sampler=ClosedSampler(), engine_sampler=ClosedSampler())
        self.collector.started = 0.

    def tearDown(self):
        if self.collector.file:
            self.collector.file.close()
        self.state.close()
        self.temp.cleanup()

    def emit(self, sample):
        result = self.collector._emit(sample)
        self.now += 1
        return result

    def host(self, pid, ram, shared, busy, foreign=False, uncertain=False, known=True):
        return {"kind": "host", "host": {"ram_used_bytes": ram, "cpu_percent": None},
            "memory": {"shared_gpu_delta_bytes": shared}, "potential_foreign_gpu_pids": [pid],
            "active_foreign_gpu_pids": [pid] if foreign else [],
            "uncertain_gpu_pids": [pid] if uncertain else [],
            "gpu_engines": {"gpu_engine_processes": [{"pid": pid, "busy_percent": busy}]},
            "foreign_workload_overlap": foreign, "foreign_workload_attribution_uncertain": uncertain,
            "foreign_workload_evidence_available": known}

    def fill_beyond_retention(self):
        self.emit({"kind": "gpu", "gpu": {"memory.used": 9000, "power.draw": None}})
        self.emit(self.host(17, 9000, 500, 90., foreign=True, uncertain=True, known=False))
        for _ in range(self.collector.samples.maxlen + 3):
            self.emit({"kind": "gpu", "gpu": {"memory.used": 100, "power.draw": None}})
        self.emit(self.host(20, 200, 5, 0.))

    def test_session_peaks_survive_eviction_recent_window_excludes_old_peaks_and_raw_records_survive(self):
        self.collector.path = Path(self.temp.name) / "all-resources.jsonl"
        self.collector.file = self.collector.path.open("w", encoding="utf-8")
        self.fill_beyond_retention()
        self.collector.file.flush()
        whole = self.collector.snapshot()
        recent = self.collector.snapshot(started=self.now - 900, finished=self.now)
        self.assertGreaterEqual(self.collector.samples.maxlen, 7200)
        self.assertEqual(len(self.collector.samples), self.collector.samples.maxlen)
        self.assertGreater(whole["discarded_sample_count"], 0)
        self.assertGreater(whole["sample_count"], len(self.collector.samples))
        self.assertTrue(whole["resource_window_complete"])
        self.assertEqual(whole["resource_summary_scope"], "whole_session_aggregate")
        self.assertEqual(whole["peak_gpu_used_mib"], 9000)
        self.assertEqual(whole["peak_ram_used_bytes"], 9000)
        self.assertEqual(whole["peak_shared_gpu_delta_bytes"], 500)
        self.assertEqual(whole["peak_gpu_engine_percent_by_pid"], {"17": 90., "20": 0.})
        self.assertTrue(whole["foreign_workload_overlap"])
        self.assertTrue(whole["foreign_workload_attribution_uncertain"])
        self.assertFalse(whole["foreign_workload_evidence_available"])
        self.assertEqual(whole["active_foreign_gpu_pids"], [17])
        self.assertTrue(recent["resource_window_complete"])
        self.assertEqual(recent["peak_gpu_used_mib"], 100)
        self.assertEqual(recent["peak_ram_used_bytes"], 200)
        self.assertEqual(recent["peak_shared_gpu_delta_bytes"], 5)
        self.assertEqual(recent["peak_gpu_engine_percent_by_pid"], {"20": 0.})
        self.assertFalse(recent["foreign_workload_overlap"])
        self.assertFalse(recent["foreign_workload_attribution_uncertain"])
        self.assertTrue(recent["foreign_workload_evidence_available"])
        raw = [json.loads(line) for line in self.collector.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(raw), whole["sample_count"])
        self.assertEqual(raw[0]["gpu"]["memory.used"], 9000)
        self.assertTrue(raw[1]["foreign_workload_overlap"])
        self.assertEqual(raw[-1]["host"]["ram_used_bytes"], 200)
        self.assertTrue(all("time" in row and "perf_counter_ns" in row for row in raw))
        self.assertIsNone(whole["peak_power_w"])
        self.assertIsNone(self.state.run["started"])

    def test_old_intervals_explicitly_report_incomplete_including_ranges_with_no_retained_samples(self):
        self.fill_beyond_retention()
        incomplete = self.collector.snapshot(started=0., finished=self.now)
        self.assertFalse(incomplete["resource_window_complete"])
        self.assertEqual(incomplete["peak_gpu_used_mib"], 100)
        empty_old = self.collector.snapshot(started=0., finished=1.)
        self.assertFalse(empty_old["resource_window_complete"])
        self.assertEqual(empty_old["sample_count"], 0)
        self.assertIsNone(empty_old["peak_gpu_used_mib"])
        self.assertFalse(self.collector.snapshot(finished=1.)["resource_window_complete"])
        retained = self.collector.snapshot(started=self.collector.samples[0]["time"])
        self.assertTrue(retained["resource_window_complete"])

    def test_same_timestamp_eviction_is_incomplete_and_snapshot_values_do_not_mutate_session(self):
        for _ in range(self.collector.samples.maxlen + 1):
            self.collector._emit(self.host(17, 10, None, None))
        self.assertFalse(self.collector.snapshot(started=0.)["resource_window_complete"])
        whole = self.collector.snapshot()
        whole["potential_foreign_gpu_pids"].append(99)
        whole["peak_gpu_engine_percent_by_pid"]["17"] = 100.
        independent = self.collector.snapshot()
        self.assertEqual(independent["potential_foreign_gpu_pids"], [17])
        self.assertEqual(independent["peak_gpu_engine_percent_by_pid"], {"17": None})
        self.assertIsNone(independent["peak_shared_gpu_delta_bytes"])

    def test_error_details_are_bounded_but_session_error_count_is_complete(self):
        for index in range(130):
            message = "mock telemetry error " + str(index)
            self.collector.errors.append(message)
            self.emit({"kind": "collector_error", "source": "mock", "error": message})
        whole = self.collector.snapshot()
        self.assertEqual(whole["collector_error_count"], 130)
        self.assertEqual(len(whole["errors"]), 64)
        self.assertEqual(whole["error_detail_limit"], 64)
        self.assertEqual(whole["errors"][-1], "mock telemetry error 129")
        recent = self.collector.snapshot(started=120.)
        self.assertEqual(recent["collector_error_count"], 10)


if __name__ == "__main__":
    unittest.main()
