import copy
import ctypes
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from localbench.config import load
from localbench.resources import (ResourceCollector, WindowsGpuEngines, WindowsGpuMemory,
    gpu_workload_evidence, parse_gpu_engine_path)
from localbench.state import State


def engine_path(pid, index=0, kind="Compute_0"):
    return rf"\GPU Engine(pid_{pid}_luid_0x00000000_0x00001234_phys_0_eng_{index}_engtype_{kind})\Utilization Percentage"


class FakePdh:
    def __init__(self, values):
        self.values, self.paths = values, {}
        self.collected, self.removed, self.closed = 0, [], 0

    def PdhAddEnglishCounterW(self, query, path, user, pointer):
        pointer._obj.value = len(self.paths) + 1
        self.paths[pointer._obj.value] = path
        return 0

    def PdhCollectQueryData(self, query):
        self.collected += 1
        return 0

    def PdhGetFormattedCounterValue(self, handle, flags, kind, pointer):
        assert flags == 0x200, "GPU Engine must use floating-point percentage, not byte format"
        status, value = self.values[self.paths[handle.value]]
        pointer._obj.status = status
        pointer._obj.value = value
        return 0

    def PdhRemoveCounter(self, handle):
        self.removed.append(handle.value)

    def PdhCloseQuery(self, query):
        self.closed += 1


class GpuEngineResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ("project", "state", "logs", "artifacts"):
            self.config["paths"][name] = str(Path(self.temp.name) / name)
        self.state = State(self.config)

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def row(self, pid, name="desktop.exe", kind="C+G", namespace="windows"):
        return {"pid": pid, "pid_namespace": namespace, "process_name": name, "type": kind}

    def rates(self, *rows):
        return {"gpu_engine_query_valid": True, "gpu_engine_counter_coverage_complete": True,
            "gpu_engine_processes": [{"pid": pid, "pid_namespace": "windows", "busy_percent": value,
                "process_name": name} for pid, value, name in rows]}

    def test_engine_instance_parser_preserves_namespace_physical_adapter_and_type(self):
        value = parse_gpu_engine_path(engine_path(17, 3, "3D"))
        self.assertEqual(value["pid_namespace"], "windows")
        self.assertEqual(value["pid"], 17)
        self.assertEqual(value["physical_adapter"], 0)
        self.assertEqual(value["engine_index"], 3)
        self.assertEqual(value["engine_type"], "3D")
        self.assertIsNone(parse_gpu_engine_path(r"\GPU Adapter Memory(foo)\Shared Usage"))
        self.assertIsNone(parse_gpu_engine_path(r"\GPU Engine(unknown)\Utilization Percentage"))

    def test_persistent_query_collects_each_second_refreshes_each_five_and_uses_maximum(self):
        clock = [0.]
        paths = [engine_path(17), engine_path(17, 1, "Copy"), engine_path(20)]
        pdh = FakePdh({paths[0]: (0, 25.5), paths[1]: (1, 75.), paths[2]: (0, 0.)})
        def initialize(instance):
            instance.pdh, instance.query, instance.handles, instance.error = pdh, ctypes.c_void_p(3), {}, None
        with patch.object(WindowsGpuMemory, "__init__", initialize):
            sampler = WindowsGpuEngines(clock=lambda: clock[0], name_lookup=lambda pid: "desktop.exe")
        with patch.object(sampler, "_expand", return_value=paths) as expand:
            first = sampler.sample()
            clock[0] = 1.
            second = sampler.sample()
            self.assertEqual(expand.call_count, 1)
            self.assertEqual(pdh.collected, 2)
            self.assertEqual(first["gpu_engine_processes"][0]["busy_percent"],
                second["gpu_engine_processes"][0]["busy_percent"])
            by_pid = {row["pid"]: row for row in second["gpu_engine_processes"]}
            self.assertEqual(by_pid[17]["busy_percent"], 75.)
            self.assertEqual(by_pid[17]["engine_instances"], 2)
            self.assertEqual(by_pid[20]["busy_percent"], 0.)
            self.assertTrue(second["gpu_engine_query_valid"])
            self.assertTrue(second["gpu_engine_counter_coverage_complete"])
            clock[0] = 5.
            expand.return_value = paths[:2]
            sampler.sample()
            self.assertEqual(expand.call_count, 2)
            self.assertEqual(len(pdh.removed), 1)
        sampler.close()
        self.assertEqual(pdh.closed, 1)

    def test_unknown_rate_nan_and_empty_provider_remain_unknown(self):
        path = engine_path(17)
        pdh = FakePdh({path: (0xC0000BBA, 0.)})
        def initialize(instance):
            instance.pdh, instance.query, instance.handles, instance.error = pdh, ctypes.c_void_p(3), {}, None
        with patch.object(WindowsGpuMemory, "__init__", initialize):
            sampler = WindowsGpuEngines(clock=lambda: 0., name_lookup=lambda pid: None)
        with patch.object(sampler, "_expand", return_value=[path]):
            unknown = sampler.sample()
            self.assertIsNone(unknown["gpu_engine_processes"][0]["busy_percent"])
            self.assertFalse(unknown["gpu_engine_query_valid"])
            self.assertFalse(unknown["gpu_engine_counter_coverage_complete"])
            pdh.values[path] = (0, float("nan"))
            self.assertIsNone(sampler.sample()["gpu_engine_processes"][0]["busy_percent"])
        sampler.handles = {}
        empty = sampler.sample()
        self.assertFalse(empty["gpu_engine_query_valid"])
        self.assertIn("unavailable", empty["gpu_engine_error"])
        json.dumps(empty, allow_nan=False)
        sampler.close()

    def test_busy_baseline_desktop_is_foreign_but_owned_pid_is_not(self):
        rows = [self.row(17), self.row(20, "owned-server.exe", "C")]
        evidence = gpu_workload_evidence(rows, self.rates((17, 10., "desktop.exe"), (20, 100., "owned-server.exe")),
            owned_pids={20}, baseline_pids={17})
        self.assertTrue(evidence["foreign_overlap"])
        self.assertEqual(evidence["active_foreign_gpu_pids"], [17])
        self.assertEqual(evidence["attribution"]["busy_baseline_processes"][0]["pid"], 17)
        self.assertFalse(evidence["attribution_uncertain"])
        quiet = gpu_workload_evidence(rows, self.rates((17, 9.99, "desktop.exe"), (20, 100., "owned-server.exe")),
            owned_pids={20}, baseline_pids={17})
        self.assertFalse(quiet["foreign_overlap"])

    def test_new_compute_unknown_is_conservative_and_known_idle_clears_it(self):
        rows = [self.row(30, "unrelated-compute.exe", "C"), self.row(31, "new-browser.exe", "C+G")]
        unknown = gpu_workload_evidence(rows, {})
        self.assertTrue(unknown["foreign_overlap"])
        self.assertEqual(unknown["active_foreign_gpu_pids"], [])
        self.assertEqual(len(unknown["attribution"]["new_compute_contexts_with_unknown_activity"]), 1)
        idle = gpu_workload_evidence(rows, self.rates((30, 0., "unrelated-compute.exe"), (31, 0., "new-browser.exe")))
        self.assertFalse(idle["foreign_overlap"])
        active = gpu_workload_evidence(rows, self.rates((30, 0., "unrelated-compute.exe"), (31, 12., "new-browser.exe")))
        self.assertEqual(active["active_foreign_gpu_pids"], [31])

    def test_vmmem_is_uncertain_even_if_registered_owned_or_baseline(self):
        row = self.row(40, "vmmemWSL.exe", "C")
        for owned, baseline in (({40}, set()), (set(), {40}), (set(), set())):
            with self.subTest(owned=owned, baseline=baseline):
                evidence = gpu_workload_evidence([row], self.rates((40, 100., "vmmemWSL.exe")), owned, baseline)
                self.assertFalse(evidence["foreign_overlap"])
                self.assertTrue(evidence["attribution_uncertain"])
                self.assertEqual(evidence["uncertain_gpu_pids"], [40])
                self.assertEqual(evidence["active_foreign_gpu_pids"], [])
        unknown = gpu_workload_evidence([row], {})
        self.assertTrue(unknown["attribution_uncertain"])
        quiet = gpu_workload_evidence([row], self.rates((40, 0., "vmmemWSL.exe")))
        self.assertFalse(quiet["attribution_uncertain"])

    def test_shortened_table_name_uses_native_name_to_detect_vmmem_and_unknown_pid_is_uncertain(self):
        row = self.row(40, "...", "C")
        evidence = gpu_workload_evidence([row], self.rates((40, 88., r"C:\Windows\System32\vmmemWSL.exe")))
        self.assertTrue(evidence["attribution_uncertain"])
        self.assertFalse(evidence["foreign_overlap"])
        missing = gpu_workload_evidence([], self.rates((77, 100., None)))
        self.assertTrue(missing["attribution_uncertain"])
        self.assertEqual(missing["uncertain_gpu_pids"], [77])
        self.assertFalse(missing["foreign_overlap"])

    def test_counter_pid_and_linux_pid_do_not_share_ownership(self):
        linux_row = self.row(20, "linux-foreign", "C", namespace="linux")
        evidence = gpu_workload_evidence([linux_row], self.rates((20, 100., "owned-server.exe")), owned_pids={20})
        self.assertTrue(evidence["foreign_overlap"])
        self.assertEqual(evidence["active_foreign_gpu_pids"], [])
        self.assertEqual(evidence["attribution"]["owned"][0]["pid_namespace"], "windows")
        self.assertEqual(evidence["attribution"]["new_compute_contexts_with_unknown_activity"][0]["pid_namespace"], "linux")

    def test_snapshot_reports_interval_activity_and_aggregate_uncertainty(self):
        class ClosedSampler:
            def close(self):
                pass
        collector = ResourceCollector(self.config, self.state, host_sampler=ClosedSampler(),
            memory_sampler=ClosedSampler(), engine_sampler=ClosedSampler(), clock=lambda: 100.)
        collector._emit({"kind": "host", "host": {}, "memory": {}, "gpu_engines": self.rates((17, 15., "desktop.exe")),
            "potential_foreign_gpu_pids": [17], "active_foreign_gpu_pids": [17], "uncertain_gpu_pids": [],
            "foreign_workload_overlap": True, "foreign_workload_attribution_uncertain": False,
            "foreign_workload_evidence_available": True})
        collector.clock = lambda: 110.
        collector._emit({"kind": "host", "host": {}, "memory": {}, "gpu_engines": self.rates((40, None, "vmmemWSL.exe")),
            "potential_foreign_gpu_pids": [40], "active_foreign_gpu_pids": [], "uncertain_gpu_pids": [40],
            "foreign_workload_overlap": False, "foreign_workload_attribution_uncertain": True,
            "foreign_workload_evidence_available": False})
        all_samples = collector.snapshot()
        self.assertTrue(all_samples["foreign_workload_overlap"])
        self.assertTrue(all_samples["foreign_workload_attribution_uncertain"])
        self.assertEqual(all_samples["active_foreign_gpu_pids"], [17])
        self.assertEqual(all_samples["peak_gpu_engine_percent_by_pid"], {"17": 15., "40": None})
        last = collector.snapshot(started=105.)
        self.assertFalse(last["foreign_workload_overlap"])
        self.assertEqual(last["uncertain_gpu_pids"], [40])
        self.assertFalse(last["foreign_workload_evidence_available"])
        self.assertIsNone(self.state.run["started"])


if __name__ == "__main__":
    unittest.main()
