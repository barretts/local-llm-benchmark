"""Low-overhead telemetry; observation never changes another process or GPU settings."""

from __future__ import annotations

import csv
from collections import deque
import ctypes
import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time


GPU_FIELDS = ("timestamp", "index", "uuid", "memory.total", "memory.used", "memory.free",
              "utilization.gpu", "utilization.memory", "utilization.encoder",
              "utilization.decoder", "temperature.gpu", "power.draw", "clocks.sm")


def _number(value):
    try:
        value = float(str(value).strip())
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def parse_gpu_line(line, fields=GPU_FIELDS):
    """N/A is unknown, never zero. Keep the vendor timestamp beside client clocks."""
    values = next(csv.reader([line]))
    if len(values) != len(fields):
        return None
    result = {}
    for field, value in zip(fields, values):
        result[field] = value.strip() if field in {"timestamp", "uuid"} else _number(value)
    return result


def parse_processes(text):
    result = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 3:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue
        result.append({"pid": pid, "name": row[1].strip(), "used_gpu_mib": _number(row[2])})
    return result


class WindowsHost:
    """Use native Windows APIs instead of launching PowerShell at sample frequency."""

    def __init__(self):
        self.previous = None
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None

    def sample(self, pids=()):
        result = {"cpu_percent": None, "ram_total_bytes": None, "ram_available_bytes": None,
                  "ram_used_bytes": None, "owned_process_ram_bytes": None,
                  "owned_process_ram": {}}
        if self.kernel is None:
            return result
        from ctypes import wintypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total",
                 "page_available", "virtual_total", "virtual_available", "extended")]

        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if self.kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
            result.update(ram_total_bytes=memory.total, ram_available_bytes=memory.available,
                          ram_used_bytes=memory.total - memory.available)
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if self.kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            pair = lambda item: (item.dwHighDateTime << 32) | item.dwLowDateTime
            now = (pair(idle), pair(kernel) + pair(user))
            if self.previous is not None and now[1] > self.previous[1]:
                result["cpu_percent"] = max(0., min(100., 100. * (1 -
                    (now[0] - self.previous[0]) / (now[1] - self.previous[1]))))
            self.previous = now
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        class ProcessMemory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "working", "quota_peak_paged",
                 "quota_paged", "quota_peak_nonpaged", "quota_nonpaged", "pagefile",
                 "peak_pagefile", "private")]

        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        known = []
        for pid in pids:
            handle = self.kernel.OpenProcess(0x1000 | 0x10, False, pid)
            value = None
            if handle:
                try:
                    info = ProcessMemory()
                    info.cb = ctypes.sizeof(info)
                    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(info), info.cb):
                        value = info.working
                finally:
                    self.kernel.CloseHandle(handle)
            result["owned_process_ram"][str(pid)] = value
            if value is not None:
                known.append(value)
        if pids and len(known) == len(pids):
            result["owned_process_ram_bytes"] = sum(known)
        return result


class WindowsGpuMemory:
    """One persistent PDH query expands adapter/process wildcards every five seconds."""

    PATHS = (r"\GPU Adapter Memory(*)\Dedicated Usage", r"\GPU Adapter Memory(*)\Shared Usage",
             r"\GPU Process Memory(*)\Dedicated Usage", r"\GPU Process Memory(*)\Shared Usage")

    def __init__(self):
        self.pdh, self.query, self.handles = None, None, {}
        self.error = None
        if os.name != "nt":
            self.error = "Windows PDH unavailable on this OS"
            return
        from ctypes import wintypes
        try:
            self.pdh = ctypes.WinDLL("pdh")
            self.pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
            self.pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR,
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
            self.pdh.PdhExpandWildCardPathW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR,
                wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
            self.pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            self.pdh.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, wintypes.DWORD,
                ctypes.c_void_p, ctypes.c_void_p]
            self.pdh.PdhRemoveCounter.argtypes = [ctypes.c_void_p]
            self.pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
            self.query = ctypes.c_void_p()
            if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)):
                raise OSError("PdhOpenQueryW failed")
        except (OSError, AttributeError) as error:
            self.error = str(error)
            self.pdh = None

    def _expand(self, wildcard):
        from ctypes import wintypes
        size = wintypes.DWORD(0)
        self.pdh.PdhExpandWildCardPathW(None, wildcard, None, ctypes.byref(size), 0)
        if not size.value:
            return []
        buffer = ctypes.create_unicode_buffer(size.value)
        if self.pdh.PdhExpandWildCardPathW(None, wildcard, buffer, ctypes.byref(size), 0):
            return []
        return [path for path in buffer[:size.value].split("\0") if path]

    def sample(self, owned_pids=()):
        result = {"dedicated_gpu_bytes": None, "shared_gpu_bytes": None,
                  "owned_dedicated_gpu_bytes": None, "owned_shared_gpu_bytes": None,
                  "gpu_memory_counters": {}, "gpu_memory_error": self.error}
        if not self.pdh:
            return result

        class Value(ctypes.Structure):
            _fields_ = [("status", ctypes.c_ulong), ("value", ctypes.c_longlong)]

        active = {path for wildcard in self.PATHS for path in self._expand(wildcard)}
        for path in set(self.handles) - active:
            self.pdh.PdhRemoveCounter(self.handles.pop(path))
        for path in active - set(self.handles):
            handle = ctypes.c_void_p()
            if not self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle)):
                self.handles[path] = handle
        if self.pdh.PdhCollectQueryData(self.query):
            result["gpu_memory_error"] = "PdhCollectQueryData failed"
            return result
        groups = {"dedicated_gpu_bytes": [], "shared_gpu_bytes": [],
                  "owned_dedicated_gpu_bytes": [], "owned_shared_gpu_bytes": []}
        for path, handle in self.handles.items():
            value = Value()
            status = self.pdh.PdhGetFormattedCounterValue(handle, 0x400, None, ctypes.byref(value))
            observed = value.value if not status and value.status in (0, 1) else None
            result["gpu_memory_counters"][path] = observed
            lower = path.lower()
            name = "shared_gpu_bytes" if "shared usage" in lower else "dedicated_gpu_bytes"
            if "gpu adapter memory" in lower:
                groups[name].append(observed)
            else:
                match = re.search(r"pid_(\d+)_", lower)
                if match and int(match.group(1)) in owned_pids:
                    groups["owned_" + name].append(observed)
        for name, values in groups.items():
            if values and all(value is not None for value in values):
                result[name] = sum(values)
        return result

    def close(self):
        if self.pdh and self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = None


def parse_gpu_engine_path(path):
    """PDH supplies Windows PID identities, separate from NVML/Linux PIDs."""
    instance = re.search(r"GPU Engine\(([^)]+)\)\\Utilization Percentage$", path, re.I)
    if not instance:
        return None
    name = instance.group(1)
    pid = re.search(r"(?:^|_)pid_(\d+)(?:_|$)", name, re.I)
    physical = re.search(r"_phys_(\d+)(?:_|$)", name, re.I)
    engine = re.search(r"_eng_(\d+)(?:_|$)", name, re.I)
    kind = re.search(r"_engtype_(.+)$", name, re.I)
    if not pid:
        return None
    return {"pid": int(pid.group(1)), "pid_namespace": "windows", "instance": name,
        "physical_adapter": int(physical.group(1)) if physical else None,
        "engine_index": int(engine.group(1)) if engine else None,
        "engine_type": kind.group(1) if kind else None}


def _windows_process_name(pid):
    if os.name != "nt" or pid <= 0:
        return None
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        return buffer.value if kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)) else None
    finally:
        kernel.CloseHandle(handle)


class WindowsGpuEngines(WindowsGpuMemory):
    """Persistent PDH rate query: collect each second, expand every five seconds."""

    PATHS = (r"\GPU Engine(*)\Utilization Percentage",)

    def __init__(self, clock=time.monotonic, name_lookup=_windows_process_name):
        super().__init__()
        self.clock, self.name_lookup = clock, name_lookup
        self.next_refresh, self.names = 0., {}

    def sample(self):
        result = {"gpu_engine_counters": {}, "gpu_engine_processes": [], "gpu_engine_query_valid": False,
            "gpu_engine_counter_coverage_complete": False, "gpu_engine_error": self.error,
            "gpu_engine_sampled_monotonic": self.clock(), "gpu_engine_scope": "WDDM adapter/engine instance identities"}
        if not self.pdh:
            return result
        now = self.clock()
        if now >= self.next_refresh:
            self.next_refresh = now + 5
            active = {path for wildcard in self.PATHS for path in self._expand(wildcard)}
            for path in set(self.handles) - active:
                self.pdh.PdhRemoveCounter(self.handles.pop(path))
            for path in active - set(self.handles):
                handle = ctypes.c_void_p()
                if not self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle)):
                    self.handles[path] = handle
            pids = {parsed["pid"] for path in self.handles if (parsed := parse_gpu_engine_path(path)) is not None}
            self.names = {pid: self.name_lookup(pid) for pid in pids}
        if self.pdh.PdhCollectQueryData(self.query):
            result["gpu_engine_error"] = "PdhCollectQueryData failed"
            return result

        class Value(ctypes.Structure):
            _fields_ = [("status", ctypes.c_ulong), ("value", ctypes.c_double)]

        by_pid = {}
        for path, handle in self.handles.items():
            parsed = parse_gpu_engine_path(path)
            if parsed is None:
                continue
            value = Value()
            status = self.pdh.PdhGetFormattedCounterValue(handle, 0x200, None, ctypes.byref(value))
            observed = value.value if not status and value.status in (0, 1) and math.isfinite(value.value) and value.value >= 0 else None
            result["gpu_engine_counters"][path] = {**parsed, "utilization_percent": observed,
                "pdh_status": int(status), "counter_status": int(value.status)}
            by_pid.setdefault(parsed["pid"], []).append({**parsed, "utilization_percent": observed})
        for pid, rows in by_pid.items():
            known = [row["utilization_percent"] for row in rows if row["utilization_percent"] is not None]
            result["gpu_engine_processes"].append({"pid": pid, "pid_namespace": "windows", "process_name": self.names.get(pid),
                "busy_percent": max(known) if known else None, "counter_coverage_complete": len(known) == len(rows),
                "engine_types": sorted({row["engine_type"] for row in rows if row["engine_type"]}), "engine_instances": len(rows)})
        values = list(result["gpu_engine_counters"].values())
        result["gpu_engine_query_valid"] = any(row["utilization_percent"] is not None for row in values)
        result["gpu_engine_counter_coverage_complete"] = bool(values) and all(row["utilization_percent"] is not None for row in values)
        if not values:
            result["gpu_engine_error"] = "GPU Engine utilization counters unavailable"
        return result


def gpu_workload_evidence(process_rows, engine_sample, owned_pids=(), baseline_pids=(), threshold=10.):
    """Busy baseline apps remain foreign; vmmem activity has uncertain ownership."""
    from .process_attribution import classify_processes
    rows = {(row["pid_namespace"], row["pid"]): dict(row) for row in process_rows}
    activity = {}
    for engine in (engine_sample or {}).get("gpu_engine_processes", []):
        key = (engine["pid_namespace"], engine["pid"])
        value = engine.get("busy_percent")
        activity[key] = value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None
        if key not in rows:
            rows[key] = {"pid": engine["pid"], "pid_namespace": engine["pid_namespace"], "type": None,
                "process_name": engine.get("process_name") or "", "source": "Windows-PDH-GPU-Engine",
                "identity_verified": bool(engine.get("process_name")) and engine["pid"] > 0}
        elif engine.get("process_name"):
            # A full native process name recognizes aggregate vmmem even when
            # nvidia-smi shortened the display name to an ellipsis.
            rows[key]["native_process_name"] = engine["process_name"]
            if "vmmem" in engine["process_name"].lower():
                rows[key]["process_name"] = engine["process_name"]
    attribution = classify_processes(list(rows.values()), owned={("windows", pid) for pid in owned_pids},
        baseline={("windows", pid) for pid in baseline_pids}, activity=activity, minimum_activity_percent=threshold)
    aggregates = attribution["aggregate_scope_ambiguous"]
    aggregate_keys = {(row["pid_namespace"], row["pid"]) for row in aggregates}
    busy_baseline = [row for row in attribution["baseline"] if row["activity_proven"]]
    busy_other = [row for row in attribution["active_foreign"]
        if (row["pid_namespace"], row["pid"]) not in aggregate_keys and row.get("identity_verified", True)]
    unknown_identity_busy = [row for row in attribution["active_foreign"] if not row.get("identity_verified", True)]
    conservative_compute = [row for row in attribution["new_compute_contexts"]
        if (row["pid_namespace"], row["pid"]) not in aggregate_keys and row["activity_percent"] is None]
    ambiguous = [row for row in aggregates if row["activity_percent"] is None or row["activity_proven"]] + unknown_identity_busy
    active = busy_baseline + busy_other
    attribution["active_foreign_including_baseline"] = active
    attribution["busy_baseline_processes"] = busy_baseline
    attribution["new_compute_contexts_with_unknown_activity"] = conservative_compute
    potential = attribution["potential_foreign"] + busy_baseline
    return {"attribution": attribution, "foreign_overlap": bool(active or conservative_compute),
        "attribution_uncertain": bool(ambiguous), "uncertain_gpu_pids": sorted({row["pid"] for row in ambiguous}),
        "active_foreign_gpu_pids": sorted({row["pid"] for row in active}),
        "potential_foreign_gpu_pids": sorted({row["pid"] for row in potential})}


class _ResourceAggregate:
    """Keep session summaries independent of the bounded raw-record window."""

    PEAKS = ("peak_gpu_used_mib", "peak_dedicated_gpu_bytes", "peak_windows_dedicated_counter_bytes",
        "peak_shared_gpu_delta_bytes", "peak_ram_used_bytes", "peak_owned_process_ram_bytes",
        "peak_cpu_percent", "peak_gpu_utilization_percent", "peak_temperature_c", "peak_power_w")
    PID_FIELDS = ("potential_foreign_gpu_pids", "active_foreign_gpu_pids", "uncertain_gpu_pids")

    def __init__(self):
        self.count, self.host_count, self.error_count = 0, 0, 0
        self.first_time, self.last_time = None, None
        self.peaks = dict.fromkeys(self.PEAKS)
        self.pids = {name: set() for name in self.PID_FIELDS}
        self.engine_peaks = {}
        self.foreign_overlap, self.attribution_uncertain, self.evidence_complete = False, False, True

    @staticmethod
    def maximum(previous, value):
        if type(value) not in (int, float) or not math.isfinite(value):
            return previous
        return value if previous is None else max(previous, value)

    def observe(self, sample):
        self.count += 1
        self.first_time = sample["time"] if self.first_time is None else min(self.first_time, sample["time"])
        self.last_time = sample["time"] if self.last_time is None else max(self.last_time, sample["time"])
        self.foreign_overlap |= sample.get("foreign_encoder_overlap", False)
        kind = sample.get("kind")
        if kind == "collector_error":
            self.error_count += 1
        if kind == "gpu":
            gpu = sample.get("gpu") or {}
            used = gpu.get("memory.used")
            values = {"peak_gpu_used_mib": used,
                "peak_dedicated_gpu_bytes": used * 1024 * 1024 if type(used) in (int, float) else None,
                "peak_gpu_utilization_percent": gpu.get("utilization.gpu"),
                "peak_temperature_c": gpu.get("temperature.gpu"), "peak_power_w": gpu.get("power.draw")}
        elif kind == "host":
            self.host_count += 1
            host, memory = sample.get("host") or {}, sample.get("memory") or {}
            values = {"peak_windows_dedicated_counter_bytes": memory.get("dedicated_gpu_bytes"),
                "peak_shared_gpu_delta_bytes": memory.get("shared_gpu_delta_bytes"),
                "peak_ram_used_bytes": host.get("ram_used_bytes"),
                "peak_owned_process_ram_bytes": host.get("owned_process_ram_bytes"),
                "peak_cpu_percent": host.get("cpu_percent")}
            for name in self.PID_FIELDS:
                self.pids[name].update(sample.get(name, []))
            self.foreign_overlap |= sample.get("foreign_workload_overlap", False)
            self.attribution_uncertain |= sample.get("foreign_workload_attribution_uncertain", False)
            self.evidence_complete &= sample.get("foreign_workload_evidence_available", False)
            for row in sample.get("gpu_engines", {}).get("gpu_engine_processes", []):
                pid = row["pid"]
                self.engine_peaks[pid] = self.maximum(self.engine_peaks.get(pid), row.get("busy_percent"))
        else:
            values = {}
        for name, value in values.items():
            self.peaks[name] = self.maximum(self.peaks[name], value)

    def result(self):
        return {"sample_count": self.count, **self.peaks,
            **{name: sorted(values) for name, values in self.pids.items()},
            "peak_gpu_engine_percent_by_pid": {str(pid): value for pid, value in sorted(self.engine_peaks.items())},
            "foreign_workload_overlap": self.foreign_overlap,
            "foreign_workload_attribution_uncertain": self.attribution_uncertain,
            "foreign_workload_evidence_available": bool(self.host_count) and self.evidence_complete,
            "collector_error_count": self.error_count}


def compress_resource_log(path):
    """Lossless NTFS storage; keep the exact JSONL bytes and sample schedule."""
    if os.name!='nt':return False
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
        wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.DeviceIoControl.argtypes=[wintypes.HANDLE,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,
        ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p]
    kernel.DeviceIoControl.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.GetFileAttributesW.argtypes=[wintypes.LPCWSTR]
    kernel.GetFileAttributesW.restype=wintypes.DWORD
    handle=kernel.CreateFileW(str(path),0xC0000000,7,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise OSError('resource_log_compression_open_failed')
    try:
        default=ctypes.c_ushort(1);returned=wintypes.DWORD()
        if not kernel.DeviceIoControl(handle,0x9C040,ctypes.byref(default),ctypes.sizeof(default),
                None,0,ctypes.byref(returned),None):raise OSError('resource_log_compression_failed')
    finally:kernel.CloseHandle(handle)
    attributes=kernel.GetFileAttributesW(str(path))
    if attributes==0xFFFFFFFF or not attributes&0x800:raise OSError('resource_log_compression_unverified')
    return True

class ResourceCollector:
    """Persistent telemetry owned by the controller; never a model-runtime probe."""

    def __init__(self, config, state, clock=time.time, host_sampler=None, memory_sampler=None, engine_sampler=None):
        self.config, self.state, self.clock = config, state, clock
        self.host = host_sampler or WindowsHost()
        self.memory = memory_sampler or WindowsGpuMemory()
        self.engines = engine_sampler or WindowsGpuEngines()
        self.lock = threading.RLock()
        self.event = threading.Event()
        self.threads, self.process = [], None
        # About two hours at the normal one GPU + one host record per second.
        # Every original record still goes to JSONL; this only bounds RAM usage.
        self.samples = deque(maxlen=14400)
        self.session = _ResourceAggregate()
        self.discarded_samples = 0
        self.discarded_first_time, self.discarded_last_time = None, None
        self.owned_pids, self.baseline_pids = set(), set()
        self.shared_baseline = None
        self.baseline_encoder = None
        self.latest_gpu = None
        self.memory_sample = None
        self.process_sample = None
        self.engine_sample = None
        self.started = None
        self.file = None
        self.path = None
        self.errors = deque(maxlen=64)
        self._load_baseline()

    def _load_baseline(self):
        saved = self.state.get_control("doctor_baseline_gpu_pids")
        if saved is not None:
            self.baseline_pids.update(int(pid) for pid in saved)
            return
        path = Path(self.config["paths"]["artifacts"]) / "host.json"
        try:
            host = json.loads(path.read_text(encoding="utf-8"))
            rows = parse_processes(host.get("commands", {}).get("gpu_processes", {}).get("output", ""))
            self.baseline_pids.update(row["pid"] for row in rows)
        except (OSError, ValueError, TypeError):
            pass
        self.state.control("doctor_baseline_gpu_pids", sorted(self.baseline_pids))

    def mark_owned_pid(self, pid, owned=True):
        if type(pid) is not int or pid <= 0:
            raise ValueError("PID must be a positive integer")
        with self.lock:
            self.owned_pids.add(pid) if owned else self.owned_pids.discard(pid)
        self.state.control("resource_owned_pids", sorted(self.owned_pids))

    def start(self):
        if self.started is not None:
            return self
        self.started = self.clock()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = Path(self.config["paths"]["logs"]) / ("resources-" + stamp + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        compressed=compress_resource_log(self.path)
        self.file = self.path.open("a", encoding="utf-8", buffering=1)
        self.state.control("resource_log", str(self.path))
        self.memory_sample = self.memory.sample(self.owned_pids)
        self.engine_sample = self.engines.sample()
        self.shared_baseline = self.memory_sample.get("shared_gpu_bytes") if not self.owned_pids else None
        self._emit({"kind": "collector_start", "baseline_gpu_pids": sorted(self.baseline_pids),
                    "log_storage": "lossless_ntfs_compression" if compressed else "plain_jsonl",
                    "shared_memory_note": "allocation delta is diagnostic; it does not prove active spill"})
        argv = ["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
                "--format=csv,noheader,nounits", "-l", "1"]
        try:
            self.process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            gpu = threading.Thread(target=self._gpu_loop, name="localbench-gpu-telemetry", daemon=True)
            self.threads.append(gpu)
            gpu.start()
        except OSError as error:
            self.errors.append(str(error))
            self._emit({"kind": "collector_error", "source": "nvidia-smi", "error": str(error)})
        thread = threading.Thread(target=self._host_loop, name="localbench-host-telemetry", daemon=True)
        self.threads.append(thread)
        thread.start()
        return self

    def _emit(self, sample):
        sample = dict(sample)
        sample.update(time=self.clock(), perf_counter_ns=time.perf_counter_ns())
        with self.lock:
            if len(self.samples) == self.samples.maxlen:
                discarded = self.samples[0]["time"]
                self.discarded_samples += 1
                self.discarded_first_time = discarded if self.discarded_first_time is None else min(self.discarded_first_time, discarded)
                self.discarded_last_time = discarded if self.discarded_last_time is None else max(self.discarded_last_time, discarded)
            self.samples.append(sample)
            self.session.observe(sample)
            if self.file and not self.file.closed:
                self.file.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
        return sample

    def _gpu_loop(self):
        for line in self.process.stdout:
            if self.event.is_set():
                break
            parsed = parse_gpu_line(line)
            if parsed is None:
                self._emit({"kind": "gpu_raw", "raw": line.rstrip(), "gpu": None})
                continue
            with self.lock:
                self.latest_gpu = parsed
                if self.baseline_encoder is None and not self.owned_pids:
                    self.baseline_encoder = parsed.get("utilization.encoder")
            self._emit({"kind": "gpu", "raw": line.rstrip(), "gpu": parsed,
                        "foreign_encoder_overlap": self._encoder_foreign(parsed)})
        if not self.event.is_set():
            error = "persistent nvidia-smi telemetry ended unexpectedly"
            self.errors.append(error)
            self._emit({"kind": "collector_error", "source": "nvidia-smi", "error": error})

    def _encoder_foreign(self, gpu):
        value = gpu.get("utilization.encoder")
        return value is not None and value >= 10

    def _host_loop(self):
        next_memory = 0.
        interval = self.config["measurement"].get("resource_sample_seconds", 1)
        memory_interval = self.config["measurement"].get("shared_memory_sample_seconds", 5)
        while not self.event.is_set():
            tick = time.monotonic()
            try:
                with self.lock:
                    owned = set(self.owned_pids)
                host = self.host.sample(owned)
                self.engine_sample = self.engines.sample()
                if tick >= next_memory:
                    next_memory = tick + memory_interval
                    self.memory_sample = self.memory.sample(owned)
                    shared = self.memory_sample.get("shared_gpu_bytes")
                    if self.shared_baseline is None and shared is not None and not owned:
                        self.shared_baseline = shared
                    try:
                        query = subprocess.run(["nvidia-smi"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=4,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        from .process_attribution import parse_gpu_process_table
                        self.process_sample = {"query_valid": query.returncode == 0,
                            "processes": parse_gpu_process_table(query.stdout) if query.returncode == 0 else None,
                            "raw": query.stdout, "sampled_at": self.clock()}
                    except (OSError, subprocess.TimeoutExpired) as error:
                        self.process_sample = {"query_valid": False, "processes": None,
                                               "error": str(error), "sampled_at": self.clock()}
                memory = dict(self.memory_sample or {})
                shared = memory.get("shared_gpu_bytes")
                memory["shared_gpu_baseline_bytes"] = self.shared_baseline
                memory["shared_gpu_delta_bytes"] = None if shared is None or self.shared_baseline is None \
                    else shared - self.shared_baseline
                processes = self.process_sample or {}
                evidence = gpu_workload_evidence(processes.get("processes") or [], self.engine_sample,
                    owned, self.baseline_pids)
                self._emit({"kind": "host", "host": host, "memory": memory,
                    "gpu_processes": processes, "gpu_engines": self.engine_sample,
                    "potential_foreign_gpu_pids": evidence["potential_foreign_gpu_pids"],
                    "active_foreign_gpu_pids": evidence["active_foreign_gpu_pids"],
                    "gpu_process_attribution": evidence["attribution"],
                    "foreign_workload_attribution_uncertain": evidence["attribution_uncertain"],
                    "uncertain_gpu_pids": evidence["uncertain_gpu_pids"],
                    "foreign_workload_overlap": evidence["foreign_overlap"] or
                        bool(self.latest_gpu and self._encoder_foreign(self.latest_gpu)),
                    "foreign_workload_evidence_available": processes.get("query_valid", False) and
                        self.engine_sample.get("gpu_engine_query_valid", False) and
                        self.engine_sample.get("gpu_engine_counter_coverage_complete", False) and
                        not evidence["attribution_uncertain"]})
            except Exception as error:
                self.errors.append(str(error))
                self._emit({"kind": "collector_error", "source": "host", "error": str(error)})
            self.event.wait(max(.01, interval - (time.monotonic() - tick)))

    def snapshot(self, started=None, finished=None):
        with self.lock:
            whole_session = started is None and finished is None
            aggregate = self.session if whole_session else _ResourceAggregate()
            if not whole_session:
                for sample in self.samples:
                    if (started is None or sample["time"] >= started) and (finished is None or sample["time"] <= finished):
                        aggregate.observe(sample)
            result = aggregate.result()
            intersects_discarded = self.discarded_samples and \
                (started is None or started <= self.discarded_last_time) and \
                (finished is None or finished >= self.discarded_first_time)
            result.update(resource_window_complete=whole_session or not bool(intersects_discarded),
                resource_summary_scope="whole_session_aggregate" if whole_session else "retained_record_interval",
                retained_sample_count=len(self.samples), retained_sample_limit=self.samples.maxlen,
                discarded_sample_count=self.discarded_samples,
                retained_started=self.samples[0]["time"] if self.samples else None,
                retained_finished=self.samples[-1]["time"] if self.samples else None,
                errors=list(self.errors), error_detail_limit=self.errors.maxlen)
        return {**result, "log": str(self.path) if self.path else None,
            "started": started if started is not None else self.started, "finished": finished,
            "dedicated_memory_source": "NVML GPU0 memory.used;includes baseline driver/display allocation",
            "shared_gpu_baseline_bytes": self.shared_baseline,
            "gpu_engine_percent_definition": "maximum observed WDDM engine utilization counter per Windows PID;never sum parallel engines"}

    def stop(self):
        self.event.set()
        if self.process and self.process.poll() is None:
            # This is only the nvidia-smi child launched by this collector.
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for thread in self.threads:
            thread.join(timeout=6)
        self.memory.close()
        self.engines.close()
        self._emit({"kind": "collector_stop"})
        summary = self.snapshot()
        if self.file:
            self.file.close()
        self.state.control("resource_summary", summary)
        return summary
