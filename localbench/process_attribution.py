"""Pure telemetry parsing with namespace-qualified PIDs; no process operations."""

from __future__ import annotations

import math
import re


ROW = re.compile(r"^\s*\|\s*(\d+)\s+(?:(N/A|\d+)\s+(N/A|\d+)\s+)?(\d+)\s+"
                 r"(C\+G|G\+C|C|G|M)\s+(.+?)\s+(N/A|[\d.]+(?:MiB|GiB|KiB))\s*\|\s*$")


def parse_gpu_process_table(text, pid_namespace="windows"):
    """Parse process rows from full nvidia-smi; query-compute-apps lacks type."""
    if not isinstance(pid_namespace, str) or not pid_namespace:
        raise ValueError("PID namespace is required")
    output = []
    for line in text.splitlines():
        match = ROW.match(line)
        if match is None:
            continue
        gpu, gi, ci, pid, process_type, name, memory = match.groups()
        observed = None
        if memory != "N/A":
            number = float(memory[:-3])
            observed = number * {"MiB": 1., "GiB": 1024., "KiB": 1 / 1024}[memory[-3:]]
        output.append({"gpu_index": int(gpu), "gpu_instance_id": int(gi) if gi not in (None, "N/A") else None,
            "compute_instance_id": int(ci) if ci not in (None, "N/A") else None,
            "pid_namespace": pid_namespace, "pid": int(pid), "type": "C+G" if process_type == "G+C" else process_type,
            "process_name": name.strip(), "used_gpu_mib": observed, "source": "nvidia-smi-full-table"})
    return output


def _keys(identities):
    result = set()
    for identity in identities:
        if not isinstance(identity, tuple) or len(identity) != 2 or not isinstance(identity[0], str) or \
                type(identity[1]) is not int or identity[1] <= 0:
            raise ValueError("Ownership requires (PID namespace, PID), never a bare integer")
        result.add(identity)
    return result


def classify_processes(rows, owned=(), baseline=(), activity=None, minimum_activity_percent=10.):
    """Describe evidence without assuming a C+G desktop is busy or Docker-owned.

    A compute context is not an utilization sample. Caller decides whether new
    pure-C context presence alone is grounds to repeat a timing measurement.
    Container ownership must come from label-verified docker top host PIDs; an
    aggregate Windows vmmem PID never proves ownership of every WSL workload.
    """
    owned, baseline = _keys(owned), _keys(baseline)
    activity = activity or {}
    result = {"owned": [], "baseline": [], "potential_foreign": [], "new_compute_contexts": [],
              "new_mixed_contexts": [], "new_graphics_contexts": [], "active_foreign": [],
              "unknown_type_or_activity": [], "aggregate_scope_ambiguous": []}
    for row in rows:
        key = (row["pid_namespace"], row["pid"])
        item = dict(row)
        value = activity.get(key)
        item["activity_percent"] = value if type(value) in (float, int) and math.isfinite(value) and value >= 0 else None
        item["activity_proven"] = item["activity_percent"] is not None and item["activity_percent"] >= minimum_activity_percent
        # vmmem can aggregate several distributions/containers even when a
        # Windows caller registers that PID. Do not promote it to owned here.
        aggregate = "vmmem" in row.get("process_name", "").lower()
        if aggregate:
            result["aggregate_scope_ambiguous"].append(item)
        if key in owned and not aggregate:
            result["owned"].append(item)
            continue
        if key in baseline and not aggregate:
            result["baseline"].append(item)
            continue
        result["potential_foreign"].append(item)
        category = {"C": "new_compute_contexts", "C+G": "new_mixed_contexts", "G": "new_graphics_contexts"}.get(item.get("type"))
        if category:
            result[category].append(item)
        if item["activity_proven"]:
            result["active_foreign"].append(item)
        else:
            result["unknown_type_or_activity"].append(item)
    return result
