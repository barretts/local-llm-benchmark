"""Project-local, fail-closed supervision of the existing benchmark CLI.

No model/API/GPU work, credentials, State constructor, termination, or OS tasks.
Start only after review. A final endpoint still requires the normal handoff.
"""
from pathlib import Path
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from localbench.config import atomic_json, digest, file_hash, load
from localbench.recovery import _split_windows_command_line
from localbench.acquisition import _disk_headroom, _runtime_headroom

PENDING = {"baseline_measurements_pending", "current_measurements_pending"}
CREDENTIAL_FILE = r"C:\Users\barrett\lmstudio-local.txt"
IDENTITY_FIELDS = ("id", "spec_hash", "created", "started", "deadline")
BLOCK_PARTS = ("source_review", "pending_auth", "pending_load", "pending_descendants", "owned_recovery",
    "ownership", "unrelated_loaded", "disk_headroom", "storage_disk_headroom", "runtime_soft_cap",
    "weight_budget_exhausted", "budget_exhausted", "stop_after_current", "gates not verified",
    "fixtures are not verified", "another controller", "authentication_unavailable", "auth_unavailable",
    "stale_managed_", "stale_pid_", "stale_process_", "unverified_saved_process", "unverified_managed_instance",
    "owned_managed_", "refusing_unload", "refusing_termination")


def ordinary(path, root):
    path, root = Path(path), Path(root)
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("private_artifact_path_required")
    cursor = path
    while cursor != root.parent:
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise RuntimeError("ordinary_private_path_required")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    return path


def filetime_utc(text):
    # Preserve Windows' seventh fractional digit, unlike datetime microseconds.
    match = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,7}))?(?:Z|\+00:00)", text or "")
    if not match:
        raise RuntimeError("invalid_creation_receipt")
    seconds = int(datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
    return (seconds + 11644473600) * 10000000 + int((match[2] or "").ljust(7, "0"))


def read_snapshot(config):
    # URI read-only connection: never instantiate State or run recovery.
    path = ordinary(Path(config["paths"]["state"]) / "benchmark.sqlite3", PROJECT)
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("BEGIN")
        run = dict(db.execute("SELECT * FROM runs ORDER BY created DESC LIMIT 1").fetchone())
        controls = {r["key"]: json.loads(r["value"]) for r in db.execute("SELECT key,value FROM controls WHERE key IN "
            "('doctor_verified','harness_verified','fixtures_verified','search_covered','remaining_search',"
            "'unfinished_current_measurement_jobs','ownership_recovery','managed_load_preflight','waiting_for_idle','active_job')")}
        row = db.execute("SELECT data FROM entities WHERE kind='controller' AND id=?", (run["id"],)).fetchone()
        controller = json.loads(row[0]) if row else None
        protocol = (controller or {}).get("protocol_source_hashes", {})
        pending = 0
        for row in db.execute("SELECT key_json FROM jobs WHERE status IN ('pending','running')"):
            key = json.loads(row[0]); kind = key.get("kind")
            if kind not in {"quality", "capacity", "timing", "throughput", "common_byte"}:
                continue
            files = ("quality.py", "tools.py", "schema.py") if kind == "quality" else \
                ("common_byte.py", "measurement.py") if kind == "common_byte" else \
                ("measurement.py", "prompts.py", "regional_prompt.py")
            expected = digest({"protocol": {p: protocol.get(p) for p in files}, "run": run["id"],
                "fixture_hash": key.get("fixture_prompt_hash") if kind == "quality" else None,
                "kind": kind, "seed": key.get("seed"), "mode": key.get("mode"),
                "replicate": key.get("replicate"), "target": key.get("target_tokens")})
            pending += key.get("logical_plan_hash") == expected
        finished = db.execute("SELECT COUNT(*),COALESCE(MAX(id),0) FROM attempts WHERE finished IS NOT NULL").fetchone()
        running = db.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0] + \
            db.execute("SELECT COUNT(*) FROM attempts WHERE finished IS NULL").fetchone()[0]
        weights = [dict(r) for r in db.execute("SELECT id,bytes,purpose,path,acquired FROM weights ORDER BY id")]
        tuning = [(r[0], json.loads(r[1]).get("status")) for r in db.execute("SELECT id,data FROM entities "
            "WHERE kind='tuning_proposal' ORDER BY id")]
        failures = [(r[0], r[1] or "") for r in db.execute("SELECT id,reason FROM attempts "
            "WHERE status IN ('invalid','skipped') ORDER BY id DESC LIMIT 128")]
        helper_path = ordinary(Path(config["paths"]["artifacts"]) / "credential-controller-status.json", PROJECT)
        helper = json.loads(helper_path.read_text(encoding="utf-8")) if helper_path.is_file() else None
        return {"run": run, "controls": controls, "controller": controller, "pending": pending,
            "weights": weights, "failures": failures, "last_attempt": db.execute("SELECT COALESCE(MAX(id),0) FROM attempts").fetchone()[0],
            "completed_attempts": finished[0], "running": running,
            "progress": digest({"finished": list(finished), "weights": [(r["id"], r["acquired"]) for r in weights], "tuning": tuning}),
            "helper": helper, "execution_hash": load()["_execution_hash"]}
    finally:
        db.close()


class QueryHandle:
    """Hold the original process object; query rights only, never termination."""
    def __init__(self, pid):
        from ctypes import wintypes
        self.pid = pid
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.c_void_p] * 4
        self.api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.OpenProcess(0x1000 | 0x00100000, False, pid)
        if not self.handle:
            raise RuntimeError("process_identity_unavailable")
        values = [ctypes.c_ulonglong() for _ in range(4)]
        if not self.api.GetProcessTimes(self.handle, *(ctypes.byref(v) for v in values)):
            self.close(); raise RuntimeError("process_creation_unavailable")
        self.created = values[0].value
        self.parent_pid, self.argv = None, None

    def exit_code(self):
        value = ctypes.c_ulong()
        if not self.api.GetExitCodeProcess(self.handle, ctypes.byref(value)):
            raise RuntimeError("process_exit_unavailable")
        return None if value.value == 259 else value.value

    def alive(self):
        return self.exit_code() is None

    def close(self):
        if getattr(self, "handle", None):
            self.api.CloseHandle(self.handle); self.handle = None


class WindowsBackend:
    def __init__(self, config):
        self.config = config
        self.pwsh = shutil.which("pwsh")
        if os.name != "nt" or not self.pwsh:
            raise RuntimeError("Windows_PowerShell7_required")
        observer = QueryHandle(os.getpid())
        self.supervisor_created = observer.created
        observer.close()

    def preflight(self, snapshot):
        _disk_headroom(self.config, 0)
        _runtime_headroom(self.config, 0)
        if snapshot["run"]["stop_requested"] or snapshot["run"]["deadline"] <= time.time() + 7200:
            raise RuntimeError("stop_or_budget_launch_block")

    def inspect(self, pid):
        if type(pid) is not int or pid <= 0:
            raise RuntimeError("invalid_registered_pid")
        watch = QueryHandle(pid)
        try:
            # Raw command text is internal only; do not save it, even on mismatch.
            script = "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "';if($null-ne$p){$p|Select-Object ProcessId,ParentProcessId,CommandLine|ConvertTo-Json -Compress}"
            result = subprocess.run([self.pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode or not result.stdout.strip():
                raise RuntimeError("registered_process_not_live_or_unverifiable")
            metadata = json.loads(result.stdout.decode("utf-8-sig"))
            if metadata.get("ProcessId") != pid or not watch.alive():
                raise RuntimeError("registered_process_identity_changed")
            watch.parent_pid = metadata.get("ParentProcessId")
            watch.argv = _split_windows_command_line(metadata["CommandLine"])
            return watch
        except Exception:
            watch.close(); raise

    def launch_helper(self, run_id):
        ready = ordinary(Path(self.config["paths"]["artifacts"]) / "credential-controller-ready.json", PROJECT)
        value = json.loads(ready.read_text(encoding="utf-8"))
        if value.get("run_id") != run_id or value.get("ready") is not True:
            raise RuntimeError("verified_helper_readiness_required")
        helper = ordinary(PROJECT / "scripts" / "credential-controller.ps1", PROJECT)
        stamp = uuid.uuid4().hex
        paths = [ordinary(Path(self.config["paths"]["logs"]) / ("unattended-helper-" + stamp + suffix), PROJECT)
                 for suffix in ("-stdout.log", "-stderr.log")]
        with paths[0].open("wb") as out, paths[1].open("wb") as err:
            return subprocess.Popen([self.pwsh, "-NoProfile", "-STA", "-File", str(helper), "-RunId", run_id,
                "-CredentialFile", CREDENTIAL_FILE, "-WorkerCommand", "sweep", "-MaximumWaitSeconds", "60"], cwd=PROJECT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)


class Supervisor:
    def __init__(self, config, reader, backend, emit, clock=time.time, delay=45, maximum_resumes=24):
        self.config, self.reader, self.backend, self.emit, self.clock = config, reader, backend, emit, clock
        self.delay, self.maximum_resumes = delay, maximum_resumes
        self.initial = reader(); self.identity = {k: self.initial["run"][k] for k in IDENTITY_FIELDS}
        if self.identity["started"] is None or self.initial["execution_hash"] != config["_execution_hash"]:
            raise RuntimeError("original_started_execution_required")
        self.execution_hash = config["_execution_hash"]
        self.helper_hash = file_hash(PROJECT / "scripts" / "credential-controller.ps1")
        self.script_hash = file_hash(Path(__file__))
        self.source_hashes = {str(p.relative_to(PROJECT)): file_hash(p) for p in (PROJECT / "localbench").rglob("*.py")}
        for filename, expected in (self.initial.get("controller") or {}).get("protocol_source_hashes", {}).items():
            if file_hash(PROJECT / "localbench" / filename) != expected:
                raise RuntimeError("worker_protocol_source_changed_requires_review")
        self.weights = {r["id"]: r for r in self.initial["weights"]}
        self.watches, self.worker, self.logs = [], None, None
        self.progress = self.initial["progress"]; self.last_attempt = self.initial["last_attempt"]
        self.no_progress, self.resumes, self.next_launch, self.launching = 0, 0, None, None
        self.done = False
        self.cached = self.initial

    def record(self, status, reason=None, event=False, **fields):
        snapshot = self.cached
        active = snapshot["controls"].get("active_job") or {}
        self.emit({"run_id": self.identity["id"], "status": status, "reason": reason,
            "timestamp_utc": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
            "supervisor_pid": os.getpid(), "resumes": self.resumes, "no_progress_cycles": self.no_progress,
            "supervisor_created_filetime": getattr(self.backend, "supervisor_created", None),
            "supervisor_script_sha256": self.script_hash, "current_job_id": active.get("job"),
            "current_attempt_id": snapshot.get("last_attempt"),
            "deadline": self.identity["deadline"], "remaining_seconds": max(0, self.identity["deadline"] - self.clock()),
            "completed_attempts": snapshot.get("completed_attempts"), "pending_current_jobs": snapshot["pending"],
            "new_weight_bytes_reserved": sum(r["bytes"] for r in snapshot["weights"]),
            "execution_hash": self.execution_hash, "credential_helper_sha256": self.helper_hash,
            "worker_source_manifest_sha256": digest(self.source_hashes), "phase": active.get("kind"),
            "active_job": {k: active.get(k) for k in ("kind", "fixture", "seed", "engine", "model", "configuration_id")},
            **fields}, event)

    def halt(self, reason, completed=False):
        self.done = True
        self.record("complete_final_review_required" if completed else "attention_required", reason, True)

    def check(self, snapshot):
        self.cached = snapshot
        if any(snapshot["run"][k] != v for k, v in self.identity.items()) or snapshot["execution_hash"] != self.execution_hash:
            raise RuntimeError("original_run_or_execution_identity_changed")
        if any(snapshot["controls"].get(k) is not True for k in ("doctor_verified", "harness_verified", "fixtures_verified")):
            raise RuntimeError("verification_gate_unavailable")
        total = sum(r["bytes"] for r in snapshot["weights"])
        if total > self.config["limits"]["new_model_weight_bytes"]:
            raise RuntimeError("weight_budget_exhausted")
        current = {r["id"]: r for r in snapshot["weights"]}
        for identifier, old in self.weights.items():
            new = current.get(identifier)
            if new is None or any(new[k] != old[k] for k in ("bytes", "purpose", "path")) or new["acquired"] < old["acquired"]:
                raise RuntimeError("cumulative_weight_ledger_changed")
        for identifier, item in current.items():
            if identifier not in self.weights and item["bytes"] and not Path(item["path"]).resolve().is_relative_to(Path(self.config["paths"]["new_model_root"]).resolve()):
                raise RuntimeError("new_weight_destination_changed")
        self.weights = current
        for key in ("ownership_recovery", "managed_load_preflight"):
            records = snapshot["controls"].get(key) or []
            records = [records] if isinstance(records, dict) else records
            if any(r.get("status") in {"refused", "pending_descendants", "pending_auth", "pending_load"} for r in records):
                raise RuntimeError("owned_recovery_unresolved")

    def approved(self, watch, command):
        expected = [str(PROJECT / ".venv" / "Scripts" / "python.exe"), "-X", "utf8", "-m", "localbench", command]
        if not isinstance(watch.argv, list) or len(watch.argv) != len(expected) or \
                str(Path(watch.argv[0]).resolve()).casefold() != str(Path(expected[0]).resolve()).casefold() or watch.argv[1:] != expected[1:]:
            raise RuntimeError("registered_command_identity_mismatch")
        return digest(expected)

    def adopt(self, snapshot, required_pid=None):
        self.check(snapshot)
        if snapshot["run"]["stop_requested"]:
            raise RuntimeError("stop_requested")
        registry = snapshot.get("controller") or {}
        pid = registry.get("pid")
        command = registry.get("command")
        if command not in {"resume", "sweep"} or (required_pid is not None and pid != required_pid):
            raise RuntimeError("registered_worker_identity_required")
        watch = self.backend.inspect(pid); watches = [watch]
        try:
            command_hash = self.approved(watch, command)
            helper = snapshot.get("helper") or {}
            mode = "registry_started_tolerance"
            if helper.get("run_id") != self.identity["id"] or helper.get("status") != "launched":
                raise RuntimeError("matching_nonsecret_launch_receipt_required")
            if helper.get("child_pid") == pid:
                if filetime_utc(helper.get("child_created_utc")) != watch.created:
                    raise RuntimeError("helper_creation_receipt_mismatch")
                mode = "exact_helper_creation"
            elif watch.parent_pid == helper.get("child_pid"):
                launcher = self.backend.inspect(helper["child_pid"]); watches.append(launcher)
                self.approved(launcher, command)
                if launcher.created != filetime_utc(helper.get("child_created_utc")) or launcher.parent_pid != helper.get("parent_pid") \
                        or not 0 <= watch.created - launcher.created <= 150000000:
                    raise RuntimeError("helper_parent_creation_receipt_mismatch")
                mode = "exact_helper_launcher_parent"
            else:
                raise RuntimeError("helper_parent_identity_mismatch")
            self.logs = {k: ordinary(helper[k], PROJECT / ".logs") for k in ("stdout_log", "stderr_log")}
            self.watches, self.worker = watches, watch
            self._last_worker_pid = pid
            self.record("observing", event=True, worker_pid=pid, creation_filetime=watch.created,
                command_identity_hash=command_hash, adoption_proof=mode, observed_pids=[w.pid for w in watches],
                worker_command=command, launch_receipt={"child_pid": helper["child_pid"],
                    "child_created_filetime": filetime_utc(helper["child_created_utc"]), "parent_pid": helper.get("parent_pid")})
        except Exception:
            for item in watches: item.close()
            raise

    def exit_status(self):
        if any(w.exit_code() != 0 for w in self.watches):
            return "unexpected_nonzero_exit"
        if self.logs["stdout_log"].stat().st_size > 16 * 1024 * 1024:
            return "exit_result_unavailable"
        try:
            result = json.loads(self.logs["stdout_log"].read_text(encoding="utf-8").strip())
            return result.get("status", "exit_result_unavailable")
        except (OSError, ValueError):
            return "exit_result_unavailable"

    def step(self):
        if self.done: return
        try:
            snapshot = self.reader(); self.check(snapshot)
            stopped = snapshot["run"]["stop_requested"] != 0
            expired = self.clock() + 7200 + self.config["limits"]["per_task_timeout_seconds"] >= self.identity["deadline"]
            if self.watches and any(w.alive() for w in self.watches):
                self.record("observing_stop_requested" if stopped else "observing_budget_limit" if expired else "observing",
                    worker_pid=self.worker.pid, observed_pids=[w.pid for w in self.watches])
                return
            if stopped or expired:
                self.halt("stop_requested" if stopped else "benchmark_budget_limit"); return
            if self.watches:
                outcome = self.exit_status()
                for watch in self.watches: watch.close()
                self.watches, self.worker = [], None
                self.record("worker_exited", outcome, True)
                if outcome == "covered_search_complete":
                    controls = snapshot["controls"]
                    if controls.get("search_covered") is True and controls.get("remaining_search") == [] and controls.get("unfinished_current_measurement_jobs") == []:
                        self.halt("benchmark_complete_endpoint_requires_final_review", True)
                    else: self.halt("completion_not_durably_corroborated")
                    return
                if outcome not in PENDING:
                    self.halt("source_review_required" if outcome == "pending_source_review" else outcome); return
                if not snapshot["pending"]:
                    self.halt("pending_result_without_current_pending_jobs"); return
                new_blocks = [reason for identifier, reason in snapshot["failures"] if identifier > self.last_attempt
                    and any(part in reason.lower() for part in BLOCK_PARTS)]
                if new_blocks:
                    self.halt("safety_outcome_requires_attention"); return
                self.no_progress = self.no_progress + 1 if snapshot["progress"] == self.progress else 0
                if self.no_progress >= 3 or self.resumes >= self.maximum_resumes:
                    self.halt("no_progress_limit" if self.no_progress >= 3 else "supervisor_resume_limit"); return
                self.progress, self.last_attempt = snapshot["progress"], snapshot["last_attempt"]
                self.next_launch = self.clock() + self.delay
                self.record("pending_backoff", event=True, resume_after=self.next_launch)
                return
            if self.launching:
                registry = snapshot.get("controller") or {}
                if registry.get("pid") != self.old_pid and registry.get("started", 0) >= self.launched_at:
                    self.adopt(snapshot); self.launching = None; return
                if self.clock() - self.launched_at >= 120 or self.launching.poll() not in (None, 0):
                    self.halt("helper_launch_or_adoption_failed")
                return
            if self.next_launch is None:
                self.halt("no_verified_live_worker_to_adopt"); return
            if self.clock() < self.next_launch: return
            if snapshot["running"]:
                self.halt("running_measurements_require_manual_recovery"); return
            if file_hash(PROJECT / "scripts" / "credential-controller.ps1") != self.helper_hash:
                self.halt("helper_source_changed_requires_review"); return
            if {str(p.relative_to(PROJECT)): file_hash(p) for p in (PROJECT / "localbench").rglob("*.py")} != self.source_hashes:
                self.halt("worker_source_changed_requires_review"); return
            self.old_pid = (snapshot.get("controller") or {}).get("pid")
            # A newly registered worker prevents racing an independent launch.
            if self.old_pid != self.last_worker_pid:
                self.adopt(snapshot); self.next_launch = None; return
            self.launched_at = self.clock()
            self.backend.preflight(snapshot)
            immediate = self.reader(); self.check(immediate)
            if immediate["running"]:
                self.halt("running_measurements_require_manual_recovery"); return
            if immediate["run"]["stop_requested"] or immediate["run"]["deadline"] <= self.clock() + 7200 + self.config["limits"]["per_task_timeout_seconds"]:
                self.halt("stop_or_budget_launch_block"); return
            self.launching = self.backend.launch_helper(self.identity["id"])
            self.resumes += 1; self.next_launch = None
            self.record("helper_launched", event=True, helper_pid=self.launching.pid)
        except Exception as error:
            # Never expose raw command lines, subprocess output or exception text.
            reason = str(error) if isinstance(error, RuntimeError) and re.fullmatch(r"[A-Za-z0-9_]+", str(error)) else "supervisor_observation_unavailable"
            self.halt(reason)

    @property
    def last_worker_pid(self):
        return getattr(self, "_last_worker_pid", self.initial["controller"]["pid"])

    def close(self):
        for watch in self.watches: watch.close()


class Singleton:
    def __init__(self, path): self.path, self.file = Path(path), None
    def __enter__(self):
        self.file = self.path.open("a+b")
        if self.file.tell() == 0: self.file.write(b"0"); self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt; msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl; fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close(); raise RuntimeError("supervisor_already_running")
        return self
    def __exit__(self, *_): self.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-pid", type=int, required=True)
    parser.add_argument("--delay-seconds", type=int, choices=range(30, 61), default=45)
    parser.add_argument("--maximum-resumes", type=int, choices=range(1, 65), default=24)
    args = parser.parse_args(); config = load()
    artifacts = ordinary(Path(config["paths"]["artifacts"]), PROJECT)
    def emit(record, event):
        atomic_json(ordinary(artifacts / "unattended-supervisor-status.json", PROJECT), record)
        if event:
            prefix = "unattended-attention-" if record["status"] == "attention_required" else "unattended-event-"
            atomic_json(ordinary(artifacts / (prefix + uuid.uuid4().hex + ".json"), PROJECT), record)
    with Singleton(ordinary(artifacts / "unattended-supervisor.lock", PROJECT)):
        supervisor = Supervisor(config, lambda: read_snapshot(config), WindowsBackend(config), emit,
            delay=args.delay_seconds, maximum_resumes=args.maximum_resumes)
        try:
            try: supervisor.adopt(supervisor.initial, args.adopt_pid)
            except Exception: supervisor.halt("initial_worker_adoption_failed")
            while not supervisor.done:
                supervisor.step(); time.sleep(1 if supervisor.launching else args.delay_seconds)
        finally: supervisor.close()
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception: print("Unattended supervisor unavailable; no worker was terminated.", file=sys.stderr); raise SystemExit(2)
