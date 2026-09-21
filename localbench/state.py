from contextlib import contextmanager
from pathlib import Path
import json
import os
import sqlite3
import time
import uuid
from .config import atomic_json, canonical, digest

TERMINAL = {"passed", "failed", "invalid", "skipped"}


class State:
    def __init__(self, config, clock=time.time):
        self.config, self.clock = config, clock
        self.root = Path(config["paths"]["state"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "benchmark.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, spec_hash TEXT NOT NULL, created REAL NOT NULL, started REAL, deadline REAL, stop_requested INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS entities(kind TEXT, id TEXT, data TEXT NOT NULL, PRIMARY KEY(kind,id));
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, key_json TEXT NOT NULL, status TEXT NOT NULL, reason TEXT, result TEXT);
        CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, job_id TEXT REFERENCES jobs(id), number INTEGER NOT NULL, status TEXT NOT NULL, started REAL NOT NULL, finished REAL, reason TEXT, result TEXT);
        CREATE TABLE IF NOT EXISTS weights(id TEXT PRIMARY KEY, bytes INTEGER NOT NULL CHECK(bytes>=0), purpose TEXT NOT NULL, path TEXT NOT NULL, acquired INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS controls(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        row = self.db.execute("SELECT * FROM runs ORDER BY created DESC LIMIT 1").fetchone()
        if row is None:
            self.db.execute("INSERT INTO runs(id,spec_hash,created) VALUES(?,?,?)", ("localbench-" + uuid.uuid4().hex[:12], config["_spec_hash"], clock()))
            self.db.commit()
        elif row["spec_hash"] != config["_spec_hash"]:
            raise ValueError("spec changed: reconcile existing run explicitly; deadline may not reset")

    @property
    def run(self):
        return dict(self.db.execute("SELECT * FROM runs ORDER BY created DESC LIMIT 1").fetchone())

    def entity(self, kind, id, data):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO entities VALUES(?,?,?)", (kind, id, canonical(data)))

    def get_control(self, key):
        row = self.db.execute("SELECT value FROM controls WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def control(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO controls VALUES(?,?)", (key, canonical(value)))

    def start_execution(self, reason):
        if not self.get_control("harness_verified"):
            raise RuntimeError("real probes/downloads require verified harness")
        with self.db:
            run = self.run
            if run["started"] is None:
                now = self.clock()
                self.db.execute("UPDATE runs SET started=?, deadline=? WHERE id=?", (now, now + self.config["limits"]["benchmark_elapsed_seconds"], run["id"]))
                self.control("clock_start_reason", reason)
        self.snapshot()

    def check_budget(self, estimated_seconds=0, reserve_report=True):
        run = self.run
        if run["stop_requested"]:
            raise RuntimeError("stop_after_current")
        reserve = 7200 if reserve_report else 0
        if run["deadline"] is not None and self.clock() + estimated_seconds + reserve > run["deadline"]:
            raise RuntimeError("budget_exhausted")

    def reserve_weight(self, artifact_id, size, purpose, path):
        if type(size) is not int or size < 0:
            raise ValueError("invalid reservation")
        with self.db:
            row = self.db.execute("SELECT * FROM weights WHERE id=?", (artifact_id,)).fetchone()
            if row:
                if row["bytes"] != size or row["path"] != str(path) or row["purpose"] != purpose:
                    raise ValueError("reservation identity changed")
                return
            total = self.db.execute("SELECT COALESCE(SUM(bytes),0) FROM weights").fetchone()[0]
            if total + size > self.config["limits"]["new_model_weight_bytes"]:
                raise RuntimeError("weight_budget_exhausted")
            self.db.execute("INSERT INTO weights(id,bytes,purpose,path) VALUES(?,?,?,?)", (artifact_id, size, purpose, str(path)))
        self.snapshot()

    def enqueue(self, key):
        required = {"engine", "model", "effective_settings", "profile", "fixture_prompt_hash", "tokenizer", "seed", "mode", "replicate"}
        if required - key.keys():
            raise ValueError("incomplete job identity: " + str(required - key.keys()))
        job_id = digest(key)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO jobs(id,key_json,status) VALUES(?,?,'pending')", (job_id, canonical(key)))
        return job_id

    def begin(self, job_id):
        self.check_budget()
        with self.db:
            job = self.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not job or job[0] != "pending":
                raise RuntimeError("job is not pending")
            count = self.db.execute("SELECT COUNT(*) FROM attempts WHERE job_id=?", (job_id,)).fetchone()[0]
            if count >= self.config["limits"]["retry_transient_attempts"] + 1:
                raise RuntimeError("retry limit exhausted")
            cur = self.db.execute("INSERT INTO attempts(job_id,number,status,started) VALUES(?,?,'running',?)", (job_id, count + 1, self.clock()))
            self.db.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
        self.snapshot()
        return cur.lastrowid

    def finish(self, attempt_id, status, result=None, reason=None, transient=False):
        if status not in TERMINAL:
            raise ValueError("invalid final status")
        with self.db:
            row = self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if not row or row["status"] != "running":
                raise RuntimeError("attempt already finalized")
            self.db.execute("UPDATE attempts SET status=?,finished=?,reason=?,result=? WHERE id=?", (status, self.clock(), reason, canonical(result or {}), attempt_id))
            retry = transient and row["number"] <= self.config["limits"]["retry_transient_attempts"]
            self.db.execute("UPDATE jobs SET status=?,reason=?,result=? WHERE id=?", ("pending" if retry else status, reason, canonical(result or {}), row["job_id"]))
        self.snapshot()

    def recover(self):
        with self.db:
            for row in self.db.execute("SELECT * FROM attempts WHERE status='running'").fetchall():
                self.db.execute("UPDATE attempts SET status='invalid',finished=?,reason='controller_interrupted' WHERE id=?", (self.clock(), row["id"]))
                retry = row["number"] <= self.config["limits"]["retry_transient_attempts"]
                self.db.execute("UPDATE jobs SET status=?,reason='controller_interrupted' WHERE id=?", ("pending" if retry else "invalid", row["job_id"]))
            self.db.execute("UPDATE runs SET stop_requested=0 WHERE id=?", (self.run["id"],))
        self.snapshot()

    def stop(self):
        with self.db:
            self.db.execute("UPDATE runs SET stop_requested=1 WHERE id=?", (self.run["id"],))
        self.snapshot()

    def snapshot(self):
        run = self.run
        result = {"run": run, "now": self.clock(), "remaining_seconds": None if run["deadline"] is None else max(0, run["deadline"] - self.clock()), "jobs": {r[0]: r[1] for r in self.db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")}, "new_weight_bytes_reserved": self.db.execute("SELECT COALESCE(SUM(bytes),0) FROM weights").fetchone()[0], "controls": {r[0]: json.loads(r[1]) for r in self.db.execute("SELECT key,value FROM controls")}}
        atomic_json(self.root / "summary.json", result)
        atomic_json(Path(self.config["paths"]["artifacts"]) / "weight-ledger.json", [dict(r) for r in self.db.execute("SELECT * FROM weights")])
        return result

    def close(self):
        self.db.close()


@contextmanager
def gpu_lock(config):
    path = Path(config["paths"]["state"]) / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+b")
    if path.stat().st_size == 0:
        f.write(b"0")
        f.flush()
    f.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError("another controller owns GPU lock")
    try:
        yield
    finally:
        f.seek(0)
        if os.name == "nt":
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
