"""Conservatively charge this completed research, without changing runtime state."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from localbench.config import load, atomic_json

cfg = load()
report = root / 'artifacts/reddit-speed-investigation-20260920.md'
sha = hashlib.sha256(report.read_bytes()).hexdigest()
key = 'discovery_receipt:reddit-investigation-20260920:' + sha
db = sqlite3.connect(Path(cfg['paths']['state']) / 'benchmark.sqlite3', timeout=30)
try:
    db.execute('BEGIN IMMEDIATE')
    previous = db.execute('SELECT value FROM controls WHERE key=?', (key,)).fetchone()
    if previous:
        print(previous[0])
        db.rollback()
    else:
        run = db.execute('SELECT id,started,deadline FROM runs ORDER BY created DESC LIMIT 1').fetchone()
        if run != ('localbench-90f326f86086', 1789753599.011325, 1790358399.011325):
            raise RuntimeError('original_run_required')
        used = json.loads(db.execute('SELECT value FROM controls WHERE key=?', ('discovery_seconds_used',)).fetchone()[0])
        if used != 2738:
            raise RuntimeError('expected_discovery_checkpoint_required')
        end = datetime.now(timezone.utc)
        # Exact turn-start clock was not retained across compaction. Charge the
        # entire interval since the preceding ranking snapshot as an upper bound.
        start = datetime.fromisoformat('2026-09-20T05:26:57.1510034+00:00')
        seconds = math.ceil((end - start).total_seconds())
        if seconds < 0 or used + seconds > cfg['limits']['discovery_seconds']:
            raise RuntimeError('discovery_budget_exhausted')
        receipt = dict(schema='conservative_discovery_elapsed_receipt_v1',
            started_utc=start.isoformat(), ended_utc=end.isoformat(),
            elapsed_seconds_charged=seconds, accounting='upper_bound_includes_pre_research_time',
            artifact=str(report), artifact_sha256=sha, seconds_before=used,
            seconds_after=used+seconds, cap_seconds=cfg['limits']['discovery_seconds'],
            new_weights_bytes=0, runtime_probes=0, source_changes=0,
            run_id=run[0], original_started=run[1], original_deadline=run[2])
        db.execute('UPDATE controls SET value=? WHERE key=?', (json.dumps(used+seconds), 'discovery_seconds_used'))
        db.execute('INSERT INTO controls VALUES(?,?)', (key, json.dumps(receipt)))
        db.commit()
        atomic_json(root / 'artifacts/reddit-speed-investigation-20260920-receipt.json', receipt)
        print(json.dumps(receipt))
finally:
    db.close()
