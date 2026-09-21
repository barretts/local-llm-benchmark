"""Read-only guard for reverifying a deployed setup after storage repair."""
from pathlib import Path
import json
import sqlite3

from scripts.speculation_setup_contract import checkpoint_ready

def clean_installed_stop(cfg):
    db=sqlite3.connect((Path(cfg['paths']['state'])/'benchmark.sqlite3').as_uri()+'?mode=ro',uri=True)
    try:
        run=db.execute('SELECT id,started,deadline,stop_requested FROM runs ORDER BY created DESC LIMIT 1').fetchone()
        row=db.execute("SELECT value FROM controls WHERE key='speculation_checkpoint_pause'").fetchone()
        active=db.execute("SELECT COUNT(*) FROM attempts WHERE status='running'").fetchone()[0]
        if not checkpoint_ready(run,json.loads(row[0]) if row else None,active):
            raise RuntimeError('clean_authorized_installed_stop_required')
    finally:db.close()
