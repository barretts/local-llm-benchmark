"""One-shot pause after both current 27B quality checkpoints; never kill work."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localbench.config import atomic_json, digest, load

IDS = ('05c146133e492ac2e0df5f3637ecee19c2b9d797d579431a2321d54c5f91e46f',
       '79de48fe4c1c209671f80683dd4458e2325547181c883d43b4f099dd423c6267')
RUN = ('localbench-90f326f86086', 1789753599.011325, 1790358399.011325)

def checkpoints(db, cfg):
    row = db.execute("SELECT data FROM entities WHERE kind='controller' AND id=?", (RUN[0],)).fetchone()
    protocol = json.loads(row[0])['protocol_source_hashes']
    expected = {g: {(f,s) for f in cfg['grading'][g+'_fixture_ids'] for s in cfg['grading']['seeds']}
                for g in ('regular','long')}
    out = {}
    for identifier in IDS:
        groups = {g: {} for g in expected}
        for row in db.execute("SELECT key_json,json_extract(result,'$.valid'),json_extract(result,'$.passed'),"
                "json_extract(result,'$.group'),json_extract(result,'$.fixture'),json_extract(result,'$.seed') "
                "FROM jobs WHERE json_extract(key_json,'$.configuration_id')=? "
                "AND json_extract(key_json,'$.kind')='quality' AND status IN ('passed','failed')", (identifier,)):
            key = json.loads(row[0])
            current = digest({'protocol':{p:protocol[p] for p in ('quality.py','tools.py','schema.py')},
                'run':RUN[0], 'fixture_hash':key.get('fixture_prompt_hash'), 'kind':'quality',
                'seed':key.get('seed'), 'mode':key.get('mode'), 'replicate':key.get('replicate'),
                'target':key.get('target_tokens')})
            group, pair = row[3], (row[4],row[5])
            if row[1] == 1 and group in expected and pair in expected[group] \
                    and key.get('replicate') == 0 and key.get('mode') == 'fresh' \
                    and key.get('logical_plan_hash') == current:
                groups[group][pair] = row[2] == 1
        stats = {g:dict(total=len(v),passed=sum(v.values()),failed=len(v)-sum(v.values())) for g,v in groups.items()}
        stats['complete'] = stats['regular']['failed'] >= cfg['grading']['early_elimination_regular_failures'] \
            or (stats['regular']['total'] == len(expected['regular']) and
                (stats['long']['total'] == len(expected['long']) or
                 stats['long']['failed'] >= cfg['grading']['early_elimination_long_failures']))
        out[identifier] = stats
    return out

def main():
    cfg = load()
    if cfg['_execution_hash'] != '187076fef263fb85bf1972b7b06617aed11b04e75f3b9f5cf50a5e75ff3dc180':
        raise RuntimeError('original_execution_spec_required')
    started = time.time(); deadline = min(started+7200,RUN[2]-7200)
    status_path = ROOT/'artifacts/speculation-checkpoint-pause-status.json'
    db = sqlite3.connect(Path(cfg['paths']['state'])/'benchmark.sqlite3',timeout=30)
    try:
        while time.time() < deadline:
            run = db.execute('SELECT id,started,deadline,stop_requested FROM runs ORDER BY created DESC LIMIT 1').fetchone()
            if run[:3] != RUN: raise RuntimeError('original_execution_clock_required')
            if run[3]:
                atomic_json(status_path,dict(status='existing_stop_observed',run_id=RUN[0],timestamp_utc=datetime.now(timezone.utc).isoformat()))
                return
            progress = checkpoints(db,cfg)
            if all(x['complete'] for x in progress.values()):
                db.execute('BEGIN IMMEDIATE')
                with db:
                    current = db.execute('SELECT id,started,deadline,stop_requested FROM runs ORDER BY created DESC LIMIT 1').fetchone()
                    if current[:3] != RUN: raise RuntimeError('run_changed_before_checkpoint_pause')
                    proof = dict(status='pause_requested',reason='user_authorized_speculation_integration',
                        run_id=RUN[0],timestamp_utc=datetime.now(timezone.utc).isoformat(),checkpoints=progress,
                        original_started=RUN[1],original_deadline=RUN[2])
                    db.execute('UPDATE runs SET stop_requested=1 WHERE id=?',(RUN[0],))
                    db.execute('INSERT OR REPLACE INTO controls VALUES(?,?)',('speculation_checkpoint_pause',json.dumps(proof)))
                atomic_json(status_path,proof)
                print(json.dumps(proof)); return
            atomic_json(status_path,dict(status='waiting_for_quality_checkpoint',run_id=RUN[0],
                timestamp_utc=datetime.now(timezone.utc).isoformat(),checkpoints=progress,deadline=deadline))
            time.sleep(5)
        atomic_json(status_path,dict(status='attention_required',reason='checkpoint_wait_limit_reached',run_id=RUN[0]))
        raise RuntimeError('checkpoint_wait_limit_reached')
    finally: db.close()

if __name__ == '__main__': main()
