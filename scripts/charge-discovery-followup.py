"""Charge one completed research receipt without changing execution or weight clocks."""
from datetime import datetime
from pathlib import Path
import json
import math
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localbench.config import atomic_json, canonical, file_hash, load
from localbench.state import State


def main():
    config = load()
    artifacts = Path(config['paths']['artifacts'])
    receipt_path = artifacts / 'discovery-followup-20260918T213727Z-receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt_sha = file_hash(receipt_path)
    findings = artifacts / 'discovery-followup-20260918T213727Z.json'
    if receipt['artifact_sha256'] != 'ec915ca541f1011f9eb734a55e86b1675d69a1496d97ef10de8ee7a6e040cb11' \
            or file_hash(findings) != receipt['artifact_sha256']:
        raise RuntimeError('exact_reviewed_findings_required')
    duration = (datetime.fromisoformat(receipt['ended_utc']) -
                datetime.fromisoformat(receipt['started_utc'])).total_seconds()
    if receipt.get('schema') != 'bounded_discovery_elapsed_receipt_v1' \
            or receipt['checkpoint_seconds_used'] != 2135 \
            or type(receipt['elapsed_seconds']) is not int \
            or not math.ceil(duration) <= receipt['elapsed_seconds'] <= 1200 \
            or receipt['discovery_cap_seconds'] != config['limits']['discovery_seconds']:
        raise RuntimeError('bounded_completed_research_receipt_required')
    state = State(config)
    try:
        run = state.run
        if run['id'] != 'localbench-90f326f86086' or run['started'] != 1789753599.011325 \
                or run['deadline'] != 1790358399.011325:
            raise RuntimeError('original_execution_clock_required')
        key = 'discovery_receipt:' + receipt_sha
        previous = state.get_control(key)
        if previous:
            print(json.dumps({**previous, 'reused': True}, indent=2))
            return
        state.db.execute('BEGIN IMMEDIATE')
        with state.db:
            used = state.get_control('discovery_seconds_used')
            if type(used) is not int or used != receipt['checkpoint_seconds_used']:
                raise RuntimeError('exact_discovery_checkpoint_required')
            after = used + receipt['elapsed_seconds']
            if after > config['limits']['discovery_seconds']:
                raise RuntimeError('discovery_budget_exhausted')
            charge = {'run_id': run['id'], 'recorded_at': time.time(), 'reused': False,
                'receipt': str(receipt_path), 'receipt_sha256': receipt_sha,
                'findings_sha256': receipt['artifact_sha256'], 'seconds_before': used,
                'seconds_charged': receipt['elapsed_seconds'], 'seconds_after': after,
                'cap_seconds': config['limits']['discovery_seconds'],
                'original_started': run['started'], 'original_deadline': run['deadline'],
                'candidate_ids': receipt['candidate_ids'], 'all_local_gates_pending': True}
            state.db.execute('INSERT OR REPLACE INTO controls VALUES(?,?)',
                             ('discovery_seconds_used', canonical(after)))
            state.db.execute('INSERT OR REPLACE INTO controls VALUES(?,?)', (key, canonical(charge)))
            current = state.run
            if any(current[field] != run[field] for field in ('id', 'spec_hash', 'created', 'started', 'deadline')):
                raise RuntimeError('execution_clock_must_remain_unchanged')
        atomic_json(artifacts / f'discovery-followup-charge-{time.time_ns()}.json', charge)
        state.snapshot()
        print(json.dumps(charge, indent=2))
    finally:
        state.close()


if __name__ == '__main__':
    main()
