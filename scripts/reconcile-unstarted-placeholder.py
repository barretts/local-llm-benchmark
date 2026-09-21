"""Retire one proven stop-time placeholder while preserving its real packed job."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localbench.config import atomic_json, canonical, digest, load
from localbench.scheduler import prior_result
from localbench.state import State, gpu_lock

ORPHAN = 'fc5c21b5fe16de0dce92366c56c103969d18c7b849a8469f4c9cdc40f64d045c'
PACKED = 'c53d1ecf609a52ee5c556f79a65fbedfd8f34f99d3d0bd65648fba2e9ffe5ae1'
EXPECTED = {
    'configuration_id': 'f9144256cb8304e98b66e7c6d0dd9c834d6ef963db9149db44fce91d30466afe',
    'logical_plan_hash': '8210db5813397fdf00127b417d6f51069b1f9ced2cd4050d4b5f9e2f53d3231e',
    'engine': '939f308c41a9f9f88a5e2a960c6540962a180a6e605794d4fa149c4bc92f16f3',
    'model': '3c5483e8749f4865f1aae2c36b796e7b8c43ec02c5a74d663a82a5fe916b2298',
    'kind': 'capacity', 'seed': 17, 'mode': 'fresh', 'replicate': 0,
    'target_tokens': 61440,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    config = load()
    state = State(config)
    try:
        with gpu_lock(config):
            run = state.run
            if run['id'] != 'localbench-90f326f86086' or run['stop_requested'] != 1:
                raise RuntimeError('original_stopped_run_required')
            if state.get_control('active_job') or state.db.execute(
                    "SELECT COUNT(*) FROM attempts WHERE status='running'").fetchone()[0]:
                raise RuntimeError('no_active_measurement_required')
            rows = []
            for job_id in (ORPHAN, PACKED):
                row = state.db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
                if row is None or row['status'] != 'pending':
                    raise RuntimeError('exact_unstarted_pending_job_required:' + job_id)
                if state.db.execute('SELECT COUNT(*) FROM attempts WHERE job_id=?',
                                    (job_id,)).fetchone()[0]:
                    raise RuntimeError('zero_attempts_required:' + job_id)
                rows.append(dict(row))
            orphan_key, packed_key = [json.loads(row['key_json']) for row in rows]
            base = {key: value for key, value in orphan_key.items() if key != 'fixture_prompt_hash'}
            packed_base = {key: value for key, value in packed_key.items() if key != 'fixture_prompt_hash'}
            if base != packed_base or any(base.get(key) != value for key, value in EXPECTED.items()):
                raise RuntimeError('exact_same_logical_measurement_identity_required')
            if digest(orphan_key) != ORPHAN or digest(packed_key) != PACKED:
                raise RuntimeError('job_identity_hash_mismatch')
            if orphan_key['fixture_prompt_hash'] != digest({'unpacked_plan': base}):
                raise RuntimeError('proven_unpacked_placeholder_required')
            if packed_key['fixture_prompt_hash'] != '7c8c401957e8c84a6cad456713303035df6de17a64e6b03955619c78c10db139':
                raise RuntimeError('exact_packed_prompt_required')
            if prior_result(state, base) is not None:
                raise RuntimeError('no_existing_measurement_result_required')
            stamp = time.time_ns()
            receipt = {
                'run_id': run['id'], 'applied': args.apply, 'recorded_at': stamp / 1e9,
                'reason': 'stop_before_execution_unpacked_placeholder', 'unstarted': True,
                'retired_job_id': ORPHAN, 'preserved_packed_job_id': PACKED,
                'job_identity': base, 'attempt_counts': {ORPHAN: 0, PACKED: 0},
                'before': rows, 'original_started': run['started'],
                'original_deadline': run['deadline'],
                'new_weight_bytes_reserved': state.db.execute(
                    'SELECT COALESCE(SUM(bytes),0) FROM weights').fetchone()[0],
            }
            if args.apply:
                with state.db:
                    changed = state.db.execute(
                        "UPDATE jobs SET status='skipped',reason=?,result=? WHERE id=? AND status='pending'",
                        (receipt['reason'], canonical({'unstarted': True,
                         'superseded_by': PACKED, 'reason': receipt['reason']}), ORPHAN)).rowcount
                    if changed != 1 or prior_result(state, base) is not None:
                        raise RuntimeError('retirement_must_not_replace_real_measurement')
                    unchanged = dict(state.db.execute('SELECT * FROM jobs WHERE id=?',
                                                       (PACKED,)).fetchone())
                    if unchanged != rows[1] or state.run != run:
                        raise RuntimeError('packed_job_and_original_run_must_remain_unchanged')
                    state.db.execute('INSERT OR REPLACE INTO controls VALUES(?,?)',
                        ('unstarted_placeholder_reconciliation', canonical(receipt)))
                atomic_json(Path(config['paths']['artifacts']) /
                            f'unstarted-placeholder-reconciliation-{stamp}.json', receipt)
                state.snapshot()
            print(json.dumps(receipt, indent=2, ensure_ascii=True))
    finally:
        state.close()


if __name__ == '__main__':
    main()
