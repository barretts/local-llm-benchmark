"""Resume root-requested harness maintenance with the original run ledger intact."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localbench.config import atomic_json, file_hash, load
from localbench.state import State, gpu_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reason', choices=('regular-budget', 'gemma-control', 'ollama-storage', 'source-directory', 'extra-candidates', 'reasoning-intake', 'probe-timeout'), default='regular-budget')
    args = parser.parse_args()
    config = load()
    state = State(config)
    try:
        with gpu_lock(config):
            before = state.run
            if before['id'] != 'localbench-90f326f86086' or before['stop_requested'] != 1:
                raise RuntimeError('original_run_maintenance_stop_required')
            if state.get_control('active_job') or state.db.execute(
                    "SELECT COUNT(*) FROM attempts WHERE status='running'").fetchone()[0]:
                raise RuntimeError('active_job_must_finish_before_maintenance_resume')
            for gate in ('doctor_verified', 'fixtures_verified', 'harness_verified'):
                if state.get_control(gate) is not True:
                    raise RuntimeError('verification_gate_required:' + gate)
            version = state.get_control('fixture_version')
            if version != '01e78658e121c4053eb6564d26a213bfc815906fca66e9d01cb8e2e178a64982':
                raise RuntimeError('frozen_fixture_version_required')
            weights = state.db.execute('SELECT COALESCE(SUM(bytes),0) FROM weights').fetchone()[0]
            with state.db:
                state.db.execute('UPDATE runs SET stop_requested=0 WHERE id=?', (before['id'],))
                state.check_budget(1800)
                if any(state.run[key] != before[key] for key in
                       ('id', 'spec_hash', 'created', 'started', 'deadline')):
                    raise RuntimeError('original_run_identity_and_clock_must_remain')
            stamp = time.time_ns()
            receipt = {'resumed_at': stamp/1e9, 'run_id': before['id'],
                'reason': {'regular-budget': 'root-requested maintenance: regular token budget and total inference deadline corrected',
                           'gemma-control': 'root-requested maintenance: add missing diagnostic Gemma 64K control and bounded reduced reasoning followup',
                           'ollama-storage': 'root-requested maintenance: place private Ollama fallback weight copies and staging on the user-authorized E download volume',
                           'source-directory': 'root-requested maintenance: normalize an allowed source directory trailing separator and invalidate affected quality jobs while retaining all raw attempts',
                           'extra-candidates': 'root-requested maintenance: integrate two pinned discovery GGUF leads after installed-model baselines using unchanged grading and measurement gates',
                           'reasoning-intake': 'root-requested maintenance: allow context-valid native marker-only defaults into the bounded three-family reasoning-profile exploration using unchanged grading and measurement gates',
                           'probe-timeout': 'root-requested maintenance: enforce the declared 300-second small-probe HTTP deadline while preserving full-context and coding-task budgets and under-limit completed measurements'}[args.reason],
                'base_spec_hash': config['_spec_hash'], 'execution_hash': config['_execution_hash'],
                'fixture_version': version, 'original_started': before['started'],
                'original_deadline': before['deadline'], 'new_weight_bytes_reserved': weights,
                'quality_source_sha256': file_hash(Path(config['paths']['project'])/'localbench/quality.py'),
                'search_source_sha256': file_hash(Path(config['paths']['project'])/'localbench/search.py'),
                'screen_source_sha256': file_hash(Path(config['paths']['project'])/'localbench/search_core.py'),
                'http_source_sha256': file_hash(Path(config['paths']['project'])/'localbench/adapters/base.py')}
            state.control('harness_maintenance_resume', receipt)
            atomic_json(Path(config['paths']['artifacts'])/f'harness-maintenance-resume-{stamp}.json', receipt)
            state.snapshot()
            print(json.dumps(receipt, indent=2))
    finally:
        state.close()


if __name__ == '__main__':
    main()
