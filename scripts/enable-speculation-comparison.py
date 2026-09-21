"""Commit a tested comparison after the original worker's checkpoint pause."""
from pathlib import Path
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from localbench.config import atomic_json,digest,file_hash,load
from localbench.state import State,gpu_lock
from localbench.speculation_search import make_plan,PLAN_NAME
from scripts.speculation_setup_contract import setup_files,test_files

def main():
    cfg=load();state=State(cfg)
    try:
        with gpu_lock(cfg):
            before=state.run
            if (before['id'],before['started'],before['deadline'],before['stop_requested']) != \
                    ('localbench-90f326f86086',1789753599.011325,1790358399.011325,1):
                raise RuntimeError('original_checkpoint_stop_required')
            pause=state.get_control('speculation_checkpoint_pause')
            if not pause or pause.get('reason')!='user_authorized_speculation_integration':
                raise RuntimeError('authorized_checkpoint_pause_proof_required')
            if state.get_control('active_job') or state.db.execute("SELECT COUNT(*) FROM attempts WHERE status='running'").fetchone()[0]:
                raise RuntimeError('active_work_must_finish_before_speculation_enable')
            spec=importlib.util.spec_from_file_location('checkpoint_pause',ROOT/'scripts/speculation-checkpoint-pause.py')
            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
            if not all(x['complete'] for x in helper.checkpoints(state.db,cfg).values()):
                raise RuntimeError('current_27b_checkpoint_incomplete')
            for gate in ('doctor_verified','fixtures_verified','harness_verified'):
                if state.get_control(gate) is not True:raise RuntimeError('verification_gate_required:'+gate)
            controller=json.loads(state.db.execute("SELECT data FROM entities WHERE kind='controller' AND id=?",(before['id'],)).fetchone()[0])
            for name,sha in controller['protocol_source_hashes'].items():
                if file_hash(ROOT/'localbench'/name)!=sha:raise RuntimeError('frozen_measurement_or_grading_source_changed')
            if state.get_control('fixture_version')!='01e78658e121c4053eb6564d26a213bfc815906fca66e9d01cb8e2e178a64982':
                raise RuntimeError('frozen_fixtures_required')
            verified=json.loads((ROOT/'artifacts/speculation-harness-verification.json').read_text())
            sources={p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted((ROOT/'localbench').rglob('*.py'))}
            if verified.get('passed') is not True or verified.get('sources')!=sources \
                    or file_hash(verified['log'])!=verified['log_sha256'] or verified['execution_hash']!=cfg['_execution_hash'] \
                    or verified.get('setup_scripts')!=setup_files(ROOT) or verified.get('test_sources')!=test_files(ROOT):
                raise RuntimeError('exact_tested_speculation_sources_required')
            plan_path=ROOT/'artifacts'/PLAN_NAME;atomic_json(plan_path,make_plan(cfg))
            weights=state.db.execute('SELECT COALESCE(SUM(bytes),0) FROM weights').fetchone()[0]
            phase=state.get_control('tuning_phase')
            receipt=dict(enabled=True,run_id=before['id'],reason='user_authorized_mtp_dflash_after_quality_checkpoint',
                plan_sha256=file_hash(plan_path),sources_sha256=digest(sources),harness_test_count=verified['test_count'],
                original_started=before['started'],original_deadline=before['deadline'],
                original_tuning_deadline=phase.get('deadline') if phase else None,new_weight_bytes_before=weights,
                fixture_version=state.get_control('fixture_version'),recorded_at=time.time())
            with state.db:
                state.db.execute('UPDATE runs SET stop_requested=0 WHERE id=?',(before['id'],))
                state.check_budget(1800)
                for field in ('id','created','spec_hash','started','deadline'):
                    if state.run[field]!=before[field]:raise RuntimeError('original_run_and_clock_must_remain')
                state.db.execute('INSERT OR REPLACE INTO controls VALUES(?,?)',('speculation_comparison_enabled',json.dumps(receipt)))
            atomic_json(ROOT/'artifacts/speculation-comparison-enable-receipt.json',receipt)
            state.snapshot();print(json.dumps(receipt))
    finally:state.close()

if __name__=='__main__':main()
