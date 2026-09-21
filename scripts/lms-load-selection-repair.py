"""One-time guarded repair of the observed pre-load CLI picker failure.

Reads the explicitly authorized key only into this process environment.
No credentials, API bodies, or arbitrary exception strings are logged.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from localbench.config import load, atomic_json, digest
from localbench.state import State, gpu_lock
from localbench.adapters.lms import _logged_command, _without_credentials
from localbench.adapters.base import LocalHTTP
from localbench.recovery import _lm_instances, recover_processes

RUN = 'localbench-90f326f86086'
IDENTIFIER = 'localbench-72754dfb82e748969fca9125ce797812'
VARIANT = 'google/gemma-4-12b-qat@q4_0'
BLOCK = 'owned_managed_load_unresolved:'


def main():
    config = load()
    state = State(config)
    previous_token = os.environ.get('LM_BENCH_TOKEN')
    try:
        with gpu_lock(config):
            before = state.run
            if before['id'] != RUN or before['stop_requested'] != 1 or before['started'] != 1789753599.011325 or before['deadline'] != 1790358399.011325:
                raise RuntimeError('original_maintenance_stop_and_clock_required')
            if state.get_control('active_job') or state.db.execute("SELECT COUNT(*) FROM attempts WHERE status='running'").fetchone()[0]:
                raise RuntimeError('running_job_requires_attention')
            for gate in ('doctor_verified', 'harness_verified', 'fixtures_verified'):
                if state.get_control(gate) is not True:
                    raise RuntimeError('verification_gate_required')
            controller = json.loads(state.db.execute("SELECT data FROM entities WHERE kind='controller' AND id=?", (RUN,)).fetchone()[0])
            # A live controller, or an outstanding client using this identifier,
            # forbids repair. Emit counts only, never unrelated process argv.
            inspect = "$p=Get-CimInstance Win32_Process;@($p|Where-Object { $_.ProcessId -eq " + str(controller['pid']) + " -or ($_.Name -eq 'lms.exe' -and $_.CommandLine -like '*" + IDENTIFIER + "*') }).Count"
            count = subprocess.run(['pwsh.exe', '-NoProfile', '-NonInteractive', '-Command', inspect],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            if count.returncode or count.stdout.strip() != b'0':
                raise RuntimeError('worker_or_load_client_still_live')
            handle = Path(config['paths']['state']) / 'owned-lm-studio-instance.json'
            old = json.loads(handle.read_text(encoding='utf8'))
            if old.get('owner') != 'localbench' or old.get('run_id') != RUN or old.get('instance_id') != IDENTIFIER or old.get('phase') != 'pending_instance_config' or old.get('selected_model_variant') != VARIANT or old.get('load_completed') is not False:
                raise RuntimeError('observed_pending_handle_required')
            argv = old.get('argv', [])
            if argv[:3] != [config['installed_tools']['lms'], 'load', VARIANT] or '--yes' in argv or argv[argv.index('--identifier') + 1] != IDENTIFIER:
                raise RuntimeError('observed_interactive_cli_identity_required')
            weights_before = [dict(r) for r in state.db.execute('SELECT * FROM weights ORDER BY id')]
            stamp = str(time.time_ns())
            receipt_path = Path(config['paths']['artifacts']) / ('lms-load-selection-repair-' + stamp + '.json')
            # Replay only model selection/estimation, never a model load.
            replay = _logged_command(config, 'lms-no-match-estimate-proof',
                [config['installed_tools']['lms'], 'load', VARIANT, '--estimate-only', '--yes'], timeout=15)
            if replay['exit_code'] != 1 or 'Model not found' not in replay['output'] or 'No model found that matches model key' not in replay['output'] or VARIANT not in replay['output']:
                raise RuntimeError('definitive_cli_no_match_proof_required')
            try:
                subprocess.run([config['installed_tools']['lms'], 'load', VARIANT, '--estimate-only'],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=_without_credentials(), timeout=4, creationflags=subprocess.CREATE_NO_WINDOW)
                raise RuntimeError('interactive_picker_replay_required')
            except subprocess.TimeoutExpired as timed:
                message = (timed.stdout or b'').decode('utf8', errors='replace')
                if ('Cannot find a model matching the provided model key (' + VARIANT + ')') not in message or 'Select a model to estimate' not in message:
                    raise RuntimeError('pre_load_picker_proof_required')
            os.environ['LM_BENCH_TOKEN'] = Path(r'C:\Users\barrett\lmstudio-local.txt').read_text(encoding='utf8').strip()
            if not os.environ['LM_BENCH_TOKEN']:
                raise RuntimeError('authorized_credential_unavailable')
            client = LocalHTTP('http://127.0.0.1:1234', 15, credential_env='LM_BENCH_TOKEN')
            inventories = [_lm_instances(client.json('/api/v1/models')) for _ in range(2)]
            if any(inventories):
                raise RuntimeError('loaded_instance_requires_separate_review')
            stages = []
            for row in state.db.execute("SELECT key,value FROM controls WHERE key LIKE 'planned_screen:%'"):
                value = json.loads(row['value'])
                if str(value.get('reason', '')).startswith((BLOCK, 'lm_studio_pending_load_completion_unknown')):
                    stages.append(dict(row))
            proposals = []
            for row in state.db.execute("SELECT id,data FROM entities WHERE kind='tuning_proposal'"):
                value = json.loads(row['data'])
                if str(value.get('reason', '')).startswith(BLOCK):
                    proposals.append(dict(row))
            atomic_json(receipt_path, {'run_id': RUN, 'status': 'repair_prepared', 'timestamp': time.time(),
                'old_owned_handle': old, 'blocked_stages': stages, 'blocked_tuning_proposals': proposals,
                'selection_replay_log': replay['log'], 'proof': 'CLI key rejected before load; interactive estimate-only picker reproduced; two authenticated inventories empty; original worker and client absent',
                'original_run': before, 'weight_ledger_sha256': digest(weights_before)})
            atomic_json(handle, {**old, 'unloaded': True, 'recovered': True, 'load_not_started': True,
                'recovered_at': time.time(), 'repair_receipt': str(receipt_path),
                'unload_proof': 'reviewed pre-load interactive selection failure; repeated authenticated absence'})
            recovery = recover_processes(config, state)
            if any(r.get('status') in ('refused', 'pending_load', 'pending_auth', 'pending_descendants') for r in recovery):
                raise RuntimeError('other_owned_recovery_requires_attention')
            with state.db:
                for row in stages:
                    state.db.execute('DELETE FROM controls WHERE key=? AND value=?', (row['key'], row['value']))
                for row in proposals:
                    value = json.loads(row['data'])
                    value.update(status='pending_measurements', repair_receipt=str(receipt_path))
                    state.db.execute("UPDATE entities SET data=? WHERE kind='tuning_proposal' AND id=? AND data=?",
                        (json.dumps(value), row['id'], row['data']))
                state.db.execute('UPDATE runs SET stop_requested=0 WHERE id=?', (RUN,))
                state.check_budget(1800)
                if any(state.run[k] != before[k] for k in ('id', 'spec_hash', 'created', 'started', 'deadline')) or [dict(r) for r in state.db.execute('SELECT * FROM weights ORDER BY id')] != weights_before:
                    raise RuntimeError('clock_or_cumulative_ledger_changed')
            state.control('ownership_recovery', recovery)
            state.control('managed_load_preflight', {'status': 'no_owned_orphan', 'repair_receipt': str(receipt_path)})
            state.control('lms_selection_repair', {'receipt': str(receipt_path), 'stages_retried': len(stages), 'proposals_retried': len(proposals)})
            state.snapshot()
            print(json.dumps({'status': 'repaired', 'receipt': str(receipt_path), 'blocked_stages_retried': len(stages),
                'blocked_proposals_retried': len(proposals), 'original_deadline': before['deadline'],
                'new_weight_bytes_reserved': sum(r['bytes'] for r in weights_before)}))
    finally:
        if previous_token is None:
            os.environ.pop('LM_BENCH_TOKEN', None)
        else:
            os.environ['LM_BENCH_TOKEN'] = previous_token
        state.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Guarded LM Studio repair stopped; review required.', file=sys.stderr)
        raise SystemExit(2)
