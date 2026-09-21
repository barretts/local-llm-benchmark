"""Install only the verified comparison repair at a clean owned checkpoint."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localbench.config import atomic_json,digest,file_hash,load
from localbench.state import State,gpu_lock
from scripts.install_speculation_check import clean_installed_stop
from scripts.speculation_setup_contract import RUN,process_absent

def sources(root):
    return {p.relative_to(root).as_posix():file_hash(p) for p in sorted((root/'localbench').rglob('*.py'))}

def main():
    cfg=load();clean_installed_stop(cfg)
    spec=importlib.util.spec_from_file_location('repair_supervisor',ROOT/'scripts/unattended-supervisor.py')
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    pwsh=shutil.which('pwsh')
    if not pwsh:raise RuntimeError('installed_powershell7_required')
    # Query-only process handles. A reused PID grants no termination authority.
    for pid,created in ((22164,134343650078202930),(30948,134343650095422179)):
        try:watch=supervisor.QueryHandle(pid)
        except RuntimeError:
            if not process_absent(pwsh,pid):raise RuntimeError('original_exit_unverified') from None
            continue
        try:
            if watch.created==created and watch.alive():raise RuntimeError('original_worker_or_supervisor_still_live')
        finally:watch.close()
    latest=json.loads((ROOT/'artifacts/unattended-supervisor-status.json').read_text())
    for pid in (latest.get('worker_pid'),latest.get('supervisor_pid')):
        if not isinstance(pid,int):continue
        try:watch=supervisor.QueryHandle(pid)
        except RuntimeError:
            if not process_absent(pwsh,pid):raise RuntimeError('latest_registered_exit_unverified') from None
            continue
        try:
            if watch.alive():raise RuntimeError('latest_registered_worker_or_supervisor_still_live')
        finally:watch.close()
    receipt=json.loads((ROOT/'artifacts/speculation-repair-deployment.json').read_text())
    if receipt.get('verified') is not True or receipt.get('execution_hash')!=cfg['_execution_hash'] \
            or receipt.get('harness',{}).get('passed') is not True or sources(ROOT)!=receipt['before_sources']:
        raise RuntimeError('verified_repair_and_unchanged_sources_required')
    codex=Path(r'C:\Users\barrett\AppData\Local\nvm\v24.13.1\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log=ROOT/'.logs'/('speculation-repair-install-'+stamp+'.log')
    env=os.environ.copy();env.pop(cfg['safety']['credential_env'],None)
    with gpu_lock(cfg):
        clean_installed_stop(cfg)
        with log.open('wb') as output:
            for pin in receipt['patches']:
                path=Path(pin['path'])
                if not path.resolve().is_relative_to((ROOT/'.patches').resolve()) or file_hash(path)!=pin['sha256']:
                    raise RuntimeError('reviewed_patch_changed')
                raw=path.read_text(encoding='utf-8')
                if len(raw)>24000:raise RuntimeError('patch_too_large')
                done=subprocess.run([str(codex),'--codex-run-as-apply-patch',raw],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=60,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if done.returncode:raise RuntimeError('reviewed_patch_install_failed')
            if sources(ROOT)!=receipt['after_sources']:raise RuntimeError('installed_source_mismatch')
            done=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/verify-speculation-harness.py')],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=660,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if done.returncode:raise RuntimeError('installed_harness_verification_failed')
        verified=json.loads((ROOT/'artifacts/speculation-harness-verification.json').read_text())
        if verified.get('passed') is not True or verified.get('sources')!=sources(ROOT):
            raise RuntimeError('installed_harness_binding_failed')
        state=State(cfg)
        try:
            if (state.run['id'],state.run['started'],state.run['deadline'])!=RUN or not state.run['stop_requested']:
                raise RuntimeError('original_stopped_execution_clock_required')
            if state.db.execute("SELECT count(*) FROM attempts WHERE status='running'").fetchone()[0]:
                raise RuntimeError('clean_checkpoint_required')
            original_protocol=json.loads(state.db.execute("SELECT data FROM entities WHERE kind='controller' AND id=?",
                (RUN[0],)).fetchone()[0])['protocol_source_hashes']
            if any(file_hash(ROOT/'localbench'/p)!=sha for p,sha in original_protocol.items()):
                raise RuntimeError('frozen_grading_or_measurement_changed')
            state.control('harness_verified',True)
            enabled=state.get_control('speculation_comparison_enabled')
            state.control('speculation_comparison_enabled',{**enabled,
                'sources_sha256':verified['sources_sha256'],'harness_test_count':verified['test_count'],
                'repair_reason':'measure_experimental_arms_without_forgiving_control_coding_failures'})
            state.recover();state.snapshot()
        finally:state.close()
    with log.open('ab') as output:
        done=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/release-verified-controller.py')],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=60,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if done.returncode:raise RuntimeError('verified_controller_release_failed')
    paths=[ROOT/'.logs'/('speculation-repair-helper-'+stamp+suffix) for suffix in ('-stdout.log','-stderr.log')]
    with paths[0].open('wb') as out,paths[1].open('wb') as err:
        helper=subprocess.Popen([pwsh,'-NoProfile','-STA','-File',str(ROOT/'scripts/credential-controller.ps1'),
            '-RunId',RUN[0],'-CredentialFile',r'C:\Users\barrett\lmstudio-local.txt',
            '-WorkerCommand','sweep','-MaximumWaitSeconds','60'],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    db=sqlite3.connect((Path(cfg['paths']['state'])/'benchmark.sqlite3').as_uri()+'?mode=ro',uri=True)
    try:
        limit=time.time()+120;pid=None
        while time.time()<limit:
            row=db.execute("SELECT data FROM entities WHERE kind='controller' AND id=?",(RUN[0],)).fetchone()
            controller=json.loads(row[0]) if row else {}
            if controller.get('started',0)>=verified['finished']:
                pid=controller['pid'];break
            if helper.poll() not in (None,0):raise RuntimeError('private_credential_helper_failed')
            time.sleep(1)
    finally:db.close()
    if not pid:raise RuntimeError('new_worker_registration_missing')
    # The supervisor verifies exact command, kernel creation and helper receipt.
    paths=[ROOT/'.logs'/('speculation-repair-supervisor-'+stamp+suffix) for suffix in ('-stdout.log','-stderr.log')]
    with paths[0].open('wb') as out,paths[1].open('wb') as err:
        observer=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'scripts/unattended-supervisor.py'),
            '--adopt-pid',str(pid)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW)
    status=dict(status='resumed_with_repaired_speculation_comparison',worker_pid=pid,
        supervisor_launcher_pid=observer.pid,helper_pid=helper.pid,
        sources_sha256=verified['sources_sha256'],harness_test_count=verified['test_count'],
        original_started=RUN[1],original_deadline=RUN[2],timestamp_utc=datetime.now(timezone.utc).isoformat(),
        install_log=str(log),qualification=False)
    atomic_json(ROOT/'artifacts/speculation-repair-status.json',status)
    print(json.dumps(status))
    return 0

if __name__=='__main__':raise SystemExit(main())
