"""One-time reviewed maintenance handoff; no paid API, OS task or global edits."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from localbench.config import atomic_json,digest,file_hash,load
from localbench.state import State,gpu_lock
from scripts.speculation_setup_contract import RUN,checkpoint_ready,original_handles,process_absent,setup_files,test_files

def sources():return {p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted((ROOT/'localbench').rglob('*.py'))}

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value

def read_run(cfg):
    db=sqlite3.connect((Path(cfg['paths']['state'])/'benchmark.sqlite3').as_uri()+'?mode=ro',uri=True)
    try:
        run=db.execute('SELECT id,started,deadline,stop_requested FROM runs ORDER BY created DESC LIMIT 1').fetchone()
        row=db.execute("SELECT value FROM controls WHERE key='speculation_checkpoint_pause'").fetchone()
        active=db.execute("SELECT COUNT(*) FROM attempts WHERE status='running'").fetchone()[0]
        return run,json.loads(row[0]) if row else None,active
    finally:db.close()

def main():
    cfg=load();bundle_path=ROOT/'artifacts/speculation-preparation-deployment.json'
    bundle=json.loads(bundle_path.read_text(encoding='utf-8'));stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    status_path=ROOT/'artifacts/speculation-setup-status.json';watches=[];changed=False
    expected=RUN
    env=os.environ.copy();env.pop(cfg['safety']['credential_env'],None)
    def record(stage,**extra):
        item=dict(status=stage,run_id=expected[0],timestamp_utc=datetime.now(timezone.utc).isoformat(),
            setup_pid=os.getpid(),bundle_sha256=file_hash(bundle_path),**extra)
        atomic_json(status_path,item);return item
    def run_script(name,timeout=900):
        path=ROOT/'scripts'/name
        if file_hash(path)!=bundle['setup_scripts'].get(name):raise RuntimeError('reviewed_setup_script_changed:'+name)
        log=ROOT/'.logs'/('speculation-setup-'+stamp+'-'+name+'.log')
        with log.open('wb') as f:
            done=subprocess.run([sys.executable,'-X','utf8',str(path)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if done.returncode:raise RuntimeError('setup_step_failed:'+name+':'+str(log))
        return log
    try:
        if bundle.get('verified') is not True or bundle.get('execution_hash')!=cfg['_execution_hash'] \
                or sources()!=bundle['before_sources'] or bundle.get('run_id')!=RUN[0] \
                or bundle.get('original_started')!=RUN[1] or bundle.get('original_deadline')!=RUN[2] \
                or setup_files(ROOT)!=bundle.get('setup_scripts') or test_files(ROOT)!=bundle.get('test_sources'):
            raise RuntimeError('exact_shadow_verified_deployment_required')
        if file_hash(Path(__file__))!=bundle['setup_scripts'].get(Path(__file__).name):
            raise RuntimeError('reviewed_setup_runner_changed')
        supervisor=module('speculation_supervisor',ROOT/'scripts/unattended-supervisor.py')
        backend=supervisor.WindowsBackend(cfg)
        watches=original_handles(supervisor.QueryHandle,bundle.get('original_processes'),
            lambda:checkpoint_ready(*read_run(cfg)),lambda pid:process_absent(backend.pwsh,pid))
        deadline=min(time.time()+7200,expected[2]-7200)
        while time.time()<deadline:
            run,pause,active=read_run(cfg)
            if run[:3]!=expected:raise RuntimeError('original_execution_clock_changed')
            if checkpoint_ready(run,pause,active) and all(w.exit_code() is not None for w in watches):break
            if run[3] and not pause:raise RuntimeError('other_stop_request_requires_attention')
            record('waiting_for_quality_checkpoint',active_attempts=active,original_deadline=run[2])
            time.sleep(5)
        else:raise RuntimeError('checkpoint_wait_limit_reached')
        record('applying_verified_patches')
        with gpu_lock(cfg):
            if not checkpoint_ready(*read_run(cfg)):raise RuntimeError('authorized_checkpoint_stop_changed')
            if sources()!=bundle['before_sources']:raise RuntimeError('live_source_changed_before_install')
            if setup_files(ROOT)!=bundle['setup_scripts'] or test_files(ROOT)!=bundle['test_sources']:
                raise RuntimeError('reviewed_setup_or_tests_changed_before_install')
            exe=Path(bundle['patch_executable'])
            if file_hash(exe)!=bundle['patch_executable_sha256']:raise RuntimeError('patch_executable_changed')
            changed=True
            for pin in bundle['patches']:
                path=Path(pin['path'])
                if not path.resolve().is_relative_to((ROOT/'.patches').resolve()) or file_hash(path)!=pin['sha256']:
                    raise RuntimeError('reviewed_patch_changed')
                raw=path.read_text(encoding='utf-8')
                log=ROOT/'.logs'/('speculation-setup-'+stamp+'-'+path.name+'.log')
                with log.open('wb') as f:
                    done=subprocess.run([str(exe),'--codex-run-as-apply-patch',raw],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                        stdout=f,stderr=subprocess.STDOUT,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if done.returncode:raise RuntimeError('reviewed_patch_failed:'+str(log))
            if sources()!=bundle['after_sources']:raise RuntimeError('installed_sources_differ_from_verified_shadow')
        record('verifying_installed_harness')
        run_script('verify-speculation-harness.py',600)
        record('enabling_comparison')
        run_script('enable-speculation-comparison.py',120)
        record('checking_local_activation')
        run_script('speculation-activation-probe.py',3600)
        record('releasing_unattended_worker')
        # This established helper handles no key. The credential controller
        # privately reads only the file explicitly authorized by the user.
        with (ROOT/'.logs'/('speculation-setup-'+stamp+'-release.log')).open('wb') as f:
            done=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/release-verified-controller.py')],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=120,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if done.returncode:raise RuntimeError('verified_controller_release_failed')
        helper=backend.launch_helper(expected[0]);worker=None
        for _ in range(90):
            snapshot=supervisor.read_snapshot(cfg)
            controller=snapshot.get('controller') or {}
            if controller.get('pid') not in (None,26500) and controller.get('command')=='sweep':
                worker=backend.inspect(controller['pid']);break
            if helper.poll() is not None and helper.returncode not in (None,0):
                raise RuntimeError('private_credential_helper_failed')
            time.sleep(1)
        if worker is None:raise RuntimeError('new_worker_readiness_timeout')
        command=[sys.executable,'-X','utf8',str(ROOT/'scripts/unattended-supervisor.py'),'--adopt-pid',str(worker.pid)]
        with (ROOT/'.logs'/('speculation-supervisor-'+stamp+'-stdout.log')).open('wb') as out,\
                (ROOT/'.logs'/('speculation-supervisor-'+stamp+'-stderr.log')).open('wb') as err:
            observer=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        time.sleep(2)
        if observer.poll() is not None:raise RuntimeError('new_supervisor_start_failed')
        record('resumed_with_speculation_comparison',worker_pid=worker.pid,worker_created_filetime=worker.created,
            supervisor_launcher_pid=observer.pid,activation_status=str(ROOT/'artifacts/speculation-activation-status.json'),
            original_started=expected[1],original_deadline=expected[2],source_manifest_sha256=digest(sources()))
        worker.close();return 0
    except (RuntimeError,OSError,ValueError,subprocess.TimeoutExpired) as error:
        if changed:
            state=State(cfg)
            try:state.stop()
            finally:state.close()
        record('attention_required',reason=str(error),installed_changes=changed)
        print(json.dumps({'status':'attention_required','reason':str(error)}));return 2
    finally:
        for watch in watches:watch.close()

if __name__=='__main__':raise SystemExit(main())
