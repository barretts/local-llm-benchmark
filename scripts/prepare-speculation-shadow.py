"""Verify staged patches in an isolated copy; leave the running worker alone."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from localbench.config import atomic_json,digest,file_hash,load
from scripts.speculation_setup_contract import PROCESS_PINS,setup_files,test_files

PATCHES=('069-native-speculation-integration.patch','071-native-speculation-final-gates.patch','073-speculation-engine-match.patch')
CODEX=Path(r'C:\Users\barrett\AppData\Local\nvm\v24.13.1\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe')

def sources(root):return {p.relative_to(root).as_posix():file_hash(p) for p in sorted((root/'localbench').rglob('*.py'))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--installed',action='store_true');args=parser.parse_args()
    cfg=load();shadow=ROOT/'.staging'/('speculation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    patch_names=PATCHES
    if args.installed:
        previous=json.loads((ROOT/'artifacts/speculation-preparation-deployment.json').read_text())
        current=sources(ROOT);expected=previous['after_sources']
        if previous.get('verified') is not True or previous.get('execution_hash')!=cfg['_execution_hash'] \
                or set(current)!=set(expected) or any(current[p]!=sha for p,sha in expected.items() if p!='localbench/resources.py'):
            raise RuntimeError('previously_verified_installed_controller_required')
        from scripts.install_speculation_check import clean_installed_stop
        clean_installed_stop(cfg)
        patch_names=()
    if not shadow.resolve().is_relative_to(ROOT.resolve()) or shadow.exists():raise RuntimeError('fresh_private_shadow_required')
    shadow.mkdir(parents=True)
    before=sources(ROOT)
    scripts_before=setup_files(ROOT);tests_before=test_files(ROOT)
    for name in ('localbench','tests','scripts','fixtures','grader'):
        if (ROOT/name).is_dir():shutil.copytree(ROOT/name,shadow/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('benchmark-spec.json','execution-overrides.json'):
        if (ROOT/name).is_file():shutil.copyfile(ROOT/name,shadow/name)
    (shadow/'artifacts').mkdir();(shadow/'.logs').mkdir()
    pins=[]
    env=os.environ.copy();env.pop(cfg['safety']['credential_env'],None)
    for name in patch_names:
        path=ROOT/'.patches'/name
        raw=path.read_text(encoding='utf-8')
        if len(raw)>24000:raise RuntimeError('patch_command_length_limit')
        done=subprocess.run([str(CODEX),'--codex-run-as-apply-patch',raw],cwd=shadow,env=env,stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (shadow/'.logs'/(name+'.log')).write_bytes(done.stdout)
        if done.returncode:print(done.stdout.decode('utf-8','replace'));return 1
        pins.append(dict(path=str(path),sha256=file_hash(path)))
    done=subprocess.run([sys.executable,'-X','utf8',str(shadow/'scripts/verify-speculation-harness.py')],
        cwd=shadow,env=env,stdin=subprocess.DEVNULL,timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if done.returncode:return done.returncode
    after=sources(shadow)
    if sources(ROOT)!=before:raise RuntimeError('live_sources_changed_during_shadow_verification')
    if setup_files(ROOT)!=scripts_before or test_files(ROOT)!=tests_before:
        raise RuntimeError('live_setup_or_tests_changed_during_shadow_verification')
    deployment=dict(schema_version=1,run_id='localbench-90f326f86086',execution_hash=cfg['_execution_hash'],
        original_started=1789753599.011325,original_deadline=1790358399.011325,
        before_sources=before,after_sources=after,patches=pins,patch_executable=str(CODEX),
        patch_executable_sha256=file_hash(CODEX),shadow=str(shadow),verified=True,
        original_processes=PROCESS_PINS,
        setup_scripts=scripts_before,test_sources=tests_before)
    atomic_json(ROOT/'artifacts/speculation-preparation-deployment.json',deployment)
    print(json.dumps({'verified':True,'shadow':str(shadow),'after_sources_sha256':digest(after),'staged_patches':len(pins)}))
    return 0

if __name__=='__main__':raise SystemExit(main())
