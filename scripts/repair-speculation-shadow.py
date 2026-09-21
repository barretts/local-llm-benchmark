"""Verify the bounded comparison repair in a private copy, without inference."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localbench.config import atomic_json,digest,file_hash,load

PATCHES=('093-mtp-dflash-real-timing-contract.patch',)
CODEX=Path(r'C:\Users\barrett\AppData\Local\nvm\v24.13.1\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe')

def sources(root):
    return {p.relative_to(root).as_posix():file_hash(p) for p in sorted((root/'localbench').rglob('*.py'))}

def main():
    cfg=load();before=sources(ROOT)
    shadow=ROOT/'.staging'/('speculation-repair-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if shadow.exists() or not shadow.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError('fresh_private_shadow_required')
    shadow.mkdir(parents=True)
    for name in ('localbench','tests','scripts','fixtures','grader'):
        shutil.copytree(ROOT/name,shadow/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('benchmark-spec.json','execution-overrides.json'):
        if (ROOT/name).is_file():shutil.copyfile(ROOT/name,shadow/name)
    (shadow/'.logs').mkdir();(shadow/'artifacts').mkdir()
    env=os.environ.copy();env.pop(cfg['safety']['credential_env'],None)
    pins=[]
    for name in PATCHES:
        path=ROOT/'.patches'/name;raw=path.read_text(encoding='utf-8')
        if len(raw)>24000:raise RuntimeError('patch_too_large')
        done=subprocess.run([str(CODEX),'--codex-run-as-apply-patch',raw],cwd=shadow,env=env,
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        log=shadow/'.logs'/(name+'.log');log.write_bytes(done.stdout)
        if done.returncode:
            print(done.stdout.decode('utf-8','replace'));return 1
        pins.append(dict(path=str(path),sha256=file_hash(path)))
    log=shadow/'.logs/verification-launch.log'
    with log.open('wb') as output:
        done=subprocess.run([sys.executable,'-X','utf8',str(shadow/'scripts/verify-speculation-harness.py')],
            cwd=shadow,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=660,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if done.returncode:
        print(log.read_text(encoding='utf-8',errors='replace')[-9000:]);return 1
    verification=json.loads((shadow/'artifacts/speculation-harness-verification.json').read_text())
    after=sources(shadow)
    if sources(ROOT)!=before or verification.get('passed') is not True or verification.get('sources')!=after:
        raise RuntimeError('source_or_verification_drift')
    receipt=dict(schema_version=1,verified=True,shadow=str(shadow),patches=pins,
        execution_hash=cfg['_execution_hash'],before_sources=before,after_sources=after,
        after_sources_sha256=digest(after),harness=verification,
        original_started=1789753599.011325,original_deadline=1790358399.011325)
    atomic_json(ROOT/'artifacts/speculation-repair-deployment.json',receipt)
    print(json.dumps(dict(verified=True,shadow=str(shadow),test_count=verification['test_count'],
        after_sources_sha256=digest(after))))
    return 0

if __name__=='__main__':raise SystemExit(main())
