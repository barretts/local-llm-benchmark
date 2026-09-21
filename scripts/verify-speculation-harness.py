"""Run the full mock harness and bind the result to exact controller sources."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localbench.config import atomic_json,digest,file_hash,load
from scripts.speculation_setup_contract import setup_files,test_files

def main():
    cfg=load();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log=ROOT/'.logs'/('speculation-harness-tests-'+stamp+'.log')
    sources={p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted((ROOT/'localbench').rglob('*.py'))}
    scripts=setup_files(ROOT);tests=test_files(ROOT)
    env=os.environ.copy();env.pop(cfg['safety']['credential_env'],None)
    started=time.time()
    with log.open('wb') as f:
        done=subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-v'],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=600,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    raw=log.read_text(encoding='utf-8',errors='replace');match=re.search(r'^Ran (\d+) tests? in ',raw,re.MULTILINE)
    current={p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted((ROOT/'localbench').rglob('*.py'))}
    receipt=dict(passed=done.returncode==0 and current==sources and scripts==setup_files(ROOT) and tests==test_files(ROOT),
        test_count=int(match[1]) if match else None,setup_scripts=scripts,test_sources=tests,
        sources=sources,sources_sha256=digest(sources),log=str(log),log_sha256=file_hash(log),
        started=started,finished=time.time(),spec_hash=cfg['_spec_hash'],execution_hash=cfg['_execution_hash'])
    atomic_json(ROOT/'artifacts/speculation-harness-verification.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='sources'}))
    if not receipt['passed']:
        print(raw[-16000:]);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
