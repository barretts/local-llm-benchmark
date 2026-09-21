"""Verify the harness before the focused experiment; never start its clock."""
from pathlib import Path
import json
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localbench.config import atomic_json, digest, file_hash, load
from localbench.focused_qwen38 import OUT, CONTRACT

OUT.mkdir(parents=True, exist_ok=True)
prior = json.loads((ROOT / 'artifacts' / 'speculation-harness-verification.json').read_text(encoding='utf-8'))
files = list((ROOT / 'localbench').rglob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
files += [ROOT / 'scripts' / n for n in ('run-qwen38-16k.py', 'verify-qwen38-16k.py')]
for n in prior.get('setup_scripts', {}):
    p = Path(n)
    if not p.is_absolute():p = ROOT / p
    if p.is_file():files.append(p)
sources = {str(p.relative_to(ROOT)).replace('\\', '/'): file_hash(p) for p in sorted(set(files))}
started = time.time()
log = OUT / 'harness-tests.log'
with log.open('wb') as out:
    p = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
unchanged = all(file_hash(ROOT / n) == sha for n, sha in sources.items())
raw = log.read_text(encoding='utf-8', errors='replace')
matches = re.findall(r'Ran (\d+) tests', raw)
cfg = load()
receipt = {'passed': p.returncode == 0 and unchanged, 'sources': sources,
    'sources_sha256': digest(sources), 'test_count': int(matches[-1]) if matches else None,
    'log': str(log), 'log_sha256': file_hash(log), 'started': started, 'finished': time.time(),
    'parent_spec_hash': cfg['_spec_hash'], 'parent_execution_hash': cfg['_execution_hash'],
    'contract_sha256': digest(CONTRACT), 'source_unchanged_during_tests': unchanged}
atomic_json(OUT / 'harness-verification.json', receipt)
print(json.dumps({k: v for k, v in receipt.items() if k != 'sources'}), flush=True)
raise SystemExit(0 if receipt['passed'] else 1)
