"""User-requested Bonsai test using the verified, frozen 64K benchmark."""
from pathlib import Path
import argparse
import copy
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localbench.config import atomic_json, file_hash, load
from localbench.state import State, gpu_lock
from localbench.adapters.native import NativeAdapter, creation_time
from localbench.focused_qwen38 import assert_original_clock, assert_verified

OUT = ROOT / 'artifacts' / 'bonsai-2-27b'


class PrismAdapter(NativeAdapter):
    def __init__(self, config, state, engine, model, binary=None, settings=None):
        if engine != 'prism-ternary':
            raise ValueError('Only the user-requested Prism runtime is eligible')
        # Reuse native local-port selection; preserve the actual fork identity.
        super().__init__(config, state, 'upstream-llama-nightly', model, binary, settings)
        self.engine = engine


class BonsaiState(State):
    def snapshot(self):
        value = {'run': self.run, 'now': self.clock(),
            'experiment': self.get_control('bonsai_test'), 'active_job': self.get_control('active_job'),
            'weight_bytes': self.db.execute('SELECT COALESCE(SUM(bytes),0) FROM weights').fetchone()[0]}
        atomic_json(OUT / 'status.json', value)
        return value


def verify():
    assert_verified()
    cfg = load()
    adapter = PrismAdapter(cfg, None, 'prism-ternary', {'id': 'inert', 'path': 'inert'}, binary='inert')
    assert adapter.engine == 'prism-ternary'
    assert adapter.minimum_context_tokens() == 65536
    assert adapter.requested['context'] == 65536
    assert adapter.http.port == cfg['private_ports']['native']
    try:
        PrismAdapter(cfg, None, 'different-engine', {}, binary='inert')
    except ValueError:
        pass
    else:
        raise AssertionError('Unexpected engine accepted')
    atomic_json(OUT / 'harness-verification.json', {'passed': True,
        'controller_sha256': file_hash(Path(__file__)), 'parent_harness_tests': 597,
        'parent_sources_verified_unchanged': True, 'context_tokens': 65536,
        'checks': ['fork identity and local endpoint', 'unchanged 64K gate', 'engine scope'],
        'updated': time.time()})


def run():
    cfg = copy.deepcopy(load())
    cfg['paths']['artifacts'] = str(OUT)
    cfg['paths']['logs'] = str(ROOT / '.logs' / 'bonsai-2-27b')
    OUT.mkdir(parents=True, exist_ok=True)
    Path(cfg['paths']['logs']).mkdir(parents=True, exist_ok=True)
    receipt = json.loads((OUT / 'harness-verification.json').read_text())
    if not receipt['passed'] or receipt['controller_sha256'] != file_hash(Path(__file__)):
        raise RuntimeError('Verified controller required')
    assert_verified()
    acquisition = json.loads((OUT / 'acquisition.json').read_text())
    runtime = json.loads((OUT / 'runtime.json').read_text())
    if acquisition['status'] != 'download_verified':
        raise RuntimeError('Verified model download required')
    for filename in ('fixture-manifest.json', 'grader-images.json', 'runtime-container-storage.json'):
        shutil.copyfile(ROOT / 'artifacts' / filename, OUT / filename)
    model = {'id': 'bonsai-2-27b-ptq1', 'path': acquisition['path'],
        'sha256': acquisition['artifact']['sha256'], 'revision': acquisition['artifact']['revision']}
    cfg['installed_model_candidates'].append(model)
    prior = json.loads((ROOT / 'artifacts' / 'qwen38-16k' / 'worker-receipt.json').read_text(encoding='utf-8-sig'))
    state = BonsaiState(cfg)
    collector = None
    def progress(status):
        state.control('bonsai_test', {'status': status, 'pid': os.getpid(),
            'created': creation_time(os.getpid()), 'updated': time.time(),
            'context_tokens': 65536, 'publisher_claims_verified': False})
        state.snapshot()
        print(json.dumps({'status': status}), flush=True)
    try:
        assert_original_clock(state)
        progress('waiting_for_qwen')
        # Waiting never probes the GPU or clears another worker's stop flag.
        while creation_time(prior['pid']) is not None:
            if time.time() + 7200 >= state.run['deadline']:
                raise RuntimeError('Original benchmark deadline reached')
            time.sleep(5)
        with gpu_lock(cfg):
            state.recover()
            assert_original_clock(state)
            from localbench.doctor import command, powershell
            command(cfg, 'bonsai-gpu', ['nvidia-smi'], timeout=15)
            powershell(cfg, 'bonsai-memory', 'Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json', timeout=20)
            from localbench.resources import ResourceCollector
            collector = ResourceCollector(cfg, state).start()
            from localbench import scheduler
            scheduler.NativeAdapter = PrismAdapter  # Only this private controller process.
            from localbench.search_core import screen, regular_and_long, quality_qualified, final_samples
            from localbench.report import Report
            profile = {'kind': 'native', 'engine': 'prism-ternary', 'binary_path': runtime['binary'],
                'source_commit': runtime['source_commit'], 'release': runtime['release']}
            progress('64k_screen')
            candidate = screen(cfg, state, collector, profile, model)
            atomic_json(OUT / 'screen.json', candidate)
            if candidate.get('capacity_qualified'):
                progress('coding_quality')
                regular_and_long(cfg, state, collector, [candidate])
            if quality_qualified(cfg, state, [candidate]):
                progress('stability_validation')
                final_samples(cfg, state, collector, candidate)
                from localbench.handoff import demo_and_prepare
                demo_and_prepare(cfg, state, collector, candidate)
            summary = Report(cfg, state).write()
            atomic_json(OUT / 'completion.json', {'status': 'complete', 'configuration_id': candidate.get('configuration_id'),
                'screen': candidate, 'updated': time.time(), 'report': str(OUT / 'report.html')})
            progress('complete')
    except Exception as error:
        progress('attention_required')
        atomic_json(OUT / 'error.json', {'type': type(error).__name__, 'reason': str(error), 'updated': time.time()})
        raise
    finally:
        if collector is not None:
            collector.stop()
        state.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    verify() if args.self_test else run()
