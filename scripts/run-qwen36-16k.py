"""Qwen3.6 27B IQ4_XS at 16K, queued behind the requested Bonsai test."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localbench.config import atomic_json, digest, file_hash
from localbench.adapters.native import NativeAdapter, creation_time

# A separate module instance keeps the running Bonsai/Qwen3.8 controllers intact.
spec = importlib.util.spec_from_file_location('localbench.focused_qwen36_runtime',
    ROOT / 'localbench' / 'focused_qwen38.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
PARENT_OUT = core.OUT
core.OUT = ROOT / 'artifacts' / 'qwen36-16k'
core.CONTRACT = copy.deepcopy(core.CONTRACT)
core.CONTRACT.update(experiment='qwen36-27b-16k-iq4xs',
    user_instruction='stop Qwen3.8; test Qwen3.6 of the same 27B size',
    model_repository='unsloth/Qwen3.6-27B-GGUF',
    model_revision='82d411acf4a06cfb8d9b073a5211bf410bfc29bf', new_variant_limit=1,
    selection='User-requested Qwen3.6 27B IQ4_XS; full offload, f16 KV and 512 MiB measured headroom.')
original_config = core.focused_config


def config():
    cfg = original_config()
    cfg['paths']['logs'] = str(ROOT / '.logs' / 'qwen36-16k')
    return cfg


class Adapter(core.FocusedAdapter):
    def launch(self, config=None):
        if self.requested['context'] != 16384 or self.model['id'] != 'qwen36-27b-iq4xs':
            raise ValueError('qwen36_16k_scope_required')
        value = NativeAdapter.launch(self, config)
        if self.effective.get('effective_context') != 16384:
            self.unload_owned()
            raise RuntimeError('qwen36_effective_context_mismatch')
        return value


class State(core.FocusedState):
    def snapshot(self):
        active = self.get_control('active_job')
        if active and active.get('model') != 'qwen36-27b-iq4xs':
            active = None
        value = {'run': self.run, 'now': self.clock(), 'active_job': active,
            'experiment': self.get_control('focused_qwen36_16k'),
            'weight_bytes': self.db.execute('SELECT COALESCE(SUM(bytes),0) FROM weights').fetchone()[0]}
        atomic_json(core.OUT / 'status.json', value)
        return value


def progress(state, status, **details):
    state.control('focused_qwen36_16k', {'status': status, 'updated': time.time(),
        'contract_sha256': digest(core.CONTRACT), **details})
    state.snapshot()
    print(json.dumps({'status': status, **details}), flush=True)


def choose(cfg, state, collector, selection):
    if selection.get('selected'):
        return selection['selected']
    a = json.loads((core.OUT / 'acquisition.json').read_text())
    if a['status'] != 'download_verified':
        raise RuntimeError('Verified Qwen3.6 download required')
    model = {'id': 'qwen36-27b-iq4xs', 'path': a['path'],
        'bytes': a['artifact']['bytes'], 'sha256': a['artifact']['sha256'], 'revision': a['revision']}
    progress(state, 'fit_probe')
    probe = core.footprint_probe(cfg, state, collector, model)
    selection.setdefault('probes', []).append(probe)
    if probe['fit']['fits']:
        selection['selected'] = probe
    atomic_json(core.OUT / 'selection.json', selection)
    return selection.get('selected')


original_results = core.write_results


def results(*args, **kwargs):
    value = original_results(*args, **kwargs)
    if value['qualified_16k']:
        path = core.OUT / 'launch.json'
        launch = json.loads(path.read_text())
        launch['command'] = str(ROOT / '.venv/Scripts/python.exe') + ' ' + str(Path(__file__)) + ' --serve'
        atomic_json(path, launch)
    return value


core.focused_config = config
core.FocusedAdapter = Adapter
core.FocusedState = State
core.progress = progress
core.choose_weight = choose
core.write_results = results


def verify():
    parent = json.loads((PARENT_OUT / 'harness-verification.json').read_text())
    if parent['passed'] is not True:
        raise RuntimeError('Verified parent harness required')
    for name, expected in parent['sources'].items():
        if file_hash(ROOT / name) != expected:
            raise RuntimeError('Parent source changed: ' + name)
    cfg = config()
    assert cfg['measurement']['context_window_tokens'] == 16384
    assert cfg['measurement']['full_prompt_tokens'] == 12288
    assert cfg['paths']['artifacts'] == str(core.OUT)
    assert Adapter.minimum_context_tokens(None) == 16384
    assert NativeAdapter.minimum_context_tokens(None) == 65536
    for model, context in (('qwen38-27b-ud-iq4xs', 16384), ('qwen36-27b-iq4xs', 65536)):
        a = Adapter(cfg, None, core.ENGINE, {'id': model, 'path': 'inert'}, binary='inert', settings={'context': context})
        try:
            a.launch()
        except ValueError:
            pass
        else:
            raise AssertionError('Out-of-scope launch accepted')
    receipt = copy.deepcopy(parent)
    receipt['sources']['scripts/run-qwen36-16k.py'] = file_hash(Path(__file__))
    receipt.update(contract_sha256=digest(core.CONTRACT), parent_receipt=str(PARENT_OUT / 'harness-verification.json'),
        wrapper_checks_passed=True, wrapper_verified_at=time.time())
    atomic_json(core.OUT / 'harness-verification.json', receipt)
    print('Qwen3.6 wrapper verified against the unchanged 597-test harness', flush=True)


def run(serve=False):
    cfg = config()
    state = State(cfg)
    try:
        core.assert_original_clock(state)
        if not serve:
            prior = json.loads((ROOT / 'artifacts/bonsai-2-27b/worker-receipt.json').read_text(encoding='utf-8-sig'))
            progress(state, 'waiting_for_bonsai_and_download')
            while True:
                a = json.loads((core.OUT / 'acquisition.json').read_text())
                if creation_time(prior['pid']) is None and a['status'] == 'download_verified':
                    break
                if time.time() + 7200 >= state.run['deadline']:
                    raise RuntimeError('Original benchmark deadline reached')
                time.sleep(5)
        for name in ('fixture-manifest.json', 'grader-images.json', 'runtime-container-storage.json'):
            shutil.copyfile(ROOT / 'artifacts' / name, core.OUT / name)
    finally:
        state.close()
    return core.run(serve)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--serve', action='store_true')
    args = p.parse_args()
    verify() if args.self_test else run(args.serve)
