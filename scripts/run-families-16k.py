"""Sequential installed-family 16K trials with a four-run tool-error screen."""
from pathlib import Path
import argparse
import copy
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localbench.config import atomic_json, digest, file_hash, load
from localbench.adapters.native import NativeAdapter
from localbench.recovery import WindowsProcess

OUT = ROOT / 'artifacts/families-16k'
IDS = ['devstral-small2-24b-q3', 'ministral3-14b-reasoning-q4',
       'gptoss20b-q4ks', 'gemma4-12b-qat-q4', 'gemma4-e4b-q6',
       'nemotron3nano-4b-q6']
TOOL_ERRORS = {'malformed_or_unpermitted_tool_stream',
               'long_requires_exactly_one_final_action', 'missing_long_final_action'}
SOURCE = 'scripts/run-families-16k.py'


def screen(rows):
    runs = [r['result'] for r in rows if r['result'].get('kind') == 'quality'
            and r['result'].get('group') == 'regular'
            and r['job_key'].get('replicate') == 0
            and r['job_key'].get('mode') == 'fresh']
    first = runs[:4]
    errors = sum(r.get('reason') in TOOL_ERRORS for r in first)
    return {'runs': len(first), 'tool_errors': errors,
            'stop': len(first) == 4 and errors >= 2,
            'rule': 'At least two tool-call errors in the first four coding runs.'}


def configure(model_id):
    if model_id not in IDS:
        raise ValueError('Unapproved family model')
    models = {m['id']: m for m in load()['installed_model_candidates']}
    model = copy.deepcopy(models[model_id])
    if model_id == 'gemma4-12b-qat-q4':
        assert Path(model['path']) == Path('F:/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf')
    if not Path(model['path']).is_file():
        raise RuntimeError('Installed model missing: ' + model['path'])
    spec = importlib.util.spec_from_file_location('localbench.family_runtime',
                                                ROOT / 'localbench/focused_qwen38.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.OUT = OUT / model_id
    core.CONTRACT = copy.deepcopy(core.CONTRACT)
    core.CONTRACT.update(experiment='installed-family-16k-' + model_id,
        user_instruction='Test installed families at 16K; stop at two tool errors in first four coding runs.',
        selection='Installed weights only; full GPU offload, f16 KV, 512 MiB measured headroom.',
        new_variant_limit=0, model_repository=None, model_revision=None,
        installed_model_path=model['path'], first_pass_runs=4,
        first_pass_tool_error_limit=2, first_pass_tool_error_reasons=sorted(TOOL_ERRORS))
    base_config = core.focused_config

    def config():
        cfg = base_config()
        cfg['paths']['logs'] = str(ROOT / '.logs/families-16k' / model_id)
        return cfg

    class Adapter(core.FocusedAdapter):
        def launch(self, config=None):
            if self.requested['context'] != 16384 or self.model['id'] != model_id:
                raise ValueError('family_16k_scope_required')
            value = NativeAdapter.launch(self, config)
            if self.effective.get('effective_context') != 16384:
                self.unload_owned()
                raise RuntimeError('family_effective_context_mismatch')
            return value

    control = 'family_16k_' + model_id

    class State(core.FocusedState):
        def snapshot(self):
            active = self.get_control('active_job')
            if active and active.get('model') != model_id:
                active = None
            value = {'run': self.run, 'now': self.clock(), 'active_job': active,
                     'experiment': self.get_control(control),
                     'weight_bytes': self.db.execute('SELECT COALESCE(SUM(bytes),0) FROM weights').fetchone()[0]}
            atomic_json(core.OUT / 'status.json', value)
            return value

    def progress(state, status, **details):
        state.control(control, {'status': status, 'updated': time.time(),
                                'contract_sha256': digest(core.CONTRACT), **details})
        state.snapshot()
        print(json.dumps({'model': model_id, 'status': status, **details}), flush=True)

    def choose(cfg, state, collector, selection):
        if selection.get('selected'):
            return selection['selected']
        if selection.get('probes'):
            return None
        from localbench.scheduler import model_identity
        pinned = model_identity(cfg, state, model)
        progress(state, 'fit_probe')
        probe = core.footprint_probe(cfg, state, collector, pinned)
        selection.setdefault('probes', []).append(probe)
        if probe['fit']['fits']:
            selection['selected'] = probe
        atomic_json(core.OUT / 'selection.json', selection)
        return selection.get('selected')

    original_quality = core.run_quality
    original_results = core.write_results

    def results(cfg, state, identifier, selection, status):
        value = original_results(cfg, state, identifier, selection, status)
        value['first_pass'] = screen(core.rows_for(state, identifier)) if identifier else None
        atomic_json(core.OUT / 'results.json', value)
        if value['qualified_16k']:
            launch = json.loads((core.OUT / 'launch.json').read_text())
            launch['command'] = str(ROOT / '.venv/Scripts/python.exe') + ' ' + str(Path(__file__)) + ' --model ' + model_id + ' --serve'
            atomic_json(core.OUT / 'launch.json', launch)
        return value

    def quality(cfg, state, adapter, collector, identifier, selection):
        seed = cfg['grading']['seeds'][0]
        for fixture in cfg['grading']['regular_fixture_ids'][:4]:
            verdict = screen(core.rows_for(state, identifier))
            if verdict['stop']:
                return results(cfg, state, identifier, selection, 'first_pass_tool_errors_eliminated')
            for unused in range(cfg['limits']['retry_transient_attempts'] + 1):
                result = core.coding_job(cfg, state, adapter, collector, identifier, fixture, seed)
                if result.get('valid') or result.get('reason') != 'inference_infrastructure_error':
                    break
            value = results(cfg, state, identifier, selection, 'first_pass_in_progress')
            progress(state, 'first_pass_result', fixture=fixture, passed=result.get('passed'),
                     reason=result.get('reason'), first_pass=value['first_pass'])
        if screen(core.rows_for(state, identifier))['stop']:
            progress(state, 'first_pass_tool_errors_eliminated')
            return results(cfg, state, identifier, selection, 'first_pass_tool_errors_eliminated')
        return original_quality(cfg, state, adapter, collector, identifier, selection)

    core.focused_config = config
    core.FocusedAdapter = Adapter
    core.FocusedState = State
    core.progress = progress
    core.choose_weight = choose
    core.write_results = results
    core.run_quality = quality
    # Preserve the verified execution flow, adding only the authorized immediate screen stop.
    run_source = inspect.getsource(core.run)
    needle = '            value = run_quality(cfg, state, adapter, collector, identifier, selection)\n'
    assert run_source.count(needle) == 1
    run_source = run_source.replace(needle, needle +
        "            if value['status'] == 'first_pass_tool_errors_eliminated':\n"
        "                return value\n")
    exec(compile(run_source, SOURCE + ':scoped_run', 'exec'), core.__dict__)
    return core


def verify():
    parent_path = ROOT / 'artifacts/qwen38-16k/harness-verification.json'
    parent = json.loads(parent_path.read_text())
    assert parent['passed'] is True
    for name, expected in parent['sources'].items():
        if file_hash(ROOT / name) != expected:
            raise RuntimeError('Verified parent source changed: ' + name)
    for errors, stop in ((0, False), (1, False), (2, True), (3, True), (4, True)):
        rows = [{'result': {'kind': 'quality', 'group': 'regular',
                            'reason': 'malformed_or_unpermitted_tool_stream' if i < errors else 'hidden_tests_failed'},
                 'job_key': {'replicate': 0, 'mode': 'fresh'}} for i in range(4)]
        assert screen(rows)['stop'] == stop
        assert screen(rows[:3])['stop'] is False
    for model_id in IDS:
        core = configure(model_id)
        cfg = core.focused_config()
        assert cfg['measurement']['context_window_tokens'] == 16384
        assert cfg['measurement']['full_prompt_tokens'] == 12288
        assert NativeAdapter.minimum_context_tokens(None) == 65536
        assert core.FocusedAdapter.minimum_context_tokens(None) == 16384
        for ident, context in ((model_id, 65536), ('wrong-model', 16384)):
            adapter = core.FocusedAdapter(cfg, None, core.ENGINE,
                {'id': ident, 'path': 'inert'}, binary='inert', settings={'context': context})
            try:
                adapter.launch()
            except ValueError:
                pass
            else:
                raise AssertionError('Out-of-scope launch accepted')
        core.OUT.mkdir(parents=True, exist_ok=True)
        receipt = copy.deepcopy(parent)
        receipt['sources'][SOURCE] = file_hash(Path(__file__))
        receipt.update(parent_receipt=str(parent_path),
            parent_sources_sha256=parent['sources_sha256'],
            sources_sha256=digest(receipt['sources']),
            contract_sha256=digest(core.CONTRACT), wrapper_checks_passed=True,
            wrapper_verified_at=time.time())
        atomic_json(core.OUT / 'harness-verification.json', receipt)
    atomic_json(OUT / 'manifest.json', {'models': IDS, 'context': 16384,
        'first_pass_runs': 4, 'stop_tool_errors': 2, 'downloads': False,
        'original_deadline': 1790358399.011325, 'source_sha256': file_hash(Path(__file__))})
    print('Verified: unchanged 597-test parent, family scopes, four-run cutoff, and source pins.', flush=True)


def child(model_id, serve=False):
    core = configure(model_id)
    core.assert_verified()
    for name in ('fixture-manifest.json', 'grader-images.json', 'runtime-container-storage.json'):
        shutil.copyfile(ROOT / 'artifacts' / name, core.OUT / name)
    return core.run(serve)


def queue():
    verify()
    status_path = OUT / 'queue-status.json'
    saved = json.loads(status_path.read_text()) if status_path.exists() else {}
    completed = saved.get('completed', {})
    atomic_json(OUT / 'worker-receipt.json', {'pid': os.getpid(), 'started': time.time(),
        'source_sha256': file_hash(Path(__file__)), 'original_deadline': 1790358399.011325})
    for model_id in IDS:
        if model_id in completed:
            continue
        if time.time() + 7200 >= 1790358399.011325:
            atomic_json(status_path, {'status': 'deadline_report_reserve', 'completed': completed})
            return
        atomic_json(status_path, {'status': 'running', 'model': model_id,
                                 'updated': time.time(), 'completed': completed})
        folder = OUT / model_id
        with (folder / 'worker-stdout.log').open('a') as stdout, (folder / 'worker-stderr.log').open('a') as stderr:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--model', model_id],
                                        cwd=str(ROOT), stdout=stdout, stderr=stderr)
            code = process.wait()
        result_path = folder / 'results.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        completed[model_id] = {'exit_code': code, 'status': result.get('status', 'worker_error'),
            'qualified_16k': result.get('qualified_16k', False), 'regular': result.get('regular'),
            'long': result.get('long'), 'first_pass': result.get('first_pass')}
        atomic_json(status_path, {'status': 'between_models', 'updated': time.time(), 'completed': completed})
        if code != 0:
            # A failed child may have ownership issues: require review before another GPU load.
            atomic_json(status_path, {'status': 'attention_required', 'model': model_id,
                                     'updated': time.time(), 'completed': completed})
            return
    atomic_json(status_path, {'status': 'complete', 'updated': time.time(), 'completed': completed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--model', choices=IDS)
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        verify()
    elif args.model:
        child(args.model, args.serve)
    else:
        queue()

