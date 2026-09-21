"""Explicit 16K Qwen3.8 experiment; the primary 64K contract stays frozen."""
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import time

from .adapters.native import NativeAdapter, creation_time
from .config import atomic_json, digest, file_hash, load
from .state import State, gpu_lock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'qwen38-16k'
ORIGINAL = ('localbench-90f326f86086', 1789753599.011325, 1790358399.011325)
REVISION = '4ca720788d1e01f1bff70c033e0d0028fd02e502'
REPOSITORY = 'unsloth/Qwen3.8-27B-GGUF'
ENGINE = 'upstream-llama-nightly'
BINARY = Path('F:/local-llm-benchmark-runtime/upstream-5b335f413e4f73b0809c4fe39af894efbcc6a0d2/bin/llama-server.exe')
PINS = [
    ('ud-q4km', 'Qwen3.8-27B-UD-Q4_K_M.gguf', 16464440224, '322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482'),
    ('ud-q4ks', 'Qwen3.8-27B-UD-Q4_K_S.gguf', 15358213024, '75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017'),
    ('ud-iq4xs', 'Qwen3.8-27B-UD-IQ4_XS.gguf', 14252845984, '40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199'),
]
CONTRACT = {
    'experiment': 'qwen38-27b-16k-higher-precision',
    'user_instruction': 'drop to 16k, increase to the highest quant you can and test just qwen 3.8 27b',
    'context_window_tokens': 16384, 'full_prompt_tokens': 12288,
    'reserved_output_tokens': 4096, 'regular_minimum': 27, 'regular_total': 36,
    'long_minimum': 9, 'long_total': 12, 'capacity_values_required': 9,
    'final_fresh': 20, 'final_cached': 20, 'minimum_gpu_headroom_mib': 512,
    'selection': 'Descending weight precision with the existing f16 cache; full layer offload and measured headroom.',
    'cache_profiles': ['f16'],
    'new_variant_limit': 3, 'model_revision': REVISION, 'model_repository': REPOSITORY,
    'scope': 'Separately labelled 16K experiment; never a 64K qualification.',
    'budgets': 'Shared original cumulative weight ledger and original 168-hour deadline.',
}


def focused_config():
    cfg = copy.deepcopy(load())
    cfg['measurement'].update(context_window_tokens=16384, full_prompt_tokens=12288,
        smaller_prompt_targets=[4096, 8192], cached_stable_prefix_tokens_target=10240,
        cached_variable_suffix_tokens_target=2048)
    cfg['paths']['artifacts'] = str(OUT)
    cfg['paths']['logs'] = str(ROOT / '.logs' / 'qwen38-16k')
    cfg['_focused_contract_sha256'] = digest(CONTRACT)
    return cfg


class FocusedState(State):
    def snapshot(self):
        # Avoid rewriting the primary result set or its large summary.
        value = {'run': self.run, 'now': self.clock(),
            'new_weight_bytes_reserved': self.db.execute('SELECT COALESCE(SUM(bytes),0) FROM weights').fetchone()[0],
            'experiment': self.get_control('focused_qwen38_16k'),
            'active_job': self.get_control('active_job')}
        atomic_json(OUT / 'status.json', value)
        ledger = [dict(r) for r in self.db.execute('SELECT * FROM weights')]
        atomic_json(OUT / 'weight-ledger.json', ledger)
        atomic_json(ROOT / 'artifacts' / 'weight-ledger.json', ledger)
        return value


class FocusedAdapter(NativeAdapter):
    def minimum_context_tokens(self):
        return 16384

    def launch(self, config=None):
        if self.requested['context'] != 16384 or not self.model['id'].startswith('qwen38-27b-'):
            raise ValueError('focused_qwen38_16k_scope_required')
        result = super().launch(config)
        if self.effective.get('effective_context') != 16384:
            self.unload_owned()
            raise RuntimeError('focused_effective_context_mismatch')
        return result


def make_adapter(cfg, state, collector, model, cache='f16'):
    from .scheduler import model_identity, register_configuration
    expected_sha = model.get('sha256')
    model = model_identity(cfg, state, model)
    if expected_sha and model['sha256'] != expected_sha:
        raise RuntimeError('focused_model_pin_changed')
    settings = {'context': 16384, 'cache_k': cache, 'cache_v': cache,
        'focused_contract_sha256': cfg['_focused_contract_sha256']}
    adapter = FocusedAdapter(cfg, state, ENGINE, model, binary=BINARY, settings=settings)
    adapter.collector = collector
    try:
        adapter.discover_capabilities()
        adapter.launch()
        return adapter, register_configuration(cfg, state, adapter)
    except BaseException:
        adapter.unload_owned()
        raise


def assert_original_clock(state):
    r = state.run
    if (r['id'], r['started'], r['deadline']) != ORIGINAL:
        raise RuntimeError('original_execution_clock_required')


def assert_verified():
    receipt = json.loads((OUT / 'harness-verification.json').read_text(encoding='utf-8'))
    if receipt.get('passed') is not True:
        raise RuntimeError('focused_harness_verification_required')
    for name, expected in receipt['sources'].items():
        if file_hash(ROOT / name) != expected:
            raise RuntimeError('focused_verified_source_changed:' + name)


def progress(state, status, **details):
    old = state.get_control('focused_qwen38_16k') or {}
    value = {**old, **details, 'status': status, 'updated': time.time(),
        'contract_sha256': digest(CONTRACT), 'original_deadline': ORIGINAL[2]}
    state.control('focused_qwen38_16k', value)
    state.snapshot()
    print(json.dumps({'status': status, **details}), flush=True)


def percentile(values, p):
    values = sorted(v for v in values if isinstance(v, (float, int)) and math.isfinite(v))
    return values[max(0, math.ceil(len(values) * p) - 1)] if values else None


def validated_measurement(cfg, result):
    from .prompts import validate_context
    evidence = result.get('token_evidence', {})
    reasons = validate_context(evidence.get('expected_prompt_tokens'), result.get('metrics', {}),
        cfg['measurement'], evidence.get('effective_context_tokens'), evidence.get('truncated', False))
    if evidence.get('context_shift') is not False:
        reasons.append('context_shift_unverified')
    if evidence.get('tokenization_verified') is not True:
        reasons.append('tokenization_unverified')
    if reasons:
        result.update(valid=False, passed=False, reason=';'.join(reasons))
    return result


def gpu_reading(cfg, label):
    from .doctor import command
    r = command(cfg, label, ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], timeout=15)
    if r['exit_code'] != 0:
        raise RuntimeError('gpu_memory_query_unavailable')
    total, used, free = map(float, r['output'].strip().split(','))
    return {'total_mib': total, 'used_mib': used, 'free_mib': free, 'log': r['log']}


def memory_verdict(adapter, before, after, resources):
    layers = adapter.effective.get('offload_layers')
    full = isinstance(layers, list) and len(layers) == 2 and layers[0] == layers[1] and layers[1] > 0
    peak = resources.get('peak_gpu_used_mib')
    peak = max(after['used_mib'], peak if isinstance(peak, (float, int)) else 0)
    available = after['used_mib'] + after['free_mib']
    reserved = max(0, after['total_mib'] - available)
    headroom = min(after['free_mib'], available - peak)
    return {'fits': full and headroom >= CONTRACT['minimum_gpu_headroom_mib'],
        'full_layer_offload': full, 'offload_layers': layers, 'peak_used_mib': peak,
        'minimum_free_mib': headroom, 'reserved_mib': reserved,
        'available_mib': available, 'before': before, 'after': after,
        'shared_allocation_is_diagnostic': True, 'resources': resources}


def coding_job(cfg, state, adapter, collector, identifier, fixture_id, seed, replicate=0, mode='fresh'):
    from .fixtures import load_fixture
    from .quality import quality_task
    from .scheduler import key_base, prior_result, result_status
    fixture = load_fixture(cfg, fixture_id, seed)
    target = cfg['measurement']['full_prompt_tokens'] if fixture_id.startswith('long') else None
    base = key_base(cfg, adapter, identifier, 'quality', seed, mode, replicate, target, fixture['fixture_hash'])
    key = {**base, 'fixture_prompt_hash': fixture['fixture_hash']}
    prior = prior_result(state, key)
    if prior is not None:
        return prior
    state.check_budget(cfg['limits']['per_task_timeout_seconds'])
    job = state.enqueue(key)
    attempt = state.begin(job)
    began = time.time()
    state.control('active_job', {'job': job, 'kind': 'quality', 'fixture': fixture_id,
        'seed': seed, 'model': adapter.model['id'], 'engine': ENGINE,
        'configuration_id': identifier, 'started': began})
    progress(state, 'coding', fixture=fixture_id, seed=seed, configuration_id=identifier)
    try:
        result = quality_task(cfg, state, adapter, fixture, identifier, job, attempt)
    except (RuntimeError, ValueError, OSError) as error:
        result = {'kind': 'quality', 'configuration_id': identifier, 'fixture': fixture_id,
            'seed': seed, 'group': 'long' if target else 'regular', 'passed': False,
            'valid': False, 'reason': 'inference_infrastructure_error', 'error': str(error)}
    result['experiment_contract_sha256'] = digest(CONTRACT)
    result['resources'] = collector.snapshot(began, time.time())
    if result['resources'].get('foreign_workload_overlap'):
        result.update(timing_valid=False, latency_diagnostic_only=True)
    state.finish(attempt, result_status(result), result, reason=result.get('reason'),
        transient=result.get('reason') == 'inference_infrastructure_error')
    state.control('active_job', None)
    return result


def rows_for(state, identifier):
    rows = state.db.execute('SELECT a.*,j.key_json FROM attempts a JOIN jobs j ON j.id=a.job_id '
        "WHERE json_extract(j.key_json,'$.configuration_id')=? ORDER BY a.id", (identifier,)).fetchall()
    latest = {}
    for r in rows:
        x = dict(r)
        x['result'] = json.loads(x['result'] or '{}')
        x['job_key'] = json.loads(x.pop('key_json'))
        latest[x['job_id']] = x
    return list(latest.values())


def quality_stats(cfg, rows, group):
    expected = {(f, s) for f in cfg['grading'][group + '_fixture_ids'] for s in cfg['grading']['seeds']}
    completed = {}
    for r in rows:
        d, k = r['result'], r['job_key']
        pair = (d.get('fixture'), d.get('seed'))
        if d.get('kind') == 'quality' and d.get('group') == group and d.get('valid') is True \
                and pair in expected and k.get('replicate') == 0 and k.get('mode') == 'fresh':
            completed[pair] = d.get('passed') is True
    passed = sum(completed.values())
    return {'passed': passed, 'attempted': len(completed), 'total': len(expected),
        'failed': len(completed) - passed, 'maximum_possible': passed + len(expected) - len(completed)}


def measured(cfg, state, adapter, collector, identifier, kind, seed, mode, replicate, operation):
    from .scheduler import measured_job
    result = None
    for unused in range(cfg['limits']['retry_transient_attempts'] + 1):
        result = measured_job(cfg, state, adapter, collector, identifier, kind, seed, mode, replicate,
            12288, lambda callback: validated_measurement(cfg, operation(callback)))
        if result.get('valid') or result.get('reason') not in ('foreign_workload_overlap', 'inference_infrastructure_error'):
            break
    return result


def write_results(cfg, state, identifier, selection, status):
    rows = rows_for(state, identifier) if identifier else []
    regular, long = (quality_stats(cfg, rows, g) for g in ('regular', 'long'))
    caps = {r['result'].get('seed'): r['result'] for r in rows
        if r['result'].get('kind') == 'capacity' and r['job_key'].get('target_tokens') == 12288}
    timing = [r['result'] for r in rows if r['result'].get('kind') == 'timing']
    fresh = [r for r in timing if r.get('mode') == 'fresh' and r.get('passed') is True
        and r.get('valid') is True and r.get('final_validation') is True
        and r.get('token_evidence', {}).get('cache_isolation_verified') is True]
    cached = [r for r in timing if r.get('mode') == 'cached' and r.get('passed') is True
        and r.get('valid') is True and r.get('final_validation') is True
        and r.get('token_evidence', {}).get('controlled_prefix_experiment_verified') is True]
    demos = [r['result'] for r in rows if r['job_key'].get('mode') == 'isolated-demo']
    config_row = state.db.execute("SELECT data FROM entities WHERE kind='configuration' AND id=?",
        (identifier,)).fetchone() if identifier else None
    configuration = json.loads(config_row[0]) if config_row else None
    gates = {'regular_coverage': regular['attempted'] == 36, 'regular_quality': regular['passed'] >= 27,
        'long_coverage': long['attempted'] == 12, 'long_quality': long['passed'] >= 9,
        'capacity_all_nine': all(caps.get(s, {}).get('passed') is True and caps[s].get('valid') is True
            and caps[s].get('markers_passed') == 3 for s in cfg['grading']['capacity_marker_seeds']),
        'twenty_valid_fresh': len(fresh) == 20, 'twenty_valid_cached': len(cached) == 20,
        'isolated_demo': bool(demos) and demos[-1].get('passed') is True and demos[-1].get('valid') is True,
        'gpu_headroom': bool(selection.get('selected', {}).get('fit', {}).get('fits'))}
    selected_peak = max([r['result'].get('resources', {}).get('peak_gpu_used_mib') or 0 for r in rows] or [0])
    after = selection.get('selected', {}).get('fit', {}).get('after', {})
    available = after.get('used_mib', 0) + after.get('free_mib', 0)
    if selected_peak and available - selected_peak < CONTRACT['minimum_gpu_headroom_mib']:
        gates['gpu_headroom'] = False
    value = {'status': status, 'updated': time.time(), 'contract': CONTRACT, 'parent_run': state.run,
        'configuration_id': identifier, 'configuration': configuration, 'selection': selection,
        'regular': regular, 'long': long, 'gates': gates, 'qualified_16k': all(gates.values()),
        'qualified_64k': False, 'failed_gates': [k for k, v in gates.items() if not v],
        'fresh_p95_first_action_seconds': percentile([r['metrics'].get('first_action_seconds') for r in fresh], .95),
        'cached_p95_first_action_seconds': percentile([r['metrics'].get('first_action_seconds') for r in cached], .95),
        'results': rows}
    atomic_json(OUT / 'results.json', value)
    text = ['# Qwen 3.8 27B at 16K', '', 'Status: ' + status,
        'Qualified at 16K: ' + str(value['qualified_16k']), 'This is a separately labelled 16K experiment.', '',
        f"Regular coding: {regular['passed']}/{regular['attempted']} measured; gate 27/36.",
        f"Context tasks: {long['passed']}/{long['attempted']} measured; gate 9/12.",
        'Failed gates: ' + ', '.join(value['failed_gates']), '',
        'Exact configuration, commands, grading records and raw log references: results.json.']
    (OUT / 'results.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    if configuration:
        atomic_json(OUT / 'launch.json', {'configuration': configuration,
            'command': str(ROOT / '.venv' / 'Scripts' / 'python.exe') + ' ' + str(ROOT / 'scripts' / 'run-qwen38-16k.py') + ' --serve',
            'qualified_16k': value['qualified_16k'], 'effective_context': 16384,
            'api': 'http://127.0.0.1:38201/v1', 'model': 'local-model',
            'launch_argv': configuration['effective_settings'].get('launch_argv')})
    return value


def weight_model(cfg, state, pin):
    from .acquisition import download
    name, filename, size, sha = pin
    path = Path(cfg['paths']['new_model_root']) / 'unsloth--Qwen3.8-27B-GGUF' / REVISION / filename
    progress(state, 'downloading', variant=name, bytes=size)
    artifact = {'bytes': size, 'sha256': sha,
        'url': f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{filename}'}
    last = None
    for unused in range(cfg['limits']['retry_transient_attempts'] + 1):
        try:
            download(cfg, state, artifact, path, weight=True)
            return {'id': 'qwen38-27b-' + name, 'path': str(path), 'bytes': size,
                'sha256': sha, 'repo': REPOSITORY, 'revision': REVISION, 'reuse_new_bytes': size}
        except (OSError, RuntimeError) as error:
            last = error
            if any(s in str(error) for s in ('budget', 'headroom', 'identity', 'mismatch', 'stop_after_current')):
                raise
    raise RuntimeError('focused_download_unavailable:' + type(last).__name__)


def footprint_probe(cfg, state, collector, model, cache='f16'):
    from .measurement import capacity
    before = gpu_reading(cfg, 'fit-before-' + model['id'])
    adapter = None
    began = time.time()
    try:
        adapter, identifier = make_adapter(cfg, state, collector, model, cache)
        cap = measured(cfg, state, adapter, collector, identifier, 'capacity', 17, 'fresh', 0,
            lambda cb: capacity(cfg, state, adapter, identifier, 17, target=12288, on_packed=cb))
        after = gpu_reading(cfg, 'fit-after-' + model['id'])
        resources = collector.snapshot(began, time.time())
        verdict = memory_verdict(adapter, before, after, resources)
        return {'model': model, 'cache': cache, 'configuration_id': identifier,
            'fit': verdict, 'capacity': cap, 'metadata': adapter.metadata()}
    except (RuntimeError, ValueError, OSError) as error:
        return {'model': model, 'cache': cache, 'fit': {'fits': False}, 'error': str(error)}
    finally:
        if adapter is not None:
            adapter.unload_owned()


def choose_weight(cfg, state, collector, selection):
    if selection.get('selected'):
        return selection['selected']
    from .search_core import checkpoint_model
    if 'installed_footprint' not in selection:
        model = checkpoint_model(state, 'qwen38-27b-iq3', cfg['installed_model_candidates'])
        progress(state, 'installed_16k_memory_probe')
        selection['installed_footprint'] = footprint_probe(cfg, state, collector, model)
        atomic_json(OUT / 'selection.json', selection)
    for pin in PINS:
        previous = next((p for p in selection.get('probes', []) if p['model']['id'] == 'qwen38-27b-' + pin[0]), None)
        if previous:
            if previous['fit']['fits']:
                selection['selected'] = previous
                return previous
            continue
        model = weight_model(cfg, state, pin)
        progress(state, 'fit_probe', variant=pin[0])
        probe = footprint_probe(cfg, state, collector, model)
        selection.setdefault('probes', []).append(probe)
        if probe['fit']['fits']:
            selection['selected'] = probe
        atomic_json(OUT / 'selection.json', selection)
        if selection.get('selected'):
            return probe
    return None


def run_quality(cfg, state, adapter, collector, identifier, selection):
    for group in ('regular', 'long'):
        threshold = cfg['grading']['early_elimination_' + group + '_failures']
        for seed in cfg['grading']['seeds']:
            for fixture in cfg['grading'][group + '_fixture_ids']:
                stats = quality_stats(cfg, rows_for(state, identifier), group)
                if stats['failed'] >= threshold:
                    return write_results(cfg, state, identifier, selection, group + '_quality_eliminated')
                result = None
                for unused in range(cfg['limits']['retry_transient_attempts'] + 1):
                    result = coding_job(cfg, state, adapter, collector, identifier, fixture, seed)
                    if result.get('valid') or result.get('reason') != 'inference_infrastructure_error':
                        break
                value = write_results(cfg, state, identifier, selection, 'quality_in_progress')
                progress(state, 'coding_result', fixture=fixture, seed=seed,
                    passed=result.get('passed'), reason=result.get('reason'),
                    regular=value['regular'], long=value['long'])
        stats = quality_stats(cfg, rows_for(state, identifier), group)
        if stats['attempted'] != stats['total'] or stats['passed'] < cfg['grading'][group + '_minimum_passes']:
            return write_results(cfg, state, identifier, selection, group + '_quality_incomplete_or_failed')
    return write_results(cfg, state, identifier, selection, 'quality_passed')


def run_final(cfg, state, adapter, collector, identifier, selection):
    from .measurement import cached_prefix, latency
    for i in range(20):
        progress(state, 'final_fresh', replicate=i)
        result = measured(cfg, state, adapter, collector, identifier, 'timing', 1042 + i, 'fresh', 1000 + i,
            lambda cb, r=i: latency(cfg, state, adapter, identifier, 1000 + r,
                final=True, target=12288, on_packed=cb))
        write_results(cfg, state, identifier, selection, 'stability_in_progress')
        if result.get('passed') is not True:
            return write_results(cfg, state, identifier, selection, 'final_fresh_stability_failed')
    adapter.fresh_restart()
    prefix = cached_prefix(cfg, adapter, state, identifier)
    prime = validated_measurement(cfg, latency(cfg, state, adapter, identifier, -1,
        mode='cached', stable_prefix=prefix, target=12288))
    m = prime['metrics']
    prefix['primed_verified'] = prime.get('passed') is True and m.get('stream_complete') is True \
        and m.get('first_action_valid') is True and type(m.get('prompt_tokens')) is int \
        and 12224 <= m['prompt_tokens'] <= 12288
    prefix['owned_pid'] = adapter.process.pid
    atomic_json(OUT / 'cache-prime.json', {'prefix': {k: v for k, v in prefix.items() if k != 'prefix'}, 'result': prime})
    if not prefix['primed_verified']:
        return write_results(cfg, state, identifier, selection, 'cache_prime_unverified')
    for i in range(20):
        progress(state, 'final_cached', replicate=i)
        result = measured(cfg, state, adapter, collector, identifier, 'timing', 2042 + i, 'cached', 2000 + i,
            lambda cb, r=i: latency(cfg, state, adapter, identifier, 2000 + r,
                mode='cached', final=True, stable_prefix=prefix, target=12288, on_packed=cb))
        write_results(cfg, state, identifier, selection, 'stability_in_progress')
        if result.get('passed') is not True:
            return write_results(cfg, state, identifier, selection, 'final_cached_stability_failed')
    adapter.fresh_restart()
    coding_job(cfg, state, adapter, collector, identifier, 'py02', 42, replicate=1, mode='isolated-demo')
    return write_results(cfg, state, identifier, selection, 'validation_complete')


def prepare(cfg, state):
    assert_verified()
    assert_original_clock(state)
    for key in ('doctor_verified', 'fixtures_verified', 'harness_verified'):
        if state.get_control(key) is not True:
            raise RuntimeError('original_' + key + '_required')
    row = state.db.execute("SELECT data FROM entities WHERE kind='controller' AND id=?", (ORIGINAL[0],)).fetchone()
    if row:
        pid = json.loads(row[0]).get('pid')
        if pid and creation_time(pid) is not None:
            raise RuntimeError('legacy_worker_must_exit_before_focused_run')
    supervisor = ROOT / 'artifacts' / 'unattended-supervisor-status.json'
    if supervisor.exists():
        old = json.loads(supervisor.read_text(encoding='utf-8'))
        if old.get('status') not in ('attention_required', 'complete', 'covered_search_complete'):
            raise RuntimeError('legacy_supervisor_must_halt_before_focused_run')
    OUT.mkdir(parents=True, exist_ok=True)
    Path(cfg['paths']['logs']).mkdir(parents=True, exist_ok=True)
    for filename in ('grader-images.json', 'runtime-container-storage.json'):
        source = ROOT / 'artifacts' / filename
        if source.exists():
            shutil.copyfile(source, OUT / filename)
    from .scheduler import PROTOCOL_HASHES
    proof = {'contract': CONTRACT, 'contract_sha256': digest(CONTRACT), 'protocol_source_hashes': PROTOCOL_HASHES,
        'fixture_version': state.get_control('fixture_version'), 'parent_run': state.run,
        'sampling': cfg['default_sampling'], 'binary': str(BINARY), 'binary_sha256': file_hash(BINARY)}
    plan = OUT / 'plan.json'
    if plan.exists():
        saved = json.loads(plan.read_text(encoding='utf-8'))
        for key in ('contract_sha256', 'protocol_source_hashes', 'fixture_version', 'sampling', 'binary_sha256'):
            if saved[key] != proof[key]:
                raise RuntimeError('focused_plan_changed_requires_reconciliation:' + key)
    else:
        atomic_json(plan, proof)
    # A user-authorized focused resume clears only the cooperative stop request.
    state.recover()
    assert_original_clock(state)
    state.entity('focused_controller', digest(CONTRACT), {'pid': os.getpid(),
        'created': creation_time(os.getpid()), 'started': time.time(), 'contract_sha256': digest(CONTRACT)})


@contextmanager
def owned_gpu(cfg, cleanup):
    with gpu_lock(cfg):
        try:
            yield
        finally:
            # Keep the GPU lock until the original model handle has exited.
            cleanup()


def run(serve=False):
    cfg = focused_config()
    state = FocusedState(cfg)
    adapter = collector = None
    selection = {}
    identifier = None
    def cleanup():
        if adapter is not None:
            adapter.unload_owned()
        if collector is not None:
            collector.stop()
        state.control('active_job', None)
        state.stop()
    try:
        with owned_gpu(cfg, cleanup):
            prepare(cfg, state)
            from .doctor import command, powershell
            command(cfg, 'focused-gpu', ['nvidia-smi'], timeout=15)
            powershell(cfg, 'focused-host', 'Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json', timeout=20)
            powershell(cfg, 'focused-memory', 'Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json', timeout=20)
            docker = command(cfg, 'focused-docker', [cfg['installed_tools']['docker'], 'info', '--format', '{{.ServerVersion}}'], timeout=20)
            if docker['exit_code'] != 0:
                raise RuntimeError('isolated_docker_grading_unavailable')
            from .resources import ResourceCollector
            collector = ResourceCollector(cfg, state).start()
            path = OUT / 'selection.json'
            selection = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'contract_sha256': digest(CONTRACT), 'probes': []}
            if selection['contract_sha256'] != digest(CONTRACT):
                raise RuntimeError('focused_selection_contract_changed')
            selected = choose_weight(cfg, state, collector, selection)
            if not selected:
                progress(state, 'no_higher_precision_profile_fits')
                return write_results(cfg, state, None, selection, 'no_higher_precision_profile_fits')
            identifier = selected['configuration_id']
            if serve:
                previous = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
                if previous.get('qualified_16k') is not True:
                    raise RuntimeError('focused_qualified_16k_result_required_for_serve')
            adapter, actual = make_adapter(cfg, state, collector, selected['model'], selected['cache'])
            if actual != identifier:
                raise RuntimeError('focused_configuration_identity_changed')
            if serve:
                progress(state, 'serving', endpoint='http://127.0.0.1:38201/v1')
                while True:
                    state.check_budget(reserve_report=False)
                    if not adapter.health():
                        raise RuntimeError('focused_endpoint_health_failed')
                    time.sleep(2)
            from .measurement import capacity, latency, throughput
            for seed in cfg['grading']['capacity_marker_seeds']:
                progress(state, '16k_capacity', seed=seed, configuration_id=identifier)
                measured(cfg, state, adapter, collector, identifier, 'capacity', seed, 'fresh', 0,
                    lambda cb, s=seed: capacity(cfg, state, adapter, identifier, s, target=12288, on_packed=cb))
            value = run_quality(cfg, state, adapter, collector, identifier, selection)
            for i in range(3):
                progress(state, 'fresh_action_screen', replicate=i)
                measured(cfg, state, adapter, collector, identifier, 'timing', 42 + i, 'fresh', i,
                    lambda cb, r=i: latency(cfg, state, adapter, identifier, r, target=12288, on_packed=cb))
            measured(cfg, state, adapter, collector, identifier, 'throughput', 42, 'fresh', 0,
                lambda cb: throughput(cfg, adapter, identifier, target=12288, on_packed=cb))
            if not (value['gates']['regular_quality'] and value['gates']['long_quality'] and value['gates']['capacity_all_nine']):
                progress(state, 'complete_no_qualification', regular=value['regular'], long=value['long'])
                return write_results(cfg, state, identifier, selection, 'complete_no_qualification')
            value = run_final(cfg, state, adapter, collector, identifier, selection)
            progress(state, 'complete_qualified_16k' if value['qualified_16k'] else 'complete_no_qualification',
                regular=value['regular'], long=value['long'])
            return value
    except BaseException as error:
        progress(state, 'attention_required', reason=type(error).__name__ + ':' + str(error))
        write_results(cfg, state, identifier, selection, 'attention_required')
        raise
    finally:
        state.close()
