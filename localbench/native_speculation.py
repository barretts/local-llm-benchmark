"""Pinned native draft contracts and real, per-request acceptance evidence."""
from pathlib import Path
import math
import re
import time

from .config import atomic_json, file_hash

TARGET = 'qwen36-27b-iq2'
REVISION = '8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8'
DRAFT_PINS = {
    'draft-mtp': dict(id='qwen36-27b-mtp-q4', repo='ggml-org/Qwen3.6-27B-GGUF',
        revision=REVISION, filename='mtp-Qwen3.6-27B-Q4_0.gguf', bytes=1680270560,
        sha256='3d593f9e2788d59bb30d6024706b1efd5219fea466b6397c46159e3540937173',
        target_model_id=TARGET, maximum_draft_tokens=2, role='existing_target_mtp_heads'),
    'draft-dflash': dict(id='qwen36-27b-dflash-q8', repo='ggml-org/Qwen3.6-27B-GGUF',
        revision=REVISION, filename='dflash-Qwen3.6-27B-Q8_0.gguf', bytes=1849481440,
        sha256='a31adddb37adaca315b94a18d96d124135ee15b76b7249986e77057267b01909',
        target_model_id=TARGET, maximum_draft_tokens=15, role='new_trained_draft_model'),
}
FIELDS = ('draft_model', 'draft_n_max', 'draft_cache_k', 'draft_cache_v',
          'draft_gpu_layers', 'speculation_contract_sha256', 'speculation_baseline_configuration_id')
COUNTERS = {
    'drafted_tokens': 'llamacpp:spec_decode_num_draft_tokens_total',
    'accepted_tokens': 'llamacpp:spec_decode_num_accepted_tokens_total',
    'verification_steps': 'llamacpp:spec_decode_num_drafts_total',
}

def contract_hash():
    return file_hash(Path(__file__))

def section(text, flag):
    lines = text.splitlines()
    pattern = re.compile(r'(?<![\w-])'+re.escape(flag)+r'(?![\w-])')
    for n, line in enumerate(lines):
        if not pattern.search(line): continue
        found = [line]
        for following in lines[n+1:]:
            if following.lstrip().startswith('-') and re.search(r'--[a-z0-9-]+', following): break
            found.append(following)
        return '\n'.join(found)
    return ''

def supports(text, mode):
    return mode in DRAFT_PINS and bool(re.search(r'(?<![\w-])'+re.escape(mode)+r'(?![\w-])',section(text,'--spec-type')))

def clean_environment(env):
    # Synthetic acceptance can be inherited from environment variables too.
    return {k:v for k,v in env.items() if not k.upper().startswith(('LLAMA_ARG_SPEC_', 'LLAMA_ARG_DRAFT_'))}

def arguments(config, state, model, settings, help_text):
    mode = settings.get('speculation', 'off')
    if mode == 'off':
        if any(k in settings for k in FIELDS): raise ValueError('draft_controls_require_active_speculation')
        return []
    if mode not in DRAFT_PINS or model.get('id') != TARGET:
        raise ValueError('unverified_speculative_type_or_target')
    if settings.get('speculation_contract_sha256') != contract_hash():
        raise RuntimeError('speculation_contract_changed_reconciliation_required')
    baseline = settings.get('speculation_baseline_configuration_id')
    if not isinstance(baseline,str) or not re.fullmatch(r'[a-f0-9]{64}',baseline):
        raise ValueError('matched_speculation_control_required')
    row=state.db.execute("SELECT data FROM entities WHERE kind='configuration' AND id=?",(baseline,)).fetchone()
    import json
    data=json.loads(row[0]) if row else None
    expected={k:v for k,v in settings.items() if k not in FIELDS}
    expected['speculation']='off'
    if not data or data.get('model_id')!=model['id'] or data.get('requested_settings')!=expected:
        raise ValueError('matched_control_settings_or_target_mismatch')
    if not supports(help_text,mode): raise RuntimeError('unsupported_native_speculative_type:'+mode)
    draft = settings.get('draft_model')
    pin = DRAFT_PINS[mode]
    if not isinstance(draft,dict) or set(draft) != set(pin)|{'path'} \
            or any(draft.get(k) != v for k,v in pin.items()):
        raise ValueError('draft_identity_or_target_pin_mismatch')
    path = Path(draft['path'])
    if not path.is_absolute() or path.name != pin['filename'] or not path.is_file():
        raise RuntimeError('verified_local_draft_file_required')
    roots = [Path(config['paths'][k]).resolve() for k in ('new_model_root','installed_model_root')]
    roots.append((Path.home()/'.cache/huggingface').resolve())
    if not any(path.resolve().is_relative_to(r) for r in roots):
        raise ValueError('draft_outside_authorized_model_roots')
    stat = path.stat()
    if stat.st_size != pin['bytes']: raise RuntimeError('draft_weight_size_mismatch')
    identity = dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    cached = state.get_control('model_hash:'+str(path))
    if not cached or any(cached.get(k)!=v for k,v in identity.items()) or cached.get('sha256')!=pin['sha256']:
        state.check_budget()
        if file_hash(path) != pin['sha256']: raise RuntimeError('draft_weight_hash_mismatch')
        state.control('model_hash:'+str(path),{**identity,'sha256':pin['sha256']})
    n = settings.get('draft_n_max')
    if type(n) is not int or not 1 <= n <= pin['maximum_draft_tokens']:
        raise ValueError('invalid_or_clamped_draft_width')
    if settings.get('draft_gpu_layers') != 99 or settings.get('flash_attention') != 'on':
        raise ValueError('initial_speculation_requires_cuda_draft_and_flash_attention')
    flags = dict((('--spec-type',mode),('--spec-draft-model',str(path)),
        ('--spec-draft-n-max',n),('--spec-draft-n-min',0),('--spec-draft-p-min',0.0),
        ('--spec-draft-ngl',99),('--spec-draft-device','CUDA0')))
    for key,flag in (('draft_cache_k','--spec-draft-type-k'),('draft_cache_v','--spec-draft-type-v')):
        value = settings.get(key)
        if value not in ('f16','q8_0','q4_0') or not re.search(r'\b'+re.escape(value)+r'\b',section(help_text,flag)):
            raise RuntimeError('unsupported_draft_cache_control:'+key)
        flags[flag] = value
    for flag in flags:
        if not section(help_text,flag): raise RuntimeError('unsupported_native_draft_flag:'+flag)
    return [item for flag,value in flags.items() for item in (flag,str(value))]

def startup_evidence(raw, settings):
    # This proof says a draft loaded. Real use is proved by request counters.
    if settings.get('speculation','off') == 'off': return None
    forbidden = ('synthetic speculative acceptance is enabled','synthetic acceptance:',
        'no implementations specified for speculative decoding')
    if any(v in raw.lower() for v in forbidden):
        raise RuntimeError('invalid_or_synthetic_speculative_runtime')
    filename = settings['draft_model']['filename']
    if filename not in raw or 'loading draft model' not in raw.lower():
        raise RuntimeError('draft_load_not_verified_in_startup_logs')
    return dict(status='draft_loaded_counters_required',kind=settings['speculation'],
        draft_sha256=settings['draft_model']['sha256'],draft_weight_bytes=settings['draft_model']['bytes'],
        configured_n_max=settings['draft_n_max'],draft_context='inherits_target_context_in_pinned_engine')

def parse_counters(raw):
    found = {}
    for key,name in COUNTERS.items():
        match = re.findall(r'^'+re.escape(name)+r'\s+([^\s]+)\s*$',raw,re.MULTILINE)
        if len(match) != 1: raise RuntimeError('missing_or_ambiguous_speculative_counter:'+key)
        try: number = float(match[0])
        except ValueError: raise RuntimeError('invalid_speculative_counter:'+key) from None
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            raise RuntimeError('invalid_speculative_counter:'+key)
        found[key] = int(number)
    if found['accepted_tokens'] > found['drafted_tokens']:
        raise RuntimeError('impossible_speculative_acceptance_counter')
    return found

def verify_baseline_engine(config,state,settings,metadata):
    import json
    from .scheduler import engine_identity
    row=state.db.execute("SELECT data FROM entities WHERE kind='configuration' AND id=?",
        (settings['speculation_baseline_configuration_id'],)).fetchone()
    baseline=json.loads(row[0]) if row else None
    if not baseline or engine_identity(baseline)!=engine_identity(metadata) \
            or baseline.get('identity',{}).get('sampler')!=config['default_sampling']:
        raise ValueError('matched_control_engine_or_sampler_mismatch')

def counter_delta(before,after):
    delta = {k:after[k]-before[k] for k in COUNTERS}
    if any(v<0 for v in delta.values()) or delta['accepted_tokens'] > delta['drafted_tokens']:
        raise RuntimeError('speculative_counter_reset_or_invalid_delta')
    if delta['drafted_tokens'] and not delta['verification_steps']:
        raise RuntimeError('draft_tokens_without_target_verification')
    return {**delta,'status':'real_drafting_verified' if delta['drafted_tokens'] else 'no_drafting_for_request',
        'acceptance_fraction':delta['accepted_tokens']/delta['drafted_tokens'] if delta['drafted_tokens'] else None,
        'synthetic_acceptance':False,'source':'owned_server_prometheus_counter_delta'}

def read_counters(http):
    conn,response = http.request('/metrics',deadline=time.monotonic()+2)
    try:
        raw = response.read(512*1024+1).decode('utf-8')
        if len(raw)>512*1024: raise RuntimeError('oversized_speculative_metrics')
        return parse_counters(raw),raw
    finally:
        response.close();conn.close()

def measured_stream(adapter,request,kwargs):
    prefix = Path(str(kwargs['log_prefix'])+'-speculation.json')
    evidence = dict(kind=adapter.requested['speculation'],owned_pid=adapter.process.pid,
        draft_sha256=adapter.requested['draft_model']['sha256'],contract_sha256=contract_hash())
    try:
        before,raw_before = read_counters(adapter.http)
        response = adapter.http.measure(request,**kwargs)
        after,raw_after = read_counters(adapter.http)
        telemetry = counter_delta(before,after)
        evidence.update(before=before,after=after,raw_before=raw_before,raw_after=raw_after,delta=telemetry)
        response['speculation'] = {**telemetry,'kind':adapter.requested['speculation'],'evidence_path':str(prefix)}
        response['metrics'].update(speculation_counters_verified=True,
            speculation_drafted_tokens=telemetry['drafted_tokens'],speculation_accepted_tokens=telemetry['accepted_tokens'],
            speculation_verification_steps=telemetry['verification_steps'],speculation_acceptance_fraction=telemetry['acceptance_fraction'],
            speculation_evidence_path=str(prefix))
        return response
    except (OSError,RuntimeError,ValueError) as error:
        evidence.update(status='verification_failed',error_type=type(error).__name__,reason=str(error))
        raise RuntimeError('speculation_evidence_verification_failed:'+str(error)) from error
    finally:
        atomic_json(prefix,evidence)

def real_drafting(screen):
    rows = screen.get('capacity',[])+screen.get('timing',[])+[screen.get('throughput') or {}]
    return any(r.get('valid') is True and r.get('metrics',{}).get('speculation_counters_verified') is True
        and r['metrics'].get('speculation_drafted_tokens',0)>0 for r in rows)
