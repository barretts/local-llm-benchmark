"""Finite resumable search: installed baselines, all engine probes, qualification."""
from pathlib import Path
import json
import re
from .acquisition import acquire_gguf, acquire_hf_subset
from .config import digest, file_hash
from .runtime_catalog import native_upstream, acquire_image, cuda_access_probe
from .search_core import screen, unavailable, regular_and_long, quality_qualified, screen_latency, final_samples, gemma_marker_control, gemma_control_timings_complete
from .report import Report
from .scheduler import PROTOCOL_HASHES,model_identity
from .extra_discovery import extra_gguf_screens


def runtime(config,state,engine):
    if engine=='bundled-llama':return {'engine':engine,'kind':'native','binary_path':config['installed_tools']['bundled_llama_server']}
    row=state.db.execute("SELECT data FROM entities WHERE kind='runtime_catalog' AND id=?",(engine,)).fetchone()
    if row:return dict(json.loads(row['data']),engine=engine)
    if engine=='upstream-llama-nightly':return {'engine':engine,'kind':'native',**native_upstream(config,state)}
    if engine in ('ik-llama','vllm','sglang'):
        cuda_access_probe(config,state)
        return {'engine':engine,'kind':'container',**acquire_image(config,state,engine)}
    if engine in ('ollama','lm-studio'):return {'engine':engine,'kind':engine}
    if engine=='turboquant-cuda':
        from .runtime_build import prepare_pinned_source, detect_windows_toolchain, build_pinned_source
        prepared=prepare_pinned_source(config,state,engine)
        review=state.get_control('source_review:'+engine)
        if not review or review.get('fingerprint')!=prepared['review_fingerprint']:
            state.control('pending_source_review:'+engine,prepared)
            raise RuntimeError('source_review_pending:'+prepared['review_bundle'])
        built=build_pinned_source(config,state,prepared,review['fingerprint'],toolchain=detect_windows_toolchain(config))
        record={'engine':engine,'kind':'native',**built}
        state.entity('runtime_catalog',engine,record)
        return record
    if engine=='exllamav3-tabby':
        from .tabby_setup import prepare_tabby_source, install_tabby
        prepared=prepare_tabby_source(config,state)
        review=state.get_control('source_review:'+engine)
        if not review or review.get('fingerprint')!=prepared['review_fingerprint']:
            state.control('pending_source_review:'+engine,prepared)
            raise RuntimeError('source_review_pending:'+prepared['review_bundle'])
        built=install_tabby(config,state,prepared,review['fingerprint'])
        record={'engine':engine,'kind':'tabby',**built}
        state.entity('runtime_catalog',engine,record)
        return record
    raise RuntimeError('runtime_candidate_not_integrated:'+engine)


def saved_screens(state,runtimes):
    result=[]
    for row in state.db.execute("SELECT data FROM entities WHERE kind='screen'"):
        item=json.loads(row['data'])
        if item.get('configuration_id') and item.get('engine') in runtimes:
            item.setdefault('runtime',runtimes[item['engine']])
            result.append(item)
    return result


def stage_key(runtime,model,settings):
    binary=runtime.get('binary_path')
    native={'binary':file_hash(binary),'dlls':{p.name:file_hash(p) for p in sorted(Path(binary).parent.glob('*.dll'))}} if binary and Path(binary).is_file() else None
    if native and runtime['engine']=='bundled-llama':
        dependencies=Path.home()/'AppData/Local/Programs/Ollama/lib/ollama/cuda_v13'
        native['process_cuda_dependencies']={p.name:file_hash(p) for p in sorted(dependencies.glob('cublas*.dll'))}
    identity=runtime.get('image_digest') or runtime.get('runtime_manifest_sha256') or native or digest(runtime)
    return digest({'engine':runtime['engine'],'identity':identity,'model':model.get('sha256') or model.get('revision') or model,
                   'settings':settings,'screen_protocol':'sequential-screen-v1','protocol_sources':PROTOCOL_HASHES})


def pending_for_stage(state,configuration_id):
    """Only unfinished jobs belonging to the currently loaded protocol count."""
    if not configuration_id:return []
    unfinished=[]
    for row in state.db.execute("SELECT id,key_json FROM jobs WHERE status IN ('pending','running')"):
        key=json.loads(row['key_json']);kind=key.get('kind')
        if key.get('configuration_id')!=configuration_id or kind not in ('capacity','timing','throughput','common_byte','quality'):continue
        names=('quality.py','tools.py','schema.py') if kind=='quality' else ('common_byte.py','measurement.py') if kind=='common_byte' else ('measurement.py','prompts.py','regional_prompt.py')
        logical=digest({'protocol':{name:PROTOCOL_HASHES[name] for name in names},'run':state.run['id'],
            'fixture_hash':key.get('fixture_prompt_hash') if kind=='quality' else None,'kind':kind,
            'seed':key.get('seed'),'mode':key.get('mode'),'replicate':key.get('replicate'),'target':key.get('target_tokens')})
        if key.get('logical_plan_hash')==logical:unfinished.append(row['id'])
    return unfinished


def gemma_cached_small(config,state,result):
    """Read an existing current 16K pass; old screen records omitted that result."""
    if result.get('small_capacity'):
        return result['small_capacity']
    for row in state.db.execute("SELECT key_json,result FROM jobs WHERE status='passed' AND json_extract(key_json,'$.configuration_id')=?",(result.get('configuration_id'),)):
        key=json.loads(row['key_json'])
        if key.get('kind')=='capacity' and key.get('target_tokens')==16384 and key.get('seed')==17 and key.get('mode')=='fresh' and key.get('replicate')==0 and Report(config,state)._current_protocol(key,'capacity'):
            value=json.loads(row['result'] or '{}')
            if value.get('passed') is True and value.get('valid') is True:
                return value
    return None


def gemma_missing_control(config,state,rt,model,settings,result):
    if model.get('id')!='gemma4-12b-qat-q4' or settings is not None or rt.get('kind')!='native' or rt.get('engine') not in ('bundled-llama','upstream-llama-nightly'):
        return False
    return gemma_marker_control(config,rt,model,settings,gemma_cached_small(config,state,result),result.get('capacity',[])) and not gemma_control_timings_complete(config,result)


def run_screen(config,state,collector,rt,model,settings=None,tuning=False):
    if Path(model.get('path','')).is_file():model=model_identity(config,state,model)
    key='planned_screen:'+stage_key(rt,model,settings)
    done=state.get_control(key)
    if done and not pending_for_stage(state,done.get('configuration_id')) and not (not tuning and gemma_missing_control(config,state,rt,model,settings,done)):
        return {**done,'planned_screen_key':key}
    identifier=done.get('configuration_id') if done else None
    for retry in range(config['limits']['retry_transient_attempts']+1):
        result=screen(config,state,collector,rt,model,settings,tuning)
        identifier=result.get('configuration_id') or identifier
        result={**result,'planned_screen_key':key}
        if not pending_for_stage(state,identifier):
            state.control(key,result)
            break
    return result


def gemma_reduced_followup(config,state,collector,rt,model,baseline):
    """One supported requested-256 profile, using every normal tuning gate."""
    if model.get('id')!='gemma4-12b-qat-q4' or rt.get('kind')!='native' or rt.get('engine') not in ('bundled-llama','upstream-llama-nightly'):
        return None
    if pending_for_stage(state,baseline.get('configuration_id')) or not gemma_control_timings_complete(config,baseline):
        return None
    if not gemma_marker_control(config,rt,model,None,gemma_cached_small(config,state,baseline),baseline.get('capacity',[])):
        return None
    if baseline.get('settings',{}).get('reasoning')!='default':
        return None
    row=state.db.execute("SELECT data FROM entities WHERE kind='configuration' AND id=?",(baseline.get('configuration_id'),)).fetchone()
    metadata=json.loads(row['data']) if row else {}
    help_path=Path(metadata.get('help_log',''))
    supported=help_path.is_file() and bool(re.search(r'(?m)^\s*--reasoning-budget\s+N\b',help_path.read_text(encoding='utf-8-sig')))
    proposal={'kind':'gemma_reduced_reasoning_followup','engine':rt['engine'],'model':model['id'],
        'default_configuration_id':baseline.get('configuration_id'),'requested_reasoning_budget_tokens':256,
        'requested_method':'--reasoning-budget 256','effective_budget_enforcement':'unverified',
        'limitations':['Help and launch acceptance do not prove effective reasoning-budget enforcement; inspect props/logs and actual reasoning counters.'],
        'help_log':str(help_path) if supported else None}
    decision=digest(['gemma-reduced-followup',rt['engine'],baseline.get('configuration_id')])
    if not supported:
        state.entity('selection',decision,{**proposal,'selected':False,'reason':'unsupported_reasoning_budget_in_pinned_help'})
        return None
    settings={**baseline['settings'],'reasoning':'reduced'}
    pinned=model_identity(config,state,model) if Path(model.get('path','')).is_file() else model
    marker='planned_screen:'+stage_key(rt,pinned,settings)
    count=state.db.execute("SELECT COUNT(*) FROM entities WHERE kind='configuration'").fetchone()[0]
    if count>=config['limits']['maximum_unique_runtime_configurations'] and not state.get_control(marker):
        state.entity('selection',decision,{**proposal,'selected':False,'reason':'configuration_budget_exhausted'})
        return None
    result=run_screen(config,state,collector,rt,pinned,settings=settings,tuning=True)
    state.entity('selection',decision,{**proposal,'selected':True,'configuration_id':result.get('configuration_id'),
        'reason':'separate_profile_uses_all_normal_capacity_smoke_and_quality_gates'})
    return result


def memory_screen(config,state,collector,rt,model):
    result=run_screen(config,state,collector,rt,model)
    attempts=[result]
    followup=gemma_reduced_followup(config,state,collector,rt,model,result)
    if followup is not None:attempts.append(followup)
    if not result.get('eligible') and 'oom' in str(result.get('reason','')).lower():
        for cache in ('q8_0','q4_0'):
            settings={'cache_k':cache,'cache_v':cache} if rt['kind']!='container' or rt['engine']=='ik-llama' else {'kv_cache_dtype':'fp8','gpu_memory_utilization':.8}
            if rt['kind']=='tabby':settings={'cache_mode':'8,8' if cache=='q8_0' else '4,4'}
            result=run_screen(config,state,collector,rt,model,settings=settings)
            attempts.append(result)
            if result.get('eligible'):break
            if rt['kind']=='container' and rt['engine'] in ('vllm','sglang'):break
    # A smaller scratch allocation or one bounded layer reduction can make a
    # real 64K configuration feasible. These are separate measured settings.
    if rt['kind']=='native' and not any(s.get('eligible') or s.get('capacity_qualified') for s in attempts):
        marked=[s for s in attempts if 'oom' in str(s.get('reason','')).lower()]
        if marked:
            base=dict(marked[-1].get('settings') or {})
            batch,ubatch=min(config['tuning']['batch_ubatch_initial'],key=lambda pair:(pair[0],pair[1]))
            reduced={**base,'batch':batch,'ubatch':ubatch}
            maximum=config['limits']['maximum_unique_runtime_configurations']
            count=state.db.execute("SELECT COUNT(*) FROM entities WHERE kind='configuration'").fetchone()[0]
            if count>=maximum:
                state.entity('selection',digest(['native-memory',rt['engine'],model['id']]),{'selected':False,'reason':'configuration_budget_exhausted','model':model['id'],'engine':rt['engine']})
                return attempts
            result=run_screen(config,state,collector,rt,model,settings=reduced)
            attempts.append(result)
            if not result.get('eligible') and 'oom' in str(result.get('reason','')).lower():
                from .doctor import gguf_metadata
                try:
                    metadata=gguf_metadata(model['path']);architecture=metadata.get('general.architecture')
                    blocks=metadata.get(str(architecture)+'.block_count')
                except (OSError,ValueError):blocks=None
                if type(blocks) is int and blocks>1:
                    layers=max(1,blocks*3//4)
                    count=state.db.execute("SELECT COUNT(*) FROM entities WHERE kind='configuration'").fetchone()[0]
                    if count<maximum:
                        result=run_screen(config,state,collector,rt,model,settings={**reduced,'gpu_layers':layers})
                        attempts.append(result)
                    else:
                        state.entity('selection',digest(['native-offload',rt['engine'],model['id']]),{'selected':False,'reason':'configuration_budget_exhausted','model':model['id'],'engine':rt['engine']})
                else:
                    state.entity('selection',digest(['native-offload',rt['engine'],model['id']]),{'selected':False,'reason':'exact_layer_count_unavailable_for_bounded_offload','model':model['id'],'engine':rt['engine']})
    return attempts


def acquire_model(config,state,candidate,format='gguf'):
    state.check_budget(1800)
    try:return acquire_gguf(config,state,candidate) if format=='gguf' else acquire_hf_subset(config,state,candidate)
    except (RuntimeError,OSError,ValueError) as error:
        if str(error) in ('budget_exhausted','stop_after_current'):raise
        unavailable(config,state,'model-acquisition',candidate,'pinned_model_acquisition_unavailable:'+str(error),evidence=candidate)
        return None


def full_search(config,state,collector):
    runtimes={};screens=[];pending=[]
    # Reconcile exact current checkpoints, including those from the earlier
    # native worker. A legacy Boolean cannot prove the current plans finished.
    baseline_screens=[]
    state.control('search_covered',False)
    installed=sorted(config['installed_model_candidates'],key=lambda m:m['priority'])
    for engine in ('bundled-llama','upstream-llama-nightly'):
        try:runtimes[engine]=runtime(config,state,engine)
        except (RuntimeError,OSError,ValueError) as error:
            if str(error) in ('stop_after_current','budget_exhausted'):raise
            unavailable(config,state,engine,{'id':'installed-baselines'},str(error));continue
        for model in installed:
            if Path(model['path']).is_file():
                attempts=memory_screen(config,state,collector,runtimes[engine],model)
                screens.extend(attempts);baseline_screens.extend(attempts)
    baselines_done=all(s.get('planned_screen_key') and state.get_control(s['planned_screen_key']) and not pending_for_stage(state,s.get('configuration_id')) for s in baseline_screens)
    state.control('installed_baselines_completed',baselines_done)
    state.control('native_screens_completed',baselines_done)
    state.control('installed_baseline_stage_keys',[s.get('planned_screen_key') for s in baseline_screens])
    screens+=saved_screens(state,runtimes)
    if not baselines_done:
        return {'status':'baseline_measurements_pending','pending_engines':[], 'summary':Report(config,state).write()}
    # Reuse exact installed/HF files before acquiring these two small plain
    # files. They provide the same GGUF control for upstream and ik.
    ggufs=[]
    for candidate in config['download_weights']:
        if candidate['priority']!=1:continue
        model=acquire_model(config,state,candidate)
        if model:ggufs.append(model)
    for engine in ('bundled-llama','upstream-llama-nightly'):
        if engine in runtimes:
            for model in ggufs:screens.extend(memory_screen(config,state,collector,runtimes[engine],model))
    try:
        runtimes['ik-llama']=runtime(config,state,'ik-llama')
        for model in ggufs:screens.extend(memory_screen(config,state,collector,runtimes['ik-llama'],model))
    except (RuntimeError,OSError,ValueError) as error:
        if str(error) in ('stop_after_current','budget_exhausted'):raise
        unavailable(config,state,'ik-llama',{'id':'plain-9b-control'},str(error))
    try:
        runtimes['turboquant-cuda']=runtime(config,state,'turboquant-cuda')
        for model in ggufs:
            control=run_screen(config,state,collector,runtimes['turboquant-cuda'],model)
            screens.append(control)
            if 'unsupported' not in str(control.get('reason','')).lower():
                screens.append(run_screen(config,state,collector,runtimes['turboquant-cuda'],model,settings={'cache_k':'tbqp3','cache_v':'tbq3'},tuning=True))
    except (RuntimeError,OSError,ValueError) as error:
        if str(error) in ('stop_after_current','budget_exhausted'):raise
        if str(error).startswith('source_review_pending'):pending.append('turboquant-cuda')
        else:unavailable(config,state,'turboquant-cuda',{'id':'plain-9b-control'},str(error))
    linux_models=[]
    for candidate in config['linux_checkpoint_probes']:
        model=acquire_model(config,state,candidate,format='hf')
        if model:linux_models.append(model)
    for engine in ('vllm','sglang'):
        try:
            runtimes[engine]=runtime(config,state,engine)
            for model in linux_models:screens.extend(memory_screen(config,state,collector,runtimes[engine],model))
        except (RuntimeError,OSError,ValueError) as error:
            if str(error) in ('stop_after_current','budget_exhausted'):raise
            unavailable(config,state,engine,{'id':'prequantized-9b-snapshot'},str(error))
    # New ready EXL3 is bounded discovery variant 1. Exact public file pins,
    # model provenance and license are preserved by the subset acquisition.
    discovery=json.loads((Path(config['paths']['artifacts'])/'engine-discovery.json').read_text(encoding='utf-8-sig'))
    leads=[lead for lead in discovery['new_weight_leads'] if lead['id']=='ornith15-9b-exl3-hq4']
    try:
        runtimes['exllamav3-tabby']=runtime(config,state,'exllamav3-tabby')
        for lead in leads:
            selected=state.get_control('discovered_weight_variants') or []
            if lead['id'] not in selected:
                if len(selected)>=config['limits']['maximum_new_discovery_weight_variants']:raise RuntimeError('discovery_variant_budget_exhausted')
                state.control('discovered_weight_variants',selected+[lead['id']])
            model=acquire_model(config,state,dict(lead,bytes=lead['weight_bytes']),format='hf')
            if model:screens.extend(memory_screen(config,state,collector,runtimes['exllamav3-tabby'],model))
    except (RuntimeError,OSError,ValueError) as error:
        if str(error) in ('stop_after_current','budget_exhausted'):raise
        if str(error).startswith('source_review_pending'):pending.append('exllamav3-tabby')
        else:unavailable(config,state,'exllamav3-tabby',{'id':'ornith15-9b-exl3-hq4'},str(error))
    # Managed controls use a separate owned CPU tokenizer, never a second GPU.
    # Optional newly discovered GGUFs use the same acquired weights and every
    # ordinary capacity/quality/tuning/final/demo gate on existing native engines.
    screens.extend(extra_gguf_screens(config,state,collector,runtimes,baseline_screens,
        acquire_model=acquire_model,memory_screen=memory_screen,unavailable=unavailable))
    control_models=[m for m in config['installed_model_candidates'] if m['priority']==1]
    for engine in ('lm-studio','ollama'):
        try:
            runtimes[engine]=runtime(config,state,engine)
            if engine=='lm-studio':
                import os
                if not os.environ.get('LM_BENCH_TOKEN'):raise RuntimeError('lm_studio_authentication_unavailable:new_process_local_key_required')
            for model in control_models[:2]:screens.extend(memory_screen(config,state,collector,runtimes[engine],model))
        except (RuntimeError,OSError,ValueError) as error:
            if str(error) in ('stop_after_current','budget_exhausted'):raise
            unavailable(config,state,engine,{'id':'installed-priority1-controls'},str(error))
    # Keep every settings outcome, but schedule one copy per exact cfg.
    unique={s['configuration_id']:s for s in screens if s.get('configuration_id')}
    screens=list(unique.values())
    state.control('remaining_search',pending)
    from .speculation_search import comparison, final_controls, baseline_quality_screens, comparison_due
    # The user-enabled comparison has a shared deadline. Resume it before
    # unrelated baseline work can consume the time reserved for its experiments.
    prioritized_comparison=comparison_due(config,state)
    if prioritized_comparison:screens.extend(comparison(config,state,collector,screens))
    regular_and_long(config,state,collector,baseline_quality_screens(state,screens))
    if not prioritized_comparison:screens.extend(comparison(config,state,collector,screens))
    # Higher-bit upgrades and dense/MoE comparisons follow the measured scores.
    scores={s['configuration_id']:s for s in Report(config,state).summarize()['configurations']}
    extra=[]
    for candidate in config['download_weights']:
        if candidate['priority']!=2:continue
        family=candidate['id'].split('-q')[0]
        near=any(s.get('model','').split('-q')[0]==family and scores.get(s.get('configuration_id'),{}).get('regular',{}).get('passed',0)>=24 for s in screens)
        comparison=candidate['id'] in ('ornith10-35b-iq2','qwen35-35ba3b-iq3','qwen38-27b-iq3')
        if not near and not comparison:
            state.entity('selection',candidate['id'],{'selected':False,'reason':'higher_bit_upgrade_not_indicated_by_current_quality_scores','candidate':candidate});continue
        model=acquire_model(config,state,candidate)
        if model and 'upstream-llama-nightly' in runtimes:extra.extend(memory_screen(config,state,collector,runtimes['upstream-llama-nightly'],model))
    if extra:regular_and_long(config,state,collector,extra);screens.extend(extra)
    from .tuning import tune
    tuned=tune(config,state,collector,baseline_quality_screens(state,screens))
    if tuned:
        screens=list({s['configuration_id']:s for s in screens+tuned if s.get('configuration_id')}.values())
    finalists=sorted(quality_qualified(config,state,screens),key=screen_latency)[:config['limits']['maximum_final_configurations']]
    state.control('finalist_configuration_ids',[s['configuration_id'] for s in finalists])
    final_controls(config,state,collector,finalists,screens)
    for candidate in finalists:final_samples(config,state,collector,candidate)
    from .handoff import demo_and_prepare
    for candidate in finalists:demo_and_prepare(config,state,collector,candidate)
    summary=Report(config,state).write()
    unfinished=sorted({job for s in screens for job in pending_for_stage(state,s.get('configuration_id'))})
    state.control('unfinished_current_measurement_jobs',unfinished)
    if not pending and not unfinished:
        state.control('search_covered',True)
        state.control('search_completed_at',state.clock())
    return {'status':'pending_source_review' if pending else 'current_measurements_pending' if unfinished else 'covered_search_complete','pending_engines':pending,'unfinished_jobs':unfinished,'summary':summary}
