"""Sequential engine-independent qualification and bounded configuration search."""
from pathlib import Path
import json
import time
from .config import atomic_json, digest, file_hash
from .fixtures import load_fixture
from .measurement import capacity, latency, throughput, cached_prefix
from .quality import request_for
from .schema import tool_definitions, validate_tool
from .scheduler import make_native, register_configuration, measured_job, coding_job
from .report import Report, context_reasons
from .runtime_catalog import setup_budget, compatible


def close_adapter(adapter):
    try:adapter.unload_owned()
    finally:
        helper=getattr(adapter,'owned_tokenizer_helper',None)
        if helper:helper.unload_owned()


def checkpoint_model(state,model_id,installed):
    found=next((m for m in installed if m['id']==model_id),None)
    row=state.db.execute("SELECT data FROM entities WHERE kind='model' AND id=?",(model_id,)).fetchone()
    if row:found=json.loads(row['data'])
    if found is None:raise RuntimeError('model_registry_entry_missing:'+model_id)
    return found


def make_adapter(config,state,collector,runtime,model,settings=None):
    managed_handle=Path(config['paths']['state'])/'owned-lm-studio-instance.json'
    if managed_handle.exists() and not json.loads(managed_handle.read_text(encoding='utf-8')).get('unloaded'):
        from .recovery import recover_lm_studio
        recovered=recover_lm_studio(config,state)
        state.control('managed_load_preflight',recovered)
        if recovered.get('status') not in ('recovered','already_unloaded','no_owned_orphan'):
            raise RuntimeError('owned_managed_load_unresolved: no additional GPU workload permitted')
    engine=runtime['engine']
    if runtime['kind']=='native':
        return make_native(config,state,collector,engine,model,settings,runtime.get('binary_path'))
    helper=None;adapter=None
    try:
        if runtime['kind']=='container':
            from .adapters.container import ContainerAdapter
            adapter=ContainerAdapter(config,state,engine,model,image_digest=runtime['image_digest'],settings=settings)
            state.control('last_container_recovery:'+engine,adapter.recover_owned())
        elif runtime['kind']=='tabby':
            from .adapters.tabby import TabbyAdapter
            adapter=TabbyAdapter(config,state,model,runtime,settings=settings)
        elif runtime['kind'] in ('ollama','lm-studio'):
            from .scheduler import model_identity
            from .tokenizer_helper import GGUFTokenizerHelper
            model=model_identity(config,state,model)
            helper=GGUFTokenizerHelper(config,state,model)
            helper.launch()
            if runtime['kind']=='ollama':
                from .adapters.ollama import OllamaAdapter
                adapter=OllamaAdapter(config,state,model,settings=settings,exact_tokenizer=helper)
            else:
                from .adapters.lms import LMStudioAdapter
                adapter=LMStudioAdapter(config,state,model,settings=settings,exact_tokenizer=helper)
            adapter.engine=engine
        else:raise RuntimeError('unknown_runtime_kind')
        adapter.collector=collector
        adapter.discover_capabilities();adapter.launch()
        if helper:
            # Verify actual templated input counts before allowing the helper
            # to be treated as valid for this managed engine.
            tools=tool_definitions(include_answer=False)
            messages=[{'role':'system','content':'native-tools-verification:'+state.run['id']},
                      {'role':'user','content':'Call read_file with path src/cache.py. This is a synthetic tool test.'}]
            response=adapter.stream_chat(request_for(config,messages,tools,42,256),
                log_prefix=Path(config['paths']['logs'])/(engine+'-template-verification-'+str(time.time_ns())),validator=validate_tool,mode='exploratory')
            if not response['valid_stream']:raise RuntimeError('managed_native_tool_template_probe_failed')
            helper.verify_against(engine,messages,tools,response['metrics'].get('prompt_tokens'))
            helper_metadata=helper.metadata()
            adapter._metadata['tokenizer_helper_sha256']=digest({'binary_sha256':helper_metadata.get('binary_sha256'),'adjacent_dlls':helper_metadata.get('adjacent_dlls'),'model_sha256':helper.model_sha256})
            adapter.owned_tokenizer_helper=helper
        identifier=register_configuration(config,state,adapter)
        return adapter,identifier
    except BaseException:
        if adapter:adapter.unload_owned()
        if helper:helper.unload_owned()
        raise


def unavailable(config,state,engine,model,reason,settings=None,kind='engine_probe',evidence=None):
    """Unavailability is a recorded setup outcome, never a quality/latency loss."""
    result={'kind':kind,'engine':engine,'model':model.get('id'),'passed':False,'valid':False,
            'eligible':False,'reason':reason,'settings':settings,'evidence':evidence}
    key={'engine':digest({'engine':engine,'runtime':evidence}),'model':model.get('sha256') or model.get('revision') or digest(model),
         'effective_settings':settings or {'not_loaded':True},'profile':config['default_sampling'],
         'fixture_prompt_hash':digest({'setup':kind,'reason':reason}),'tokenizer':'not_verified','seed':0,'mode':'fresh','replicate':0}
    job=state.enqueue(key)
    if state.db.execute('SELECT status FROM jobs WHERE id=?',(job,)).fetchone()[0]=='pending':
        state.finish(state.begin(job),'skipped',result,reason=reason)
    state.entity('engine_outcome',digest([engine,model.get('id'),settings,reason]),result)
    return result


def gemma_marker_control(config,runtime,model,settings,small,capacities):
    """Only the installed default Gemma baseline may retain failed-marker diagnostics."""
    installed=next((item for item in config['installed_model_candidates']
        if item['id']=='gemma4-12b-qat-q4'),None)
    if settings is not None or runtime.get('kind')!='native' or runtime.get('engine') not in ('bundled-llama','upstream-llama-nightly'):
        return False
    if not installed or model.get('id')!=installed['id'] or Path(model.get('path','')).resolve()!=Path(installed['path']).resolve():
        return False
    if not small or small.get('passed') is not True or small.get('valid') is not True:
        return False
    seeds=config['grading']['capacity_marker_seeds']
    if len(capacities)!=len(seeds) or {item.get('seed') for item in capacities}!=set(seeds):
        return False
    failed=False
    for item in capacities:
        metrics=item.get('metrics',{})
        if item.get('kind')!='capacity' or item.get('target_tokens')!=config['measurement']['full_prompt_tokens'] or item.get('valid') is not True:
            return False
        if metrics.get('stream_complete') is not True or type(metrics.get('invalid_tool_calls')) is not int or metrics['invalid_tool_calls']!=0:
            return False
        # A clean bounded output exhaustion is a functional missing-answer failure.
        if metrics.get('completion_reason') not in ('stop','tool_calls','length') or context_reasons(item,config,'fresh'):
            return False
        if item.get('passed') is True and item.get('reason')=='passed':
            continue
        if item.get('passed') is not False or item.get('reason')!='capacity_marker_or_evidence_failure':
            return False
        failed=True
    return failed


def gemma_control_timings_complete(config,result):
    rows=result.get('timing',[])
    count=config['measurement']['fresh_repetitions_screen']
    return len(rows)==count and {row.get('replicate') for row in rows}==set(range(count)) and all(
        row.get('kind')=='timing' and row.get('mode')=='fresh' and row.get('final_validation') is not True for row in rows)


def screen(config,state,collector,runtime,model,settings=None,tuning=False):
    engine=runtime['engine'];adapter=None
    try:
        setup=setup_budget(config,state,engine)
        adapter,identifier=make_adapter(config,state,collector,runtime,model,settings)
        caps=[];timings=[];smoke=[]
        small=measured_job(config,state,adapter,collector,identifier,'capacity',17,'fresh',0,16384,
            lambda cb:capacity(config,state,adapter,identifier,17,16384,on_packed=cb))
        if not small.get('passed'):
            result={'engine':engine,'model':model['id'],'configuration_id':identifier,'eligible':False,'reason':small.get('reason'),'capacity':[small],'settings':adapter.requested,'runtime':runtime}
            state.entity('screen',digest([engine,model['id'],adapter.requested]),result)
            return result
        compatible(state,engine,{'configuration_id':identifier,'runtime_feasibility_only':True,'streamed_native_tool_at_16k':True,'effective_configuration':adapter.effective})
        for seed in config['grading']['capacity_marker_seeds']:
            caps.append(measured_job(config,state,adapter,collector,identifier,'capacity',seed,'fresh',0,61440,
                lambda cb,s=seed:capacity(config,state,adapter,identifier,s,on_packed=cb)))
        capacity_pass=all(c.get('passed') for c in caps)
        diagnostic=not tuning and gemma_marker_control(config,runtime,model,settings,small,caps)
        if capacity_pass:
            for job in config['grading']['smoke_jobs']:
                smoke.append(coding_job(config,state,adapter,collector,identifier,job['fixture'],job['seed']))
        smoke_pass=bool(smoke) and all(s.get('passed') for s in smoke)
        if (capacity_pass and (not tuning or smoke_pass)) or diagnostic:
            for rep in range(config['measurement']['fresh_repetitions_screen']):
                timings.append(measured_job(config,state,adapter,collector,identifier,'timing',rep+42,'fresh',rep,61440,
                    lambda cb,r=rep:dict(latency(config,state,adapter,identifier,r,on_packed=cb),diagnostic_gemma_control=diagnostic)))
        speed=None
        if capacity_pass and timings and all(t.get('passed') for t in timings):
            for target in config['measurement']['smaller_prompt_targets']+[61440]:
                sample=measured_job(config,state,adapter,collector,identifier,'throughput',42,'fresh',0,target,
                    lambda cb,t=target:throughput(config,adapter,identifier,target=t,on_packed=cb))
                if target==61440:speed=sample
                generated=sample.get('metrics',{}).get('generated_tokens')
                if type(generated) is int and generated<64:
                    sample=measured_job(config,state,adapter,collector,identifier,'throughput',42,'fresh',1,target,
                        lambda cb,t=target:throughput(config,adapter,identifier,target=t,retry_longer=True,on_packed=cb))
                    if target==61440:speed=sample
            from .common_byte import common_byte
            measured_job(config,state,adapter,collector,identifier,'common_byte',42,'fresh',0,None,
                lambda cb:common_byte(config,state,adapter,identifier,on_packed=cb))
        eligible=capacity_pass and (not tuning or smoke_pass) and bool(timings) and all(t.get('passed') for t in timings)
        result={'engine':engine,'model':model['id'],'configuration_id':identifier,'eligible':eligible,'capacity_qualified':capacity_pass,
            'diagnostic_gemma_control':diagnostic,'small_capacity':small,
            'exploration_smoke_passed':smoke_pass,'reason':'screen_passed' if eligible else 'capacity_or_action_or_smoke_gate_failed',
            'capacity':caps,'timing':timings,'throughput':speed,'smoke':smoke,'settings':dict(adapter.requested),'runtime':runtime}
        if capacity_pass and smoke_pass:compatible(state,engine,{'configuration_id':identifier,'native_tool_round_trip':True,'all_capacity_markers':True,'six_smoke_jobs':True})
        state.entity('screen',digest([engine,model['id'],adapter.requested]),result)
        return result
    except (OSError,RuntimeError,ValueError) as error:
        if str(error) in ('stop_after_current','budget_exhausted'):raise
        return unavailable(config,state,engine,model,str(error),settings,evidence=runtime)
    finally:
        if adapter:close_adapter(adapter)
        Report(config,state).write()


def regular_and_long(config,state,collector,screens,phase_deadline=None):
    active=[s for s in screens if s.get('eligible') or s.get('capacity_qualified')]
    failures={s['configuration_id']:{'regular':0,'long':0} for s in active}
    for group,ids,limit in (('regular',config['grading']['regular_fixture_ids'],config['grading']['early_elimination_regular_failures']),
                           ('long',config['grading']['long_fixture_ids'],config['grading']['early_elimination_long_failures'])):
        for seed in config['grading']['seeds']:
            for fixture in ids:
                for candidate in active:
                    identifier=candidate['configuration_id']
                    # Once either group is mathematically impossible, additional
                    # quality work cannot qualify this exact configuration.
                    if failures[identifier]['regular']>=config['grading']['early_elimination_regular_failures'] or failures[identifier]['long']>=config['grading']['early_elimination_long_failures']:continue
                    if phase_deadline is not None and state.clock()+config['limits']['per_task_timeout_seconds']+config['limits']['server_start_timeout_seconds']>phase_deadline:
                        state.control('speculation_quality_phase_budget_exit',{'deadline':phase_deadline,'stopped_at':state.clock()})
                        return Report(config,state).write()
                    model=checkpoint_model(state,candidate['model'],config['installed_model_candidates'])
                    adapter=None
                    try:
                        adapter,actual=make_adapter(config,state,collector,candidate['runtime'],model,candidate['settings'])
                        if actual!=identifier:raise RuntimeError('configuration_identity_changed')
                        for retry in range(config['limits']['retry_transient_attempts']+1):
                            result=coding_job(config,state,adapter,collector,identifier,fixture,seed)
                            if result.get('valid'):break
                        if result.get('valid') and not result.get('passed'):failures[identifier][group]+=1
                    except (OSError,RuntimeError,ValueError) as error:
                        if str(error) in ('stop_after_current','budget_exhausted'):raise
                        unavailable(config,state,candidate['engine'],model,'qualification_infrastructure_unavailable:'+str(error),candidate['settings'],evidence=candidate['runtime'])
                    finally:
                        if adapter:close_adapter(adapter)
                    Report(config,state).write()
    return Report(config,state).write()


def quality_qualified(config,state,screens):
    scores={s['configuration_id']:s for s in Report(config,state).summarize()['configurations']}
    return [s for s in screens if s.get('configuration_id') in scores and all(scores[s['configuration_id']]['gates'][gate] for gate in
        ('regular_coverage','regular_quality','long_coverage','long_quality','capacity_all_nine_values'))]


def screen_latency(candidate):
    values=[r.get('metrics',{}).get('first_action_seconds') for r in candidate.get('timing',[]) if r.get('passed')]
    values=[v for v in values if type(v) in (int,float)]
    return max(values) if values else float('inf')


def final_samples(config,state,collector,candidate):
    adapter=None;model=checkpoint_model(state,candidate['model'],config['installed_model_candidates'])
    try:
        adapter,identifier=make_adapter(config,state,collector,candidate['runtime'],model,candidate['settings'])
        for i in range(config['measurement']['final_fresh_repetitions']):
            for retry in range(config['limits']['retry_transient_attempts']+1):
                row=measured_job(config,state,adapter,collector,identifier,'timing',1042+i,'fresh',1000+i,61440,
                    lambda cb,r=i:latency(config,state,adapter,identifier,1000+r,final=True,on_packed=cb))
                if row.get('valid') or row.get('reason')!='foreign_workload_overlap':break
        adapter.fresh_restart();prefix=cached_prefix(config,adapter,state,identifier)
        prime=latency(config,state,adapter,identifier,-1,mode='cached',stable_prefix=prefix)
        metrics=prime['metrics']
        prefix['primed_verified']=metrics.get('stream_complete') is True and metrics.get('first_action_valid') is True and type(metrics.get('prompt_tokens')) is int and 61376<=metrics['prompt_tokens']<=61440
        prefix['owned_pid']=adapter.process.pid
        atomic_json(Path(config['paths']['logs'])/('cache-prime-'+identifier+'-'+str(time.time_ns())+'.json'),{'prefix':{k:v for k,v in prefix.items() if k!='prefix'},'response':prime})
        for i in range(config['measurement']['final_cached_repetitions']):
            for retry in range(config['limits']['retry_transient_attempts']+1):
                row=measured_job(config,state,adapter,collector,identifier,'timing',2042+i,'cached',2000+i,61440,
                    lambda cb,r=i:latency(config,state,adapter,identifier,2000+r,mode='cached',final=True,stable_prefix=prefix,on_packed=cb))
                if row.get('valid') or row.get('reason')!='foreign_workload_overlap':break
    finally:
        if adapter:close_adapter(adapter)
    return Report(config,state).write()
