"""Bounded, resumable MTP/DFlash comparison inside the original tuning budget."""
from pathlib import Path
from contextlib import contextmanager
import json

from .config import atomic_json, digest, file_hash
from .native_speculation import DRAFT_PINS, TARGET, contract_hash, real_drafting, supports
from .report import Report
from .search_core import checkpoint_model, regular_and_long, screen_latency

PLAN_NAME = 'speculation-comparison-plan.json'

class PhaseDeadline(Exception):
    """A local phase deadline must not be mistaken for model/stream failure."""

@contextmanager
def phase_budget(state,deadline):
    original=state.check_budget
    had_override='check_budget' in state.__dict__
    override=state.__dict__.get('check_budget')
    def check_budget(estimated_seconds=0,reserve_report=True):
        original(estimated_seconds,reserve_report=reserve_report)
        if state.clock()+estimated_seconds>deadline:raise PhaseDeadline('speculation_tuning_phase_budget_exhausted')
    state.check_budget=check_budget
    try:yield
    finally:
        if had_override:state.check_budget=override
        else:del state.check_budget

def checkpoint_screens(state,identifiers):
    if not identifiers:return []
    slots=','.join('?' for _ in identifiers)
    rows=state.db.execute("SELECT value FROM controls WHERE key LIKE 'planned_screen:%' AND "
        "json_extract(value,'$.configuration_id') IN ("+slots+")",identifiers)
    found={}
    for row in rows:
        value=json.loads(row[0])
        found[value['configuration_id']]=value
    return [found[x] for x in identifiers if x in found]

def baseline_quality_screens(state,screens):
    phase=state.get_control('speculation_comparison_phase') or {}
    comparison_ids=set(phase.get('screens',[]))
    return [s for s in screens if s.get('configuration_id') not in comparison_ids
            and s.get('settings',{}).get('speculation','off') not in DRAFT_PINS]

def make_plan(config):
    return dict(schema_version=1,target_model_id=TARGET,engine='upstream-llama-nightly',
        cache_pairs=[['q8_0','q8_0'],['q4_0','q4_0']],draft_widths={'draft-mtp':2,'draft-dflash':7},
        draft_cache=['q8_0','q8_0'],maximum_new_configurations=6,
        contract_sha256=contract_hash(),draft_pins=DRAFT_PINS,
        mtp_head_accounting='planned_native_mtp_heads_of_existing_target;all_bytes_in_weight_ledger',
        dflash_accounting='one_new_discovery_weight_variant',
        fresh_action_regression_limit_fraction=config['tuning']['speculation_cold_p95_regression_limit_fraction'])

def enabled_plan(config,state):
    enabled = state.get_control('speculation_comparison_enabled')
    if not enabled or enabled.get('enabled') is not True: return None
    path = Path(config['paths']['artifacts'])/PLAN_NAME
    if file_hash(path) != enabled.get('plan_sha256'): raise RuntimeError('speculation_plan_changed_requires_review')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan != make_plan(config): raise ValueError('unexpected_speculation_comparison_plan')
    return plan

def control_timings_complete(config,control):
    def full_fresh(row):
        evidence=row.get('token_evidence',{})
        count=evidence.get('expected_prompt_tokens')
        return type(count) is int and 61440-config['measurement']['prompt_target_absolute_tolerance_tokens']<=count<=61440 \
            and evidence.get('effective_context_tokens')==65536 and evidence.get('context_shift') is False \
            and evidence.get('truncated') is False and evidence.get('tokenization_verified') is True \
            and evidence.get('cache_isolation_verified') is True
    rows=control.get('timing',[])
    count=config['measurement']['fresh_repetitions_screen']
    return len(rows)==count and {r.get('replicate') for r in rows}==set(range(count)) and all(
        r.get('kind')=='timing' and r.get('mode')=='fresh' and full_fresh(r)
        and r.get('valid') is True and r.get('passed') is True
        and r.get('metrics',{}).get('first_action_valid') is True for r in rows)

def matched_control(config,state,collector,control):
    """Measure a capacity-valid comparator without forgiving its coding failures."""
    if control.get('capacity_qualified') is not True:return control
    if not control_timings_complete(config,control):
        from .search_core import make_adapter,close_adapter
        from .scheduler import measured_job
        from .measurement import latency
        model=checkpoint_model(state,control['model'],config['installed_model_candidates'])
        adapter=None;rows=[]
        try:
            adapter,actual=make_adapter(config,state,collector,control['runtime'],model,control['settings'])
            if actual!=control['configuration_id']:raise RuntimeError('matched_control_identity_changed')
            for rep in range(config['measurement']['fresh_repetitions_screen']):
                for retry in range(config['limits']['retry_transient_attempts']+1):
                    row=measured_job(config,state,adapter,collector,actual,'timing',42+rep,'fresh',rep,61440,
                        lambda cb,r=rep:latency(config,state,adapter,actual,r,on_packed=cb))
                    if row.get('valid') or row.get('reason')!='foreign_workload_overlap':break
                rows.append(row)
        finally:
            if adapter:close_adapter(adapter)
        control={**control,'timing':rows}
    control={**control,'comparison_control_ready':control_timings_complete(config,control)}
    control['eligible']=control['comparison_control_ready'] and control.get('exploration_smoke_passed') is True
    # These are diagnostic controls, not smoke-qualified candidates. The arm
    # must independently pass every ordinary tuning and qualification gate.
    if control.get('exploration_smoke_passed') is not True:
        control.update(eligible=False,reason='matched_control_coding_smokes_failed;diagnostic_timings_only')
    elif control['eligible']:control['reason']='screen_passed'
    state.control(control['planned_screen_key'],control)
    state.entity('screen',digest([control['engine'],control['model'],control['settings']]),control)
    return control

def comparison_due(config,state):
    """An unfinished enabled comparison gets the remaining shared tuning time."""
    if enabled_plan(config,state) is None:return False
    phase=state.get_control('speculation_comparison_phase') or {}
    parent=state.get_control('tuning_phase') or {}
    if state.clock()>=parent.get('deadline',float('inf')):return False
    expected=2*len(make_plan(config)['cache_pairs'])
    return phase.get('policy_version')!=2 or len(phase.get('arm_outcomes',{}))<expected or phase.get('status') in (
        'screening','quality','qualification_budget_exhausted')

def comparison(config,state,collector,screens):
    plan = enabled_plan(config,state)
    if plan is None: return []
    from .search import run_screen
    from .acquisition import acquire_gguf
    anchors = [s for s in screens if s.get('model')==TARGET and s.get('engine')==plan['engine']
               and s.get('capacity_qualified') is True and s.get('settings',{}).get('speculation')=='off']
    if not anchors: raise RuntimeError('speculation_target_baseline_not_available')
    anchor = min(anchors,key=screen_latency)
    parent = state.get_control('tuning_phase')
    if parent is None:
        now = state.clock()
        parent = dict(started=now,deadline=min(now+config['phase_budget_hours']['tuning_engine_comparisons']*3600,
            state.run['deadline']-7200),selected_families=None,status='active')
        state.control('tuning_phase',parent)
    if parent.get('selected_families') is not None and TARGET not in parent['selected_families']:
        raise RuntimeError('existing_three_family_tuning_limit_excludes_speculation_target')
    key = 'speculation_comparison:'+digest(plan)
    phase = state.get_control(key)
    if phase is None:
        phase = dict(started=state.clock(),deadline=parent['deadline'],status='screening',
                     plan_sha256=file_hash(Path(config['paths']['artifacts'])/PLAN_NAME),screens=[],decisions=[])
    if phase['deadline'] != parent['deadline']: raise RuntimeError('speculation_tuning_deadline_changed')
    phase['policy_version']=2
    phase.setdefault('arm_outcomes',{})
    outcomes = checkpoint_screens(state,phase.get('screens',[]))
    def save():
        phase['screens'] = list(dict.fromkeys(phase.get('screens',[])+
            [s['configuration_id'] for s in outcomes if s.get('configuration_id')]))
        state.control(key,phase)
        state.control('speculation_comparison_phase',phase)
        atomic_json(Path(config['paths']['artifacts'])/'speculation-comparison-status.json',phase)
    def available(estimate=0):
        state.check_budget(estimate)
        if state.clock()+estimate > phase['deadline']:
            phase['status']='tuning_budget_exhausted';save();return False
        return True
    def decision(mode,cache,reason):
        item=dict(mode=mode,cache=cache,reason=reason)
        if item not in phase['decisions']: phase['decisions'].append(item)
        save()
    def append(result):
        for index,old in enumerate(outcomes):
            if old.get('configuration_id')==result.get('configuration_id'):
                outcomes[index]=result;break
        else:outcomes.append(result)
        save()
    def limited(operation):
        try:
            with phase_budget(state,phase['deadline']):return operation()
        except PhaseDeadline:
            phase['status']='tuning_budget_exhausted';save();return None
    estimate=config['limits']['server_start_timeout_seconds']+config['limits']['per_64k_request_timeout_seconds']
    model=checkpoint_model(state,TARGET,config['installed_model_candidates'])
    runtime=anchor['runtime']
    metadata=json.loads(state.db.execute("SELECT data FROM entities WHERE kind='configuration' AND id=?",
        (anchor['configuration_id'],)).fetchone()[0])
    help_text=Path(metadata['help_log']).read_text(encoding='utf-8-sig')
    phase['status']='screening';save()
    for cache in plan['cache_pairs']:
        if not available(estimate): return outcomes
        settings={**anchor['settings'],'cache_k':cache[0],'cache_v':cache[1]}
        phase['active_arm']=dict(mode='off',cache=cache);save()
        control=limited(lambda:run_screen(config,state,collector,runtime,model,settings=settings,tuning=True))
        if control is None:return outcomes
        control=limited(lambda:matched_control(config,state,collector,control))
        if control is None:return outcomes
        append(control)
        if control.get('comparison_control_ready') is not True:
            for mode in plan['draft_widths']:
                phase['arm_outcomes'][mode+':'+','.join(cache)]=dict(status='blocked',reason='matched_control_capacity_or_fresh_action_failed')
                decision(mode,cache,'matched_control_capacity_or_fresh_action_failed')
            continue
        if control.get('exploration_smoke_passed') is not True:
            decision('off',cache,'control_smoke_failures_retained;arm_requires_its_own_six_smokes')
        for mode,width in plan['draft_widths'].items():
            if not available(estimate): return outcomes
            if not supports(help_text,mode):
                phase['arm_outcomes'][mode+':'+','.join(cache)]=dict(status='unavailable',reason='unsupported_mode_in_pinned_native_help')
                decision(mode,cache,'unsupported_mode_in_pinned_native_help');continue
            pin=DRAFT_PINS[mode]
            if mode=='draft-dflash':
                selected=state.get_control('discovered_weight_variants') or []
                if pin['id'] not in selected:
                    if len(selected)>=config['limits']['maximum_new_discovery_weight_variants']:
                        phase['arm_outcomes'][mode+':'+','.join(cache)]=dict(status='unavailable',reason='discovery_variant_budget_exhausted')
                        decision(mode,cache,'discovery_variant_budget_exhausted');continue
                    state.control('discovered_weight_variants',selected+[pin['id']])
            else:
                state.control('planned_mtp_head_artifact',pin)
            acquired=limited(lambda:acquire_gguf(config,state,pin))
            if acquired is None:return outcomes
            draft={**pin,'path':acquired['path']}
            requested={**settings,'speculation':mode,'draft_model':draft,'draft_n_max':width,
                'draft_gpu_layers':99,'draft_cache_k':plan['draft_cache'][0],'draft_cache_v':plan['draft_cache'][1],
                'speculation_contract_sha256':plan['contract_sha256'],
                'speculation_baseline_configuration_id':control['configuration_id']}
            phase['active_arm']=dict(mode=mode,cache=cache);save()
            result=limited(lambda:run_screen(config,state,collector,runtime,model,settings=requested,tuning=True))
            if result is None:return outcomes
            if result.get('eligible') is True and not real_drafting(result):
                result.update(eligible=False,capacity_qualified=False,reason='real_speculative_drafting_not_observed')
                state.control(result['planned_screen_key'],result)
            if result.get('eligible') is True and screen_latency(result)>screen_latency(control)*(1+plan['fresh_action_regression_limit_fraction']):
                result.update(eligible=False,capacity_qualified=False,reason='exploratory_fresh_action_regression_over_limit')
                state.control(result['planned_screen_key'],result)
            if result.get('configuration_id'):
                state.entity('screen',digest([result['engine'],result['model'],result['settings']]),result)
            phase['arm_outcomes'][mode+':'+','.join(cache)]=dict(
                status='screen_passed' if result.get('eligible') is True else 'screen_failed',
                configuration_id=result.get('configuration_id'),reason=result.get('reason'),
                capacity_qualified=result.get('capacity_qualified'),
                exploration_smoke_passed=result.get('exploration_smoke_passed'))
            append(result)
    # Qualify one matched control and each algorithm's best screened arm.
    # Quality is measured, never inferred from target/draft equivalence claims.
    chosen=[]
    for mode in ('off','draft-mtp','draft-dflash'):
        pool=[s for s in outcomes if s.get('eligible') is True and s.get('settings',{}).get('speculation')==mode]
        if pool: chosen.append(min(pool,key=screen_latency))
    phase['status']='quality';phase['qualification_ids']=[s['configuration_id'] for s in chosen];save()
    if not available(): return outcomes
    # Give the common qualifier the original shared deadline as an extra check.
    regular_and_long(config,state,collector,chosen,phase_deadline=phase['deadline'])
    exit_record=state.get_control('speculation_quality_phase_budget_exit')
    phase['status']='qualification_budget_exhausted' if exit_record and exit_record.get('deadline')==phase['deadline'] else (
        'screening_and_quality_complete' if any(s['settings']['speculation'] in DRAFT_PINS for s in chosen)
        else 'no_speculative_arm_passed_screening');save()
    Report(config,state).write()
    return outcomes

def final_controls(config,state,collector,finalists,screens):
    """Twenty matched fresh controls enforce the final <=10% regression gate."""
    from .search_core import make_adapter,close_adapter
    from .scheduler import measured_job
    from .measurement import latency
    needed={s['settings']['speculation_baseline_configuration_id'] for s in finalists
        if s.get('settings',{}).get('speculation') in DRAFT_PINS}
    if not needed:return None
    by_id={s.get('configuration_id'):s for s in screens}
    for identifier in sorted(needed):
        control=by_id.get(identifier)
        if not control: raise RuntimeError('matched_speculation_control_missing')
        model=checkpoint_model(state,control['model'],config['installed_model_candidates']);adapter=None
        try:
            adapter,actual=make_adapter(config,state,collector,control['runtime'],model,control['settings'])
            if actual!=identifier: raise RuntimeError('matched_control_identity_changed')
            for i in range(config['measurement']['final_fresh_repetitions']):
                for retry in range(config['limits']['retry_transient_attempts']+1):
                    row=measured_job(config,state,adapter,collector,identifier,'timing',1042+i,'fresh',1000+i,61440,
                        lambda cb,r=i:latency(config,state,adapter,identifier,1000+r,final=True,on_packed=cb))
                    if row.get('valid') or row.get('reason')!='foreign_workload_overlap':break
        finally:
            if adapter:close_adapter(adapter)
    return Report(config,state).write()
