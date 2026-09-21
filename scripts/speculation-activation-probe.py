"""Owned sequential launch/tool/draft checks; never a 64K qualification claim."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from localbench.config import atomic_json,load
from localbench.state import State,gpu_lock
from localbench.native_speculation import DRAFT_PINS,TARGET,contract_hash
from localbench.acquisition import acquire_gguf
from localbench.doctor import doctor
from localbench.recovery import recover_processes
from localbench.resources import ResourceCollector
from localbench.search import runtime
from localbench.search_core import make_adapter,close_adapter,checkpoint_model
from localbench.fixtures import create_workspace,load_fixture
from localbench.tools import ToolExecutor
from localbench.quality import request_for
from localbench.schema import tool_definitions,validate_tool

def main():
    cfg=load();state=State(cfg);collector=None;results=[]
    status=ROOT/'artifacts/speculation-activation-status.json'
    def save(stage):
        atomic_json(status,dict(status=stage,run_id=state.run['id'],timestamp_utc=datetime.now(timezone.utc).isoformat(),
            qualification=False,results=results))
    try:
        with gpu_lock(cfg):
            enabled=state.get_control('speculation_comparison_enabled')
            if not enabled or enabled.get('enabled') is not True:raise RuntimeError('tested_speculation_enable_required')
            state.check_budget(1800)
            recovery=recover_processes(cfg,state)
            if any(x.get('status') in ('refused','pending_load','pending_auth','pending_descendants') for x in recovery):
                raise RuntimeError('owned_recovery_unresolved')
            inspected=doctor(cfg,hash_weights=False)
            if not inspected['disk_headroom_ok']:raise RuntimeError('disk_headroom_failed')
            host=json.loads((ROOT/'artifacts/host.json').read_text())
            try:loaded=json.loads(host['commands']['lms_loaded']['output'])
            except (KeyError,ValueError):raise RuntimeError('loaded_model_inventory_unavailable') from None
            if loaded:raise RuntimeError('unrelated_loaded_models')
            collector=ResourceCollector(cfg,state).start()
            model=checkpoint_model(state,TARGET,cfg['installed_model_candidates']);rt=runtime(cfg,state,'upstream-llama-nightly')
            baseline=json.loads(state.db.execute("SELECT data FROM entities WHERE kind='configuration' AND id=?",
                ('05c146133e492ac2e0df5f3637ecee19c2b9d797d579431a2321d54c5f91e46f',)).fetchone()[0])
            base=baseline['requested_settings']
            drafts={}
            for mode,pin in DRAFT_PINS.items():
                if mode=='draft-dflash':
                    selected=state.get_control('discovered_weight_variants') or []
                    if pin['id'] not in selected:
                        if len(selected)>=cfg['limits']['maximum_new_discovery_weight_variants']:
                            raise RuntimeError('discovery_variant_budget_exhausted')
                        state.control('discovered_weight_variants',selected+[pin['id']])
                else:state.control('planned_mtp_head_artifact',pin)
                save('acquiring_'+mode)
                acquired=acquire_gguf(cfg,state,pin);drafts[mode]={**pin,'path':acquired['path']}
            fixture=load_fixture(cfg,'py01',42);workspace=create_workspace(cfg,fixture,'speculation-activation-'+uuid.uuid4().hex)
            controls={}
            for mode,draft in drafts.items():
                ready=False
                for cache in ('q8_0','q4_0'):
                    adapter=None;started=time.time();save('probing_'+mode+'_'+cache)
                    try:
                        settings={**base,'cache_k':cache,'cache_v':cache}
                        if cache not in controls:
                            control,identifier=make_adapter(cfg,state,collector,rt,model,settings)
                            try:controls[cache]=identifier
                            finally:close_adapter(control)
                        requested={**settings,'speculation':mode,'draft_model':draft,'draft_n_max':2 if mode=='draft-mtp' else 7,
                            'draft_gpu_layers':99,'draft_cache_k':'q8_0','draft_cache_v':'q8_0',
                            'speculation_contract_sha256':contract_hash(),'speculation_baseline_configuration_id':controls[cache]}
                        adapter,identifier=make_adapter(cfg,state,collector,rt,model,requested)
                        adapter.http.timeout=cfg['limits']['per_16k_request_timeout_seconds']
                        executor=ToolExecutor(workspace,fixture,None,adapter.tokenize_text)
                        tools=tool_definitions(include_answer=False)
                        messages=[dict(role='system',content='This is an interface setup check. Immediately use the requested native tool.'),
                            dict(role='user',content='Call read_file once with path src/cache.py. Do not edit files or run tests.')]
                        expected=adapter.tokenize(messages,tools)['count']
                        prefix=ROOT/'.logs'/('speculation-activation-'+mode+'-'+cache+'-'+uuid.uuid4().hex)
                        response=adapter.stream_chat(request_for(cfg,messages,tools,42,1024),
                            log_prefix=prefix,validator=executor.validate,mode='exploratory')
                        calls=response.get('calls',[])
                        permitted=response['valid_stream'] and calls and calls[0]['name']=='read_file' \
                            and calls[0]['arguments']=={'path':'src/cache.py'} and executor.execute('read_file',calls[0]['arguments'])['ok']
                        counts=response['metrics'].get('prompt_tokens')==expected
                        text=[dict(role='user',content='Write 40 small Python functions named item_0 through item_39. Each returns its own integer index. Output the complete code.')]
                        coding=adapter.stream_chat(request_for(cfg,text,[],42,1536),log_prefix=Path(str(prefix)+'-code'),
                            validator=validate_tool,mode='exploratory')
                        drafted=sum(x['metrics'].get('speculation_drafted_tokens',0) for x in (response,coding))
                        ready=bool(permitted and counts and coding['valid_stream'] and drafted>0)
                        result=dict(kind=mode,cache=cache,configuration_id=identifier,passed=ready,qualification=False,
                            native_tool_round_trip=bool(permitted),exact_short_prompt_count=bool(counts),drafted_tokens=drafted,
                            accepted_tokens=sum(x['metrics'].get('speculation_accepted_tokens',0) for x in (response,coding)),
                            effective_context=adapter.effective['effective_context'],settings=adapter.requested,
                            launch_argv=adapter.effective['launch_argv'],evidence_prefix=str(prefix))
                        results.append(result)
                    except (RuntimeError,OSError,ValueError) as error:
                        if str(error) in ('stop_after_current','budget_exhausted'):raise
                        results.append(dict(kind=mode,cache=cache,passed=False,qualification=False,reason=str(error)))
                    finally:
                        if adapter:close_adapter(adapter)
                    results[-1]['resources']=collector.snapshot(started,time.time());save('activation_progress')
                    if ready:break
            state.control('speculation_activation_probe',dict(results=results,qualification=False))
            save('activation_checks_complete')
            print(json.dumps({'activation_ready':{mode:any(r.get('kind')==mode and r.get('passed') is True for r in results) for mode in drafts},
                              'qualification':False,'artifact':str(status)}))
            return 0
    except (RuntimeError,OSError,ValueError) as error:
        save('attention_required');print(json.dumps({'status':'attention_required','reason':str(error)}));return 2
    finally:
        if collector:collector.stop()
        state.close()

if __name__=='__main__':raise SystemExit(main())
