"""Lifecycle for a verified local endpoint; inference remains in its adapter."""
from pathlib import Path
from contextlib import nullcontext
import time
from .config import atomic_json
from .search_core import close_adapter


def serve(config,state,collector,plan_path,sleep=time.sleep):
    from .handoff import launch_plan
    adapter=None
    status=None
    outcome='stopped'
    failure_reason=None
    phase='launch'
    try:
        startup=state.startup_io() if hasattr(state,'startup_io') else nullcontext()
        with startup:
            adapter,identifier=launch_plan(config,state,collector,plan_path)
            phase='interface'
            interface=adapter.handoff_launch_evidence['interface']
            endpoint='http://127.0.0.1:'+str(adapter.http.port)
            if interface['endpoint']!=endpoint:raise RuntimeError('handoff_endpoint_interface_disagrees')
        status={'status':'serving','configuration_id':identifier,'engine':adapter.engine,
                'serving_session_id':getattr(state,'session_id',None),
                'endpoint':endpoint,'chat_path':interface['chat_path'],'models_path':interface['models_path'],
                'wire_protocol':interface['wire_protocol'],'plan':str(Path(plan_path).resolve()),
                'effective_settings':adapter.effective,'ownership':getattr(adapter,'saved',None),
                'started':time.time(),'sampler':config['default_sampling'],
                'credential':'process-local LM_BENCH_TOKEN required' if adapter.engine=='lm-studio' else 'none',
                'served_model':interface['model'],
                'agent_demo_command':['.venv/Scripts/python.exe','-m','localbench','agent-demo','--configuration',identifier]}
        phase='initial_health'
        readiness=state.startup_io() if hasattr(state,'startup_io') else nullcontext()
        with readiness:
            if not adapter.health():raise RuntimeError('handoff_endpoint_not_ready')
            if hasattr(state,'finish_startup'):state.finish_startup()
        state.control('live_agent_endpoint',status)
        atomic_json(Path(config['paths']['artifacts'])/'agent-endpoint.json',status)
        phase='health_monitor'
        while not state.run['stop_requested']:
            sleep(1)
            if state.run['stop_requested']:break
            if not adapter.health():raise RuntimeError('handoff_endpoint_health_failed')
        return {'status':'endpoint_stopped','configuration_id':identifier}
    except BaseException as error:
        outcome='stopped' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed'
        failure_reason='controller_shutdown' if outcome=='stopped' else 'endpoint_'+phase+'_failed'
        raise
    finally:
        cleaned=False
        try:
            if adapter:
                original=getattr(adapter,'process',None)
                close_adapter(adapter)
                # Native Popen handles expose Windows PIDs; container handles
                # expose string identities and verify removal in their adapter.
                if original is not None and type(getattr(original,'pid',None)) is int and original.poll() is None:
                    raise RuntimeError('owned_endpoint_stop_unverified')
                if getattr(adapter,'instance_id',None) is not None:
                    raise RuntimeError('owned_endpoint_instance_unload_unverified')
                cleaned=True
        except BaseException:
            outcome='cleanup_failed'
            failure_reason='owned_endpoint_cleanup_failed'
            raise
        finally:
            # A launch rejected before returning an adapter owns no previous
            # endpoint descriptor and must not mark another handle stopped.
            if status is not None:
                final={**status,'status':outcome,'stopped_at':time.time() if cleaned else None,
                       'cleanup_verified':cleaned,'failure_reason':failure_reason}
                state.control('live_agent_endpoint',final)
                atomic_json(Path(config['paths']['artifacts'])/'agent-endpoint.json',final)
