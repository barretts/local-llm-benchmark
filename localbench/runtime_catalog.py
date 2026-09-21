"""Pinned local runtime acquisition and durable compatibility setup budgets."""
from pathlib import Path
import json
import re
from .acquisition import acquire_upstream, _disk_headroom, _runtime_headroom
from .config import atomic_json, digest
from .doctor import command
from .runtime_build import run_owned
import shutil

TAGS={'ik-llama':'ghcr.io/ikawrakow/ik-llama-cpp:cu12-server',
      'vllm':'vllm/vllm-openai:v0.29.0','sglang':'lmsysorg/sglang:v0.5.19-cu129',
      'cuda-access':'nvidia/cuda:12.6.3-base-ubuntu24.04'}


def setup_budget(config,state,engine):
    key='engine_probe_budget:'+engine
    record=state.get_control(key)
    if record is None:
        limit=next(e['probe_minutes'] for e in config['runtime_candidates'] if e['id']==engine) if engine in {e['id'] for e in config['runtime_candidates']} else 15
        first=state.get_control('runtime_first_probe:'+engine)
        first=first if type(first) in (int,float) else state.run['started'] if state.get_control('clock_start_reason')=='runtime_probe:'+engine else state.clock()
        receipt=None
        configurations={row['id']:json.loads(row['data']) for row in state.db.execute("SELECT id,data FROM entities WHERE kind='configuration'")}
        for row in state.db.execute("SELECT result,started FROM attempts WHERE status='passed' ORDER BY started"):
            result=json.loads(row['result'] or '{}')
            observed=configurations.get(result.get('configuration_id'),{})
            if observed.get('engine_id')==engine and result.get('kind')=='capacity' and result.get('passed') and result.get('valid'):
                effective=observed.get('effective_settings',{})
                if effective.get('effective_context')==65536 and effective.get('effective_slots')==1:
                    receipt={'configuration_id':result['configuration_id'],'preserved_runtime_feasibility_probe':True,'attempt_started':row['started']}
                    break
        record={'started':first,'deadline':first+limit*60,'compatible':receipt is not None,'phase':'setup','evidence':receipt}
        state.control(key,record)
    state.check_budget()
    if not record.get('compatible') and state.clock()>=record['deadline']:
        raise RuntimeError('engine_probe_budget_exhausted:'+engine)
    return record


def compatible(state,engine,evidence):
    record=state.get_control('engine_probe_budget:'+engine) or {}
    state.control('engine_probe_budget:'+engine,{**record,'compatible':True,'verified_at':state.clock(),'evidence':evidence})


def select_manifest(payload):
    entries=payload if isinstance(payload,list) else [payload]
    for entry in entries:
        descriptor=entry.get('Descriptor',entry.get('descriptor',{}))
        platform=descriptor.get('platform',{})
        pin=descriptor.get('digest')
        if platform.get('architecture')=='amd64' and platform.get('os')=='linux' and re.fullmatch(r'sha256:[a-f0-9]{64}',str(pin)):
            return pin
    if isinstance(payload,dict):
        for entry in payload.get('manifests',[]):
            platform=entry.get('platform',{})
            if platform.get('architecture')=='amd64' and platform.get('os')=='linux' and re.fullmatch(r'sha256:[a-f0-9]{64}',str(entry.get('digest'))):
                return entry['digest']
    raise RuntimeError('docker_linux_amd64_manifest_not_established')


def acquire_image(config,state,engine):
    """Resolve an immutable linux/amd64 digest before downloading its layers."""
    tag=TAGS[engine]
    record=setup_budget(config,state,engine)
    control='runtime_image:'+engine
    saved=state.get_control(control)
    if saved is None:
        resolved=command(config,engine+'-manifest',[config['installed_tools']['docker'],'manifest','inspect','--verbose',tag],timeout=min(120,max(1,int(record['deadline']-state.clock()))))
        if resolved['exit_code']!=0:raise RuntimeError('image_manifest_unavailable:'+resolved['log'])
        pin=select_manifest(json.loads(resolved['output']))
        repository=tag.rsplit(':',1)[0]
        saved={'engine':engine,'tag':tag,'image_digest':repository+'@'+pin,'manifest_log':resolved['log'],'resolved_at':state.clock(),'platform':'linux/amd64'}
        state.control(control,saved)
        atomic_json(Path(config['paths']['artifacts'])/(engine+'-image-pin.json'),saved)
    image=saved['image_digest']
    storage_path=Path(config['paths']['artifacts'])/'runtime-container-storage.json'
    storage=json.loads(storage_path.read_text(encoding='utf-8')) if storage_path.exists() else {'images':{},'accounting':'Full uncompressed image sizes conservatively include shared layers; failed pulls retain reservations. Docker disk assumed C for headroom.'}
    inspected=command(config,engine+'-existing-image',[config['installed_tools']['docker'],'image','inspect','--format','{{.Id}}',image])
    if inspected['exit_code']!=0:
        estimate=(1 if engine=='cuda-access' else 24)*1024**3
        prior=storage['images'].get(image,{}).get('accounted_bytes',0)
        extra=max(0,estimate-prior)
        _disk_headroom(config,extra);_runtime_headroom(config,extra)
        disk_root=config.get('acquisition_disk_roots',{}).get('C','C:\\')
        if shutil.disk_usage(disk_root).free-extra<config['limits']['minimum_free_c_bytes']:raise RuntimeError('docker_storage_disk_headroom_c')
        storage['images'][image]={'accounted_bytes':max(estimate,prior),'status':'reserved_pull','engine':engine}
        atomic_json(storage_path,storage)
        remaining=record['deadline']-state.clock()
        if remaining<=0:raise RuntimeError('engine_probe_budget_exhausted:'+engine)
        state.start_execution('runtime_image_probe_setup:'+engine)
        result=run_owned(config,state,engine+'-image-pull',[config['installed_tools']['docker'],'pull','--platform','linux/amd64',image],timeout=remaining,setup_deadline=record['deadline'])
        if result['exit_code']!=0:raise RuntimeError('pinned_image_pull_failed:'+result['log'])
    size=command(config,engine+'-image-size',[config['installed_tools']['docker'],'image','inspect','--format','{{.Size}}',image])
    if size['exit_code']!=0 or not size['output'].strip().isdigit():raise RuntimeError('docker_runtime_size_not_established')
    actual=int(size['output'].strip())
    storage['images'][image]={'accounted_bytes':actual,'status':'acquired','engine':engine,'size_log':size['log'],'acquired_by_benchmark':storage['images'].get(image,{}).get('acquired_by_benchmark',inspected['exit_code']!=0)}
    atomic_json(storage_path,storage)
    _disk_headroom(config,0);_runtime_headroom(config,0)
    state.entity('runtime_catalog',engine,{'kind':'container',**saved})
    return saved


def native_upstream(config,state):
    setup_budget(config,state,'upstream-llama-nightly')
    result=acquire_upstream(config,state)
    state.entity('runtime_catalog','upstream-llama-nightly',{'kind':'native',**result})
    return result


def cuda_access_probe(config,state):
    if state.get_control('docker_cuda_access_verified'):return state.get_control('docker_cuda_access_verified')
    pin=acquire_image(config,state,'cuda-access')['image_digest']
    job='cuda-access-'+digest([state.run['id'],pin])[:16]
    argv=[config['installed_tools']['docker'],'run','--rm','--pull','never','--gpus','device=0',
        '--network','none','--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges',
        '--memory','1g','--cpus','2','--pids-limit','128','--label','localbench.owner=localbench',
        '--label','localbench.run='+state.run['id'],'--label','localbench.job='+job,pin,
        'nvidia-smi','--query-gpu=name,driver_version,compute_cap,memory.total','--format=csv,noheader,nounits']
    result=command(config,'docker-cuda-access',argv,timeout=60)
    if result['exit_code']!=0 or 'RTX 4060 Ti' not in result['output'] or '8.9' not in result['output']:
        raise RuntimeError('docker_cuda_access_failed:'+result['log'])
    compatible(state,'cuda-access',{'isolated_cuda_probe':True,'compute_capability':'8.9'})
    state.control('docker_cuda_access_verified',{'image_digest':pin,'log':result['log'],'output':result['output'],'verified_at':state.clock(),'ubuntu_started':False})
    return state.get_control('docker_cuda_access_verified')
