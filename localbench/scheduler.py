from pathlib import Path
from contextlib import contextmanager
import json
import os
import time
from .adapters.native import NativeAdapter
from .config import atomic_json, canonical, digest, file_hash
from .doctor import command, doctor
from .fixtures import load_fixture
from .measurement import capacity, latency, throughput, cached_prefix
from .quality import quality_task
from .resources import ResourceCollector
from .report import Report

# Capture the imported controller protocol once. Updating a source file while
# an owned worker is active must not relabel its loaded code as a new version.
PROTOCOL_HASHES={p:file_hash(Path(__file__).parent/p) for p in ("measurement.py","prompts.py","regional_prompt.py","quality.py","tools.py","schema.py","common_byte.py")}


class CachedJob(Exception):
    def __init__(self,result):self.result=result


def engine_identity(metadata):
    if "binary_sha256" in metadata and "adjacent_dlls" in metadata:
        value={"binary_sha256":metadata["binary_sha256"],"adjacent_dlls":metadata["adjacent_dlls"],"version":metadata.get("version_output")}
        if metadata.get("runtime_dependencies"):value["runtime_dependencies"]=metadata["runtime_dependencies"]
        return value
    keys=("binary_sha256","adjacent_dlls","runtime_dependencies","version_output","backend_runtime_versions","image_digest","image_id","image_revision","help_sha256","daemon_version","dependency_versions","source_commit","tabby_commit","exllamav3_commit","exllamav3_wheel_sha256","runtime_manifest_sha256","dependency_lock_sha256","packages")
    value={key:metadata[key] for key in keys if key in metadata}
    if not value:raise RuntimeError("unverified_engine_identity")
    return value


def tokenizer_identity(metadata):
    if "binary_sha256" in metadata and "adjacent_dlls" in metadata and not metadata.get("tokenizer_helper_sha256"):
        return metadata["binary_sha256"]
    return digest({"engine":engine_identity(metadata),"checkpoint":metadata.get("model_sha256"),"helper":metadata.get("tokenizer_helper_sha256")})


def model_identity(config,state,model):
    path=Path(model["path"])
    if not path.is_file():raise RuntimeError("missing_model_file")
    stat=path.stat()
    identity={"path":str(path),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
    cached=state.get_control("model_hash:"+str(path))
    sha=cached["sha256"] if cached and all(cached.get(k)==v for k,v in identity.items()) else file_hash(path)
    state.control("model_hash:"+str(path),{**identity,"sha256":sha})
    result={**model,**identity,"sha256":sha,"reuse_new_bytes":model.get("reuse_new_bytes",0)}
    state.entity("model",result["id"],result)
    return result


def key_base(config,adapter,configuration_id,kind,seed,mode,replicate,target,fixture_hash=None):
    metadata=adapter.metadata()
    engine=digest(engine_identity(metadata))
    protocol_files=("quality.py","tools.py","schema.py") if kind=="quality" else ("common_byte.py", "measurement.py") if kind=="common_byte" else ("measurement.py","prompts.py","regional_prompt.py")
    protocol={p:PROTOCOL_HASHES[p] for p in protocol_files}
    return {"engine":engine,"model":metadata["model_sha256"],"effective_settings":adapter.requested,"profile":{"sampling":config["default_sampling"],"reasoning":adapter.requested.get("reasoning","default")},"tokenizer":tokenizer_identity(metadata),"seed":seed,"mode":mode,"replicate":replicate,"kind":kind,"target_tokens":target,"configuration_id":configuration_id,"logical_plan_hash":digest({"protocol":protocol,"run":adapter.state.run["id"],"fixture_hash":fixture_hash,"kind":kind,"seed":seed,"mode":mode,"replicate":replicate,"target":target})}


def prior_result(state,base):
    # An unstarted cancellation retirement is forensic bookkeeping, not a
    # measured result that can replace a legitimate packed job on resume.
    for row in state.db.execute("SELECT j.key_json,j.result,j.status FROM jobs j WHERE j.status IN ('passed','failed','invalid','skipped') AND EXISTS (SELECT 1 FROM attempts a WHERE a.job_id=j.id)"):
        key=json.loads(row["key_json"])
        if all(key.get(k)==v for k,v in base.items()):
            return json.loads(row["result"] or "{}")
    return None


def result_status(result):
    return "passed" if result.get("passed") else "failed" if result.get("valid",False) else "invalid"


@contextmanager
def probe_request_timeout(config,adapter,kind,target):
    # Coding tasks own a separate total-task deadline and clamp every turn to
    # its remaining budget. Only known-size measurement probes use this scope.
    if kind=="quality" or type(target) is not int or target<=0:
        yield
        return
    limit=config["limits"]["per_16k_request_timeout_seconds"] if target<=16384 else config["limits"]["per_64k_request_timeout_seconds"]
    previous=adapter.http.timeout
    try:
        adapter.http.timeout=limit
        yield
    finally:
        adapter.http.timeout=previous


def measured_job(config,state,adapter,collector,configuration_id,kind,seed,mode,replicate,target,operation,fixture_hash=None):
    base=key_base(config,adapter,configuration_id,kind,seed,mode,replicate,target,fixture_hash)
    prior=prior_result(state,base)
    if prior is not None:return prior
    state.check_budget(config["limits"]["per_64k_request_timeout_seconds"] if target==61440 else config["limits"]["per_16k_request_timeout_seconds"])
    holder={}
    def on_packed(packed):
        # Packing can outlast a new stop request or the remaining job budget.
        # Check again before creating a real measurement job/attempt.
        state.check_budget(config["limits"]["per_64k_request_timeout_seconds"] if target==61440 else config["limits"]["per_16k_request_timeout_seconds"])
        key={**base,"fixture_prompt_hash":packed["prompt_hash"]}
        job=state.enqueue(key)
        row=state.db.execute("SELECT status,result FROM jobs WHERE id=?",(job,)).fetchone()
        if row["status"]!="pending":raise CachedJob(json.loads(row["result"] or "{}"))
        holder.update(job=job,attempt=state.begin(job),started=time.time())
        state.control("active_job",{"job":job,"kind":kind,"model":adapter.model["id"],"engine":adapter.engine,"configuration_id":configuration_id,"started":holder["started"]})
    try:
        with probe_request_timeout(config,adapter,kind,target):
            result=operation(on_packed)
    except CachedJob as cached:return cached.result
    except (OSError,RuntimeError,ValueError) as error:
        if str(error) in ("stop_after_current","budget_exhausted"):raise
        result={"kind":kind,"configuration_id":configuration_id,"seed":seed,"mode":mode,"replicate":replicate,"target_tokens":target,"passed":False,"valid":False,"reason":str(error),"error_type":type(error).__name__}
        if kind=="timing" and replicate>=1000:result.update(final_validation=True,stability_passed=False)
        if not holder:
            job=state.enqueue({**base,"fixture_prompt_hash":digest({"unpacked_plan":base})})
            holder.update(job=job,attempt=state.begin(job),started=time.time())
    result["resources"]=collector.snapshot(holder["started"],time.time())
    if result["resources"]["foreign_workload_overlap"]:
        result.update(timing_valid=False,latency_diagnostic_only=True,timing_invalid_reason="foreign_workload_overlap")
        # Capacity measures retrieved values and verified context. Preserve its
        # functional verdict; its contaminated speed metrics are diagnostic.
        if kind!="capacity":result.update(passed=False,valid=False,reason="foreign_workload_overlap")
    transient=result.get("reason") in ("foreign_workload_overlap","inference_infrastructure_error")
    state.finish(holder["attempt"],result_status(result),result,reason=result.get("reason"),transient=transient)
    state.control("active_job",None)
    return result


def coding_job(config,state,adapter,collector,configuration_id,fixture_id,seed,replicate=0,mode="fresh"):
    fixture=load_fixture(config,fixture_id,seed)
    base=key_base(config,adapter,configuration_id,"quality",seed,mode,replicate,61440 if fixture_id.startswith("long") else None,fixture["fixture_hash"])
    key={**base,"fixture_prompt_hash":fixture["fixture_hash"]}
    prior=prior_result(state,key)
    if prior is not None:return prior
    state.check_budget(config["limits"]["per_task_timeout_seconds"])
    job=state.enqueue(key);attempt=state.begin(job);began=time.time()
    state.control("active_job",{"job":job,"kind":"quality","fixture":fixture_id,"seed":seed,"model":adapter.model["id"],"engine":adapter.engine,"configuration_id":configuration_id,"started":began})
    try:
        result=quality_task(config,state,adapter,fixture,configuration_id,job,attempt)
    except (RuntimeError,ValueError,OSError) as error:
        result={"kind":"quality","configuration_id":configuration_id,"fixture":fixture_id,"seed":seed,"group":"long" if fixture_id.startswith("long") else "regular","passed":False,"valid":False,"reason":"inference_infrastructure_error","error":str(error)}
    result["resources"]=collector.snapshot(began,time.time())
    # Functional grades remain valid under unrelated workload; their latency is
    # diagnostic only. Final latency samples are rejected in measured_job.
    state.finish(attempt,result_status(result),result,reason=result.get("reason"),transient=result.get("reason")=="inference_infrastructure_error")
    state.control("active_job",None)
    return result


def register_configuration(config,state,adapter):
    metadata=adapter.metadata()
    identity={"engine":engine_identity(metadata),"model_sha256":metadata["model_sha256"],"settings":adapter.requested,"sampler":config["default_sampling"],"fixture_version":state.get_control("fixture_version")}
    if "binary_sha256" in metadata and "adjacent_dlls" in metadata:
        identity={"engine_binary":metadata["binary_sha256"],"engine_dlls":metadata["adjacent_dlls"],"model_sha256":metadata["model_sha256"],"settings":adapter.requested,"sampler":config["default_sampling"],"fixture_version":state.get_control("fixture_version")}
        if metadata.get("runtime_dependencies"):identity["runtime_dependencies"]=metadata["runtime_dependencies"]
    identifier=digest(identity)
    row=state.db.execute("SELECT id FROM entities WHERE kind='configuration' AND id=?",(identifier,)).fetchone()
    count=state.db.execute("SELECT COUNT(*) FROM entities WHERE kind='configuration'").fetchone()[0]
    if row is None and count>=config["limits"]["maximum_unique_runtime_configurations"]:
        raise RuntimeError("configuration_budget_exhausted")
    state.entity("configuration",identifier,{**metadata,"configuration_id":identifier,"model_id":adapter.model["id"],"identity":identity,"engine_id":adapter.engine})
    return identifier


def make_native(config,state,collector,engine,model,settings=None,binary=None):
    model=model_identity(config,state,model)
    adapter=NativeAdapter(config,state,engine,model,binary=binary,settings=settings)
    adapter.collector=collector
    try:
        adapter.discover_capabilities()
        adapter.launch()
        identifier=register_configuration(config,state,adapter)
    except BaseException:
        adapter.unload_owned()
        raise
    return adapter,identifier


def screen_native(config,state,collector,engine,model,settings=None,binary=None):
    adapter=None
    probe_started=time.time()
    try:
        adapter,identifier=make_native(config,state,collector,engine,model,settings,binary)
        caps=[];timings=[];smoke=[]
        target16=config["measurement"]["smaller_prompt_targets"][-1]
        small=measured_job(config,state,adapter,collector,identifier,"capacity",17,"fresh",0,target16,lambda cb:capacity(config,state,adapter,identifier,17,target16,on_packed=cb))
        if not small.get("passed"):
            return {"engine":engine,"model":model["id"],"configuration_id":identifier,"eligible":False,"reason":small.get("reason"),"capacity":[small]}
        for seed in config["grading"]["capacity_marker_seeds"]:
            caps.append(measured_job(config,state,adapter,collector,identifier,"capacity",seed,"fresh",0,61440,lambda cb,s=seed:capacity(config,state,adapter,identifier,s,on_packed=cb)))
        # Timings remain useful diagnostics even when capacity fails.
        for i in range(config["measurement"]["fresh_repetitions_screen"]):
            timings.append(measured_job(config,state,adapter,collector,identifier,"timing",i+42,"fresh",i,61440,lambda cb,r=i:latency(config,state,adapter,identifier,r,on_packed=cb)))
        speed=measured_job(config,state,adapter,collector,identifier,"throughput",42,"fresh",0,61440,lambda cb:throughput(config,adapter,identifier,on_packed=cb))
        if speed.get("metrics",{}).get("generated_tokens",256)<64:
            speed=measured_job(config,state,adapter,collector,identifier,"throughput",42,"fresh",1,61440,lambda cb:throughput(config,adapter,identifier,retry_longer=True,on_packed=cb))
        eligible=all(c.get("passed") for c in caps) and all(t.get("passed") for t in timings)
        if eligible:
            for job in config["grading"]["smoke_jobs"]:
                smoke.append(coding_job(config,state,adapter,collector,identifier,job["fixture"],job["seed"]))
        result={"engine":engine,"model":model["id"],"configuration_id":identifier,"eligible":eligible,"reason":"screen_passed" if eligible else "capacity_or_streamed_action_gate_failed","capacity":caps,"timing":timings,"throughput":speed,"smoke":smoke,"probe_seconds":time.time()-probe_started,"settings":adapter.requested,"binary":str(adapter.binary)}
        state.entity("screen",digest([engine,model["id"],adapter.requested]),result)
        return result
    except (RuntimeError,ValueError,OSError) as error:
        result={"kind":"engine_probe","engine":engine,"model":model["id"],"settings":settings,"eligible":False,"passed":False,"valid":False,"reason":str(error),"probe_seconds":time.time()-probe_started}
        # Unsupported launches still have durable attempt evidence.
        key={"engine":file_hash(binary or config["installed_tools"]["bundled_llama_server"]) if Path(binary or config["installed_tools"]["bundled_llama_server"]).is_file() else engine,"model":model_identity(config,state,model)["sha256"],"effective_settings":settings or {"baseline":"f16"},"profile":config["default_sampling"],"fixture_prompt_hash":digest({"probe":"compatibility","model":model["id"]}),"tokenizer":"unavailable","seed":0,"mode":"fresh","replicate":0}
        job=state.enqueue(key)
        row=state.db.execute("SELECT status FROM jobs WHERE id=?",(job,)).fetchone()
        if row[0]=="pending":state.finish(state.begin(job),"invalid",result,reason=result["reason"])
        state.entity("screen",digest([engine,model["id"],settings]),result)
        return result
    finally:
        if adapter is not None:adapter.unload_owned()
        Report(config,state).write()


def complete_quality_round_robin(config,state,collector,screens,engine_binary):
    active=[s for s in screens if s.get("eligible")]
    failures={s["configuration_id"]:{"regular":0,"long":0} for s in active}
    for group,ids,threshold in (("regular",config["grading"]["regular_fixture_ids"],config["grading"]["early_elimination_regular_failures"]),("long",config["grading"]["long_fixture_ids"],config["grading"]["early_elimination_long_failures"])):
        for seed in config["grading"]["seeds"]:
            for fixture in ids:
                for screen in active:
                    identifier=screen["configuration_id"]
                    if failures[identifier][group]>=threshold:continue
                    model=next(m for m in config["installed_model_candidates"] if m["id"]==screen["model"])
                    adapter=None
                    try:
                        adapter,actual=make_native(config,state,collector,screen["engine"],model,screen["settings"],engine_binary.get(screen["engine"]))
                        if actual!=identifier:raise RuntimeError("effective_configuration_identity_changed")
                        result=coding_job(config,state,adapter,collector,identifier,fixture,seed)
                        if result.get("valid") and not result.get("passed"):failures[identifier][group]+=1
                    finally:
                        if adapter is not None:adapter.unload_owned()
                    Report(config,state).write()
    return Report(config,state).write()


def final_validation(config,state,collector,screen,model,binary):
    adapter=None
    try:
        adapter,identifier=make_native(config,state,collector,screen["engine"],model,screen["settings"],binary)
        for i in range(config["measurement"]["final_fresh_repetitions"]):
            measured_job(config,state,adapter,collector,identifier,"timing",1042+i,"fresh",1000+i,61440,lambda cb,r=i:latency(config,state,adapter,identifier,1000+r,final=True,on_packed=cb))
        adapter.fresh_restart();prefix=cached_prefix(config,adapter,state,identifier)
        prime=latency(config,state,adapter,identifier,-1,mode="cached",stable_prefix=prefix)
        m=prime["metrics"]
        prefix["primed_verified"]=m.get("stream_complete",False) and m.get("first_action_valid",False) and type(m.get("prompt_tokens")) is int and 61376<=m["prompt_tokens"]<=61440
        prefix["owned_pid"]=adapter.process.pid
        atomic_json(Path(config["paths"]["logs"])/("cache-prime-"+identifier+".json"),{"prefix":{k:v for k,v in prefix.items() if k!="prefix"},"response":prime})
        for i in range(config["measurement"]["final_cached_repetitions"]):
            measured_job(config,state,adapter,collector,identifier,"timing",2042+i,"cached",2000+i,61440,lambda cb,r=i:latency(config,state,adapter,identifier,2000+r,mode="cached",final=True,stable_prefix=prefix,on_packed=cb))
        # Reproducible launch is tested from a stopped owned instance.
        adapter.fresh_restart()
        demo=coding_job(config,state,adapter,collector,identifier,"py02",42,replicate=1,mode="isolated-demo")
        result={"kind":"demo","configuration_id":identifier,"passed":demo.get("passed",False),"valid":demo.get("valid",False),"isolated":True,"reason":demo.get("reason"),"transcript_source":demo.get("source_diff"),"launch_tested_from_stopped_owned_instance":True}
        key={**key_base(config,adapter,identifier,"demo",42,"fresh",0,None),"fixture_prompt_hash":digest({"demo":"py02","fixture_version":state.get_control("fixture_version")})}
        job=state.enqueue(key)
        if state.db.execute("SELECT status FROM jobs WHERE id=?",(job,)).fetchone()[0]=="pending":state.finish(state.begin(job),result_status(result),result,reason=result.get("reason"))
    finally:
        if adapter is not None:adapter.unload_owned()
    return Report(config,state).write()


def run(config,state,args):
    state.entity("controller",state.run["id"],{"pid":os.getpid(),"command":args.command,"started":time.time(),"protocol_source_hashes":PROTOCOL_HASHES})
    # Reinspect hardware, active engines and workloads immediately before probes.
    from .recovery import recover_processes
    recovery=recover_processes(config,state)
    state.control('ownership_recovery',recovery)
    if any(item.get('status') in ('refused','pending_descendants','pending_auth','pending_load') for item in recovery):
        raise RuntimeError('owned_recovery_unresolved: cannot safely start another GPU workload')
    inspected=doctor(config,hash_weights=False)
    if not inspected["disk_headroom_ok"]:raise RuntimeError("disk_headroom_failed")
    host=json.loads((Path(config["paths"]["artifacts"])/"host.json").read_text(encoding="utf-8"))
    loaded=host["commands"]["lms_loaded"]
    try:loaded_models=json.loads(loaded["output"])
    except ValueError:loaded_models=None
    if loaded_models:
        state.control("waiting_for_idle",{"reason":"unrelated_loaded_models","models":loaded_models})
        raise RuntimeError("unrelated_loaded_models: waiting for an agreed idle window")
    collector=ResourceCollector(config,state).start()
    try:
        from .search import full_search,runtime
        from .search_core import screen,checkpoint_model
        if args.command=='agent-endpoint':
            from .endpoint import serve
            return serve(config,state,collector,args.plan)
        if args.command=='agent-demo':
            from .handoff import demo_and_prepare
            candidate=None
            for row in state.db.execute("SELECT data FROM entities WHERE kind='screen'"):
                item=json.loads(row['data'])
                if item.get('configuration_id')==args.configuration:candidate=item;break
            if candidate is None:raise ValueError('configuration_screen_not_found')
            candidate.setdefault('runtime',runtime(config,state,candidate['engine']))
            return demo_and_prepare(config,state,collector,candidate)
        if args.command=='probe':
            model=checkpoint_model(state,args.model,config['installed_model_candidates'])
            return screen(config,state,collector,runtime(config,state,args.engine),model)
        return full_search(config,state,collector)
    finally:
        collector.stop()
        state.control("active_job",None)
        Report(config,state).write()
