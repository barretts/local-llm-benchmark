from pathlib import Path
import random
import time
import uuid
from .config import atomic_json, digest
from .fixtures import create_workspace, load_fixture
from .prompts import AGENT_SYSTEM, pack, validate_context
from .quality import request_for, token_evidence
from .regional_prompt import regional_prompt
from .schema import tool_definitions, validate_tool
from .tools import ToolExecutor


def warm(config, adapter, seed):
    nonce="disjoint-warmup:"+uuid.uuid4().hex
    tools=[]
    packed=pack(lambda m:adapter.tokenize(m,tools),lambda filler:[{"role":"system","content":nonce},{"role":"user","content":"Read this unrelated obsolete archive, then say READY.\n"+filler}],seed,config["measurement"]["warmup_input_tokens"],config["measurement"]["prompt_target_absolute_tolerance_tokens"])
    request=request_for(config,packed["messages"],tools,seed,config["measurement"]["warmup_output_tokens"])
    request.pop("tool_choice",None)
    response=adapter.stream_chat(request,log_prefix=Path(config["paths"]["logs"])/("warmup-"+uuid.uuid4().hex),validator=validate_tool,mode="exploratory")
    if not response["valid_stream"]:
        raise RuntimeError("warmup_stream_failed")
    return response["metrics"]


def capacity(config, state, adapter, configuration_id, seed, target=61440, on_packed=None):
    adapter.fresh_restart()
    warm(config,adapter,seed)
    rng=random.Random(seed)
    values=["marker-"+"".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(24)) for _ in range(3)]
    ids=[f"capacity-{seed}-{i}" for i in range(3)]
    regions=[f"SIGNED ACTIVE CAPACITY DOCUMENT id={eid}\nregion_index={i}\nexact_value={value}\n" for i,(eid,value) in enumerate(zip(ids,values))]
    answer_schema={"type":"object","properties":{"markers":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":3}},"required":["markers"],"additionalProperties":False}
    tools=[tool for tool in tool_definitions(answer_schema=answer_schema) if tool["function"]["name"]=="submit_answer"]
    def validator(name,args):
        validate_tool(name,args,answer_schema)
        if name!="submit_answer":raise ValueError("capacity requires submit_answer")
        return args
    query=f"Return all three exact_value strings from documents {', '.join(ids)} in their source appearance order. Call submit_answer once with answer={{markers:[the three exact values]}} and evidence_ids containing those three document ids. Obsolete/example values do not apply."
    system="nonce:"+digest([state.run["id"],configuration_id,"capacity",seed,target])+"\n"+AGENT_SYSTEM
    packed=regional_prompt(adapter,tools,system,regions,query,seed,target,config["measurement"]["prompt_target_absolute_tolerance_tokens"])
    if on_packed:on_packed(packed)
    response=adapter.stream_chat(request_for(config,packed["messages"],tools,seed,4096),log_prefix=Path(config["paths"]["logs"])/("capacity-"+configuration_id+"-"+str(seed)+"-"+str(target)+"-"+uuid.uuid4().hex[:8]),validator=validator,mode="fresh")
    metrics=response["metrics"]
    if target==config["measurement"]["full_prompt_tokens"]:
        reasons=validate_context(packed["expected_tokens"],metrics,config["measurement"],adapter.effective["effective_context"])
    else:
        actual=metrics.get("prompt_tokens");tol=config["measurement"]["prompt_target_absolute_tolerance_tokens"]
        reasons=[] if type(actual) is int and target-tol<=actual<=target and abs(actual-packed["expected_tokens"])<=tol else ["small_context_token_count_mismatch"]
    cached=metrics.get("cached_tokens")
    if cached is not None and cached>config["measurement"]["cold_prefix_expected_cached_tokens_max"]:
        reasons.append("fresh_prefix_cache_reuse")
    calls=response["calls"]
    answer=calls[0]["arguments"] if len(calls)==1 else None
    retrieved=answer.get("answer",{}).get("markers",[]) if answer else []
    evidence=answer.get("evidence_ids") if answer else None
    markers=sum(a==b for a,b in zip(retrieved,values)) if len(retrieved)==3 else 0
    passed=response["valid_stream"] and not reasons and markers==3 and evidence==ids
    return {"kind":"capacity","configuration_id":configuration_id,"seed":seed,"target_tokens":target,"passed":bool(passed),"valid":response["valid_stream"] and not reasons,"reason":"passed" if passed else ";".join(reasons) or "capacity_marker_or_evidence_failure","metrics":metrics,"token_evidence":token_evidence(adapter,packed,metrics,fresh_process=True),"markers_passed":markers,"retrieved_values":retrieved,"expected_values":values,"evidence_ids":evidence,"region_positions":packed["region_positions"],"prompt_hash":packed["prompt_hash"]}


def latency(config, state, adapter, configuration_id, replicate, mode="fresh", final=False, stable_prefix=None, target=61440, on_packed=None):
    fixture=load_fixture(config,"py01",42)
    workspace=create_workspace(config,fixture,"latency-"+uuid.uuid4().hex)
    executor=ToolExecutor(workspace,fixture,None,adapter.tokenize_text)
    tools=tool_definitions(include_answer=False)
    if mode=="fresh":
        adapter.fresh_restart();warm(config,adapter,replicate+30017)
        system="nonce:"+digest([state.run["id"],configuration_id,"timing",replicate,final,target])+"\n"+AGENT_SYSTEM
        prefix=""
    else:
        if not stable_prefix:raise ValueError("controlled cached latency requires a primed prefix")
        system=stable_prefix["system"];prefix=stable_prefix["prefix"]
    query="\nFINAL REQUEST: Your first action must be one native read_file call with path src/cache.py. Do not edit code or run tests in this latency probe."
    packed=pack(lambda m:adapter.tokenize(m,tools),lambda filler:[{"role":"system","content":system},{"role":"user","content":prefix+filler+query}],1000003+replicate*7919,target,config["measurement"]["prompt_target_absolute_tolerance_tokens"])
    if on_packed:on_packed(packed)
    response=adapter.stream_chat(request_for(config,packed["messages"],tools,replicate+42,4096),log_prefix=Path(config["paths"]["logs"])/("timing-"+configuration_id+"-"+mode+"-"+str(replicate)+"-"+uuid.uuid4().hex[:8]),validator=executor.validate,mode=mode)
    metrics=response["metrics"]
    reasons=validate_context(packed["expected_tokens"],metrics,config["measurement"],adapter.effective["effective_context"]) if target==61440 else []
    cached=metrics.get("cached_tokens")
    prefix_reuse=mode=="cached" and type(cached) is int and cached>64
    if mode=="fresh" and type(cached) is int and cached>64:reasons.append("fresh_prefix_cache_reuse")
    controlled_verified=mode=="cached" and type(cached) is int and cached>=0 and bool(stable_prefix.get("primed_verified")) and stable_prefix.get("owned_pid")==adapter.process.pid
    if mode=="cached" and not controlled_verified:reasons.append("controlled_cache_experiment_unverified")
    calls=response["calls"]
    expected_action=bool(calls and calls[0]["name"]=="read_file" and calls[0]["arguments"]=={"path":"src/cache.py"})
    passed=response["valid_stream"] and not reasons and metrics.get("first_action_valid",False) and expected_action
    if expected_action and response["valid_stream"]:
        # Demonstrate real permitted tool execution, no source code execution.
        action=executor.execute(calls[0]["name"],calls[0]["arguments"])
        passed=passed and action["ok"]
    evidence=token_evidence(adapter,packed,metrics,fresh_process=mode=="fresh",prefix_reuse=prefix_reuse)
    evidence["controlled_prefix_experiment_verified"]=controlled_verified
    evidence["observed_prefix_cache_hit"]=prefix_reuse
    return {"kind":"timing","configuration_id":configuration_id,"mode":mode,"replicate":replicate,"final_validation":final,"passed":bool(passed),"valid":response["valid_stream"] and not reasons,"reason":"passed" if passed else ";".join(reasons) or "missing_or_incorrect_first_action","stability_passed":bool(passed),"metrics":metrics,"token_evidence":evidence,"prompt_hash":packed["prompt_hash"],"common_byte_corpus_bytes":sum(len(m["content"].encode()) for m in packed["messages"])}


def cached_prefix(config, adapter, state=None, configuration_id=None):
    tools=tool_definitions(include_answer=False)
    system="controlled-stable-prefix:"+digest([state.run["id"] if state else "exploratory",configuration_id,"controlled_cached"])+"\n"+AGENT_SYSTEM
    packed=pack(lambda m:adapter.tokenize(m,tools),lambda filler:[{"role":"system","content":system},{"role":"user","content":filler}],170101,config["measurement"]["cached_stable_prefix_tokens_target"],config["measurement"]["prompt_target_absolute_tolerance_tokens"])
    return {"system":system,"prefix":packed["messages"][1]["content"],"expected_prefix_tokens":packed["expected_tokens"],"prefix_hash":packed["prompt_hash"]}


def throughput(config, adapter, configuration_id, seed=42, target=61440, retry_longer=False, on_packed=None):
    adapter.fresh_restart();warm(config,adapter,seed+91009)
    tools=[]
    query="\nWrite complete Python source for a robust ordered bounded asynchronous worker pool. Include input validation, cancellation cleanup, and detailed examples. Produce at least 150 lines of substantive source. Output only source."
    if retry_longer:query+="\nInclude three complete additional example workers and a complete demonstration main."
    system="nonce:"+digest([configuration_id,"throughput",seed,target,retry_longer])+"\nYou are a coding assistant."
    packed=pack(lambda m:adapter.tokenize(m,tools),lambda filler:[{"role":"system","content":system},{"role":"user","content":filler+query}],seed,target,config["measurement"]["prompt_target_absolute_tolerance_tokens"])
    if on_packed:on_packed(packed)
    # System nonce must remain constant through packing iterations.
    request=request_for(config,packed["messages"],tools,seed,config["measurement"]["throughput_max_output_tokens"])
    request.pop("tool_choice",None)
    response=adapter.stream_chat(request,log_prefix=Path(config["paths"]["logs"])/("throughput-"+configuration_id+"-"+str(seed)+"-"+uuid.uuid4().hex[:8]),validator=validate_tool,mode="fresh")
    metrics=response["metrics"]
    reasons=validate_context(packed["expected_tokens"],metrics,config["measurement"],adapter.effective["effective_context"]) if target==61440 else []
    valid=response["valid_stream"] and not reasons
    return {"kind":"throughput","configuration_id":configuration_id,"seed":seed,"target_tokens":target,"passed":bool(valid),"valid":bool(valid),"reason":"passed" if valid else ";".join(reasons) or "throughput_stream_failure","metrics":metrics,"token_evidence":token_evidence(adapter,packed,metrics,fresh_process=True),"prompt_hash":packed["prompt_hash"],"short_output_unqualified":not metrics.get("sustained_throughput_qualified",False)}
