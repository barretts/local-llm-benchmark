from pathlib import Path
import difflib
import json
import time
import uuid
from .config import atomic_json, canonical, digest, file_hash
from .fixtures import create_workspace, score_answer
from .grading import DockerGrader
from .prompts import AGENT_SYSTEM, validate_context
from .regional_prompt import regional_prompt
from .schema import tool_definitions
from .tools import ToolExecutor, serialize_tool_response


def request_for(config, messages, tools, seed, max_tokens=4096):
    return {"model":"local-model","messages":messages,"tools":tools,"tool_choice":"auto","stream":True,"stream_options":{"include_usage":True},"max_tokens":max_tokens,"seed":seed,**config["default_sampling"]}


def token_evidence(adapter, packed, metrics, fresh_process=False, prefix_reuse=False):
    return {"effective_context_tokens":adapter.effective.get("effective_context"),"expected_prompt_tokens":packed["expected_tokens"],"prompt_tokens":metrics.get("prompt_tokens"),"tokenization_verified":True,"tokenization_provenance":packed["tokenization"],"truncated":False,"context_shift":not adapter.effective.get("context_shift_disabled",False),"cached_tokens":metrics.get("cached_tokens"),"evaluated_tokens":metrics.get("evaluated_tokens"),"cache_isolation_verified":fresh_process,"prefix_reuse_verified":prefix_reuse}


def source_scope(workspace, fixture, marker_hash):
    workspace=Path(workspace)
    marker=workspace/".localbench-workspace.json"
    if not marker.is_file() or file_hash(marker)!=marker_hash:
        return "workspace_ownership_tampering"
    writable=set(fixture.get("writable",fixture["source"]))
    for relative,initial in fixture["source"].items():
        from .paths import safe_source
        try:path=safe_source(workspace,relative)
        except ValueError:return "source_path_violation"
        if not path.is_file():return "missing_source"
        if relative not in writable and path.read_text(encoding="utf-8")!=initial:
            return "readonly_dependency_tampering"
    for path in workspace.rglob("*"):
        relative=path.relative_to(workspace).as_posix()
        if path.is_symlink() or (hasattr(path,"is_junction") and path.is_junction()):
            return "workspace_link_violation"
        if path.is_file() and relative not in fixture["source"] and relative!=".localbench-workspace.json" and not (fixture["language"]=="typescript" and relative.startswith(".build/") and path.suffix==".js"):
            return "unauthorized_workspace_file"
    return None


def archive_diff(config, workspace, fixture, attempt_name):
    chunks=[]
    for relative,original in fixture["source"].items():
        from .paths import safe_source
        try:
            path=safe_source(workspace,relative)
            changed=path.read_text(encoding="utf-8") if path.is_file() else ""
        except (ValueError,OSError):
            chunks.append("Unsafe or unavailable source omitted: "+relative+"\n")
            continue
        chunks.extend(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile="a/"+relative,tofile="b/"+relative))
    path=Path(config["paths"]["logs"])/(attempt_name+"-source.diff")
    path.write_text("".join(chunks),encoding="utf-8",newline="\n")
    return str(path)


def quality_task(config, state, adapter, fixture, configuration_id, job_id, attempt_id):
    name="quality-"+job_id+"-a"+str(attempt_id)
    workspace=create_workspace(config,fixture,name)
    marker_hash=file_hash(workspace/".localbench-workspace.json")
    grader=DockerGrader(config,state.run["id"])
    execute=ToolExecutor(workspace,fixture,lambda root,phase="public":grader.grade(root,fixture,phase=phase,log_id=name+"-"+uuid.uuid4().hex[:8]),adapter.tokenize_text,config["limits"]["maximum_tool_output_tokens_per_turn"])
    tools=tool_definitions(include_answer=fixture["id"].startswith("long"),answer_schema=fixture.get("answer_schema"))
    seed=fixture["seed"]
    system="nonce:"+digest([state.run["id"],configuration_id,fixture["fixture_hash"],seed,"quality"])+"\n"+AGENT_SYSTEM
    is_long=fixture["id"].startswith("long")
    if is_long:
        adapter.fresh_restart()
        packed=regional_prompt(adapter,tools,system,fixture["documents"],fixture["contract"],seed,config["measurement"]["full_prompt_tokens"],config["measurement"]["prompt_target_absolute_tolerance_tokens"])
        messages=packed["messages"]
    else:
        messages=[{"role":"system","content":system},{"role":"user","content":fixture["contract"]+"\nFiles available:\n"+"\n".join(fixture["source"])+"\nThe public assertions described by this contract fail in the initial source. Repair the required defects. Public feedback is available through run_tests."}]
        packed=None
    began=time.monotonic();generated=0;generated_complete=True;all_metrics=[];failure=None;infra=False;first_edit=None;first_public_pass=None;first_action=None;reasoning=[]
    max_turns=1 if is_long else config["limits"]["maximum_fixture_turns"]
    for turn in range(max_turns):
        if time.monotonic()-began>=config["limits"]["per_task_timeout_seconds"]:
            failure="task_timeout";break
        expected=adapter.tokenize(messages,tools)
        if time.monotonic()-began>=config["limits"]["per_task_timeout_seconds"]:
            failure="task_timeout";break
        remaining_generated=config["limits"]["maximum_fixture_generated_tokens"]-generated
        output_budget=min(config["measurement"]["quality_long_response_max_tokens"] if is_long else remaining_generated,remaining_generated,config["measurement"]["context_window_tokens"]-expected["count"])
        if output_budget<=0:
            failure="task_context_or_generated_budget_exhausted";break
        adapter.http.timeout=max(1,min(config["limits"]["per_64k_request_timeout_seconds"],config["limits"]["per_task_timeout_seconds"]-(time.monotonic()-began)))
        response=adapter.stream_chat(request_for(config,messages,tools,seed,output_budget),log_prefix=Path(config["paths"]["logs"])/(name+"-turn"+str(turn)),validator=execute.validate,mode="fresh" if turn==0 else "fixture_cached")
        metrics=response["metrics"];all_metrics.append(metrics)
        count=metrics.get("generated_tokens")
        if type(count) is int and count>=0:generated+=count
        else:generated_complete=False
        if first_action is None and metrics.get("first_action_seconds") is not None:
            first_action=time.monotonic()-began-metrics["wall_seconds"]+metrics["first_action_seconds"]
        if not response["valid_stream"]:
            if "TimeoutError" in response.get("error","") and time.monotonic()-began>=config["limits"]["per_task_timeout_seconds"]:
                failure="task_timeout"
            else:
                failure="malformed_or_unpermitted_tool_stream" if "StreamError" in response.get("error","") else "inference_infrastructure_error"
            infra=failure=="inference_infrastructure_error";break
        actual=metrics.get("prompt_tokens")
        if type(count) is not int or count<0 or type(actual) is not int or actual<0 or abs(actual-expected["count"])>config["measurement"]["prompt_target_absolute_tolerance_tokens"]:
            failure="unverified_generation_or_input_counters";infra=True;break
        if count>output_budget or actual+count>config["measurement"]["context_window_tokens"] or generated>config["limits"]["maximum_fixture_generated_tokens"]:
            failure="task_token_budget_violation";break
        if time.monotonic()-began>=config["limits"]["per_task_timeout_seconds"]:
            failure="task_timeout";break
        if is_long:
            reasons=validate_context(packed["expected_tokens"],metrics,config["measurement"],adapter.effective["effective_context"])
            if reasons:
                failure=";".join(reasons);infra=True;break
            if len(response["calls"])!=1:
                failure="long_requires_exactly_one_final_action";break
        calls=response["calls"]
        if not calls:
            if is_long:failure="missing_long_final_action"
            break
        assistant={"role":"assistant","content":response["content"],"tool_calls":[]}
        reasoning_text="".join(e["text"] for e in response.get("events",[]) if e["type"]=="reasoning")
        if reasoning_text:assistant["reasoning_content"]=reasoning_text
        for index,call in enumerate(calls):
            assistant["tool_calls"].append({"id":call.get("id") or "localbench-"+str(turn)+"-"+str(index),"type":"function","function":{"name":call["name"],"arguments":canonical(call["arguments"])}})
        messages.append(assistant)
        for wire,call in zip(assistant["tool_calls"],calls):
            if time.monotonic()-began>=config["limits"]["per_task_timeout_seconds"]:
                failure="task_timeout";break
            result=execute.execute(call["name"],call["arguments"])
            if not result["ok"]:
                failure="tool_execution_or_path_violation";break
            if call["name"] in ("write_file","apply_patch") and first_edit is None:first_edit=time.monotonic()-began
            if call["name"]=="run_tests" and execute.last_grade_result and execute.last_grade_result["passed"] and first_public_pass is None:first_public_pass=time.monotonic()-began
            messages.append({"role":"tool","tool_call_id":wire["id"],"content":serialize_tool_response(result)})
            if time.monotonic()-began>=config["limits"]["per_task_timeout_seconds"]:
                failure="task_timeout";break
        atomic_json(Path(config["paths"]["logs"])/(name+"-checkpoint.json"),{"turn":turn,"generated_tokens":generated,"messages":messages,"first_action_seconds":first_action,"first_edit_seconds":first_edit,"first_public_pass_seconds":first_public_pass})
        if failure or is_long:break
    scope=source_scope(workspace,fixture,marker_hash)
    failure=failure or scope
    public=hidden=None
    if scope is None:
        if fixture["language"]=="answer":
            scoring=score_answer(fixture,execute.submitted_answer)
            passed=not failure and scoring["passed"]
            failure=failure or (None if passed else scoring["reason"])
        else:
            public=grader.grade(workspace,fixture,"public",log_id=name+"-final")
            hidden=grader.grade(workspace,fixture,"hidden",log_id=name+"-final")
            scope=source_scope(workspace,fixture,marker_hash)
            passed=not failure and public["passed"] and hidden["passed"] and scope is None
            if not passed:failure=failure or scope or "deterministic_test_failure"
    else:passed=False
    # Mandatory scoring still runs after the agent deadline, but cannot turn a
    # task whose execution/scoring exceeded its elapsed cap into a pass.
    if not failure and time.monotonic()-began>=config["limits"]["per_task_timeout_seconds"]:
        failure="task_timeout";passed=False
    metrics={"first_action_seconds":first_action,"first_edit_seconds":first_edit,"first_public_pass_seconds":first_public_pass,"wall_seconds":time.monotonic()-began,"generated_tokens":generated if generated_complete else None,"generated_tokens_complete":generated_complete,"generated_tokens_known":generated,"generation_token_count_provenance":"server-reported response counters; unknown partial responses are excluded from known count"}
    result={"kind":"quality","configuration_id":configuration_id,"fixture":fixture["id"],"seed":seed,"group":"long" if is_long else "regular","passed":bool(passed),"valid":not infra,"reason":failure or "passed","metrics":metrics,"turn_metrics":all_metrics,"workspace":str(workspace),"source_diff":archive_diff(config,workspace,fixture,name),"public_grading":public,"hidden_grading":hidden,"fixture_version":fixture["version"],"tool_interface":"native OpenAI function calls; identical schemas","reasoning_history":"preserved as assistant.reasoning_content"}
    if is_long and all_metrics:
        result["token_evidence"]=token_evidence(adapter,packed,all_metrics[0],fresh_process=True)
        result["region_positions"]=packed["region_positions"]
    return result
