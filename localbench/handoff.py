"""Immutable, verified launch plans and a real demo on the delivered interface.

No runtime is launched by load_plan or by saving a plan.  launch_plan returns an
owned live adapter; its caller must call close_adapter before releasing the GPU
lock.  The demo itself always stops its instances before returning.
"""
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
import hashlib
import json
import os
import re
import uuid

from .config import atomic_json, canonical, digest
from .scheduler import coding_job, engine_identity, tokenizer_identity, result_status
from .search_core import make_adapter, close_adapter, checkpoint_model

SCHEMA_VERSION = 1
HASH = re.compile(r"[a-f0-9]{64}")
REVISION = re.compile(r"[a-f0-9]{40}")
TERMINAL = {"passed", "failed", "invalid", "skipped"}
SOURCE_FILES = (
    "handoff.py", "search_core.py", "scheduler.py", "config.py", "state.py",
    "quality.py", "grading.py", "fixtures.py", "fixture_python.py",
    "fixture_typescript.py", "tools.py", "schema.py", "streams.py", "metrics.py",
    "ownership.py", "tokenizer_helper.py", "adapters/base.py",
    "adapters/native.py", "adapters/container.py", "adapters/tabby.py",
    "adapters/ollama.py", "adapters/lms.py", "runtime_build.py", "tabby_setup.py",
    "serving.py", "endpoint.py", "runner.py", "report.py", "winner_launch.py",
)


def _hash_file(path, state=None, git_blob=False):
    path = Path(path)
    before = path.stat()
    if not path.is_file():
        raise RuntimeError("handoff_pin_is_not_file")
    sha = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        sha.update(("blob " + str(before.st_size) + "\0").encode("ascii"))
    with path.open("rb") as stream:
        while True:
            if state is not None:
                state.check_budget()
            chunk = stream.read(16 * 1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("handoff_file_changed_while_hashing")
    return sha.hexdigest()


def _pin(path, role, expected=None, size=None, state=None):
    path = Path(path).resolve(strict=True)
    actual = _hash_file(path, state)
    actual_size = path.stat().st_size
    if expected is not None and actual != expected:
        raise RuntimeError("handoff_" + role + "_hash_mismatch")
    if size is not None and actual_size != size:
        raise RuntimeError("handoff_" + role + "_size_mismatch")
    return {"path": str(path), "role": role, "bytes": actual_size, "sha256": actual}


def _source_pins(state=None):
    root = Path(__file__).resolve().parent
    files = [root / name for name in SOURCE_FILES]
    files.append(root/'native_speculation.py')
    files += [root.parent / "scripts" / name for name in ("grade_python.py", "grade_typescript.sh")]
    return [_pin(path, "controller", state=state) for path in files]


def _no_secrets(value):
    """Pins contain public URLs and credential environment names, never values."""
    if isinstance(value, dict):
        forbidden = {"api_key", "apikey", "authorization", "password", "access_token", "hf_token"}
        if any(str(key).lower() in forbidden for key in value):
            raise ValueError("handoff_secret_field_forbidden")
        for item in value.values():
            _no_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _no_secrets(item)
    elif isinstance(value, str):
        if value.lower().startswith(("http://", "https://")):
            parts = urlsplit(value)
            if parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError("handoff_credential_or_signed_url_forbidden")
        if re.search(r"(?i)\bbearer\s+\S+|(?:^|\s)--(?:api-key|access-token|password)(?:=|\s|$)", value):
            raise ValueError("handoff_secret_argument_forbidden")


def _interface(adapter, runtime):
    saved = getattr(adapter, "saved", None) or {}
    endpoint = saved.get("endpoint")
    if endpoint is None:
        endpoint = "http://127.0.0.1:" + str(adapter.http.port)
    parts = urlsplit(endpoint)
    if parts.scheme != "http" or parts.hostname != "127.0.0.1" or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise RuntimeError("handoff_endpoint_must_be_loopback")
    if parts.port != adapter.http.port:
        raise RuntimeError("handoff_endpoint_port_disagrees")
    ollama = runtime["kind"] == "ollama"
    model_name = getattr(adapter, "local_name", None) if ollama else getattr(adapter, "instance_id", None) if runtime["kind"] == "lm-studio" else None
    if runtime["kind"] == "tabby":
        model_name = Path(adapter.model["path"]).name
    if runtime["kind"] == "native" or runtime["kind"] == "container" and runtime["engine"] == "ik-llama":
        model_name = "local-model"
    model_name = model_name or adapter.model["id"]
    return {"endpoint": endpoint.rstrip("/"), "chat_path": "/api/chat" if ollama else "/v1/chat/completions",
            "models_path": "/api/tags" if ollama else "/v1/models",
            "wire_protocol": "Ollama native streaming NDJSON" if ollama else "OpenAI-compatible streaming SSE",
            "tool_arguments": "native function tools; controller validates arguments against the declared schemas",
            "model": model_name, "authentication": "environment credential when required; value never saved"}


def _config_identity(config, state):
    return {"spec_hash": config["_spec_hash"], "execution_hash":config.get('_execution_hash'),"private_ports": config["private_ports"],
            "installed_tools": config["installed_tools"], "default_sampling": config["default_sampling"],
            "fixture_version": state.get_control("fixture_version")}


def _checkpoint_pins(model, metadata, state):
    root = Path(model["path"]).resolve(strict=True)
    if root.is_file():
        pins=[_pin(root, "model", metadata["model_sha256"], model.get("bytes"), state)]
        draft=metadata.get('requested_settings',{}).get('draft_model')
        if draft:
            pins.append(_pin(draft['path'],'speculative-draft',draft['sha256'],draft['bytes'],state))
        return pins
    if not root.is_dir() or not REVISION.fullmatch(str(model.get("revision", ""))):
        raise RuntimeError("handoff_checkpoint_requires_immutable_revision")
    files = model.get("files") or []
    if not files:
        raise RuntimeError("handoff_checkpoint_requires_explicit_file_manifest")
    pins = []
    seen = set()
    for item in files:
        name = item.get("filename", "")
        relative = PurePosixPath(name)
        if not name or "\\" in name or relative.is_absolute() or any(part in (".", "..") for part in relative.parts) or ":" in name or name.casefold() in seen:
            raise RuntimeError("handoff_checkpoint_unsafe_filename")
        seen.add(name.casefold())
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise RuntimeError("handoff_checkpoint_file_escape")
        expected = item.get("sha256")
        blob = item.get("git_blob_sha1")
        if expected is None and (not blob or _hash_file(path, state, git_blob=True) != blob):
            raise RuntimeError("handoff_checkpoint_file_lacks_valid_pin")
        if type(item.get("bytes")) is not int or item["bytes"] < 0:
            raise RuntimeError("handoff_checkpoint_file_lacks_size")
        pins.append(_pin(path, "model", expected, item["bytes"], state))
    # Every local tokenizer/template/config file used by offline helpers must be
    # pinned too, including a config added by an installed snapshot inventory.
    pinned = {entry["path"] for entry in pins}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jinja", ".txt", ".model"):
            selected = path.resolve(strict=True)
            if not selected.is_relative_to(root):
                raise RuntimeError("handoff_tokenizer_file_escape")
            if str(selected) not in pinned:
                pins.append(_pin(selected, "tokenizer", state=state))
    return pins


def _runtime_pins(runtime, metadata, requested, state):
    pins = []
    binary = metadata.get("binary") or runtime.get("binary_path")
    if binary:
        pins.append(_pin(binary, "runtime", metadata.get("binary_sha256"), state=state))
        for name, sha in metadata.get("adjacent_dlls", {}).items():
            if Path(name).name != name or "/" in name or "\\" in name:
                raise RuntimeError("handoff_unsafe_dependency_name")
            pins.append(_pin(Path(binary).parent / name, "runtime", sha, state=state))
    dependencies = metadata.get("runtime_dependencies") or {}
    if dependencies:
        directory = requested.get("cuda_dependency_directory")
        if not directory:
            raise RuntimeError("handoff_runtime_dependency_directory_missing")
        for name, sha in dependencies.items():
            if Path(name).name != name or "/" in name or "\\" in name:
                raise RuntimeError("handoff_unsafe_dependency_name")
            pins.append(_pin(Path(directory) / name, "runtime", sha, state=state))
    if runtime["kind"] == "container":
        if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", str(runtime.get("image_digest", ""))):
            raise RuntimeError("handoff_container_requires_repository_digest")
        if metadata.get("image_digest") != runtime["image_digest"]:
            raise RuntimeError("handoff_container_digest_disagrees")
    if runtime["kind"] == "tabby":
        pins.append(_pin(runtime["manifest_path"], "runtime_manifest", metadata["runtime_manifest_sha256"], state=state))
        pins.append(_pin(runtime["python"], "runtime_python", state=state))
        pins.append(_pin(runtime["dependency_lock"], "dependency_lock", runtime["dependency_lock_sha256"], state=state))
        pins.append(_pin(runtime["requirements_lock"], "dependency_lock", runtime["requirements_lock_sha256"], state=state))
        source = Path(runtime["source"]).resolve(strict=True)
        for name, sha in runtime["source_script_hashes"].items():
            path = (source / name).resolve(strict=True)
            if not path.is_relative_to(source):
                raise RuntimeError("handoff_runtime_source_escape")
            pins.append(_pin(path, "runtime_source", sha, state=state))
    for item in (runtime.get("engine_manifest") or {}).get("files", []):
        if item.get("path") and item.get("sha256"):
            pins.append(_pin(item["path"], "runtime", item["sha256"], item.get("bytes"), state))
    if runtime["kind"] in ("native", "ollama", "lm-studio") and not binary:
        raise RuntimeError("handoff_runtime_binary_missing")
    return pins


def _owned_evidence(adapter, state):
    saved = getattr(adapter, "saved", None)
    if not saved or saved.get("owner") != "localbench" or saved.get("run_id") != state.run["id"]:
        raise RuntimeError("handoff_instance_is_not_owned")
    argv = adapter.effective.get("launch_argv") or saved.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        raise RuntimeError("handoff_exact_launch_command_missing")
    evidence = {"owned_handle": json.loads(canonical(saved)), "argv": list(argv),
                "effective_settings": json.loads(canonical(adapter.effective)), "at": state.clock()}
    _no_secrets(evidence)
    return evidence


def _close_and_prove(adapter, state):
    evidence = _owned_evidence(adapter, state)
    original = getattr(adapter, "process", None)
    close_adapter(adapter)
    if original is not None and original.poll() is None:
        raise RuntimeError("handoff_owned_process_stop_unverified")
    if getattr(adapter, "instance_id", None) is not None:
        raise RuntimeError("handoff_owned_instance_unload_unverified")
    completion = getattr(original, "removal_evidence", None) if original is not None else None
    if completion is not None:
        if completion.get("removed") is not True or completion.get("absence_verified") is not True or completion.get("container_id") != evidence["owned_handle"].get("container_id"):
            raise RuntimeError("handoff_original_container_completion_unverified")
        evidence["completion_evidence"] = completion
    evidence.update(stopped=True, stopped_at=state.clock(), stop_method="adapter.unload_owned with original identity verification")
    return evidence


def _save_immutable(path, envelope):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical(envelope) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError("handoff_immutable_plan_already_differs")
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # atomic creation without replacing a plan
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError("handoff_immutable_plan_already_differs")
    finally:
        temporary.unlink(missing_ok=True)


def load_plan(path):
    """Read a content-addressed plan; no probes, downloads or process actions."""
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("handoff_plan_size_invalid")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or set(envelope) != {"schema_version", "plan_id", "plan"} or envelope["schema_version"] != SCHEMA_VERSION:
        raise ValueError("handoff_plan_schema_invalid")
    plan = envelope["plan"]
    if not isinstance(plan, dict) or not HASH.fullmatch(str(envelope["plan_id"])) or digest(plan) != envelope["plan_id"]:
        raise ValueError("handoff_plan_hash_mismatch")
    required = {"run_id", "configuration_id", "engine", "model", "runtime", "requested_settings", "config_identity", "engine_identity", "model_identity", "tokenizer_identity", "file_pins", "interface", "observed_launch"}
    if required - plan.keys() or not HASH.fullmatch(str(plan["configuration_id"])) or not HASH.fullmatch(str(plan["model_identity"])):
        raise ValueError("handoff_plan_fields_invalid")
    _no_secrets(envelope)
    return envelope


def _validate_files(plan, state=None):
    pins = plan["file_pins"]
    if not isinstance(pins, list) or not pins:
        raise ValueError("handoff_file_pins_missing")
    seen = {}
    for pin in pins:
        if not isinstance(pin, dict) or set(pin) != {"path", "role", "bytes", "sha256"} or type(pin["bytes"]) is not int or pin["bytes"] < 0 or not HASH.fullmatch(str(pin["sha256"])):
            raise ValueError("handoff_file_pin_invalid")
        path = Path(pin["path"])
        if not path.is_absolute() or not path.is_file() or str(path.resolve(strict=True)) != pin["path"]:
            raise RuntimeError("handoff_pinned_file_missing_or_redirected")
        if pin["path"] in seen and seen[pin["path"]] != (pin["bytes"], pin["sha256"]):
            raise ValueError("handoff_conflicting_file_pins")
        seen[pin["path"]] = (pin["bytes"], pin["sha256"])
        if path.stat().st_size != pin["bytes"] or _hash_file(path, state) != pin["sha256"]:
            raise RuntimeError("handoff_pinned_file_changed:" + pin["role"])
    controller = {pin["path"] for pin in pins if pin["role"] == "controller"}
    if controller != {pin["path"] for pin in _source_pins(state)}:
        raise RuntimeError("handoff_controller_source_set_changed")
    binary = next((pin for pin in pins if pin["role"] == "runtime" and pin["path"] == str(Path(plan["runtime"].get("binary_path", "")).resolve())), None)
    if binary is None and plan["engine_identity"].get("adjacent_dlls") is not None:
        expected_binary = plan["engine_identity"].get("binary_sha256")
        binary = next((pin for pin in pins if pin["role"] == "runtime" and pin["sha256"] == expected_binary), None)
    if binary is not None and "adjacent_dlls" in plan["engine_identity"]:
        actual_names = {path.name for path in Path(binary["path"]).parent.glob("*.dll") if path.is_file()}
        if actual_names != set(plan["engine_identity"]["adjacent_dlls"]):
            raise RuntimeError("handoff_adjacent_dependency_file_set_changed")
    model_root = Path(plan["model"]["path"]).resolve()
    if model_root.is_dir():
        expected = {pin["path"] for pin in pins if pin["role"] in ("model", "tokenizer") and Path(pin["path"]).suffix in (".json", ".jinja", ".txt", ".model")}
        actual = {str(path.resolve()) for path in model_root.rglob("*") if path.is_file() and path.suffix in (".json", ".jinja", ".txt", ".model")}
        if expected != actual:
            raise RuntimeError("handoff_checkpoint_support_file_set_changed")
        if plan["runtime"]["kind"] == "tabby":
            expected_weights = {str((model_root / item["filename"]).resolve()) for item in plan["model"]["files"] if item["filename"].endswith(".safetensors") and item.get("weight", True)}
            if expected_weights != {str(path.resolve()) for path in model_root.glob("*.safetensors") if path.is_file()}:
                raise RuntimeError("handoff_exl3_weight_file_set_changed")


def launch_plan(config, state, collector, path):
    """Launch the exact pinned plan and return (owned live adapter, config ID).

    The caller holds gpu_lock and keeps the controller alive until shutdown.
    Every failed validation stops the newly launched instance before returning.
    """
    envelope = load_plan(path)
    plan = envelope["plan"]
    registry = _entity(state, "handoff_plan", plan["configuration_id"])
    if not registry or registry.get("plan_id") != envelope["plan_id"] or Path(registry["path"]).resolve() != Path(path).resolve() or not (registry.get("original_instance") or {}).get("stopped"):
        raise RuntimeError("handoff_plan_is_not_registered_and_stopped")
    if plan["run_id"] != state.run["id"] or plan["config_identity"] != _config_identity(config, state):
        raise RuntimeError("handoff_run_or_launch_config_changed")
    state.check_budget(config["limits"]["server_start_timeout_seconds"])
    _validate_files(plan, state)
    adapter = None
    try:
        adapter, identifier = make_adapter(config, state, collector, plan["runtime"], plan["model"], plan["requested_settings"])
        metadata = adapter.metadata()
        if identifier != plan["configuration_id"] or adapter.requested != plan["requested_settings"] or engine_identity(metadata) != plan["engine_identity"] or metadata["model_sha256"] != plan["model_identity"] or tokenizer_identity(metadata) != plan["tokenizer_identity"]:
            raise RuntimeError("handoff_effective_configuration_identity_changed")
        if adapter.effective.get("effective_context", 0) < 65536 or adapter.effective.get("effective_slots") != 1:
            raise RuntimeError("handoff_effective_64k_or_single_slot_missing")
        interface = _interface(adapter, plan["runtime"])
        # LM Studio regenerates the owned instance ID; it is recorded literally
        # in each live descriptor instead of reusing a stopped instance name.
        expected = dict(plan["interface"])
        if plan["runtime"]["kind"] == "lm-studio":
            expected["model"] = interface["model"]
        if interface != expected:
            raise RuntimeError("handoff_delivered_interface_changed")
        evidence = {"plan_id": envelope["plan_id"], "configuration_id": identifier,
                    "interface": interface, "launch": _owned_evidence(adapter, state),
                    "source_and_model_pins_verified": True}
        evidence_id = digest(evidence)
        state.entity("handoff_launch", evidence_id, evidence)
        adapter.handoff_launch_evidence = {**evidence, "evidence_id": evidence_id}
        return adapter, identifier
    except BaseException:
        if adapter is not None:
            close_adapter(adapter)
        raise


def _isolated_grading(config, result):
    workspace = result.get("workspace")
    root = Path(config["paths"]["agent_workspaces"]).resolve()
    if not workspace or not Path(workspace).resolve().is_relative_to(root):
        return False
    for phase in ("public_grading", "hidden_grading"):
        grade = result.get(phase) or {}
        argv = grade.get("launch_argv")
        if not grade.get("complete") or not isinstance(argv, list) or len(argv) < 2 or argv[0] != config["installed_tools"]["docker"] or argv[1] != "run":
            return False
        for flag, value in (("--network", "none"), ("--user", "65534:65534"), ("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges"), ("--memory", "2g"), ("--cpus", "2"), ("--pids-limit", "128")):
            if flag not in argv or argv.index(flag) + 1 >= len(argv) or argv[argv.index(flag) + 1] != value:
                return False
        if "--read-only" not in argv or any(arg == "--privileged" or arg.startswith("--gpus") for arg in argv):
            return False
        if "localbench.owner=localbench" not in argv:
            return False
        if not any(re.fullmatch(r"(?:[^\s@]+@)?sha256:[a-f0-9]{64}", str(arg)) for arg in argv):
            return False
    return True


def _transcript(config, result, interface, state):
    logs = Path(config["paths"]["logs"]).resolve(strict=True)
    diff = Path(result.get("source_diff", "")).resolve(strict=True)
    if not diff.is_relative_to(logs) or not diff.name.endswith("-source.diff"):
        raise RuntimeError("handoff_demo_diff_not_owned")
    prefix = diff.name[:-len("-source.diff")]
    requests = sorted(logs.glob(prefix + "-turn*-request.json"))
    if not requests:
        raise RuntimeError("handoff_demo_actual_wire_transcript_missing")
    files = [_pin(diff, "demo_diff", state=state)]
    actual_tool_call = False
    for request in requests:
        turn = request.name[:-len("-request.json")]
        for suffix in ("-request.json", "-raw.bin", "-events.jsonl", "-result.json"):
            path = logs / (turn + suffix)
            if not path.is_file() or (suffix == "-raw.bin" and path.stat().st_size == 0):
                raise RuntimeError("handoff_demo_actual_wire_transcript_incomplete")
            files.append(_pin(path, "wire_transcript", state=state))
        for line in (logs / (turn + "-events.jsonl")).read_text(encoding="utf-8").splitlines():
            event = json.loads(line).get("event") or {}
            actual_tool_call |= event.get("type") == "tool_call"
    checkpoint = logs / (prefix + "-checkpoint.json")
    if not checkpoint.is_file() or not actual_tool_call:
        raise RuntimeError("handoff_demo_native_tool_transcript_missing")
    conversation = json.loads(checkpoint.read_text(encoding="utf-8")).get("messages")
    if not isinstance(conversation, list) or not any(message.get("role") == "tool" for message in conversation):
        raise RuntimeError("handoff_demo_executed_tool_response_missing")
    files.append(_pin(checkpoint, "wire_checkpoint", state=state))
    transcript = {"interface": interface, "fixture": "py02", "seed": 42, "replicate": 1,
                  "mode": "isolated-demo", "native_tool_call_observed": True,
                  "files": files, "conversation_checkpoint": str(checkpoint),
                  "public_grading": result.get("public_grading"), "hidden_grading": result.get("hidden_grading")}
    path = logs / (prefix + "-handoff-transcript.json")
    atomic_json(path, transcript)
    return {**transcript, "manifest": _pin(path, "transcript_manifest", state=state)}


def _demo_key(config, plan, plan_id):
    return {"engine": digest(plan["engine_identity"]), "model": plan["model_identity"],
            "effective_settings": plan["requested_settings"],
            "profile": {"sampling": config["default_sampling"], "reasoning": plan["requested_settings"].get("reasoning", "default")},
            "fixture_prompt_hash": digest({"plan_id": plan_id, "fixture": "py02", "seed": 42, "fixture_version": plan["config_identity"]["fixture_version"]}),
            "tokenizer": plan["tokenizer_identity"], "seed": 42, "mode": "isolated-demo", "replicate": 1,
            "kind": "demo", "configuration_id": plan["configuration_id"], "logical_plan_hash": plan_id}


def _entity(state, kind, identifier):
    row = state.db.execute("SELECT data FROM entities WHERE kind=? AND id=?", (kind, identifier)).fetchone()
    return json.loads(row["data"]) if row else None


def _demo_and_prepare(config, state, collector, candidate):
    """Checkpoint a literal plan, stop/relaunch it, and grade py02 in isolation."""
    identifier = candidate["configuration_id"]
    registry = _entity(state, "handoff_plan", identifier)
    original = None
    if registry is None:
        state.check_budget(2 * config["limits"]["server_start_timeout_seconds"] + config["limits"]["per_task_timeout_seconds"])
        model = checkpoint_model(state, candidate["model"], config.get("installed_model_candidates", []))
        try:
            original, actual = make_adapter(config, state, collector, candidate["runtime"], model, candidate["settings"])
            if actual != identifier:
                raise RuntimeError("handoff_candidate_configuration_changed")
            if original.effective.get("effective_context", 0) < 65536 or original.effective.get("effective_slots") != 1:
                raise RuntimeError("handoff_original_effective_64k_or_single_slot_missing")
            metadata = original.metadata()
            pins = _source_pins(state) + _checkpoint_pins(original.model, metadata, state) + _runtime_pins(candidate["runtime"], metadata, original.requested, state)
            grader_pins = Path(config["paths"]["artifacts"]) / "grader-images.json"
            if not grader_pins.is_file():
                raise RuntimeError("handoff_pinned_isolation_images_missing")
            pins.append(_pin(grader_pins, "grader_images", state=state))
            plan = {"run_id": state.run["id"], "configuration_id": identifier, "engine": candidate["engine"],
                    "model": original.model, "runtime": candidate["runtime"], "requested_settings": original.requested,
                    "config_identity": _config_identity(config, state), "engine_identity": engine_identity(metadata),
                    "model_identity": metadata["model_sha256"], "tokenizer_identity": tokenizer_identity(metadata),
                    "file_pins": pins, "interface": _interface(original, candidate["runtime"]),
                    "observed_launch": _owned_evidence(original, state),
                    "reproduction": "launch through localbench agent-endpoint --plan; owned names and log paths regenerated and recorded literally"}
            envelope = {"schema_version": SCHEMA_VERSION, "plan_id": digest(plan), "plan": plan}
            _no_secrets(envelope)
            path = Path(config["paths"]["artifacts"]) / "launch-plans" / (envelope["plan_id"] + ".json")
            _save_immutable(path, envelope)
            stopped = _close_and_prove(original, state)
            original = None
            registry = {"path": str(path.resolve()), "plan_id": envelope["plan_id"], "configuration_id": identifier,
                        "interface": plan["interface"], "original_instance": stopped}
            state.entity("handoff_plan", identifier, registry)
        finally:
            if original is not None:
                close_adapter(original)
    envelope = load_plan(registry["path"])
    plan = envelope["plan"]
    if plan["run_id"] != state.run["id"] or plan["config_identity"] != _config_identity(config, state):
        raise RuntimeError("handoff_run_or_launch_config_changed")
    if envelope["plan_id"] != registry["plan_id"] or plan["configuration_id"] != identifier or not registry["original_instance"].get("stopped"):
        raise RuntimeError("handoff_stopped_launch_checkpoint_invalid")
    if plan["runtime"] != candidate["runtime"] or plan["requested_settings"] != candidate["settings"]:
        raise RuntimeError("handoff_candidate_settings_or_runtime_changed")
    _validate_files(plan, state)
    key = _demo_key(config, plan, envelope["plan_id"])
    job = state.enqueue(key)
    row = state.db.execute("SELECT status,result FROM jobs WHERE id=?", (job,)).fetchone()
    if row["status"] in TERMINAL:
        prior = json.loads(row["result"] or "{}")
        if prior.get("passed"):
            transcript = prior.get("transcript") or {}
            for pin in transcript.get("files", []) + ([transcript["manifest"]] if transcript.get("manifest") else []):
                _pin(pin["path"], pin["role"], pin["sha256"], pin["bytes"], state)
            if not transcript.get("files") or not transcript.get("manifest") or not (prior.get("demo_instance") or {}).get("stopped"):
                raise RuntimeError("handoff_successful_demo_checkpoint_proof_missing")
        return prior
    state.check_budget(config["limits"]["server_start_timeout_seconds"] + config["limits"]["per_task_timeout_seconds"])
    attempt = state.begin(job)
    adapter = None
    result = {"kind": "demo", "configuration_id": identifier, "fixture": "py02", "seed": 42,
              "replicate": 1, "mode": "isolated-demo", "plan_id": envelope["plan_id"],
              "launch_plan": registry["path"], "original_instance": registry["original_instance"],
              "launch_tested_from_stopped_owned_instance": False, "isolated": False,
              "passed": False, "valid": False}
    try:
        adapter, actual = launch_plan(config, state, collector, registry["path"])
        result.update(launch_tested_from_stopped_owned_instance=True, launch_evidence=adapter.handoff_launch_evidence)
        coding = coding_job(config, state, adapter, collector, actual, "py02", 42, replicate=1, mode="isolated-demo")
        result.update(coding_result=coding, valid=bool(coding.get("valid")), reason=coding.get("reason", "demo_failed"),
                      interface=adapter.handoff_launch_evidence["interface"])
        result["isolated"] = _isolated_grading(config, coding)
        if coding.get("passed") and not result["isolated"]:
            result.update(valid=False, reason="isolated_demo_grading_evidence_missing")
        result["transcript"] = _transcript(config, coding, result["interface"], state)
        result["passed"] = bool(coding.get("passed") and result["valid"] and result["isolated"])
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        if str(error) in ("stop_after_current", "budget_exhausted"):
            raise
        result.update(passed=False, valid=False, reason=str(error), error_type=type(error).__name__)
    finally:
        if adapter is not None:
            result["demo_instance"] = _close_and_prove(adapter, state)
    state.finish(attempt, result_status(result), result, reason=result.get("reason"))
    state.entity("demo", identifier, result)
    atomic_json(Path(config["paths"]["artifacts"]) / ("agent-demo-" + identifier + ".json"), result)
    return result


def demo_and_prepare(config, state, collector, candidate):
    """Prepare/demo one finalist; record bounded setup failure and continue."""
    try:
        return _demo_and_prepare(config, state, collector, candidate)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        if str(error) in ("stop_after_current", "budget_exhausted"):
            raise
        identifier = candidate["configuration_id"]
        # Preserve attempt history, but a current demo row whose required files
        # can no longer be verified must not remain a passing handoff gate.
        with state.db:
            rows = state.db.execute("SELECT id,key_json,result FROM jobs WHERE status='passed'").fetchall()
            for row in rows:
                prior_key = json.loads(row["key_json"])
                if prior_key.get("kind") != "demo" or prior_key.get("configuration_id") != identifier:
                    continue
                invalid = json.loads(row["result"] or "{}")
                invalid.update(passed=False, valid=False, checkpoint_validation_failed=True,
                               checkpoint_validation_reason=str(error), checkpoint_validation_at=state.clock())
                state.db.execute("UPDATE jobs SET status='invalid',reason=?,result=? WHERE id=?",
                                 ("demo_checkpoint_validation_failed", canonical(invalid), row["id"]))
        result = {"kind": "demo", "configuration_id": identifier, "fixture": "py02", "seed": 42,
                  "replicate": 1, "mode": "isolated-demo", "passed": False, "valid": False,
                  "isolated": False, "launch_tested_from_stopped_owned_instance": False,
                  "reason": str(error), "error_type": type(error).__name__, "phase": "prepare_or_checkpoint_validation"}
        # A failed prerequisite is an infrastructure/demo outcome. It cannot
        # satisfy the quality or reproducible endpoint gates in the report.
        try:
            model = checkpoint_model(state, candidate["model"], config.get("installed_model_candidates", []))
        except (RuntimeError, ValueError, KeyError):
            model = {"id": candidate["model"], "unverified_registry_entry": True}
        key = {"engine": digest(candidate["runtime"]), "model": model.get("sha256") or digest(model),
               "effective_settings": candidate["settings"], "profile": config["default_sampling"],
               "fixture_prompt_hash": digest({"demo_setup": identifier, "reason": str(error)}),
               "tokenizer": "unverified_demo_setup", "seed": 42, "mode": "isolated-demo", "replicate": 1,
               "kind": "demo", "configuration_id": identifier}
        job = state.enqueue(key)
        row = state.db.execute("SELECT status FROM jobs WHERE id=?", (job,)).fetchone()
        if row["status"] == "pending":
            state.finish(state.begin(job), "invalid", result, reason=result["reason"])
        state.entity("demo", identifier, result)
        atomic_json(Path(config["paths"]["artifacts"]) / ("agent-demo-" + identifier + ".json"), result)
        return result
