"""Public, authenticated LM Studio control of one uniquely named owned instance."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
import uuid

from ..config import atomic_json, digest, file_hash
from ..doctor import sanitize
from .base import EngineAdapter, LocalHTTP


DOCS = {"load": "https://lmstudio.ai/docs/cli/local-models/load",
        "instances": "https://lmstudio.ai/docs/developer/rest/list",
        "unload": "https://lmstudio.ai/docs/developer/rest/unload",
        "sampling": "https://lmstudio.ai/docs/developer/openai-compat/chat-completions"}


def _without_credentials():
    environment = os.environ.copy()
    for name in list(environment):
        lowered = name.lower()
        if any(term in lowered for term in ("token", "password", "secret", "api_key", "apikey")):
            environment.pop(name, None)
    return environment


def _logged_command(config, label, argv, timeout=30, environment=None):
    path = Path(config["paths"]["logs"]) / (label + "-" + str(time.time_ns()) + ".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout, env=environment or _without_credentials(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = completed.stdout.decode("utf-8", errors="replace")
        code = completed.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        output, code = type(error).__name__, None
    path.write_text(sanitize(output), encoding="utf-8")
    return {"argv": argv, "exit_code": code, "output": sanitize(output), "log": str(path)}


def _verified_tokenizer(helper, model_sha256, engine):
    if helper is None or getattr(helper, "offline", False) is not True or \
            getattr(helper, "verified", False) is not True or \
            getattr(helper, "model_sha256", None) != model_sha256 or \
            engine not in getattr(helper, "verified_engines", ()):
        raise RuntimeError("exact_offline_template_tokenizer_unavailable:" + engine + ";exploratory_only")
    return helper


class LMStudioAdapter(EngineAdapter):
    """Uses existing 127.0.0.1:1234 service; never starts/stops or configures it.

    CLI --identifier is the public way to choose a unique instance identifier.
    REST load currently exposes more controls but no documented identifier.
    Therefore those extra load controls are explicitly unavailable in this safe
    adapter, rather than modifying the application's private configuration.
    """

    SUPPORTED_SETTINGS = {"context", "gpu", "ttl", "reasoning"}
    SUPPORTED_REQUEST = {"model", "messages", "tools", "tool_choice", "stream", "stream_options",
        "max_tokens", "temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty",
        "repeat_penalty", "seed", "stop", "logit_bias"}

    def __init__(self, config, state, model, settings=None, exact_tokenizer=None):
        self.config, self.state, self.model = config, state, model
        self.binary = Path(config["installed_tools"]["lms"])
        self.port = 1234
        self.http = LocalHTTP("http://127.0.0.1:1234", config["limits"]["per_64k_request_timeout_seconds"],
                              credential_env="LM_BENCH_TOKEN")
        self.requested = {"context": 65536, "gpu": "max", "ttl": 1800, "reasoning": "default"}
        unknown = set(settings or {}) - self.SUPPORTED_SETTINGS
        if unknown:
            raise RuntimeError("unsupported_public_lm_studio_load_controls:" + ",".join(sorted(unknown)))
        self.requested.update(settings or {})
        if self.requested["reasoning"] != "default":
            raise RuntimeError("unsupported_lm_studio_openai_reasoning_control")
        if type(self.requested["context"]) is not int or self.requested["context"] < 65536:
            raise ValueError("LM Studio primary context must be at least 65536")
        self.exact_tokenizer = exact_tokenizer
        self.process, self.instance_id = None, None
        self.effective, self._metadata = {}, {}
        self.snapshot, self.saved = None, None

    def _auth(self):
        if not os.environ.get("LM_BENCH_TOKEN"):
            raise RuntimeError("lm_studio_auth_unavailable:LM_BENCH_TOKEN_not_set;existing_service_unchanged")

    def _models(self):
        self._auth()
        try:
            data = self.http.json("/api/v1/models")
        except RuntimeError as error:
            if "401" in str(error) or "403" in str(error):
                raise RuntimeError("lm_studio_auth_unavailable:credential_rejected") from None
            raise
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise RuntimeError("invalid_lm_studio_model_inventory")
        return data

    @staticmethod
    def _instances(inventory):
        return {instance["id"]: {"model_key": model["key"], "config": instance.get("config", {})}
            for model in inventory["models"] for instance in model.get("loaded_instances", [])
            if isinstance(instance, dict) and isinstance(instance.get("id"), str)}

    def discover_capabilities(self):
        self._auth()
        if not self.binary.is_file():
            raise RuntimeError("missing_lm_studio_cli")
        self.state.start_execution("runtime_probe:lm-studio")
        version = _logged_command(self.config, "lm-studio-version", [str(self.binary), "--version"])
        help_result = _logged_command(self.config, "lm-studio-load-help", [str(self.binary), "load", "--help"])
        if version["exit_code"] != 0 or help_result["exit_code"] != 0:
            raise RuntimeError("lm_studio_cli_probe_failed")
        for flag in ("--identifier", "--context-length", "--gpu", "--ttl"):
            if flag not in help_result["output"]:
                raise RuntimeError("unsupported_lm_studio_cli_flag:" + flag)
        inventory = self._models()
        runtime = _logged_command(self.config, "lm-studio-runtime-versions", [str(self.binary), "runtime", "ls"])
        self._metadata = {"engine_id": "lm-studio", "binary": str(self.binary),
            "binary_sha256": file_hash(self.binary), "version_output": version["output"],
            "version_log": version["log"], "public_load_help_log": help_result["log"],
            "backend_runtime_versions": runtime["output"] if runtime["exit_code"] == 0 else None,
            "backend_runtime_log": runtime["log"], "model_path": self.model["path"],
            "model_sha256": self.model.get("sha256") or file_hash(self.model["path"]),
            "requested_settings": dict(self.requested), "documentation": DOCS,
            "credential_source": "LM_BENCH_TOKEN process environment only",
            "limitations": ["Exact templated tokenization requires a separately verified offline helper",
                "Existing LM Studio service PID is not owned; cache isolation after instance reload is unverified",
                "Unique CLI instance exposes context/GPU/TTL; extra REST batch/cache controls are not requested",
                "No public OpenAI control is assumed to disable context shifting",
                "Backend selected for this model must be pinned from actually observed runtime evidence"]}
        self.state.entity("engine", digest(self._metadata), self._metadata)
        return {"available_models": [model["key"] for model in inventory["models"]],
            "public_load_controls": sorted(self.SUPPORTED_SETTINGS), "tokenization": "verified_offline_helper_required",
            "cache_reset": "owned_instance_reload;isolation_unverified", "endpoint": "http://127.0.0.1:1234"}

    def launch(self, config=None):
        if not self._metadata:
            self.discover_capabilities()
        self.state.check_budget(self.config["limits"]["server_start_timeout_seconds"])
        inventory = self._models()
        existing = self._instances(inventory)
        if existing:
            raise RuntimeError("lm_studio_foreign_loaded_instances:refusing_GPU_overlap_or_app_state_changes")
        key = self.model.get("lms_key") or (self.model.get("candidate") or {}).get("lms_key")
        selected = [model for model in inventory["models"] if key and
            (model["key"] == key or key in model.get("variants", []))]
        if len(selected) != 1:
            raise RuntimeError("installed_lm_studio_model_key_unavailable")
        expected_key = selected[0]["key"]
        self.snapshot = inventory
        self.instance_id = "localbench-" + uuid.uuid4().hex
        argv = [str(self.binary), "load", key, "--yes", "--identifier", self.instance_id,
                "--context-length", str(self.requested["context"]), "--gpu", str(self.requested["gpu"]),
                "--ttl", str(self.requested["ttl"])]
        began = time.perf_counter()
        self.saved={'owner':'localbench','run_id':self.state.run['id'],'instance_id':self.instance_id,
            'model_key':expected_key,'selected_model_variant':key,'load_config':{},'phase':'pending_instance_config',
            'load_completed':False,'endpoint':'http://127.0.0.1:1234','argv':argv,'original_loaded_instance_ids':sorted(existing)}
        atomic_json(Path(self.config['paths']['state'])/'owned-lm-studio-instance.json',self.saved)
        loaded = _logged_command(self.config, "lm-studio-owned-load", argv,
            timeout=self.config["limits"]["server_start_timeout_seconds"])
        self.saved.update(load_completed=loaded['exit_code']==0,load_log=loaded['log'])
        # With --yes, this CLI selection failure exits before submitting a load.
        # Other failed exits/timeouts remain genuinely uncertain and fail closed.
        self.saved['load_not_started'] = (loaded['exit_code'] == 1 and
            'Model not found' in loaded.get('output', '') and
            'No model found that matches model key' in loaded.get('output', '') and
            key in loaded.get('output', ''))
        atomic_json(Path(self.config['paths']['state'])/'owned-lm-studio-instance.json',self.saved)
        observed = self._instances(self._models())
        owned = observed.get(self.instance_id)
        if owned is None:
            self.unload_owned()
            if self.saved.get('load_not_started'):
                raise RuntimeError("lm_studio_cli_model_not_found_no_load_submitted:" + loaded["log"])
            raise RuntimeError("lm_studio_owned_instance_load_failed:" + loaded["log"])
        self.saved = {"owner": "localbench", "run_id": self.state.run["id"], "instance_id": self.instance_id,
            "model_key": expected_key, "selected_model_variant": key, "load_config": owned["config"], "endpoint": "http://127.0.0.1:1234",
            "argv": argv, "load_log": loaded["log"], "original_loaded_instance_ids": sorted(existing)}
        atomic_json(Path(self.config["paths"]["state"]) / "owned-lm-studio-instance.json", self.saved)
        context = owned["config"].get("context_length")
        self.effective = {"effective_context": context, "effective_slots": owned["config"].get("parallel"),
            "public_load_config": owned["config"], "context_shift_disabled": False,
            "cache_isolation_verified": False, "startup_seconds": time.perf_counter() - began,
            "launch_argv": argv, "instance_id": self.instance_id,
            "sampler_effective": "supported public request controls;actual controls recorded per request"}
        if loaded["exit_code"] != 0 or owned["model_key"] != expected_key or type(context) is not int or context < 65536 or len(observed) != 1:
            self.unload_owned()
            raise RuntimeError("lm_studio_effective_context_or_load_unverified")
        # The scheduler registers the stable effective configuration.
        return {"endpoint": self.saved["endpoint"], "handle": self.saved, "effective": self.effective}

    def health(self):
        try:
            inventory = self._models()
            return self.instance_id in self._instances(inventory) if self.instance_id else True
        except (OSError, RuntimeError, ValueError):
            return False

    def tokenize(self, messages, tools):
        helper = _verified_tokenizer(self.exact_tokenizer, self._metadata.get("model_sha256"), "lm-studio")
        result = helper.tokenize(messages, tools)
        if not isinstance(result, dict) or type(result.get("count")) is not int or result["count"] < 0:
            raise RuntimeError("invalid_offline_tokenizer_count")
        return result

    def tokenize_text(self, text):
        return _verified_tokenizer(self.exact_tokenizer, self._metadata.get("model_sha256"), "lm-studio").tokenize_text(text)

    def stream_chat(self, request, **kwargs):
        if not self.instance_id or not self.saved:
            raise RuntimeError("lm_studio_owned_instance_not_loaded")
        unknown = set(request) - self.SUPPORTED_REQUEST
        if unknown:
            raise RuntimeError("unsupported_public_lm_studio_request_controls:" + ",".join(sorted(unknown)))
        instances = self._instances(self._models())
        if set(instances) - {self.instance_id}:
            raise RuntimeError("lm_studio_foreign_loaded_instances:refusing_GPU_overlap")
        owned = instances.get(self.instance_id)
        if owned != {"model_key": self.saved["model_key"], "config": self.saved["load_config"]}:
            raise RuntimeError("lm_studio_owned_instance_identity_changed")
        return self.http.measure({**request, "model": self.instance_id}, **kwargs)

    def reset_cache(self):
        return {"status": "owned_instance_reload_required", "verified": False}

    def fresh_restart(self):
        self.unload_owned()
        return self.launch()

    def unload_owned(self):
        if self.instance_id is None or self.saved is None:
            return
        observed = self._instances(self._models()).get(self.instance_id)
        pending=self.saved.get('phase')=='pending_instance_config'
        if pending and observed is not None:
            if observed['model_key']!=self.saved['model_key'] or self.instance_id in self.saved['original_loaded_instance_ids'] or self._instances(self._models()).get(self.instance_id)!=observed:
                raise RuntimeError('lm_studio_pending_load_identity_changed:refusing_unload')
            self.saved={**self.saved,'load_config':observed['config'],'phase':'observed_owned_instance'}
            atomic_json(Path(self.config['paths']['state'])/'owned-lm-studio-instance.json',self.saved)
        expected = {"model_key": self.saved["model_key"], "config": self.saved["load_config"]}
        if observed is None:
            if pending and not self.saved.get('load_completed') and not self.saved.get('load_not_started'):raise RuntimeError('lm_studio_pending_load_completion_unknown')
            if self._instances(self._models()).get(self.instance_id) is not None:raise RuntimeError('lm_studio_owned_absence_unverified')
            atomic_json(Path(self.config['paths']['state'])/'owned-lm-studio-instance.json',{**self.saved,'unloaded':True,'unload_proof':'owned identifier absent after completed load and repeated inventory'})
            self.instance_id = None
            return
        if not self.instance_id.startswith("localbench-") or observed != expected:
            raise RuntimeError("lm_studio_owned_instance_identity_changed:refusing_unload")
        if self._instances(self._models()).get(self.instance_id)!=expected:raise RuntimeError('lm_studio_owned_instance_identity_changed:refusing_unload')
        unloaded = self.http.json("/api/v1/models/unload", {"instance_id": self.instance_id})
        if unloaded.get("instance_id") != self.instance_id or self.instance_id in self._instances(self._models()):
            raise RuntimeError("lm_studio_owned_unload_unverified")
        self.instance_id = None
        atomic_json(Path(self.config["paths"]["state"]) / "owned-lm-studio-instance.json",
                    {**self.saved, "unloaded": True})

    def metadata(self):
        return {**self._metadata, "effective_settings": self.effective}


LMSAdapter = LMStudioAdapter
