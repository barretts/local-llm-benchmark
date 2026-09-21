"""Owned Windows TabbyAPI/EXL3 adapter with observable 64K/token gates."""
from pathlib import Path
import ctypes
import json
import os
import re
import shutil
import subprocess
import time

from .base import EngineAdapter, LocalHTTP
from .native import creation_time
from ..config import atomic_json, digest, file_hash
from ..ownership import check_port
from ..runtime_build import _WindowsJob, _environment, run_owned
from ..tabby_setup import TABBY_COMMIT, EXL3_COMMIT, WHEEL_SHA


def _cache_mode(value):
    normalized = str(value).upper().replace(" ", "")
    return {"Q8": "8,8", "Q6": "6,6", "Q4": "4,4", "FP16": "FP16"}.get(normalized, normalized)


def validate_exl3(model):
    root = Path(model["path"])
    if not root.is_dir() or not (root / "config.json").is_file():
        raise RuntimeError("exl3_checkpoint_directory_required")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or {}
    if str(quant.get("quant_method", "")).lower() != "exl3":
        raise RuntimeError("checkpoint_is_not_exl3")
    context = config.get("max_position_embeddings") or (config.get("text_config") or {}).get("max_position_embeddings")
    if type(context) is not int or context < 65536:
        raise RuntimeError("exl3_checkpoint_context_below_64k")
    if (root / "tabby_config.yml").exists() or (root / "tabby_config.yaml").exists():
        raise RuntimeError("checkpoint_has_unreviewed_tabby_overrides")
    weights = [item for item in model.get("files", []) if item["filename"].endswith(".safetensors") and item.get("weight", True)]
    if not weights:
        raise RuntimeError("exl3_checkpoint_missing_verified_weight_manifest")
    for item in weights:
        name = item["filename"]
        if "/" in name or "\\" in name or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name):
            raise RuntimeError("exl3_checkpoint_requires_root_shards")
        path = root / name
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise RuntimeError("exl3_weight_hash_or_size_mismatch")
    return {"context_claim": context, "quantization_config": quant, "config_sha256": file_hash(root / "config.json"),
            "weights": weights, "weight_manifest_sha256": digest(weights)}


def owned_config(model_path, port, settings):
    if settings["context"] != 65536 or settings["slots"] != 1 or settings["cache_size"] < 65536 or settings["chunk_size"] != 2048:
        raise ValueError("Tabby primary requires 65536 context/cache, one slot and chunk 2048")
    cache = _cache_mode(settings["cache_mode"])
    if cache != "FP16" and not re.fullmatch(r"[2-8],[2-8]", cache):
        raise ValueError("unsupported Tabby cache pair")
    reasoning = settings["reasoning"]
    if reasoning not in ("default", "reduced", "disabled"):
        raise ValueError("unsupported Tabby reasoning profile")
    path = Path(model_path)
    vars_force = {"enable_thinking": False} if reasoning == "disabled" else {}
    return {
        "network": {"host": "127.0.0.1", "port": port, "disable_auth": True, "allowed_origins": [],
                    "disable_fetch_requests": True, "send_tracebacks": False, "api_servers": ["OAI"], "access_log": False},
        "logging": {"log_prompt": False, "log_generation_params": False, "log_requests": False,
                    "log_live_status": False, "log_chat_completion_requests": False},
        "model": {"model_dir": str(path.parent), "model_name": path.name, "backend": "exllamav3",
                  "max_seq_len": 65536, "cache_size": settings["cache_size"], "cache_mode": cache,
                  "max_batch_size": 1, "chunk_size": 2048, "vision": False, "tensor_parallel": False,
                  "inline_model_loading": False, "use_dummy_models": False, "tool_format": "qwen3_5",
                  "reasoning": True, "reasoning_start_token": "<think>", "reasoning_end_token": "</think>",
                  "template_vars_force": vars_force, "reasoning_budget_tokens": 256 if reasoning == "reduced" else 0 if reasoning == "disabled" else None},
        "draft_model": {"draft_mode": "disabled"}, "sampling": {"override_preset": None},
        "memory": {"sysmem_kv_cache": 0, "sysmem_recurrent_cache": 4096, "sysmem_multimodal_cache": 0},
        "developer": {"unsafe_launch": False, "realtime_process_priority": False, "seqlog": False},
    }


class TabbyAdapter(EngineAdapter):
    def __init__(self, config, state, model, runtime, settings=None):
        self.config, self.state, self.model, self.runtime = config, state, dict(model), runtime
        self.engine = "exllamav3-tabby"
        self.port = config["private_ports"]["tabby"]
        self.http = LocalHTTP(f"http://127.0.0.1:{self.port}", config["limits"]["per_64k_request_timeout_seconds"])
        self.requested = {"context": 65536, "slots": 1, "cache_size": 65536, "cache_mode": "FP16",
                          "chunk_size": 2048, "reasoning": "default", "speculation": "off"}
        self.requested.update(settings or {})
        if "cache_k" in self.requested or "cache_v" in self.requested:
            pair = [self.requested.get("cache_k", "f16"), self.requested.get("cache_v", "f16")]
            bits = {"f16": 16, "q8_0": 8, "q4_0": 4}
            if any(value not in bits for value in pair):
                raise ValueError("unsupported common cache types for Tabby")
            if pair == ["f16", "f16"]:
                self.requested["cache_mode"] = "FP16"
            elif "f16" in pair:
                raise ValueError("Tabby mixed FP16/quantized cache is unsupported")
            else:
                self.requested["cache_mode"] = str(bits[pair[0]]) + "," + str(bits[pair[1]])
        if self.requested["speculation"] != "off":
            raise ValueError("Tabby primary speculation is disabled")
        owned_config(self.model["path"], self.port, self.requested)
        self.process, self.job, self.handles = None, None, []
        self.effective, self._metadata = {}, {}
        self.launch_count = 0
        self.chat_fields = set()

    def discover_capabilities(self):
        if self.runtime.get("tabby_commit") != TABBY_COMMIT or self.runtime.get("exllamav3_commit") != EXL3_COMMIT or self.runtime.get("wheel", {}).get("sha256") != WHEEL_SHA:
            raise RuntimeError("Tabby runtime manifest pin mismatch")
        if not Path(self.runtime["python"]).is_file() or not Path(self.runtime["entrypoint"]).is_file():
            raise RuntimeError("Tabby isolated runtime is not installed")
        for name, sha in self.runtime["source_script_hashes"].items():
            if file_hash(Path(self.runtime["source"]) / name) != sha:
                raise RuntimeError("Tabby source changed after setup")
        self.checkpoint = validate_exl3(self.model)
        self.model.setdefault("sha256", self.checkpoint["weights"][0]["sha256"] if len(self.checkpoint["weights"]) == 1 else self.checkpoint["weight_manifest_sha256"])
        self.state.start_execution("runtime_probe:exllamav3-tabby")
        if self.state.get_control('runtime_first_probe:'+self.engine) is None:self.state.control('runtime_first_probe:'+self.engine,self.state.clock())
        help_result = run_owned(self.config, self.state, "tabby-help", [self.runtime["python"], self.runtime["entrypoint"], "--help"],
                                cwd=self.runtime["source"], timeout=30)
        if help_result["exit_code"] != 0:
            raise RuntimeError("tabby_help_failed:" + help_result["log"])
        if "--config" not in help_result.get("output", ""):
            raise RuntimeError("tabby_actual_cli_missing_config_override")
        self._metadata = {"engine_id": self.engine, "tabby_commit": TABBY_COMMIT, "exllamav3_commit": EXL3_COMMIT,
                          "exllamav3_wheel_sha256": WHEEL_SHA, "runtime_manifest_path": self.runtime["manifest_path"],
                          "runtime_manifest_sha256": file_hash(self.runtime["manifest_path"]), "packages": self.runtime["packages"],
                          "dependency_lock_sha256": self.runtime["dependency_lock_sha256"], "help_log": help_result["log"],
                          "checkpoint": self.checkpoint, "model_path": self.model["path"], "model_sha256": self.model["sha256"],
                          "requested_settings": self.requested, "seed_support": "not exposed unless observed schema says otherwise"}
        return {"cache_reset": "owned_restart_required", "tokenization": "apply-template + v1/token/encode",
                "reasoning": "numeric server budget and checkpoint template variable", "help_log": help_result["log"]}

    def launch(self, config=None):
        if not self._metadata:
            self.discover_capabilities()
        self.state.check_budget(estimated_seconds=self.config["limits"]["server_start_timeout_seconds"])
        check_port(self.port)
        if os.name != "nt":
            raise RuntimeError("native Tabby adapter requires Windows")
        if ctypes.windll.shell32.IsUserAnAdmin():
            raise RuntimeError("administrator_runtime_forbidden")
        self.launch_count += 1
        instance = Path(self.runtime["root"]) / ("instance-" + digest(self.requested)[:12] + "-" + str(self.launch_count))
        instance.mkdir(parents=True, exist_ok=True)
        if (instance / "api_tokens.yml").exists():
            raise RuntimeError("unexpected_tabby_auth_file")
        local = owned_config(self.model["path"], self.port, self.requested)
        # JSON is a YAML1.2 subset and escapes Windows paths correctly.
        config_path = instance / "config.yml"
        atomic_json(config_path, local)
        for name in ("templates", "sampler-overrides"):
            source = Path(self.runtime["source"]) / name
            if source.is_dir():
                shutil.copytree(source, instance / name, dirs_exist_ok=True)
        (instance / "tmp").mkdir(exist_ok=True)
        env = _environment()
        env.update({"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "HF_HOME": str(Path(self.runtime["root"]) / "hf-cache"), "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
                    "TEMP": str(instance / "tmp"), "TMP": str(instance / "tmp"),
                    "TRITON_CACHE_DIR": str(Path(self.runtime["root"]) / "triton-cache")})
        prefix = Path(self.config["paths"]["logs"]) / ("tabby-" + str(time.time_ns()))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        stdout, stderr = Path(str(prefix) + "-stdout.log"), Path(str(prefix) + "-stderr.log")
        self.handles = [stdout.open("wb"), stderr.open("wb")]
        argv = [self.runtime["python"], self.runtime["entrypoint"], "--config", str(config_path)]
        begun = time.perf_counter()
        try:
            self.job = _WindowsJob()
            self.process = subprocess.Popen(argv, cwd=instance, env=env, stdin=subprocess.DEVNULL,
                                            stdout=self.handles[0], stderr=self.handles[1],
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | 0x4)
            self.job.attach_resume(self.process)
            self.saved = {"owner": "localbench", "run_id": self.state.run["id"], "pid": self.process.pid,
                          "created": creation_time(self.process.pid), "command_hash": digest(argv), "argv": argv,
                          "endpoint": f"http://127.0.0.1:{self.port}", "config_path": str(config_path),
                          "config_sha256": file_hash(config_path), "stdout": str(stdout), "stderr": str(stderr)}
            atomic_json(Path(self.config["paths"]["state"]) / ("owned-" + str(self.port) + ".json"), self.saved)
            if hasattr(self, "collector"):
                self.collector.mark_owned_pid(self.process.pid)
            deadline = time.monotonic() + self.config["limits"]["server_start_timeout_seconds"]
            while time.monotonic() < deadline:
                self.state.check_budget()
                if self.process.poll() is not None:
                    raise RuntimeError("tabby_runtime_crash:" + str(stderr))
                if self.health():
                    break
                time.sleep(.25)
            else:
                raise RuntimeError("tabby_server_start_timeout:" + str(stderr))
            if (instance / "api_tokens.yml").exists():
                raise RuntimeError("tabby_generated_unexpected_credentials")
            props = self.http.json("/props")
            models = self.http.json("/v1/models")
            atomic_json(str(prefix) + "-props.json", props)
            atomic_json(str(prefix) + "-models.json", models)
            self._verify_effective(props, models)
            schema = self.http.json("/openapi.json")
            atomic_json(str(prefix) + "-openapi.json", schema)
            for item in schema.get("components", {}).get("schemas", {}).values():
                fields = item.get("properties", {})
                if "messages" in fields:
                    self.chat_fields.update(fields)
            required = {"tools", "top_k", "min_p", "add_bos_token", "repetition_penalty"}
            if not required.issubset(self.chat_fields):
                raise RuntimeError("tabby_actual_schema_missing_required_controls")
            if self.requested["reasoning"] != "default" and "reasoning_budget_tokens" not in self.chat_fields:
                raise RuntimeError("tabby_actual_schema_missing_reasoning_budget")
            self.effective.update({"startup_seconds": time.perf_counter() - begun, "launch_argv": argv,
                                   "owned_config_sha256": self.saved["config_sha256"], "seed_supported": "seed" in self.chat_fields,
                                   "sampler_controls": sorted(self.chat_fields), "startup_logs": [str(stdout), str(stderr)]})
            self.state.entity("launch", digest(self.saved), self.metadata())
            return {"endpoint": self.saved["endpoint"], "handle": self.saved, "effective": self.effective}
        except BaseException:
            self.unload_owned()
            raise

    def _verify_effective(self, props, models):
        entries = [entry for entry in models.get("data", []) if entry.get("id") == Path(self.model["path"]).name]
        if len(entries) != 1:
            raise RuntimeError("tabby_loaded_model_identity_unverified")
        parameters = entries[0].get("parameters", {})
        context = props.get("default_generation_settings", {}).get("n_ctx")
        slots = props.get("total_slots")
        path = props.get("model_path")
        if not isinstance(path, str) or Path(path).resolve() != Path(self.model["path"]).resolve():
            raise RuntimeError("tabby_loaded_model_path_unverified")
        if context != 65536 or parameters.get("max_seq_len") != 65536 or slots != 1 or parameters.get("max_batch_size") != 1:
            raise RuntimeError("unverified_or_insufficient_per_slot_context")
        if parameters.get("cache_size", 0) < 65536 or parameters.get("chunk_size") != 2048 or _cache_mode(parameters.get("cache_mode")) != _cache_mode(self.requested["cache_mode"]):
            raise RuntimeError("tabby_effective_cache_or_chunk_mismatch")
        self.effective = {**self.requested, "effective_context": context, "effective_slots": slots,
                          "effective_cache_size": parameters["cache_size"], "effective_cache_mode": parameters["cache_mode"],
                          "effective_chunk_size": parameters["chunk_size"], "model_parameters": parameters,
                          "reasoning_method": "reasoning_budget_tokens=256" if self.requested["reasoning"] == "reduced" else "enable_thinking=false,reasoning_budget_tokens=0" if self.requested["reasoning"] == "disabled" else "checkpoint default",
                          "cache_isolation": "owned restart and disjoint warmup", "context_shift_disabled": True}

    def health(self):
        try:
            response = LocalHTTP(f"http://127.0.0.1:{self.port}", 2).json("/health")
            return response.get("status") == "healthy" and not response.get("issues")
        except (OSError, RuntimeError, ValueError):
            return False

    def tokenize(self, messages, tools):
        vars_ = {"enable_thinking": False} if self.requested["reasoning"] == "disabled" else {}
        rendered = self.http.json("/apply-template", {"messages": messages, "tools": tools,
                                  "add_generation_prompt": True, "template_vars": vars_})
        prompt = rendered.get("prompt")
        if not isinstance(prompt, str) or any(tool["function"]["name"] not in prompt for tool in tools):
            raise RuntimeError("tabby_template_omits_prompt_or_tools")
        encoded = self.http.json("/v1/token/encode", {"text": prompt, "add_bos_token": True, "encode_special_tokens": True})
        tokens = encoded.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens) or encoded.get("length") != len(tokens):
            raise RuntimeError("tabby_invalid_exact_tokenization")
        return {"count": len(tokens), "provenance": "owned Tabby /apply-template + /v1/token/encode add_bos_token=true; actual generation usage must match",
                "template_sha256": digest(prompt), "tokenizer_engine_sha256": self._metadata["runtime_manifest_sha256"]}

    def tokenize_text(self, text):
        result = self.http.json("/v1/token/encode", {"text": text, "add_bos_token": False, "encode_special_tokens": False})
        if not isinstance(result.get("tokens"), list) or any(type(token) is not int for token in result["tokens"]):
            raise RuntimeError("tabby_invalid_text_tokenization")
        return result["tokens"]

    def stream_chat(self, request, **kwargs):
        body = dict(request, model=Path(self.model["path"]).name, add_bos_token=True)
        if "repeat_penalty" in body:
            body["repetition_penalty"] = body.pop("repeat_penalty")
        if "seed" in body and "seed" not in self.chat_fields:
            body.pop("seed")
        if self.requested["reasoning"] == "reduced":
            body["reasoning_budget_tokens"] = 256
        elif self.requested["reasoning"] == "disabled":
            body["reasoning_budget_tokens"] = 0
            body["template_vars"] = {**body.get("template_vars", {}), "enable_thinking": False}
        return self.http.measure(body, **kwargs)

    def reset_cache(self):
        return {"status": "owned_server_restart_required", "verified": False}

    def fresh_restart(self):
        self.unload_owned()
        return self.launch()

    def unload_owned(self):
        if self.process is not None:
            expected = getattr(self, "saved", {}).get("created")
            if expected is not None and self.process.poll() is None and creation_time(self.process.pid) != expected:
                raise RuntimeError("stale_pid_identity: refusing termination")
        if self.job is not None:
            self.job.close()  # Original kernel job contains only this owned tree.
            self.job = None
        if self.process is not None and self.process.poll() is None:
            self.process.wait(timeout=15)
        if self.process is not None and hasattr(self, "collector"):
            self.collector.mark_owned_pid(self.process.pid, owned=False)
        self.process = None
        for handle in self.handles:
            handle.close()
        self.handles = []

    def metadata(self):
        return {**self._metadata, "effective_settings": self.effective}
