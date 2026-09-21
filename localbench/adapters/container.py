"""Pinned, owned Docker inference; no pulls, installers or global changes.

Saved launch leads are intentions. Every emitted engine flag must occur in the
help from the exact locally inspected image; actual startup/API observations
establish the window. Linux tokenization uses a CPU helper in the same pinned
image and checkpoint; final engine input usage must cross-check its counts.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from ..config import atomic_json, digest, file_hash
from ..doctor import sanitize
from ..ownership import check_port, validate_container
from .base import EngineAdapter, LocalHTTP


PROFILES = {
    "ik-llama": {"port": "ik", "container_port": 8080,
        "tag": "ghcr.io/ikawrakow/ik-llama-cpp:cu12-server"},
    "vllm": {"port": "vllm", "container_port": 8000,
        "tag": "vllm/vllm-openai:v0.29.0"},
    "sglang": {"port": "sglang", "container_port": 30000,
        "tag": "lmsysorg/sglang:v0.5.19-cu129"},
}
IMAGE_FORMAT = '{"id":{{json .Id}},"digests":{{json .RepoDigests}},"os":{{json .Os}},"architecture":{{json .Architecture}},"labels":{{json .Config.Labels}},"entrypoint":{{json .Config.Entrypoint}}}'
CONTAINER_FORMAT = '{"id":{{json .Id}},"created":{{json .Created}},"started":{{json .State.StartedAt}},"running":{{json .State.Running}},"exit_code":{{json .State.ExitCode}},"oom_killed":{{json .State.OOMKilled}},"image":{{json .Image}},"labels":{{json .Config.Labels}}}'
TOKENIZER_HELPER = '''import json, sys
from transformers import AutoTokenizer
request = json.load(sys.stdin)
tokenizer = AutoTokenizer.from_pretrained('/models/snapshot', local_files_only=True, trust_remote_code=False)
if request['operation'] == 'text':
    tokens = tokenizer.encode(request['text'], add_special_tokens=False)
else:
    rendered = tokenizer.apply_chat_template(request['messages'], tools=request['tools'], tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str):
        raise ValueError('template did not return text')
    if request['tools'] and not all(tool['function']['name'] in rendered for tool in request['tools']):
        raise ValueError('template omitted tool schemas')
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
print('\\nLOCALBENCH_TOKENIZER_JSON:' + json.dumps({'count': len(tokens), 'tokens': tokens, 'provenance': 'pinned container AutoTokenizer.apply_chat_template+encode add_special_tokens=False;actual usage cross-check required'}), flush=True)
'''


def classify_failure(output, oom_killed=False):
    text = str(output).lower()
    if oom_killed or any(s in text for s in ("out of memory", "cuda error 2", "failed to allocate", "not enough memory", "no available memory")):
        return {"reason": "model_oom", "transient": False}
    if "illegal instruction" in text:
        return {"reason": "unsupported_cpu_instruction_set", "transient": False}
    if any(s in text for s in ("no kernel image", "invalid device function", "requires sm90", "requires sm100", "compute capability", "unsupported architecture")):
        return {"reason": "unsupported_kernel_or_architecture", "transient": False}
    if any(s in text for s in ("unrecognized arguments", "unknown argument", "invalid choice", "unknown option")):
        return {"reason": "unsupported_engine_option", "transient": False}
    return {"reason": "container_runtime_crash", "transient": False}


def _safe_environment():
    return {key: value for key, value in os.environ.items()
            if not any(term in key.lower() for term in ("token", "secret", "password", "api_key", "apikey"))}


def _clean_output(output):
    # Inspect uses narrow formats, excluding image/container environment dumps.
    return sanitize(re.sub(r'(?i)((?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|LM_BENCH_TOKEN|access_token|auth_token|api[_-]?key|password|secret)["\s]*[:=]["\s]*)([^"\s,}]+)', r'\1[REDACTED]', str(output)))


def _mount_path(path, roots):
    path = Path(path)
    if any(character in str(path) for character in (",", "\r", "\n", "\0")):
        raise ValueError("unsafe Docker mount path")
    resolved = path.resolve()
    allowed = [Path(root).resolve() for root in roots]
    if not any(resolved != root and resolved.is_relative_to(root) for root in allowed):
        raise ValueError("mount must be a selected artifact below an allowed root")
    cursor = path
    while cursor != cursor.parent:
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise ValueError("Docker mount links and junctions are forbidden")
        cursor = cursor.parent
    return resolved


def _nested(data, names):
    if isinstance(data, dict):
        for name in names:
            if name in data:
                return data[name]
        for value in data.values():
            found = _nested(value, names)
            if found is not None:
                return found
    return None


class ContainerHandle:
    """An identity for fresh-restart comparisons, never a Windows/Linux PID."""
    def __init__(self, adapter, container_id):
        self.adapter, self.pid = adapter, container_id
        saved = adapter.saved or {}
        if saved.get("container_id") != container_id or saved.get("id") != container_id or not re.fullmatch(r"[a-f0-9]{64}", str(container_id)):
            raise RuntimeError("container_handle_identity_unverified")
        self._identity = {key: saved.get(key) for key in ("owner", "run_id", "job_id", "id", "container_id", "created", "image")}
        self._completion = None

    def poll(self):
        if self._completion is not None:
            return self._completion["exit_code"]
        saved = self.adapter.saved or {}
        if any(saved.get(key) != value for key, value in self._identity.items()):
            raise RuntimeError("stale_container_handle:refusing_inspection")
        observed = self.adapter._inspect_owned()
        return None if observed["running"] else observed["exit_code"]

    @property
    def removal_evidence(self):
        return json.loads(json.dumps(self._completion)) if self._completion is not None else None

    def _complete_removal(self, observed, proof):
        if observed.get("id") != self.pid or observed.get("created") != self._identity["created"] or observed.get("image") != self._identity["image"] or observed.get("running") is not False or type(observed.get("exit_code")) is not int or not validate_container(observed.get("labels") or {}, self._identity["run_id"], self._identity["job_id"]):
            raise RuntimeError("container_completion_identity_unverified")
        if proof.get("container_id") != self.pid or proof.get("removed") is not True or proof.get("absence_verified") is not True or proof.get("exit_code") != observed["exit_code"]:
            raise RuntimeError("container_completion_removal_unverified")
        self._completion = json.loads(json.dumps(proof))


class ContainerAdapter(EngineAdapter):
    COMMON = {"context", "slots", "memory_gib", "shm_gib", "reasoning", "prefix_cache"}
    IK = {"gpu_layers", "flash_attention", "cache_k", "cache_v", "batch", "ubatch", "threads", "threads_batch", "peg"}
    LINUX = {"gpu_memory_utilization", "prefill_tokens", "tool_parser", "reasoning_parser", "attention_backend", "kv_cache_dtype", "moe_backend", "dtype", "eager", "reasoning_effort", "generation_config", "language_model_only"}
    REQUEST = {"model", "messages", "tools", "tool_choice", "stream", "stream_options", "max_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty", "repetition_penalty", "seed", "stop", "logit_bias", "parallel_tool_calls"}

    def __init__(self, config, state, engine, model, image_digest=None, settings=None, template_helper=None):
        if engine not in PROFILES:
            raise ValueError("unsupported_container_engine")
        if image_digest is not None and not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", image_digest):
            raise ValueError("container image must be a pinned repository digest")
        self.config, self.state, self.engine, self.model = config, state, engine, model
        self.profile, self.docker = PROFILES[engine], config["installed_tools"]["docker"]
        self.image_digest, self.template_helper = image_digest, template_helper
        self.port = config["private_ports"][self.profile["port"]]
        self.http = LocalHTTP(f"http://127.0.0.1:{self.port}", config["limits"]["per_64k_request_timeout_seconds"])
        self.requested = {"context": 65536, "slots": 1, "memory_gib": 28, "shm_gib": 8, "reasoning": "default", "prefix_cache": True}
        if engine == "ik-llama":
            self.requested.update(gpu_layers=999, flash_attention="on", cache_k="f16", cache_v="f16", batch=2048, ubatch=512, threads=16, threads_batch=16, peg=False)
        else:
            self.requested.update(gpu_memory_utilization=.85, prefill_tokens=8192, tool_parser="qwen3_xml" if engine == "vllm" else "qwen3_coder", reasoning_parser="qwen3", attention_backend=None, kv_cache_dtype="auto")
            self.requested.update(moe_backend=None,dtype=None,eager=False,reasoning_effort=None,generation_config=None,language_model_only=True)
            if model.get('id')=='gptoss20b-mxfp4':
                # This pinned GPT-OSS checkpoint has no vision encoder.
                self.requested['language_model_only']=False
                self.requested.update(tool_parser='openai' if engine=='vllm' else 'gpt-oss',reasoning_parser='openai_gptoss' if engine=='vllm' else 'gpt-oss',attention_backend='TRITON_ATTN' if engine=='vllm' else 'triton',moe_backend='marlin' if engine=='vllm' else 'triton_kernel',dtype='bfloat16',eager=True,reasoning_effort='medium',generation_config='vllm' if engine=='vllm' else None)
        unknown = set(settings or {}) - (self.COMMON | (self.IK if engine == "ik-llama" else self.LINUX))
        if unknown:
            raise RuntimeError("unsupported_container_settings:" + ",".join(sorted(unknown)))
        self.requested.update(settings or {})
        self._validate_settings()
        roots = [config["paths"]["installed_model_root"], config["paths"]["new_model_root"]]
        roots += config["paths"].get("model_cache_roots", [])
        roots += [Path.home()/".cache"/"huggingface"/"hub"]
        self.model_path = _mount_path(model["path"], roots)
        self.model_roots = roots
        if engine == "ik-llama":
            if not self.model_path.is_file() or self.model_path.suffix.lower() != ".gguf" or "_XL" in self.model_path.name.upper() or not re.search(r"Q4_K_M|Q6_K", self.model_path.name, re.I):
                raise RuntimeError("ik_requires_existing_plain_Q4_K_M_or_Q6_K_GGUF")
            self.container_model = "/models/model.gguf"
        else:
            if not self.model_path.is_dir() or not (self.model_path/"config.json").is_file() or not any(self.model_path.glob("*.safetensors")):
                raise RuntimeError("complete_local_safetensors_snapshot_required")
            self.container_model = "/models/snapshot"
            index = self.model_path/"model.safetensors.index.json"
            if index.is_file():
                weights = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
                if not isinstance(weights, dict) or not weights or any(not isinstance(filename, str) or not self._selected_file(filename).is_file() for filename in weights.values()):
                    raise RuntimeError("incomplete_local_safetensors_index")
        self.process, self.saved = None, None
        self.handles, self.log_process = [], None
        self.effective, self._metadata = {}, {}
        self.help, self.entrypoint, self.entry_args = None, None, []
        self.launch_count = 0

    def _validate_settings(self):
        s = self.requested
        if type(s["context"]) is not int or s["context"] < 65536 or type(s["slots"]) is not int or s["slots"] != 1:
            raise ValueError("primary container window must be >=65536 in one slot")
        if any(type(s[key]) is not int or not 1 <= s[key] <= maximum for key, maximum in (("memory_gib", 28), ("shm_gib", 8))) or s["shm_gib"] >= s["memory_gib"]:
            raise ValueError("container RAM/shm must remain below the unchanged WSL cap")
        if type(s["prefix_cache"]) is not bool or s["reasoning"] not in ("default", "disabled", "reduced"):
            raise ValueError("invalid container cache/reasoning profile")
        if self.engine == "ik-llama":
            if s["cache_k"] not in ("f16", "q8_0", "q4_0") or s["cache_v"] not in ("f16", "q8_0", "q4_0") or s["flash_attention"] not in ("on", "off", "auto"):
                raise ValueError("unsupported initial ik cache/attention type")
            if any(type(s[key]) is not int or s[key] < 1 for key in ("gpu_layers", "batch", "ubatch", "threads", "threads_batch")) or s["ubatch"] > s["batch"] or type(s["peg"]) is not bool:
                raise ValueError("invalid ik offload/batch/thread setting")
            if not s["prefix_cache"]:
                raise RuntimeError("ik_prefix_cache_disable_unverified;use_owned_restart")
        else:
            if s["reasoning"] != "default":
                raise RuntimeError("unsupported_linux_numeric_or_disable_reasoning_control")
            if type(s['language_model_only']) is not bool:raise ValueError('invalid_language_model_only_control')
            if s['moe_backend'] not in (None,'marlin','triton_kernel') or s['dtype'] not in (None,'auto','float16','bfloat16') or type(s['eager']) is not bool or s['reasoning_effort'] not in (None,'low','medium','high') or s['generation_config'] not in (None,'vllm'):
                raise ValueError('unsupported_linux_backend_or_effort_control')
            if self.engine=='sglang' and s['moe_backend']=='marlin' and self.model.get('id')=='gptoss20b-mxfp4':raise RuntimeError('sglang_mxfp4_marlin_requires_sm90')
            fraction = s["gpu_memory_utilization"]
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or fraction not in (.8, .85, .9) or s["prefill_tokens"] not in (4096, 8192, 16384):
                raise ValueError("invalid bounded Linux memory/prefill setting")
            if s["kv_cache_dtype"] not in ("auto", "fp8", "fp8_e4m3", "fp8_e5m2"):
                raise ValueError("unsupported Linux cache dtype")
            if s["attention_backend"] in ("fa3", "FLASH_ATTN_3", "FLASHMLA", "CUTLASS_MLA", "TRTLLM_MHA"):
                raise RuntimeError("attention_backend_requires_separately_verified_Ada_support")
            for key in ("tool_parser", "reasoning_parser", "attention_backend"):
                value = s[key]
                if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value)):
                    raise ValueError("invalid parser/backend name")

    def _docker(self, label, args, timeout=30, stdin_json=None):
        path = Path(self.config["paths"]["logs"])/(self.engine+"-"+label+"-"+str(time.time_ns())+".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stdin_options = {"stdin": subprocess.DEVNULL} if stdin_json is None else {"input": json.dumps(stdin_json, ensure_ascii=False, allow_nan=False).encode("utf-8")}
            done = subprocess.run([self.docker]+list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, env=_safe_environment(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **stdin_options)
            output, code = done.stdout.decode("utf-8", errors="replace"), done.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            output, code = type(error).__name__, None
        cleaned = _clean_output(output)
        path.write_text(cleaned, encoding="utf-8")
        return {"argv": [self.docker]+list(args), "output": cleaned, "exit_code": code, "log": str(path)}

    def discover_capabilities(self):
        if not all(self.state.get_control(key) for key in ("doctor_verified", "harness_verified", "fixtures_verified")):
            raise RuntimeError("container_probe_requires_verified_host_harness_and_fixtures")
        self.state.check_budget(estimated_seconds=90)
        inspected = self._docker("image-inspect", ["image", "inspect", "--format", IMAGE_FORMAT, self.image_digest or self.profile["tag"]])
        if inspected["exit_code"] != 0:
            raise RuntimeError("pinned_container_image_unavailable;adapter_does_not_pull:"+inspected["log"])
        info = json.loads(inspected["output"])
        if info.get("os") != "linux" or info.get("architecture") != "amd64" or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(info.get("id"))):
            raise RuntimeError("unsupported_container_image_platform_or_identity")
        if self.image_digest is None:
            pins = info.get("digests") or []
            if not pins or not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", pins[0]):
                raise RuntimeError("container_image_has_no_reproducible_repository_digest")
            self.image_digest = pins[0]
        elif self.image_digest not in (info.get("digests") or []):
            raise RuntimeError("container_image_digest_mismatch")
        if self.engine == "ik-llama":
            entry = info.get("entrypoint") or []
            if len(entry) != 1 or Path(entry[0]).name not in ("llama-server", "server"):
                raise RuntimeError("ik_image_direct_server_entrypoint_unverified")
            self.entrypoint, self.entry_args = entry[0], []
            help_args = ["--help"]
        elif self.engine == "vllm":
            self.entrypoint, self.entry_args, help_args = "vllm", ["serve"], ["serve", "--help=all"]
        else:
            self.entrypoint, self.entry_args, help_args = "python3", ["-m", "sglang.launch_server"], ["-m", "sglang.launch_server", "--help"]
        self.state.start_execution("runtime_probe:"+self.engine)
        if self.state.get_control('runtime_first_probe:'+self.engine) is None:self.state.control('runtime_first_probe:'+self.engine,self.state.clock())
        job = "help-"+uuid.uuid4().hex
        helped = self._probe_help(job, help_args, info["id"])
        if helped["exit_code"] != 0:
            raise RuntimeError("container_help_failed:"+helped["log"])
        self.help = helped["output"]
        checkpoint_files = [{key: item.get(key) for key in ("filename", "bytes", "sha256", "git_blob_sha1")} for item in self.model.get("files", [])]
        model_identity = self.model.get("sha256") or (file_hash(self.model_path) if self.model_path.is_file() else digest({"repo": self.model.get("repo"), "revision": self.model.get("revision"), "files": checkpoint_files}) if checkpoint_files and all(item.get("sha256") or item.get("git_blob_sha1") for item in checkpoint_files) else None)
        self._metadata = {"engine_id": self.engine, "image_digest": self.image_digest, "image_id": info["id"], "image_platform": "linux/amd64", "image_revision": (info.get("labels") or {}).get("org.opencontainers.image.revision"), "image_inspect_log": inspected["log"], "help_log": helped["log"], "help_sha256": digest(self.help), "model_path": str(self.model_path), "model_sha256": model_identity, "model_revision": self.model.get("revision"), "requested_settings": dict(self.requested), "trust_remote_code": False, "tokenization_verified": False, "gpu_attribution": "owned container GPU0; Docker/WSL Windows process attribution unverified", "limitations": ["Help acceptance does not prove sampler, parser, cache or Ada kernel effectiveness", "Linux fully templated tokenization requires a verified offline helper", "Actual cache hit/evaluated counters and all capacity/quality gates remain required"]}
        self._metadata.update(model_sha256=model_identity, model_identity_kind="GGUF_file_sha256" if self.model_path.is_file() else "pinned_checkpoint_file_manifest" if model_identity else "unverified_checkpoint_contents", checkpoint_files=checkpoint_files, tokenizer_helper_sha256=hashlib.sha256(TOKENIZER_HELPER.encode()).hexdigest() if self.engine != "ik-llama" else None)
        self._metadata["tokenizer_file_pins"] = [{"filename": filename, "sha256": file_hash(self._selected_file(filename))} for filename in sorted(path.relative_to(self.model_path).as_posix() for path in self.model_path.rglob("*") if path.is_file() and path.suffix in (".json", ".jinja", ".txt", ".model"))] if self.model_path.is_dir() else []
        self._metadata["limitations"][1] = "Linux helper uses the checkpoint template in the pinned image;final actual usage must cross-check the count"
        self.state.entity("engine", digest(self._metadata), self._metadata)
        return {"image_digest": self.image_digest, "image_revision": self._metadata["image_revision"], "help_log": helped["log"], "supported_flags": sorted(set(re.findall(r"--[a-z0-9-]+", self.help))), "cache_reset": "owned_restart_required", "tokenization": "ik endpoints must verify;Linux verified offline helper required"}

    def _probe_help(self, job, help_args, image_id):
        # A timed-out Docker client does not stop its daemon-owned container.
        # Create first so even a timeout can only stop the verified original.
        args = ["create", "--name", "localbench-"+job, "--pull", "never", "--label", "localbench.owner=localbench", "--label", "localbench.run="+self.state.run["id"], "--label", "localbench.job="+job, "--network", "none", "--read-only", "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", "2g", "--pids-limit", "128", "--tmpfs", "/tmp:rw,nosuid,size=128m", "--env", "HOME=/tmp", "--env", "HF_HUB_OFFLINE=1", "--entrypoint", self.entrypoint, self.image_digest]+help_args
        fingerprint = self._configuration_fingerprint(self.image_digest)
        args[1:1] = ["--label", "localbench.engine="+self.engine, "--label", "localbench.config="+fingerprint]
        created = self._docker("help-create", args)
        cid = created["output"].strip()
        if created["exit_code"] != 0 or not re.fullmatch(r"[a-f0-9]{64}", cid):
            raise RuntimeError("container_help_create_failed:"+created["log"])
        initial = self._inspect_container(cid)
        if initial["id"] != cid or initial["image"] != image_id or not validate_container(initial.get("labels") or {}, self.state.run["id"], job):
            raise RuntimeError("container_help_identity_unverified:refusing_mutation")
        record = Path(self.config["paths"]["state"])/("owned-help-"+job+".json")
        saved = {"owner": "localbench", "run_id": self.state.run["id"], "job_id": job, "container_id": cid, **{key: initial[key] for key in ("id", "created", "started", "image")}, "argv": [self.docker]+args, "engine": self.engine, "image_digest": self.image_digest, "configuration_fingerprint": fingerprint, "kind": "help", "argv_hash": digest([self.docker]+args)}
        atomic_json(record, saved)
        try:
            return self._docker("help", ["start", "--attach", cid], timeout=60)
        finally:
            observed = self._inspect_container(cid)
            if any(observed[key] != initial[key] for key in ("id", "created", "image")) or not validate_container(observed.get("labels") or {}, self.state.run["id"], job):
                raise RuntimeError("container_help_identity_changed:refusing_mutation")
            saved["started"] = observed["started"]
            atomic_json(record, saved)
            before = self._inspect_container(cid)
            if any(before[key] != saved[key] for key in ("id", "created", "started", "image")) or not validate_container(before.get("labels") or {}, self.state.run["id"], job):
                raise RuntimeError("container_help_identity_changed:refusing_mutation")
            if before["running"]:
                stopped = self._docker("help-stop-owned", ["stop", "--time", "5", cid], timeout=15)
                if stopped["exit_code"] != 0:
                    raise RuntimeError("owned_help_stop_failed;identity_retained")
            before = self._inspect_container(cid)
            if any(before[key] != saved[key] for key in ("id", "created", "started", "image")) or not validate_container(before.get("labels") or {}, self.state.run["id"], job):
                raise RuntimeError("container_help_identity_changed:refusing_mutation")
            removed = self._docker("help-remove-owned", ["rm", cid], timeout=15)
            if removed["exit_code"] != 0:
                raise RuntimeError("owned_help_remove_failed;identity_retained")
            atomic_json(record, {**saved, "stopped": True, "removed_at": time.time()})

    def _engine_args(self):
        if not self.help:
            raise RuntimeError("exact_image_help_required_before_launch")
        s, port = self.requested, self.profile["container_port"]
        flags = []
        switches = []
        if self.engine == "ik-llama":
            flags = [("--model", self.container_model), ("--ctx-size", s["context"]), ("--parallel", 1), ("--n-gpu-layers", s["gpu_layers"]), ("--flash-attn", s["flash_attention"]), ("--cache-type-k", s["cache_k"]), ("--cache-type-v", s["cache_v"]), ("--batch-size", s["batch"]), ("--ubatch-size", s["ubatch"]), ("--threads", s["threads"]), ("--threads-batch", s["threads_batch"]), ("--spec-type", "none"), ("--reasoning", "off" if s["reasoning"] == "disabled" else "auto")]
            switches = ["--jinja", "--no-context-shift"] + (["--peg"] if s["peg"] else [])
            if s["reasoning"] == "reduced":
                flags += [("--reasoning-budget", 256)]
        elif self.engine == "vllm":
            flags = [("--max-model-len", s["context"]), ("--max-num-seqs", 1), ("--gpu-memory-utilization", s["gpu_memory_utilization"]), ("--max-num-batched-tokens", s["prefill_tokens"]), ("--tool-call-parser", s["tool_parser"]), ("--reasoning-parser", s["reasoning_parser"]), ("--served-model-name", self.model["id"]), ("--kv-cache-dtype", s["kv_cache_dtype"])]
            switches = ["--enable-auto-tool-choice", "--enable-prefix-caching" if s["prefix_cache"] else "--no-enable-prefix-caching"]
        else:
            flags = [("--model-path", self.container_model), ("--context-length", s["context"]), ("--max-running-requests", 1), ("--mem-fraction-static", s["gpu_memory_utilization"]), ("--chunked-prefill-size", s["prefill_tokens"]), ("--max-total-tokens", s["context"]), ("--tool-call-parser", s["tool_parser"]), ("--reasoning-parser", s["reasoning_parser"]), ("--served-model-name", self.model["id"]), ("--kv-cache-dtype", s["kv_cache_dtype"])]
            switches = [] if s["prefix_cache"] else ["--disable-radix-cache"]
        flags += [("--host", "0.0.0.0"), ("--port", port)]
        if self.engine != "ik-llama" and s["attention_backend"] is not None:
            flags += [("--attention-backend", s["attention_backend"])]
        if self.engine!='ik-llama':
            if s['language_model_only']:switches += ['--language-model-only']
            if s['moe_backend'] is not None:flags += [('--moe-backend' if self.engine=='vllm' else '--moe-runner-backend',s['moe_backend'])]
            if s['dtype'] is not None:flags += [('--dtype',s['dtype'])]
            if s['generation_config'] is not None:flags += [('--generation-config',s['generation_config'])]
            if s['eager']:switches += ['--enforce-eager' if self.engine=='vllm' else '--disable-cuda-graph']
        supported = set(re.findall(r"--[a-z0-9-]+", self.help))
        for flag in [flag for flag, value in flags]+switches:
            if flag not in supported:
                raise RuntimeError("unsupported_exact_image_flag:"+flag)
        args = list(self.entry_args)
        if self.engine == "vllm":
            args.append(self.container_model)
        for flag, value in flags:
            args += [flag, str(value)]
        return args+switches

    def _selected_file(self, relative):
        if not isinstance(relative, str) or any(character in relative for character in ("\\", ":", ",", "\r", "\n", "\0")) or relative.startswith("/") or any(part in ("", ".", "..") for part in relative.split("/")):
            raise ValueError("unsafe selected snapshot filename")
        path = self.model_path/relative
        # HF's trusted installed snapshots commonly link into the same hub's
        # blobs. Bind only each selected resolved artifact, never the whole hub.
        return _mount_path(path.resolve(), self.model_roots)

    def _model_mounts(self):
        mounts = [f"type=bind,source={self.model_path},target={self.container_model},readonly"]
        if self.model_path.is_dir():
            selected = [item["filename"] for item in self.model.get("files", [])] or [path.relative_to(self.model_path).as_posix() for path in self.model_path.rglob("*") if path.is_file()]
            for relative in selected:
                path = self.model_path/relative
                if path.is_symlink():
                    mounts.append(f"type=bind,source={self._selected_file(relative)},target={self.container_model}/{relative},readonly")
        return mounts

    def _tokenizer_script(self):
        path = Path(self.config["paths"]["artifacts"])/"container-tokenizer"/hashlib.sha256(TOKENIZER_HELPER.encode()).hexdigest()/"helper.py"
        path = _mount_path(path, [self.config["paths"]["project"]])
        if path.exists() and path.read_text(encoding="utf-8") != TOKENIZER_HELPER:
            raise RuntimeError("frozen_container_tokenizer_helper_changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(TOKENIZER_HELPER, encoding="utf-8", newline="\n")
        return path

    def build_launch_argv(self, name, job, runtime, tokenizer_script=None):
        """Pure argv construction; caller creates the owned cache before start."""
        if not re.fullmatch(r"localbench-[a-z0-9-]+", name) or not re.fullmatch(r"[a-zA-Z0-9-]+", job):
            raise ValueError("invalid container ownership names")
        runtime = _mount_path(runtime, [self.config["paths"]["runtime_root"]])
        argv = [self.docker, "create", "--name", name, "--pull", "never", "--label", "localbench.owner=localbench", "--label", "localbench.run="+self.state.run["id"], "--label", "localbench.job="+job, "--label", "localbench.engine="+self.engine, "--gpus", "device=0", "--publish", f"127.0.0.1:{self.port}:{self.profile['container_port']}", "--network", "bridge", "--read-only", "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", str(self.requested["memory_gib"])+"g", "--shm-size", str(self.requested["shm_gib"])+"g", "--cpus", "16", "--pids-limit", "512", "--tmpfs", "/tmp:rw,nosuid,size=1g", "--mount", f"type=bind,source={self.model_path},target={self.container_model},readonly", "--mount", f"type=bind,source={runtime},target=/runtime", "--workdir", "/runtime"]
        argv[2:2] = ["--label", "localbench.config="+self._configuration_fingerprint(self.image_digest)]
        for value in ("HOME=/runtime/cache", "XDG_CACHE_HOME=/runtime/cache", "HF_HOME=/runtime/cache/hf", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1", "TRITON_CACHE_DIR=/runtime/cache/triton", "TORCHINDUCTOR_CACHE_DIR=/runtime/cache/torchinductor", "VLLM_NO_USAGE_STATS=1", "DO_NOT_TRACK=1"):
            argv += ["--env", value]
        if self.engine=='vllm' and self.model.get('id')=='gptoss20b-mxfp4':argv += ['--env','VLLM_SYSTEM_START_DATE=2026-09-18']
        # Replace the initial selected-directory mount with explicit readonly
        # overlays for linked artifacts; no physical weight copy is made.
        primary_mount = argv.index(f"type=bind,source={self.model_path},target={self.container_model},readonly")
        argv[primary_mount-1:primary_mount+1] = [part for mount in self._model_mounts() for part in ("--mount", mount)]
        if self.engine != "ik-llama":
            if tokenizer_script is None:
                raise RuntimeError("readonly_tokenizer_helper_mount_required")
            tokenizer_script = _mount_path(tokenizer_script, [self.config["paths"]["project"]])
            argv += ["--mount", f"type=bind,source={tokenizer_script},target=/bench-tokenizer/helper.py,readonly"]
        return argv+["--entrypoint", self.entrypoint, self.image_digest]+self._engine_args()

    def _inspect_container(self, identity):
        result = self._docker("container-inspect", ["inspect", "--format", CONTAINER_FORMAT, identity])
        if result["exit_code"] != 0:
            raise RuntimeError("container_identity_unavailable:refusing_mutation")
        value = json.loads(result["output"])
        if not isinstance(value, dict) or not re.fullmatch(r"[a-f0-9]{64}", str(value.get("id"))) or type(value.get("running")) is not bool:
            raise RuntimeError("invalid_container_identity:refusing_mutation")
        return value

    def _inspect_owned(self):
        if not self.saved:
            raise RuntimeError("no_owned_container")
        current = self._inspect_container(self.saved["container_id"])
        if not validate_container(current.get("labels") or {}, self.saved["run_id"], self.saved["job_id"]) or any(current.get(key) != self.saved[key] for key in ("id", "created", "started", "image")):
            raise RuntimeError("stale_container_identity:refusing_mutation")
        return current

    def _configuration_fingerprint(self, image_digest):
        files = [{key: item.get(key) for key in ("filename", "bytes", "sha256", "git_blob_sha1")} for item in self.model.get("files", [])]
        model_identity = self.model.get("sha256") or (file_hash(self.model_path) if self.model_path.is_file() else digest({"repo": self.model.get("repo"), "revision": self.model.get("revision"), "files": files}) if files and all(item.get("sha256") or item.get("git_blob_sha1") for item in files) else None)
        if not model_identity:
            raise RuntimeError("recovery_requires_pinned_model_identity")
        return digest({"run_id": self.state.run["id"], "engine": self.engine, "image_digest": image_digest, "model": {"id": self.model["id"], "path": str(self.model_path), "identity": model_identity, "revision": self.model.get("revision"), "files": files}, "requested_settings": self.requested, "spec_hash": self.config.get("_spec_hash"), "tokenizer_helper_sha256": hashlib.sha256(TOKENIZER_HELPER.encode()).hexdigest() if self.engine != "ik-llama" else None})

    def recover_owned(self):
        """Explicitly remove verified original orphans; never claim a listener.

        Call under the controller's sequential GPU lease before a new launch.
        Legacy/incomplete handles and changed startup timestamps fail closed.
        No model loads, help commands, downloads or clock resets occur here.
        """
        if self.saved is not None:
            raise RuntimeError("recovery_requires_no_live_adapter_handle")
        root = Path(self.config["paths"]["state"])
        selected = [root/("owned-container-"+str(self.port)+".json")]
        selected += sorted(root.glob("owned-help-help-*.json"))
        recovered = []
        for record in selected:
            if not record.exists():
                continue
            record = _mount_path(record, [self.config["paths"]["project"]])
            saved = json.loads(record.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise RuntimeError("invalid_recovery_handle:refusing_mutation")
            if saved.get("stopped") is True or saved.get("removed_at") is not None:
                continue
            if record.name.startswith("owned-help-") and saved.get("engine") != self.engine:
                continue
            required = {"owner", "run_id", "job_id", "container_id", "id", "created", "started", "image", "image_digest", "engine", "configuration_fingerprint", "argv", "argv_hash", "kind"}
            if required-set(saved) or saved["owner"] != "localbench" or saved["run_id"] != self.state.run["id"] or saved["engine"] != self.engine or saved["kind"] not in ("engine", "help") or saved["container_id"] != saved["id"] or not re.fullmatch(r"[a-f0-9]{64}", str(saved["id"])) or not isinstance(saved["argv"], list) or not all(isinstance(item, str) for item in saved["argv"]) or saved["argv_hash"] != digest(saved["argv"]):
                raise RuntimeError("unverified_recovery_handle:refusing_mutation")
            pin = saved["image_digest"]
            if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", str(pin)) or (self.image_digest is not None and pin != self.image_digest) or saved["configuration_fingerprint"] != self._configuration_fingerprint(pin):
                raise RuntimeError("recovery_configuration_mismatch:refusing_mutation")
            if saved["kind"] == "engine" and saved.get("endpoint") != f"http://127.0.0.1:{self.port}":
                raise RuntimeError("recovery_endpoint_mismatch:refusing_mutation")
            inspected = self._docker("recovery-image-inspect", ["image", "inspect", "--format", IMAGE_FORMAT, pin])
            if inspected["exit_code"] != 0:
                raise RuntimeError("recovery_image_identity_unavailable:refusing_mutation")
            image = json.loads(inspected["output"])
            if image.get("id") != saved["image"] or pin not in (image.get("digests") or []):
                raise RuntimeError("recovery_image_identity_mismatch:refusing_mutation")
            def inspect_original():
                result = self._docker("recovery-container-inspect", ["inspect", "--format", CONTAINER_FORMAT, saved["id"]])
                if result["exit_code"] != 0:
                    if "no such object:" in result["output"].lower() and saved["id"] in result["output"]:
                        return None
                    raise RuntimeError("recovery_container_identity_unavailable:refusing_mutation")
                current = json.loads(result["output"])
                if not isinstance(current, dict) or type(current.get("running")) is not bool or any(current.get(key) != saved[key] for key in ("id", "created", "started", "image")) or not validate_container(current.get("labels") or {}, saved["run_id"], saved["job_id"]) or current["labels"].get("localbench.engine") != self.engine or current["labels"].get("localbench.config") != saved["configuration_fingerprint"]:
                    raise RuntimeError("stale_recovery_container_identity:refusing_mutation")
                return current
            current = inspect_original()
            if current is not None:
                if current["running"]:
                    stopped = self._docker("recovery-stop-owned", ["stop", "--time", "5" if saved["kind"] == "help" else "15", saved["id"]], timeout=30)
                    if stopped["exit_code"] != 0:
                        raise RuntimeError("owned_recovery_stop_failed;identity_retained")
                current = inspect_original()
                if current is not None:
                    removed = self._docker("recovery-remove-owned", ["rm", saved["id"]], timeout=15)
                    if removed["exit_code"] != 0:
                        raise RuntimeError("owned_recovery_remove_failed;identity_retained")
            atomic_json(record, {**saved, "stopped": True, "removed_at": time.time(), "recovered": True})
            self.image_digest = pin
            recovered.append({"kind": saved["kind"], "container_id": saved["id"], "record": str(record)})
        return {"status": "recovered" if recovered else "no_owned_orphans", "containers": recovered}

    def launch(self, config=None):
        if config is not None:
            raise RuntimeError("launch_configuration_override_unsupported;construct_explicit_adapter")
        if self.saved is not None:
            raise RuntimeError("owned_container_already_present;unload_before_launch")
        if not self._metadata:
            self.discover_capabilities()
        self.state.check_budget(estimated_seconds=self.config["limits"]["server_start_timeout_seconds"])
        check_port(self.port)
        job = "engine-"+uuid.uuid4().hex
        name = "localbench-"+self.engine+"-"+uuid.uuid4().hex[:16]
        runtime = _mount_path(Path(self.config["paths"]["runtime_root"])/"containers"/self.engine/job, [self.config["paths"]["runtime_root"]])
        (runtime/"cache").mkdir(parents=True, exist_ok=True)
        argv = self.build_launch_argv(name, job, runtime, self._tokenizer_script() if self.engine != "ik-llama" else None)
        began = time.perf_counter()
        created = self._docker("create", argv[1:], timeout=30)
        if created["exit_code"] != 0 or not re.fullmatch(r"[a-f0-9]{64}", created["output"].strip()):
            raise RuntimeError("container_create_failed:"+created["log"])
        cid = created["output"].strip()
        initial = self._inspect_container(cid)
        if initial["id"] != cid or initial.get("image") != self._metadata["image_id"] or not validate_container(initial.get("labels") or {}, self.state.run["id"], job):
            raise RuntimeError("new_container_identity_unverified:refusing_mutation")
        self.saved = {"owner": "localbench", "run_id": self.state.run["id"], "job_id": job, "container_id": cid, **{key: initial[key] for key in ("id", "created", "started", "image")}, "argv": argv, "endpoint": f"http://127.0.0.1:{self.port}", "runtime": str(runtime), "engine": self.engine, "image_digest": self.image_digest, "configuration_fingerprint": self._configuration_fingerprint(self.image_digest), "argv_hash": digest(argv), "kind": "engine"}
        atomic_json(Path(self.config["paths"]["state"])/("owned-container-"+str(self.port)+".json"), self.saved)
        self.process = ContainerHandle(self, cid)
        self.launch_count += 1
        prefix = Path(self.config["paths"]["logs"])/(self.engine+"-"+job)
        try:
            started = self._docker("start", ["start", cid], timeout=30)
            observed = self._inspect_container(cid)
            if observed["id"] != cid or observed["created"] != self.saved["created"] or observed["image"] != self.saved["image"] or not validate_container(observed.get("labels") or {}, self.saved["run_id"], job):
                raise RuntimeError("container_start_identity_changed:refusing_mutation")
            self.saved["started"] = observed["started"]
            atomic_json(Path(self.config["paths"]["state"])/("owned-container-"+str(self.port)+".json"), self.saved)
            if started["exit_code"] != 0:
                raise RuntimeError("container_start_failed:"+started["log"])
            stdout, stderr = Path(str(prefix)+"-stdout.log"), Path(str(prefix)+"-stderr.log")
            self.handles = [stdout.open("wb"), stderr.open("wb")]
            self.log_process = subprocess.Popen([self.docker, "logs", "--follow", cid], stdin=subprocess.DEVNULL, stdout=self.handles[0], stderr=self.handles[1], env=_safe_environment(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic()+self.config["limits"]["server_start_timeout_seconds"]
            while time.monotonic() < deadline:
                current = self._inspect_owned()
                if not current["running"]:
                    logs = self._docker("startup-failure-logs", ["logs", cid])
                    failure = classify_failure(logs["output"], current.get("oom_killed", False))
                    raise RuntimeError(failure["reason"]+":"+logs["log"])
                if self.health():
                    break
                time.sleep(.25)
            else:
                raise RuntimeError("server_start_timeout")
            logs = self._docker("startup-logs", ["logs", cid])
            self._observe_effective(logs["output"], prefix, began, argv)
            return {"endpoint": self.saved["endpoint"], "handle": dict(self.saved), "effective": dict(self.effective)}
        except (OSError, ValueError, RuntimeError, TypeError, AttributeError):
            self.unload_owned()
            raise

    def _observe_effective(self, raw, prefix, began, argv):
        observations = {}
        if self.engine == "ik-llama":
            observations = self.http.json("/props")
            context = _nested(observations, ("n_ctx",))
            slots = observations.get("total_slots")
            match = re.findall(r"n_ctx_per_seq\s*=\s*(\d+)", raw)
            context = int(match[-1]) if match else context
            match = re.findall(r"n_seq_max\s*=\s*(\d+)", raw)
            slots = int(match[-1]) if match else slots
        elif self.engine == "vllm":
            observations = self.http.json("/v1/models")
            context = _nested(observations, ("max_model_len",))
            # _nested deliberately does not search arbitrary model-list entries.
            contexts = [item.get("max_model_len") for item in observations.get("data", []) if item.get("id") == self.model["id"]]
            if contexts:
                context = contexts[0]
            slots = None
            match = re.findall(r"max_num_seqs[\"']?\s*[:=]\s*(\d+)", raw)
            slots = int(match[-1]) if match else None
            match = re.findall(r"max_model_len[\"']?\s*[:=]\s*(\d+)", raw)
            context = int(match[-1]) if match else context
        else:
            observations = self.http.json("/get_server_info")
            context = _nested(observations, ("context_length", "max_model_len"))
            slots = _nested(observations, ("max_running_requests",))
        atomic_json(str(prefix)+"-observed-config.json", observations)
        self.effective = {"effective_context": context, "effective_slots": slots, "context_shift_disabled": self.engine != "ik-llama" or "--no-context-shift" in argv, "context_evidence": str(prefix)+"-observed-config.json", "launch_argv": argv, "startup_seconds": time.perf_counter()-began, "launch_controls": dict(self.requested), "sampler_effective": "unverified until real request/counter evidence", "prefix_cache_effective": "unverified;measure actual cached/evaluated counts", "container_gpu_ownership": {"container_id": self.saved["container_id"], "device": 0, "windows_pid_attribution": "unverified"}}
        if type(context) is not int or context < self.requested["context"] or type(slots) is not int or slots != 1:
            raise RuntimeError("unverified_or_insufficient_per_slot_context")
        if self.engine!='ik-llama':
            self.effective['text_only_loading']={'requested':self.requested['language_model_only'],
                'emitted':'--language-model-only' in argv,'observed':_nested(observations,('language_model_only',)),
                'vision_encoder_memory_measurement':'not separately measured'}

    def health(self):
        try:
            probe = LocalHTTP(f"http://127.0.0.1:{self.port}", 2)
            if self.engine == "ik-llama":
                return probe.json("/health").get("status") == "ok"
            data = probe.json("/v1/models").get("data")
            return isinstance(data, list) and any(isinstance(item, dict) and item.get("id") == self.model["id"] for item in data)
        except (OSError, RuntimeError, ValueError):
            return False

    def _helper(self):
        helper = self.template_helper
        model_sha = self._metadata.get("model_sha256")
        if helper is None or getattr(helper, "offline", False) is not True or getattr(helper, "verified", False) is not True or not model_sha or getattr(helper, "model_sha256", None) != model_sha or self.engine not in getattr(helper, "verified_engines", ()):
            raise RuntimeError("exact_offline_template_tokenizer_unavailable:"+self.engine+";exploratory_only")
        return helper

    def _container_tokenize(self, request):
        self._inspect_owned()
        self._tokenizer_script()
        result = self._docker("exact-tokenize", ["exec", "--interactive", "--user", "1000:1000", "--env", "CUDA_VISIBLE_DEVICES=", "--env", "HF_HUB_OFFLINE=1", self.saved["container_id"], "python3", "/bench-tokenizer/helper.py"], timeout=60, stdin_json=request)
        if result["exit_code"] != 0:
            raise RuntimeError("container_exact_tokenizer_unavailable;exploratory_only:"+result["log"])
        records = [line[len("LOCALBENCH_TOKENIZER_JSON:"):] for line in result["output"].splitlines() if line.startswith("LOCALBENCH_TOKENIZER_JSON:")]
        if len(records) != 1:
            raise RuntimeError("invalid_container_exact_tokenization_marker")
        value = json.loads(records[0])
        if not isinstance(value, dict):
            raise RuntimeError("invalid_container_exact_tokenization_response")
        tokens = value.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens) or value.get("count") != len(tokens):
            raise RuntimeError("invalid_container_exact_tokenization_response")
        value.update(tokenizer_image_digest=self.image_digest, tokenizer_helper_sha256=self._metadata.get("tokenizer_helper_sha256"), count_cross_check="actual engine input usage required before qualification")
        return value

    def tokenize(self, messages, tools):
        self._inspect_owned()
        if self.engine == "ik-llama":
            try:
                rendered = self.http.json("/apply-template", {"messages": messages, "tools": tools, "add_generation_prompt": True})
                prompt = rendered.get("prompt")
                if not isinstance(prompt, str) or (tools and not all(tool["function"]["name"] in prompt for tool in tools)):
                    raise RuntimeError("template_endpoint_omits_tools_or_prompt")
                tokens = self.http.json("/tokenize", {"content": prompt, "add_special": True, "parse_special": True}).get("tokens")
                if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                    raise RuntimeError("invalid_exact_tokenization_response")
                return {"count": len(tokens), "provenance": "owned ik /apply-template+/tokenize;final usage cross-check required", "template_sha256": digest(prompt), "tokenizer_image_digest": self.image_digest}
            except (OSError, RuntimeError, ValueError):
                if self.template_helper is None:
                    raise RuntimeError("ik_exact_templated_tokenization_unavailable;exploratory_only") from None
        result = self._helper().tokenize(messages, tools) if self.template_helper is not None else self._container_tokenize({"operation": "chat", "messages": messages, "tools": tools})
        if not isinstance(result, dict) or type(result.get("count")) is not int or result["count"] < 0:
            raise RuntimeError("invalid_exact_helper_count")
        return result

    def tokenize_text(self, text):
        self._inspect_owned()
        if self.engine == "ik-llama":
            tokens = self.http.json("/tokenize", {"content": text, "add_special": False, "parse_special": False}).get("tokens")
        else:
            tokens = self._helper().tokenize_text(text) if self.template_helper is not None else self._container_tokenize({"operation": "text", "text": text})["tokens"]
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise RuntimeError("invalid_exact_text_tokenization")
        return tokens

    def stream_chat(self, request, **kwargs):
        self._inspect_owned()
        unknown = set(request)-self.REQUEST
        if unknown:
            raise RuntimeError("unsupported_container_request_controls:"+",".join(sorted(unknown)))
        payload = dict(request)
        if self.engine != "ik-llama":
            payload["model"] = self.model["id"]
            if self.requested['reasoning_effort'] is not None:payload['reasoning_effort']=self.requested['reasoning_effort']
            if "repeat_penalty" in payload:
                if "repetition_penalty" in payload:
                    raise ValueError("duplicate repetition sampler controls")
                payload["repetition_penalty"] = payload.pop("repeat_penalty")
        self._metadata["last_request_controls"] = {"requested": {key: value for key, value in request.items() if key not in ("messages", "tools")}, "emitted": {key: value for key, value in payload.items() if key not in ("messages", "tools")}, "effective": "unverified until engine request/counter evidence"}
        return self.http.measure(payload, **kwargs)

    def reset_cache(self):
        return {"status": "owned_server_restart_required", "verified": False}

    def fresh_restart(self):
        self.unload_owned()
        return self.launch()

    def close_logs(self):
        if self.log_process is not None and self.log_process.poll() is None:
            self.log_process.terminate()
            try:
                self.log_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.log_process.kill()
                self.log_process.wait(timeout=5)
        self.log_process = None
        for handle in self.handles:
            if not handle.closed:
                handle.close()
        self.handles = []

    def unload_owned(self):
        if self.saved is None:
            self.close_logs()
            return
        current = self._inspect_owned()
        cid = current["id"]
        stopped = None
        if current["running"]:
            stopped = self._docker("stop-owned", ["stop", "--time", "15", cid], timeout=30)
            if stopped["exit_code"] != 0:
                raise RuntimeError("owned_container_stop_failed;identity_retained")
            current = self._inspect_owned()
        if current["running"] or type(current.get("exit_code")) is not int:
            raise RuntimeError("owned_container_stopped_state_unverified;identity_retained")
        removed = self._docker("remove-owned", ["rm", cid], timeout=15)
        if removed["exit_code"] != 0:
            raise RuntimeError("owned_container_remove_failed;identity_retained")
        absent = self._docker("container-inspect", ["inspect", "--format", CONTAINER_FORMAT, cid])
        if absent["exit_code"] == 0 or "no such object:" not in absent["output"].lower() or cid not in absent["output"]:
            raise RuntimeError("owned_container_removal_unverified;identity_retained")
        proof = {"container_id": cid, "created": current["created"], "started": current["started"], "image": current["image"],
                 "run_id": self.saved["run_id"], "job_id": self.saved["job_id"], "exit_code": current["exit_code"],
                 "stopped_verified": True, "removed": True, "absence_verified": True, "removed_at": time.time(),
                 "stop_argv": stopped["argv"] if stopped is not None else None,
                 "remove_argv": removed["argv"], "remove_exit_code": removed["exit_code"], "absence_log": absent["log"]}
        self.close_logs()
        atomic_json(Path(self.config["paths"]["state"])/("owned-container-"+str(self.port)+".json"), {**self.saved, "stopped": True, "removed_at": proof["removed_at"], "cleanup_completion": proof})
        if isinstance(self.process, ContainerHandle):
            self.process._complete_removal(current, proof)
        self.saved, self.process = None, None

    def metadata(self):
        return {**self._metadata, "effective_settings": dict(self.effective)}
