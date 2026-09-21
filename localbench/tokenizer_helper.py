"""Owned, CPU-only GGUF template/tokenization service; never runs inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from .adapters.base import LocalHTTP
from .adapters.lms import _logged_command, _without_credentials
from .adapters.native import creation_time
from .adapters.ollama import _listener_pid
from .config import atomic_json, digest, file_hash


ENGINE = "upstream-llama-nightly"
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_COMMIT = re.compile(r"[a-f0-9]{40}\Z")
_TOKEN_PATHS = {"/health", "/props", "/apply-template", "/tokenize"}


def _manifest_record(config, state, requested_binary=None):
    """Resolve an already acquired pin. This never downloads or executes code."""
    root = Path(config["paths"]["runtime_root"]).resolve()
    row = state.db.execute("SELECT data FROM entities WHERE kind='runtime_catalog' AND id=?", (ENGINE,)).fetchone()
    records = []
    if row:
        records.append(json.loads(row[0]))
    else:
        for manifest_path in sorted(root.glob("upstream-*/engine-manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            relative = manifest.get("binary_relative_path", "")
            records.append({"kind": "native", "binary_path": str(manifest_path.parent / "bin" / relative),
                "engine_manifest": manifest, "manifest_path": str(manifest_path)})
    selected = []
    for record in records:
        manifest = record.get("engine_manifest", {})
        binary = Path(record.get("binary_path", "")).resolve()
        if requested_binary is not None and binary != Path(requested_binary).resolve():
            continue
        if manifest.get("id") != ENGINE or manifest.get("source") != "https://github.com/ggml-org/llama.cpp" or \
                not _COMMIT.fullmatch(str(manifest.get("commit", ""))) or \
                not _SHA.fullmatch(str(manifest.get("binary_sha256", ""))) or not manifest.get("release_tag"):
            raise RuntimeError("tokenizer_helper_unpinned_upstream_manifest")
        if not binary.is_relative_to(root) or not binary.is_file() or binary.name.lower() != "llama-server.exe":
            raise RuntimeError("tokenizer_helper_binary_outside_acquired_runtime")
        relative = manifest.get("binary_relative_path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise RuntimeError("tokenizer_helper_invalid_manifest_path")
        parts = Path(relative).parts
        if Path(relative).is_absolute() or any(part in {".", ".."} or ":" in part for part in parts):
            raise RuntimeError("tokenizer_helper_invalid_manifest_path")
        destination = binary
        for _ in parts:
            destination = destination.parent
        if (destination / relative).resolve() != binary:
            raise RuntimeError("tokenizer_helper_manifest_binary_path_mismatch")
        pinned_files = {item.get("path"): item for item in manifest.get("files", [])}
        required = [binary, *sorted(binary.parent.glob("*.dll"))]
        for path in required:
            name = path.relative_to(destination).as_posix()
            pin = pinned_files.get(name)
            if not pin or not _SHA.fullmatch(str(pin.get("sha256", ""))) or \
                    type(pin.get("bytes")) is not int or path.stat().st_size != pin["bytes"] or file_hash(path) != pin["sha256"]:
                raise RuntimeError("tokenizer_helper_acquired_file_hash_mismatch")
        if file_hash(binary) != manifest["binary_sha256"]:
            raise RuntimeError("tokenizer_helper_binary_pin_mismatch")
        selected.append((binary, manifest))
    if len(selected) != 1:
        raise RuntimeError("tokenizer_helper_acquired_upstream_missing_or_ambiguous")
    return selected[0]


def _ephemeral_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def cpu_launch_plan(binary, model, port, help_text):
    """Fail closed when any CPU isolation or local API control is absent."""
    supported = set(re.findall(r"--[a-z0-9-]+", help_text))
    values = [("--model", model), ("--ctx-size", 65536), ("--parallel", 1), ("--n-gpu-layers", 0),
        ("--device", "none"), ("--batch-size", 512), ("--ubatch-size", 128), ("--threads", 2),
        ("--threads-batch", 2), ("--host", "127.0.0.1"), ("--port", port),
        ("--cors-origins", "http://127.0.0.1:" + str(port))]
    booleans = ["--no-kv-offload", "--no-op-offload", "--no-context-shift", "--jinja", "--no-warmup", "--no-webui"]
    required = {flag for flag, _ in values} | set(booleans)
    missing = sorted(required - supported)
    if missing:
        raise RuntimeError("unsupported_tokenizer_helper_flags:" + ",".join(missing))
    argv = [str(binary)]
    for flag, value in values:
        argv.extend((flag, str(value)))
    argv.extend(booleans)
    if "--spec-type" in supported:
        argv.extend(("--spec-type", "none"))
    for flag, value in (("--cors-methods", "GET,POST"), ("--cors-headers", "Content-Type")):
        if flag in supported:
            argv.extend((flag, value))
    if "--no-cors-credentials" in supported:
        argv.append("--no-cors-credentials")
    return argv


class NativeTokenizerHelper:
    """A pinned tokenizer is intrinsic; each managed engine needs counter proof."""

    offline = True

    def __init__(self, config, state, model, binary=None):
        self.config, self.state, self.model = config, state, model
        self.requested_binary = binary
        self.model_sha256 = model.get("sha256")
        self.verified = False
        self.verified_engines = set()
        self.process, self.saved, self.http, self.port = None, None, None, None
        self.handles, self.effective, self._metadata = [], {}, {}
        self.proofs = {}

    def process_environment(self):
        env = _without_credentials()
        # All changes are confined to the owned child. Ignore inherited engine
        # controls and hide CUDA devices in addition to explicit CPU CLI flags.
        env = {key: value for key, value in env.items() if not key.upper().startswith("LLAMA_ARG_")}
        env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def discover_capabilities(self):
        self.binary, self.manifest = _manifest_record(self.config, self.state, self.requested_binary)
        source = Path(self.model["path"])
        if source.suffix.lower() != ".gguf" or not source.is_file() or \
                not _SHA.fullmatch(str(self.model_sha256 or "")) or file_hash(source) != self.model_sha256:
            raise RuntimeError("tokenizer_helper_model_hash_or_format_mismatch")
        self.state.check_budget(estimated_seconds=60)
        self.state.start_execution("runtime_probe:owned-cpu-tokenizer-helper")
        env = self.process_environment()
        helped = _logged_command(self.config, "cpu-tokenizer-help", [str(self.binary), "--help"], timeout=30, environment=env)
        if helped["exit_code"] != 0:
            raise RuntimeError("tokenizer_helper_help_failed:" + helped["log"])
        self.help = helped["output"]
        # Validate before spawning even when the selected port has not yet been
        # reserved. The actual port is substituted immediately before Popen.
        cpu_launch_plan(self.binary, source, 1, self.help)
        version = _logged_command(self.config, "cpu-tokenizer-version", [str(self.binary), "--version"], timeout=30, environment=env)
        if version["exit_code"] != 0:
            raise RuntimeError("tokenizer_helper_version_failed:" + version["log"])
        self._metadata = {"engine_id": ENGINE, "binary": str(self.binary), "binary_sha256": self.manifest["binary_sha256"],
            "commit": self.manifest["commit"], "release_tag": self.manifest["release_tag"],
            "adjacent_dlls": {path.name: file_hash(path) for path in sorted(self.binary.parent.glob("*.dll"))},
            "version_output": version["output"], "help_log": helped["log"], "version_log": version["log"],
            "model_path": str(source), "model_sha256": self.model_sha256, "cpu_only": True, "inference_calls": 0}
        return {"cpu_only": True, "offline": True, "tokenization": "/apply-template + /tokenize", "managed_engines_verified": []}

    def _check_identity(self, require_listener=True):
        if self.process is None or self.process.poll() is not None or not self.saved or self.saved["created"] is None:
            raise RuntimeError("tokenizer_helper_owned_process_unavailable")
        if creation_time(self.process.pid) != self.saved["created"]:
            raise RuntimeError("stale_tokenizer_helper_pid:refusing_operation")
        if require_listener and _listener_pid(self.port) != self.process.pid:
            raise RuntimeError("tokenizer_helper_listener_not_owned")

    def _json(self, path, body=None):
        if path not in _TOKEN_PATHS:
            raise RuntimeError("tokenizer_helper_inference_endpoint_forbidden")
        self._check_identity()
        return self.http.json(path, body)

    def health(self):
        try:
            self._check_identity()
            return LocalHTTP("http://127.0.0.1:" + str(self.port), 2).json("/health").get("status") == "ok"
        except (OSError, RuntimeError, ValueError):
            return False

    def launch(self):
        if self.process is not None:
            raise RuntimeError("tokenizer_helper_already_owned")
        if not hasattr(self, "help"):
            self.discover_capabilities()
        self.state.check_budget(estimated_seconds=self.config["limits"]["server_start_timeout_seconds"])
        self.port = _ephemeral_port()
        if _listener_pid(self.port) is not None:
            raise RuntimeError("tokenizer_helper_port_race_before_spawn")
        argv = cpu_launch_plan(self.binary, self.model["path"], self.port, self.help)
        self.http = LocalHTTP("http://127.0.0.1:" + str(self.port), self.config["limits"]["server_start_timeout_seconds"])
        prefix = Path(self.config["paths"]["logs"]) / ("cpu-tokenizer-" + str(time.time_ns()))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        stdout, stderr = Path(str(prefix) + "-stdout.log"), Path(str(prefix) + "-stderr.log")
        self.handles = [stdout.open("wb"), stderr.open("wb")]
        try:
            self.process = subprocess.Popen(argv, stdout=self.handles[0], stderr=self.handles[1], stdin=subprocess.DEVNULL,
                cwd=self.binary.parent, env=self.process_environment(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            self._close_logs()
            raise RuntimeError("tokenizer_helper_spawn_failed") from None
        self.saved = {"owner": "localbench", "run_id": self.state.run["id"], "pid": self.process.pid,
            "created": creation_time(self.process.pid), "argv": argv, "command_hash": digest(argv),
            "endpoint": "http://127.0.0.1:" + str(self.port), "stdout": str(stdout), "stderr": str(stderr)}
        atomic_json(Path(self.config["paths"]["state"]) / ("owned-tokenizer-" + str(self.port) + ".json"), self.saved)
        if hasattr(self, "collector"):
            self.collector.mark_owned_pid(self.process.pid)
        try:
            self._check_identity(require_listener=False)
            deadline = time.monotonic() + self.config["limits"]["server_start_timeout_seconds"]
            while time.monotonic() < deadline:
                self._check_identity(require_listener=False)
                listener = _listener_pid(self.port)
                if listener is not None and listener != self.process.pid:
                    raise RuntimeError("tokenizer_helper_listener_not_owned")
                if listener == self.process.pid and self.health():
                    break
                time.sleep(.25)
            else:
                raise RuntimeError("tokenizer_helper_start_timeout:" + str(stderr))
            props = self._json("/props")
            atomic_json(str(prefix) + "-props.json", props)
            raw = stdout.read_text(encoding="utf-8", errors="replace") + stderr.read_text(encoding="utf-8", errors="replace")
            ctx = re.findall(r"n_ctx_per_seq\s*=\s*(\d+)", raw)
            slots = re.findall(r"n_seq_max\s*=\s*(\d+)", raw)
            effective_context = int(ctx[-1]) if ctx else props.get("default_generation_settings", {}).get("n_ctx", props.get("n_ctx"))
            effective_slots = int(slots[-1]) if slots else props.get("total_slots")
            offload = re.findall(r"offloaded (\d+)/(\d+) layers", raw)
            gpu_buffers = re.findall(r"(?:CUDA\d+|Vulkan\d+|SYCL\d+)\s+[^\r\n]*buffer size\s*=\s*([0-9.]+)", raw)
            if any(int(loaded) for loaded, _ in offload) or any(float(size) > 0 for size in gpu_buffers):
                raise RuntimeError("tokenizer_helper_observed_gpu_allocation")
            if type(effective_context) is not int or effective_context != 65536 or effective_slots != 1:
                raise RuntimeError("tokenizer_helper_effective_context_or_slots_unverified")
            self.effective = {"effective_context": effective_context, "effective_slots": effective_slots,
                "cpu_only_controls_verified": True, "observed_offload_layers": list(map(int, offload[-1])) if offload else None,
                "gpu_allocation_observed": False, "launch_argv": argv, "props_path": str(prefix) + "-props.json"}
            self._probe_intrinsic()
            self.verified = True
            self.state.entity("tokenizer_helper", digest([self.model_sha256, self.manifest["binary_sha256"]]), self.metadata())
            return {"endpoint": self.saved["endpoint"], "handle": self.saved, "effective": self.effective}
        except BaseException:
            self.verified = False
            self.unload_owned()
            raise

    def _tokenize(self, messages, tools):
        rendered = self._json("/apply-template", {"messages": messages, "tools": tools, "add_generation_prompt": True})
        prompt = rendered.get("prompt")
        names = [tool.get("function", {}).get("name") for tool in tools]
        if not isinstance(prompt, str) or any(not isinstance(name, str) or not name or name not in prompt for name in names):
            raise RuntimeError("tokenizer_helper_template_omits_tools_or_prompt")
        tokenized = self._json("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})
        tokens = tokenized.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise RuntimeError("tokenizer_helper_invalid_exact_token_response")
        return {"count": len(tokens), "provenance": "owned CPU GGUF /apply-template + /tokenize add_special=true parse_special=true",
            "template_sha256": digest(prompt), "tokenizer_engine_sha256": self.manifest["binary_sha256"],
            "model_sha256": self.model_sha256, "offline": True}

    def _probe_intrinsic(self):
        tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a synthetic fixture file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}}]
        def messages(filler):
            return [{"role": "system", "content": "CPU tokenizer bootstrap only. Never generate or execute code."},
                {"role": "user", "content": "Inspect src/tokenizer_probe.py.\n" + filler}]
        target = self.config["measurement"]["warmup_input_tokens"]
        tolerance = self.config["measurement"]["prompt_target_absolute_tolerance_tokens"]
        filler = "// Deterministic CPU-only template tokenization probe.\n" * 1024
        empty = self._tokenize(messages(""), tools)
        large = self._tokenize(messages(filler), tools)
        if empty["count"] > target or large["count"] < target:
            raise RuntimeError("tokenizer_helper_bootstrap_cannot_pack_512")
        low, high, best = 0, len(filler), ("", empty)
        while low <= high:
            middle = (low + high) // 2
            observed = self._tokenize(messages(filler[:middle]), tools)
            if observed["count"] <= target:
                best = (filler[:middle], observed)
                if observed["count"] == target:
                    break
                low = middle + 1
            else:
                high = middle - 1
        repeated = self._tokenize(messages(best[0]), tools)
        without_tools = self._tokenize(messages(best[0]), [])
        if not target - tolerance <= best[1]["count"] <= target or repeated != best[1] or \
                without_tools["template_sha256"] == repeated["template_sha256"]:
            raise RuntimeError("tokenizer_helper_intrinsic_tokenization_unverified")
        self._metadata["intrinsic_verification"] = {"target_tokens": target, "exact_prompt_tokens": repeated["count"],
            "template_sha256": repeated["template_sha256"], "includes_tool_schema": True, "repeat_identical": True,
            "managed_usage_counter_verified": False}

    def tokenize(self, messages, tools):
        if not self.verified:
            raise RuntimeError("tokenizer_helper_intrinsic_verification_required")
        return self._tokenize(messages, tools)

    def tokenize_text(self, text):
        if not self.verified or not isinstance(text, str):
            raise RuntimeError("tokenizer_helper_intrinsic_verification_required")
        tokens = self._json("/tokenize", {"content": text, "add_special": False, "parse_special": False}).get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise RuntimeError("tokenizer_helper_invalid_exact_token_response")
        return tokens

    def verify_against(self, engine, messages, tools, actual_prompt_tokens):
        if engine not in {"ollama", "lm-studio"}:
            raise ValueError("tokenizer_helper_unknown_managed_engine")
        observed = self.tokenize(messages, tools)
        expected = observed["count"]
        target = self.config["measurement"]["full_prompt_tokens"]
        tolerance = self.config["measurement"]["prompt_target_absolute_tolerance_tokens"]
        passed = type(actual_prompt_tokens) is int and 0 < actual_prompt_tokens <= target and \
            0 < expected <= target and abs(actual_prompt_tokens - expected) <= tolerance
        proof = {"engine": engine, "model_sha256": self.model_sha256, "tokenizer_engine_sha256": self.manifest["binary_sha256"],
            "request_sha256": digest({"messages": messages, "tools": tools}), "expected_prompt_tokens": expected,
            "actual_prompt_tokens": actual_prompt_tokens, "maximum_absolute_difference": tolerance,
            "full_prompt_target": target, "template_sha256": observed["template_sha256"], "passed": passed}
        self.proofs[engine] = proof
        self.state.entity("tokenizer_verification", digest([engine, self.model_sha256, self.manifest["binary_sha256"], proof["request_sha256"]]), proof)
        if not passed:
            self.verified_engines.discard(engine)
            raise RuntimeError("tokenizer_helper_managed_prompt_count_mismatch")
        self.verified_engines.add(engine)
        return proof

    def metadata(self):
        return {**self._metadata, "effective_settings": self.effective, "intrinsic_verified": self.verified,
            "managed_engines_verified": sorted(self.verified_engines), "managed_counter_proofs": self.proofs,
            "owned_process": self.saved}

    def _close_logs(self):
        for handle in self.handles:
            if not handle.closed:
                handle.close()
        self.handles = []

    def unload_owned(self):
        self.verified = False
        if self.process is not None and self.process.poll() is None:
            if not self.saved or self.saved.get("created") is None or creation_time(self.process.pid) != self.saved["created"]:
                raise RuntimeError("stale_tokenizer_helper_pid:refusing_termination")
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                # The retained Popen handle still refers to our original child.
                self.process.kill()
                self.process.wait(timeout=15)
        self._close_logs()
        if self.process is not None and hasattr(self, "collector"):
            self.collector.mark_owned_pid(self.process.pid, owned=False)
        if self.saved and self.process is not None and self.process.poll() is not None:
            atomic_json(Path(self.config['paths']['state']) / ('owned-tokenizer-' + str(self.port) + '.json'), {**self.saved,'stopped':True,'stopped_at':time.time()})
        self.process = None

    close = unload_owned


GGUFTokenizerHelper = NativeTokenizerHelper
