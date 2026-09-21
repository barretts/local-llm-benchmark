"""Private, owned Ollama daemon and GGUF import; user's daemon is untouched."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import time

from ..acquisition import _disk_headroom
from ..config import atomic_json, digest, file_hash
from ..ownership import check_port
from .base import EngineAdapter, LocalHTTP
from .lms import _logged_command, _verified_tokenizer, _without_credentials
from .native import creation_time


DOCS = {"chat": "https://docs.ollama.com/api/chat", "import": "https://docs.ollama.com/import",
    "contexts": "https://docs.ollama.com/api/ps", "environment": "https://docs.ollama.com/faq",
    "blob_reuse_source": "https://github.com/ollama/ollama/blob/main/cmd/cmd.go"}


def _listener_pid(port):
    """Inspect the loopback listener directly; a live endpoint alone is not ownership."""
    if os.name != "nt":
        return None
    from ctypes import wintypes
    api = ctypes.WinDLL("iphlpapi")
    api.GetExtendedTcpTable.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
        wintypes.BOOL, wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
    size = wintypes.DWORD(0)
    api.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 3, 0)
    if not size.value:
        return None
    buffer = ctypes.create_string_buffer(size.value)
    if api.GetExtendedTcpTable(buffer, ctypes.byref(size), False, 2, 3, 0):
        return None
    count = struct.unpack_from("<I", buffer.raw)[0]
    for index in range(count):
        status, address, local_port, _, _, pid = struct.unpack_from("<6I", buffer.raw, 4 + index * 24)
        if status == 2 and socket.ntohs(local_port & 0xffff) == port and \
                socket.inet_ntoa(struct.pack("<I", address)) == "127.0.0.1":
            return pid
    return None


def _descendants(parent):
    """Read-only Windows process ancestry. Only daemon descendants are owned."""
    if os.name != "nt":
        return {}
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD),
            ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
            ("parent", wintypes.DWORD), ("priority", wintypes.LONG), ("flags", wintypes.DWORD),
            ("exe", wintypes.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return {}
    links = {}
    try:
        item = Entry()
        item.size = ctypes.sizeof(item)
        more = kernel.Process32FirstW(snapshot, ctypes.byref(item))
        while more:
            links[item.pid] = item.parent
            more = kernel.Process32NextW(snapshot, ctypes.byref(item))
    finally:
        kernel.CloseHandle(snapshot)
    found, frontier = {}, {parent}
    while frontier:
        new = {pid for pid, ancestor in links.items() if ancestor in frontier and pid not in found and pid != parent}
        for pid in new:
            born = creation_time(pid)
            if born is not None:
                found[pid] = {"pid": pid, "parent": links[pid], "created": born}
        frontier = new
    return found


def _terminate_recorded_child(record):
    """A creation-time check prevents a recycled PID from being terminated."""
    if os.name != "nt" or record.get("created") is None:
        return False
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.c_void_p] * 4
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000 | 1 | 0x100000, False, record["pid"])
    if not handle:
        return False
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)) or times[0].value != record["created"]:
            raise RuntimeError("stale_ollama_runner_pid:refusing_termination")
        if not kernel.TerminateProcess(handle, 0):
            raise RuntimeError("owned_ollama_runner_stop_failed")
        kernel.WaitForSingleObject(handle, 15000)
        return True
    finally:
        kernel.CloseHandle(handle)


class OllamaAdapter(EngineAdapter):
    SUPPORTED_SETTINGS = {"context", "slots", "gpu_layers", "flash_attention", "cache_k", "cache_v",
        "batch", "threads", "reasoning", "context_shift", "speculation"}
    SUPPORTED_REQUEST = {"model", "messages", "tools", "tool_choice", "stream", "stream_options", "max_tokens",
        "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty", "seed", "stop"}

    def __init__(self, config, state, model, settings=None, exact_tokenizer=None, verified_controls=None):
        self.config, self.state, self.model = config, state, model
        self.binary = Path(config["installed_tools"]["ollama"])
        self.port = config["private_ports"]["ollama"]
        self.http = LocalHTTP("http://127.0.0.1:" + str(self.port), config["limits"]["per_64k_request_timeout_seconds"])
        self.root = Path(config["paths"]["runtime_root"]) / "ollama" / state.run["id"]
        self.models_root = Path(config["paths"]["new_model_root"]) / "ollama" / state.run["id"] / "models"
        self.requested = {"context": 65536, "slots": 1, "gpu_layers": 9999, "flash_attention": "on",
            "cache_k": "f16", "cache_v": "f16", "batch": 512, "threads": 16, "reasoning": "default",
            "context_shift": False, "speculation": "off"}
        unknown = set(settings or {}) - self.SUPPORTED_SETTINGS
        if unknown:
            raise RuntimeError("unsupported_ollama_controls:" + ",".join(sorted(unknown)))
        self.requested.update(settings or {})
        if self.requested["slots"] != 1 or self.requested["context"] < 65536 or self.requested["context_shift"] or \
                self.requested["speculation"] != "off":
            raise ValueError("invalid primary Ollama configuration")
        if self.requested["cache_k"] != self.requested["cache_v"] or self.requested["cache_k"] not in {"f16", "q8_0", "q4_0"}:
            raise RuntimeError("unsupported_asymmetric_or_unknown_ollama_cache_types")
        if self.requested["flash_attention"] not in {"on", "off"}:
            raise RuntimeError("unsupported_ollama_flash_attention_value")
        self.exact_tokenizer, self.verified_controls = exact_tokenizer, dict(verified_controls or {})
        self.process, self.saved, self.handles = None, None, []
        self.effective, self._metadata, self.children = {}, {}, {}
        self.launch_count = 0

    def _environment(self):
        environment = _without_credentials()
        for name in list(environment):
            if name.startswith("OLLAMA_"):
                environment.pop(name)
        environment.update(OLLAMA_HOST="127.0.0.1:" + str(self.port), OLLAMA_MODELS=str(self.models_root),
            OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_CONTEXT_LENGTH=str(self.requested["context"]),
            OLLAMA_FLASH_ATTENTION="1" if self.requested["flash_attention"] == "on" else "0",
            OLLAMA_KV_CACHE_TYPE=self.requested["cache_k"], OLLAMA_NO_CLOUD="1")
        return environment

    def discover_capabilities(self):
        if not self.binary.is_file():
            raise RuntimeError("missing_ollama_binary")
        self.state.start_execution("runtime_probe:ollama")
        version = _logged_command(self.config, "ollama-version", [str(self.binary), "--version"], environment=self._environment())
        help_result = _logged_command(self.config, "ollama-serve-help", [str(self.binary), "serve", "--help"], environment=self._environment())
        if version["exit_code"] != 0 or help_result["exit_code"] != 0:
            raise RuntimeError("ollama_client_probe_failed")
        backend = {str(path.relative_to(self.binary.parent)): file_hash(path) for path in
            sorted(self.binary.parent.rglob("ollama*.exe")) if path != self.binary}
        libraries = {str(path.relative_to(self.binary.parent)): file_hash(path) for path in
            sorted(self.binary.parent.rglob("*.dll"))}
        self._metadata = {"engine_id": "ollama", "binary": str(self.binary), "binary_sha256": file_hash(self.binary),
            "version_output": version["output"], "version_log": version["log"], "help_log": help_result["log"],
            "backend_executable_hashes": backend, "backend_library_hashes": libraries, "model_path": self.model["path"],
            "model_sha256": self.model.get("sha256") or file_hash(self.model["path"]),
            "requested_settings": dict(self.requested), "documentation": DOCS,
            "verified_request_control_evidence": self.verified_controls,
            "limitations": ["Exact Ollama template/tokenization requires a verified offline helper",
                "Installed API support for truncate/shift must be established independently; unknown controls are not assumed",
                "Cache counters are only actual native final counters; missing counts remain unknown",
                "Automatic/partial CPU offload must be labeled from observed residency"]}
        self.local_name = "localbench-" + digest({"run": self.state.run["id"], "model": self._metadata["model_sha256"],
                                                   "settings": self.requested})[:24]
        self.state.entity("engine", digest(self._metadata), self._metadata)
        return {"private_endpoint": "http://127.0.0.1:" + str(self.port), "model_name": self.local_name,
            "cache_reset": "owned_daemon_restart", "tokenization": "verified_offline_helper_required",
            "native_stats": ["prompt_eval_count", "prompt_eval_cached_count", "eval_count", "eval_duration"]}

    def _prepare_blob(self):
        source = Path(self.model["path"])
        sha = self._metadata["model_sha256"]
        if not re.fullmatch(r"[a-fA-F0-9]{64}", sha):
            raise RuntimeError("invalid_verified_gguf_hash")
        target = self.models_root / "blobs" / ("sha256-" + sha.lower())
        partial = target.with_name(target.name + ".part")
        owned_root = Path(self.config["paths"]["new_model_root"])
        if any(path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(owned_root.resolve())
               for path in (target, partial, *target.parents[:len(target.parents) - len(owned_root.parents)])):
            raise RuntimeError("ollama_weight_path_must_be_ordinary_owned_model_storage")
        target.parent.mkdir(parents=True, exist_ok=True)
        size = source.stat().st_size
        reservation = "ollama-import-copy-" + sha.lower()
        mark_acquired, cumulative_reserved = False, 0
        if target.exists():
            if not target.is_file():
                raise RuntimeError("private_ollama_blob_is_not_a_regular_file")
            if os.path.samefile(source, target):
                method, acquired = "verified_existing_hardlink", 0
            elif target.stat().st_size == size and file_hash(target) == sha.lower():
                row = self.state.db.execute("SELECT * FROM weights WHERE id=?", (reservation,)).fetchone()
                if row is not None and (row["bytes"] != size or row["purpose"] != "Ollama physical import copy"
                        or row["path"] != str(target)):
                    raise RuntimeError("ollama_import_copy_ledger_mismatch")
                if row is not None and row["acquired"] == 1:
                    # Ordinary serving can reuse this exact acquired artifact
                    # through its read-only ledger without new authority.
                    method, acquired = "verified_existing_reserved_physical_copy", 0
                else:
                    self.state.reserve_weight(reservation, size, "Ollama physical import copy", target)
                    method, acquired, mark_acquired = "resumed_verified_physical_copy", size, True
                cumulative_reserved = size
            else:
                raise RuntimeError("private_ollama_blob_corrupt:preserving_evidence")
        else:
            try:
                os.link(source, target)
                if not os.path.samefile(source, target):
                    raise RuntimeError("ollama_hardlink_physical_reuse_unverified")
                method, acquired = "verified_same_volume_hardlink", 0
            except OSError:
                self.state.reserve_weight(reservation, size, "Ollama physical import copy", target)
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > size:
                    raise RuntimeError("invalid_resumable_ollama_copy_size")
                _disk_headroom(self.config, size - offset, shutil.disk_usage, destination=target)
                with source.open("rb") as original, partial.open("ab") as copy:
                    original.seek(offset)
                    while chunk := original.read(8 * 1024 * 1024):
                        self.state.check_budget()
                        copy.write(chunk)
                    copy.flush()
                    os.fsync(copy.fileno())
                if partial.stat().st_size != size or file_hash(partial) != sha.lower():
                    raise RuntimeError("ollama_import_copy_hash_mismatch")
                partial.replace(target)
                method, acquired = "reserved_verified_physical_copy", size
                mark_acquired, cumulative_reserved = True, size
        item = {"source": str(source), "blob": str(target), "sha256": sha.lower(), "logical_bytes": size,
                "additional_physical_weight_bytes": acquired, "cumulative_reserved_weight_bytes": cumulative_reserved,
                "reuse_method": method}
        if mark_acquired:
            with self.state.db:
                self.state.db.execute("UPDATE weights SET acquired=1 WHERE id=?", (reservation,))
            self.state.snapshot()
        self._metadata["weight_import"] = item
        self.state.entity("ollama_weight_import", sha.lower(), item)
        return target

    def _mark_descendants(self):
        if self.process is None:
            return
        descendants = _descendants(self.process.pid)
        self.children.update(descendants)
        if hasattr(self, "collector"):
            for pid in descendants:
                self.collector.mark_owned_pid(pid)
        if self.saved:
            self.saved["children"] = list(self.children.values())
            atomic_json(Path(self.config["paths"]["state"]) / ("owned-" + str(self.port) + ".json"), self.saved)

    def _verify_listener(self):
        if self.process is None or self.process.poll() is not None or \
                (os.name == "nt" and _listener_pid(self.port) != self.process.pid):
            raise RuntimeError("owned_ollama_listener_identity_unverified:refusing_existing_endpoint")

    def _manifest(self):
        paths = [path for path in (self.models_root / "manifests").rglob("latest")
                 if path.is_file() and path.parent.name == self.local_name]
        if len(paths) != 1:
            raise RuntimeError("ollama_private_import_manifest_unverified")
        manifest = json.loads(paths[0].read_text(encoding="utf-8"))
        layers = [layer for layer in manifest.get("layers", [])
                  if layer.get("mediaType") == "application/vnd.ollama.image.model"]
        if len(layers) != 1 or layers[0].get("digest") != "sha256:" + self._metadata["model_sha256"] or \
                layers[0].get("size") != Path(self.model["path"]).stat().st_size:
            raise RuntimeError("ollama_import_did_not_reuse_exact_verified_gguf_blob")
        return {"path": str(paths[0]), "sha256": file_hash(paths[0]), "data": manifest}

    def launch(self, config=None):
        if not self._metadata:
            self.discover_capabilities()
        self.state.check_budget(self.config["limits"]["server_start_timeout_seconds"])
        check_port(self.port)
        self.root.mkdir(parents=True, exist_ok=True)
        self._prepare_blob()
        self.launch_count += 1
        prefix = Path(self.config["paths"]["logs"]) / ("ollama-owned-" + str(time.time_ns()))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        stdout, stderr = Path(str(prefix) + "-stdout.log"), Path(str(prefix) + "-stderr.log")
        self.handles = [stdout.open("wb"), stderr.open("wb")]
        argv = [str(self.binary), "serve"]
        environment = self._environment()
        began = time.perf_counter()
        self.process = subprocess.Popen(argv, stdout=self.handles[0], stderr=self.handles[1], stdin=subprocess.DEVNULL,
            env=environment, cwd=self.root, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.saved = {"owner": "localbench", "run_id": self.state.run["id"], "pid": self.process.pid,
            "created": creation_time(self.process.pid), "command_hash": digest(argv), "argv": argv,
            "environment_overrides": {key: environment[key] for key in environment if key.startswith("OLLAMA_")},
            "endpoint": "http://127.0.0.1:" + str(self.port), "stdout": str(stdout), "stderr": str(stderr)}
        if os.name == "nt" and self.saved["created"] is None:
            self.process.terminate()
            self.process.wait(timeout=15)
            self.close_logs()
            raise RuntimeError("owned_ollama_process_identity_unverified")
        atomic_json(Path(self.config["paths"]["state"]) / ("owned-" + str(self.port) + ".json"), self.saved)
        if hasattr(self, "collector"):
            self.collector.mark_owned_pid(self.process.pid)
        deadline = time.monotonic() + self.config["limits"]["server_start_timeout_seconds"]
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("owned_ollama_daemon_crash:" + str(stderr))
                if self.health():
                    self._verify_listener()
                    break
                time.sleep(.25)
            else:
                raise RuntimeError("owned_ollama_start_timeout")
            daemon_version = self.http.json("/api/version").get("version")
            if not isinstance(daemon_version, str) or not daemon_version:
                raise RuntimeError("ollama_daemon_version_unverified")
            modelfile = self.root / (self.local_name + ".Modelfile")
            source = str(Path(self.model["path"]).resolve()).replace("\\", "/")
            if any(character in source for character in ('"', "\r", "\n")):
                raise RuntimeError("invalid_modelfile_source_path")
            modelfile.write_text('FROM "' + source + '"\nPARAMETER num_ctx ' + str(self.requested["context"]) + '\n', encoding="utf-8")
            imported = _logged_command(self.config, "ollama-owned-import", [str(self.binary), "create", self.local_name,
                "-f", str(modelfile)], timeout=self.config["limits"]["server_start_timeout_seconds"], environment=environment)
            if imported["exit_code"] != 0:
                raise RuntimeError("ollama_local_gguf_import_failed:" + imported["log"])
            manifest = self._manifest()
            shown = self.http.json("/api/show", {"model": self.local_name, "verbose": False})
            if shown.get("remote_host") or shown.get("remote_model"):
                raise RuntimeError("ollama_remote_model_forbidden")
            self.capabilities = shown.get("capabilities", [])
            self._metadata.update(daemon_version=daemon_version, local_model_name=self.local_name,
                model_api_details=shown, model_manifest=manifest, model_manifest_sha256=manifest["sha256"],
                import_log=imported["log"], modelfile=str(modelfile))
            self.http.json("/api/chat", {"model": self.local_name, "messages": [], "stream": False,
                "keep_alive": -1, "options": self._runner_options()})
            loaded = [model for model in self.http.json("/api/ps").get("models", [])
                if model.get("name", model.get("model")) in {self.local_name, self.local_name + ":latest"}]
            if len(loaded) != 1 or type(loaded[0].get("context_length")) is not int or loaded[0]["context_length"] < 65536:
                raise RuntimeError("ollama_effective_num_ctx_unverified_or_insufficient")
            self.effective = {**self.requested, "effective_context": loaded[0]["context_length"], "effective_slots": 1,
                "context_shift_disabled": all(control in self.verified_controls for control in ("truncate", "shift")),
                "observed_residency": {key: loaded[0].get(key) for key in ("size", "size_vram", "digest")},
                "startup_seconds": time.perf_counter() - began, "launch_argv": argv,
                "environment_overrides": self.saved["environment_overrides"], "model_manifest_sha256": manifest["sha256"],
                "thinking_control": self._think(), "sampler_effective": "native options;observed request logged"}
            self._mark_descendants()
            # The scheduler registers the stable effective configuration.
            return {"endpoint": self.saved["endpoint"], "handle": self.saved, "effective": self.effective}
        except Exception:
            self.unload_owned()
            raise

    def _runner_options(self):
        return {"num_ctx": self.requested["context"], "num_batch": self.requested["batch"],
                "num_gpu": self.requested["gpu_layers"], "num_thread": self.requested["threads"], "draft_num_predict": 0}

    def _think(self):
        profile = self.requested["reasoning"]
        if profile == "default":
            return None
        architecture = self.model.get("metadata", {}).get("general.architecture", "")
        if "thinking" not in getattr(self, "capabilities", []):
            raise RuntimeError("unsupported_ollama_model_thinking_control")
        if profile == "disabled" and "gptoss" not in architecture.replace("-", ""):
            return False
        if profile == "reduced" and "gptoss" in architecture.replace("-", ""):
            return "low"
        raise RuntimeError("unsupported_ollama_reasoning_profile:no_fictitious_numeric_budget")

    def health(self):
        try:
            return isinstance(LocalHTTP("http://127.0.0.1:" + str(self.port), 2).json("/api/version").get("version"), str)
        except (OSError, RuntimeError, ValueError):
            return False

    def tokenize(self, messages, tools):
        helper = _verified_tokenizer(self.exact_tokenizer, self._metadata.get("model_sha256"), "ollama")
        result = helper.tokenize(messages, tools)
        if not isinstance(result, dict) or type(result.get("count")) is not int or result["count"] < 0:
            raise RuntimeError("invalid_offline_tokenizer_count")
        return result

    def tokenize_text(self, text):
        return _verified_tokenizer(self.exact_tokenizer, self._metadata.get("model_sha256"), "ollama").tokenize_text(text)

    def native_request(self, request):
        unknown = set(request) - self.SUPPORTED_REQUEST
        if unknown:
            raise RuntimeError("unsupported_ollama_request_controls:" + ",".join(sorted(unknown)))
        if request.get("tool_choice", "auto") != "auto" or request.get("stream", True) is not True:
            raise RuntimeError("unsupported_ollama_tool_choice_or_nonstream_probe")
        if request.get("stream_options", {"include_usage": True}) != {"include_usage": True}:
            raise RuntimeError("unsupported_ollama_stream_options")
        messages, names = [], {}
        for message in request.get("messages", []):
            converted = {"role": message["role"], "content": message.get("content") or ""}
            if message.get("reasoning_content"):
                converted["thinking"] = message["reasoning_content"]
            if "tool_calls" in message:
                converted["tool_calls"] = []
                for call in message["tool_calls"]:
                    function = call["function"]
                    arguments = json.loads(function["arguments"]) if isinstance(function["arguments"], str) else function["arguments"]
                    if not isinstance(arguments, dict):
                        raise RuntimeError("invalid_ollama_history_tool_arguments")
                    converted["tool_calls"].append({"function": {"name": function["name"], "arguments": arguments}})
                    names[call.get("id")] = function["name"]
            if message["role"] == "tool":
                name = names.get(message.get("tool_call_id"))
                if not name:
                    raise RuntimeError("unmatched_ollama_history_tool_response")
                converted["tool_name"] = name
            messages.append(converted)
        options = self._runner_options()
        for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty", "seed", "stop"):
            if key in request:
                options[key] = request[key]
        if "max_tokens" in request:
            options["num_predict"] = request["max_tokens"]
        native = {"model": self.local_name, "messages": messages, "tools": request.get("tools", []),
                  "stream": True, "keep_alive": -1, "options": options}
        thinking = self._think()
        if thinking is not None:
            native["think"] = thinking
        for control in ("truncate", "shift"):
            if control in self.verified_controls:
                native[control] = False
        return native

    def stream_chat(self, request, **kwargs):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("owned_ollama_daemon_not_running")
        self._verify_listener()
        self._mark_descendants()
        return self.http.measure(self.native_request(request), protocol="ollama", **kwargs)

    def reset_cache(self):
        return {"status": "owned_daemon_restart_required", "verified": False}

    def fresh_restart(self):
        self.unload_owned()
        return self.launch()

    def close_logs(self):
        for handle in self.handles:
            if not handle.closed:
                handle.close()
        self.handles = []

    def unload_owned(self):
        if self.process is None:
            return
        original = self.process
        if original.poll() is None:
            if creation_time(original.pid) != self.saved["created"] or self.saved.get("owner") != "localbench":
                raise RuntimeError("stale_ollama_pid_identity:refusing_termination")
            self._mark_descendants()
            try:
                if hasattr(self, "local_name"):
                    self._verify_listener()
                    self.http.json("/api/chat", {"model": self.local_name, "messages": [], "stream": False, "keep_alive": 0})
            except (OSError, RuntimeError, ValueError):
                pass
            original.terminate()
            try:
                original.wait(timeout=15)
            except subprocess.TimeoutExpired:
                original.kill()
                original.wait(timeout=15)
        for record in reversed(list(self.children.values())):
            if creation_time(record["pid"]) == record["created"]:
                _terminate_recorded_child(record)
            if hasattr(self, "collector"):
                self.collector.mark_owned_pid(record["pid"], owned=False)
        if hasattr(self, "collector"):
            self.collector.mark_owned_pid(original.pid, owned=False)
        self.close_logs()
        self.process = None
        if self.saved:
            atomic_json(Path(self.config["paths"]["state"]) / ("owned-" + str(self.port) + ".json"),
                        {**self.saved, "stopped": True})
        self.children = {}

    def metadata(self):
        return {**self._metadata, "effective_settings": self.effective}
