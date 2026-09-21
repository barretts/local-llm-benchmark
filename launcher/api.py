"""Ordinary local serving, not a benchmark winner/qualification handoff."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localbench.state import gpu_lock
from localbench.recovery import WindowsProcess

RUNTIME = ROOT / "launcher" / "runtime"
STATUS = RUNTIME / "api-status.json"
ALIAS = "local-coder"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def settings():
    return read_json(ROOT / "launcher" / "profiles.json")


def clean_environment():
    # Prevent inherited API credentials, runtime overrides and global Aider flags.
    return {key: value for key, value in os.environ.items()
            if not any(term in key.lower() for term in
                       ("token", "secret", "password", "api_key", "apikey"))
            and not key.upper().startswith(("AIDER_", "LLAMA_", "LITELLM_", "OPENAI_", "GGML_"))
            and key.upper() not in ("PYTHONHOME", "PYTHONPATH", "CUDA_VISIBLE_DEVICES")}


def argv_for(profile, context, port):
    config = settings()
    entry = config["profiles"][profile]
    if context not in entry["contexts"]:
        raise ValueError("unsupported context for this profile")
    if not 1024 <= port <= 65535:
        raise ValueError("port must be 1024..65535")
    return [config["engine"]["path"], "--model", entry["path"],
            "--alias", ALIAS, "--ctx-size", str(context), "--parallel", "1",
            "--n-gpu-layers", "99", "--flash-attn", "on",
            "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--batch-size", "2048", "--ubatch-size", "512",
            "--threads", "16", "--threads-batch", "16",
            "--host", "127.0.0.1", "--port", str(port),
            "--jinja", "--no-context-shift", "--device", "CUDA0",
            "--spec-type", "none", "--metrics", "--verbosity", "1",
            "--cors-origins", f"http://127.0.0.1:{port}",
            "--no-cors-credentials"]


def request(port, route, payload=None, timeout=5):
    url = f"http://127.0.0.1:{port}{route}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def healthy(port):
    try:
        return request(port, "/health").get("status") == "ok"
    except (OSError, ValueError):
        return False


def port_available(port):
    with socket.socket() as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"port {port} is occupied; its owner will not be stopped") from None


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_files(profile):
    config = settings()
    engine = config["engine"]
    if file_hash(engine["path"]) != engine["sha256"]:
        raise RuntimeError("installed engine differs from the pinned binary")
    model = config["profiles"][profile]
    info = Path(model["path"]).stat()
    if info.st_size != model["bytes"]:
        raise RuntimeError("installed model size differs from the pinned weights")
    cache_path = RUNTIME / "verified-models.json"
    cache = read_json(cache_path) or {}
    identity = {"path": model["path"], "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns, "sha256": model["sha256"]}
    if cache.get(profile) != identity:
        print("Verifying installed model weights (first launch only)...", flush=True)
        if file_hash(model["path"]) != model["sha256"]:
            raise RuntimeError("installed model hash differs from the pinned weights")
        cache[profile] = identity
        write_json(cache_path, cache)


def gpu_memory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                             "--format=csv,noheader,nounits"], check=True,
                            capture_output=True, text=True, timeout=15,
                            env=clean_environment(), creationflags=subprocess.CREATE_NO_WINDOW)
    values = result.stdout.strip().splitlines()
    if len(values) != 1:
        raise RuntimeError("launcher requires the single recorded GPU")
    used, free = (int(item.strip()) for item in values[0].split(","))
    return {"used_mib": used, "free_mib": free}


class StopEvent:
    def __init__(self, session, create=False):
        if len(session) != 32 or any(c not in "0123456789abcdef" for c in session):
            raise ValueError("invalid launcher session")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateEventW.restype = wintypes.HANDLE
        self.kernel.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.OpenEventW.restype = wintypes.HANDLE
        self.kernel.SetEvent.argtypes = [wintypes.HANDLE]
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        name = "Local\\local-model-launcher-" + session
        self.handle = (self.kernel.CreateEventW(None, True, False, name) if create
                       else self.kernel.OpenEventW(0x2 | 0x100000, False, name))
        if not self.handle:
            raise RuntimeError("launcher stop event unavailable")

    def wait(self, seconds):
        return self.kernel.WaitForSingleObject(self.handle, int(seconds * 1000)) == 0

    def set(self):
        if not self.kernel.SetEvent(self.handle):
            raise RuntimeError("could not signal owned launcher")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def controller_argv(profile, context, port, session):
    return [str(ROOT / ".venv" / "Scripts" / "python.exe"),
            str(ROOT / "launcher" / "api.py"), "_serve", profile,
            "--context", str(context), "--port", str(port), "--session", session]


def owned_status():
    saved = read_json(STATUS)
    if not saved or saved.get("owner") != "local-model-launcher":
        return None
    process = WindowsProcess(saved["controller_pid"])
    try:
        if not process.alive():
            return None
        expected = controller_argv(saved["profile"], saved["context"], saved["port"], saved["session"])
        if process.created() != saved["controller_created"] or process.argv() != expected:
            raise RuntimeError("saved launcher identity could not be verified")
        return saved
    finally:
        process.close()


def stop():
    saved = owned_status()
    if not saved:
        print("No launcher model is running.")
        return
    event = StopEvent(saved["session"])
    try:
        event.set()
    finally:
        event.close()
    process = WindowsProcess(saved["controller_pid"])
    try:
        deadline = time.monotonic() + 30
        while process.alive() and time.monotonic() < deadline:
            time.sleep(0.2)
        if process.alive():
            raise RuntimeError("owned launcher did not stop; no other process was terminated")
    finally:
        process.close()
    print("Model stopped; GPU released.")


def serve(args):
    from launcher.windows_job import OwnedJob
    session = args.session
    RUNTIME.mkdir(parents=True, exist_ok=True)
    # This is the same OS lease as the benchmark, without opening its database.
    with gpu_lock({"paths": {"state": str(ROOT / "state")}}):
        event = StopEvent(session, create=True)
        process = WindowsProcess(os.getpid())
        saved = {"owner": "local-model-launcher", "session": session,
                 "profile": args.profile, "label": settings()["profiles"][args.profile]["label"],
                 "context": args.context, "port": args.port,
                 "api_base": f"http://127.0.0.1:{args.port}/v1", "model": ALIAS,
                 "controller_pid": os.getpid(), "controller_created": process.created(),
                 "status": "starting", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        process.close()
        write_json(STATUS, saved)
        job = OwnedJob()
        try:
            port_available(args.port)
            verify_files(args.profile)
            saved["gpu_before"] = gpu_memory()
            if saved["gpu_before"]["used_mib"] > 2048:
                raise RuntimeError("GPU has an active workload; finish it before launching")
            command = argv_for(args.profile, args.context, args.port)
            saved["engine_argv"] = command
            saved["engine_commit"] = settings()["engine"]["commit"]
            saved["server_log"] = str(RUNTIME / ("server-" + session + ".log"))
            write_json(STATUS, saved)
            with Path(saved["server_log"]).open("ab", buffering=0) as log:
                saved["server_pid"] = job.spawn(command, env=clean_environment(), stdout=log)
                write_json(STATUS, saved)
                deadline = time.monotonic() + 180
                while not healthy(args.port):
                    if event.wait(0.5):
                        return
                    if not job.alive():
                        raise RuntimeError("model server exited; inspect its local server log")
                    if time.monotonic() > deadline:
                        raise RuntimeError("model startup exceeded 180 seconds")
                saved["gpu_loaded"] = gpu_memory()
                if saved["gpu_loaded"]["free_mib"] < 512:
                    raise RuntimeError("model leaves less than 512 MiB of GPU headroom")
                models = request(args.port, "/v1/models")["data"]
                if not any(model["id"] == ALIAS for model in models):
                    raise RuntimeError("server alias did not match the Aider configuration")
                if not job.alive():
                    raise RuntimeError("owned server exited before readiness")
                props = request(args.port, "/props")
                if (props.get("default_generation_settings", {}).get("n_ctx") != args.context
                        or props.get("total_slots") != 1
                        or Path(props.get("model_path", "")).resolve() != Path(command[2]).resolve()):
                    raise RuntimeError("effective model/context does not match the launch profile")
                saved["effective_context"] = props["default_generation_settings"]["n_ctx"]
                saved["model_sha256"] = settings()["profiles"][args.profile]["sha256"]
                saved["status"] = "ready"
                write_json(STATUS, saved)
                while not event.wait(1):
                    if not job.alive():
                        raise RuntimeError("owned model server stopped unexpectedly")
        except Exception as error:
            saved["status"] = "failed"
            saved["error"] = str(error)
            raise
        finally:
            try:
                job.close()
            except Exception as error:
                saved["status"] = "failed"
                saved["error"] = "owned process cleanup failed: " + str(error)
                raise
            finally:
                event.close()
                if saved["status"] != "failed":
                    saved["status"] = "stopped"
                write_json(STATUS, saved)


def start(profile="qwen", context=16384, port=8080):
    argv_for(profile, context, port)  # Validate before stopping an existing model.
    saved = owned_status()
    if saved and saved["status"] == "ready" and saved["profile"] == profile and saved["context"] == context and saved["port"] == port and healthy(port):
        print(f"Already ready: {saved['label']} at {saved['api_base']}")
        return saved
    if saved:
        stop()
    port_available(port)
    # Read-only lease check; detached controller acquires and holds it again.
    with gpu_lock({"paths": {"state": str(ROOT / "state")}}):
        pass
    RUNTIME.mkdir(parents=True, exist_ok=True)
    session = uuid.uuid4().hex
    print(f"Starting {settings()['profiles'][profile]['label']} ({context} tokens)...", flush=True)
    with (RUNTIME / ("controller-" + session + ".log")).open("ab", buffering=0) as log:
        child = subprocess.Popen(controller_argv(profile, context, port, session),
                                 cwd=ROOT, env=clean_environment(), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    deadline = time.monotonic() + 210
    while time.monotonic() < deadline:
        saved = read_json(STATUS)
        if saved and saved.get("session") == session:
            if saved["status"] == "ready":
                print(f"Ready: {saved['api_base']} (model: {ALIAS})")
                return saved
            if saved["status"] == "failed":
                raise RuntimeError(saved.get("error", "launcher failed"))
        if child.poll() is not None:
            raise RuntimeError("launcher exited; inspect its controller log in launcher/runtime")
        time.sleep(0.5)
    saved = owned_status()
    if saved and saved["session"] == session:
        stop()
    raise RuntimeError("launcher startup timed out")


def aider_command(repo, saved, files=(), message=None, yes=False, no_git=False):
    python = RUNTIME / "aider-env" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError("private Aider is missing; see launcher/README.md installation instructions")
    context = saved["context"]
    input_limit = context - 4096
    metadata = RUNTIME / f"aider-metadata-{context}.json"
    model_settings = RUNTIME / f"aider-settings-{context}.json"
    write_json(metadata, {"openai/" + ALIAS: {
        "max_tokens": 4096, "max_input_tokens": input_limit, "max_output_tokens": 4096,
        "input_cost_per_token": 0, "output_cost_per_token": 0,
        "litellm_provider": "openai", "mode": "chat"}})
    write_json(model_settings, [{"name": "openai/" + ALIAS, "edit_format": "diff",
        "weak_model_name": "openai/" + ALIAS, "editor_model_name": "openai/" + ALIAS,
        "use_repo_map": True, "extra_params": {"max_tokens": 4096, "temperature": 0.2}}])
    home = RUNTIME / "aider-home"
    home.mkdir(parents=True, exist_ok=True)
    config = home / "launcher-config.yml"
    config.write_text("{}\n", encoding="utf-8")
    history = RUNTIME / "aider-sessions" / uuid.uuid4().hex
    history.mkdir(parents=True)
    command = [str(python), "-m", "aider", "--model", "openai/" + ALIAS,
               "--weak-model", "openai/" + ALIAS, "--editor-model", "openai/" + ALIAS,
               "--openai-api-base", saved["api_base"], "--openai-api-key", "local-only",
               "--config", str(config), "--model-settings-file", str(model_settings),
               "--model-metadata-file", str(metadata), "--edit-format", "diff",
               "--no-auto-commits", "--no-dirty-commits", "--no-gitignore",
               "--no-add-gitignore-files", "--no-analytics", "--no-check-update",
               "--no-show-release-notes", "--no-auto-test", "--no-auto-lint",
               "--no-detect-urls", "--no-suggest-shell-commands", "--no-browser",
               "--disable-playwright", "--no-cache-prompts", "--cache-keepalive-pings", "0",
               "--map-tokens", "1024", "--max-chat-history-tokens", "4096",
               "--input-history-file", str(history / "input.txt"),
               "--chat-history-file", str(history / "chat.md")]
    if no_git or not any((parent / ".git").exists() for parent in [repo, *repo.parents]):
        command.append("--no-git")
    if message is not None:
        command.extend(["--message", message, "--exit"])
    if yes:
        command.append("--yes-always")
    command.extend(files)
    environment = clean_environment()
    environment.update({"USERPROFILE": str(home), "PYTHON_DOTENV_DISABLED": "1",
                        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                        "LITELLM_MODE": "PRODUCTION", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                        "TIKTOKEN_CACHE_DIR": str(RUNTIME / "cache" / "tiktoken"),
                        "HF_HOME": str(RUNTIME / "cache" / "huggingface"),
                        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
    return command, environment


def run_aider(repo, profile, context, port, files=(), message=None, yes=False, no_git=False):
    repo = Path(repo).expanduser().resolve()
    if not repo.is_dir():
        raise ValueError("choose an existing coding folder")
    saved = start(profile, context, port)
    command, environment = aider_command(repo, saved, files, message, yes, no_git)
    print("Aider will edit files you add. Automatic Git commits are disabled.", flush=True)
    return subprocess.run(command, cwd=repo, env=environment).returncode


def menu():
    while True:
        print("\nLocal models + Aider\n"
              "1  Start Qwen3.6 (16K, quality)\n"
              "2  Start OxCoder (16K, faster)\n"
              "3  Start Qwen3.6 (64K)\n"
              "4  Open Aider in a coding folder\n"
              "5  API status\n6  Stop model\n0  Exit menu (API stays running)")
        choice = input("Choose: ").strip()
        try:
            if choice in ("1", "2", "3"):
                start("oxcoder" if choice == "2" else "qwen", 65536 if choice == "3" else 16384)
            elif choice == "4":
                repo = input("Coding folder (full path): ").strip().strip('"')
                saved = owned_status()
                run_aider(repo, saved["profile"] if saved else "qwen", saved["context"] if saved else 16384,
                          saved["port"] if saved else 8080)
            elif choice == "5":
                show_status()
            elif choice == "6":
                stop()
            elif choice == "0":
                return 0
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Error: {error}")


def show_status():
    saved = owned_status()
    if saved:
        print(json.dumps({**saved, "health_ok": healthy(saved["port"]), "gpu_now": gpu_memory()}, indent=2))
    else:
        print("No launcher model is running.")


def main():
    parser = argparse.ArgumentParser(description="Local OpenAI API launcher + private Aider")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "_serve", "aider"):
        command = commands.add_parser(name)
        command.add_argument("profile", choices=tuple(settings()["profiles"]), nargs="?", default="qwen")
        command.add_argument("--context", type=int, default=16384, choices=(16384, 65536))
        command.add_argument("--port", type=int, default=8080)
        if name == "_serve":
            command.add_argument("--session", required=True)
        if name == "aider":
            command.add_argument("--repo", required=True)
            command.add_argument("--message")
            command.add_argument("--yes", action="store_true", help="accept confirmations; use only for controlled batch tasks")
            command.add_argument("--file", action="append", default=[], help="file to add to Aider (repeatable)")
            command.add_argument("--no-git", action="store_true", help="use only selected files without a Git repository map")
    for name in ("stop", "status", "menu"):
        commands.add_parser(name)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("these installed profiles require Windows/CUDA")
    try:
        if args.command == "start":
            start(args.profile, args.context, args.port)
        elif args.command == "_serve":
            serve(args)
        elif args.command == "stop":
            stop()
        elif args.command == "status":
            show_status()
        elif args.command == "menu":
            return menu()
        elif args.command == "aider":
            return run_aider(args.repo, args.profile, args.context, args.port, args.file, args.message, args.yes, args.no_git)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
