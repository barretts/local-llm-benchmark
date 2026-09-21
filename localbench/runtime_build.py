"""Isolated, pinned CUDA source builds; importing/planning performs no work.

prepare_pinned_source acquires source and writes a complete script review
bundle. The controller reads that bundle before passing its fingerprint to
build_pinned_source. No custom installer or repository batch file is executed.
"""
from pathlib import Path
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time

from .acquisition import _disk_headroom, _runtime_headroom
from .config import atomic_json, digest, file_hash
from .doctor import sanitize

_PINS = {
    "turboquant-cuda": ("https://github.com/jarkevithwlad/llama.cpp-turboquant-cuda", "8ed935a092ee66ac87fdeb5cb8d5f383edb28f90"),
    "ik-llama": ("https://github.com/ikawrakow/ik_llama.cpp", "2ae132fa601ea06818ed3584f50f7eb4f72d4967"),
}


def _environment():
    # Compiler/Git processes do not inherit API tokens, SSH agents, proxy
    # credentials, arbitrary Git configuration, or Python/Node customizations.
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
               "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "LOCALAPPDATA",
               "SYSTEMDRIVE", "PROGRAMDATA", "ALLUSERSPROFILE", "PROCESSOR_ARCHITECTURE",
               "PROCESSOR_ARCHITEW6432"}
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull})
    return env


def _redact(text):
    return re.sub(r"https?://[^\s<>]+\?[^\s<>]+", "[URL query redacted]", sanitize(text))


class _WindowsJob:
    """A kernel-owned process tree, killed on handle close; no PID/name kills."""
    def __init__(self):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("cannot create owned build job")
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise OSError("cannot configure owned build job")

    def attach_resume(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            process.kill()
            process.wait(timeout=10)
            raise OSError("cannot assign owned build job")
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        if ntdll.NtResumeProcess(int(process._handle)) != 0:
            self.close()
            process.wait(timeout=10)
            raise OSError("cannot resume owned build process")

    def close(self):
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_owned(config, state, label, argv, *, cwd=None, timeout=30, setup_deadline=None,
              popen=subprocess.Popen, monotonic=time.monotonic, sleeper=time.sleep,
              job_factory=None, disk_usage=shutil.disk_usage, cleanup=False):
    """Run one explicit command, with bounded time and owned descendant cleanup."""
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or "\0" in a for a in argv):
        raise ValueError("build command must be an explicit string argument vector")
    if not cleanup:
        state.check_budget()
    log = Path(config["paths"]["logs"]) / ("setup-" + label + "-" + str(time.time_ns()) + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    begun = monotonic()
    use_job = os.name == "nt"
    job = (job_factory or _WindowsJob)() if use_job else None
    process, output, errors = None, [], []
    flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) | 0x4) if use_job else 0
    try:
        process = popen(argv, cwd=cwd, env=_environment(), stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        creationflags=flags, start_new_session=not use_job)
        if job:
            job.attach_resume(process)
        def collect():
            try:
                with log.open("w", encoding="utf-8") as target:
                    while line := process.stdout.readline(1024 * 1024):
                        # Never persist an unbounded/fragmented potential secret.
                        if len(line) >= 1024 * 1024:
                            while line and not line.endswith(b"\n"):
                                line = process.stdout.readline(1024 * 1024)
                            value = "[oversized log line redacted]\n"
                        else:
                            value = _redact(line.decode("utf-8", errors="replace"))
                        target.write(value)
                        target.flush()
                        output.append(value)
                        if sum(map(len, output)) > 1024 * 1024:
                            output.pop(0)
            except OSError:
                errors.append("build_log_collection_failed")
        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        reason = None
        while process.poll() is None:
            try:
                if not cleanup:
                    state.check_budget()
                    _disk_headroom(config, 0, disk_usage)
                    _runtime_headroom(config, 0)
                if setup_deadline is not None and state.clock() >= setup_deadline:
                    raise RuntimeError("engine_setup_budget_exhausted")
                if monotonic() - begun >= timeout:
                    raise RuntimeError("owned_build_command_timeout")
            except (RuntimeError, OSError) as exc:
                reason = str(exc)
                break
            sleeper(.2)
        if reason is not None:
            if job:
                job.close()
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if not job:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=10)
        else:
            process.wait(timeout=10)
        if job:
            job.close()  # Also removes any residual owned child after success.
        elif process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        reader.join(timeout=10)
        if reader.is_alive() or errors:
            raise RuntimeError("build_log_collection_failed")
        result = {"argv": argv, "exit_code": process.returncode, "seconds": monotonic() - begun,
                  "output": "".join(output), "log": str(log), "reason": reason}
        if reason:
            raise RuntimeError(reason + ":" + str(log))
        return result
    finally:
        if job:
            job.close()
        if process is not None and process.poll() is None:
            if not use_job:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=10)
        if process is not None and process.stdout is not None:
            process.stdout.close()


def detect_windows_toolchain(config, runner=None):
    """Read existing VS2022/CUDA12.6/CMake facts; never install or edit tools."""
    runner = runner or (lambda label, argv: subprocess.run(argv, capture_output=True, text=True, timeout=30))
    tools = config["installed_tools"]
    candidates = [tools.get("vswhere"), shutil.which("vswhere"),
                  r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"]
    vswhere = next((p for p in candidates if p and Path(p).is_file()), None)
    cmake = tools.get("cmake") or shutil.which("cmake")
    cuda_root = tools.get("cuda_root", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")
    nvcc = tools.get("nvcc") or str(Path(cuda_root) / "bin" / "nvcc.exe")
    result = {"available": False, "vswhere": vswhere, "cmake": cmake, "nvcc": nvcc, "cuda_root": cuda_root, "evidence": {}}
    if not vswhere or not cmake or not Path(nvcc).is_file():
        result["reason"] = "missing_compiler_toolkit"
        return result
    def probe(label, argv):
        raw = runner(label, argv)
        if isinstance(raw, dict):
            return raw
        return {"argv": argv, "exit_code": raw.returncode, "output": (raw.stdout or "") + (raw.stderr or "")}
    query = [vswhere, "-latest", "-products", "*", "-version", "[17.0,18.0)", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json", "-utf8"]
    vs = probe("vswhere", query)
    result["evidence"]["vswhere"] = vs
    try:
        installations = json.loads(vs["output"])
    except (ValueError, KeyError):
        installations = []
    if vs.get("exit_code") != 0 or not installations:
        result["reason"] = "missing_supported_msvc_vs2022"
        return result
    installation = installations[0]
    vsroot = Path(installation["installationPath"])
    msvc = sorted((vsroot / "VC" / "Tools" / "MSVC").glob("*/bin/Hostx64/x64/cl.exe"), reverse=True)
    if not msvc:
        result["reason"] = "missing_msvc_compiler"
        return result
    result.update({"cl": str(msvc[0]), "visual_studio": str(vsroot),
                   "visual_studio_version": installation.get("installationVersion"),
                   "generator": "Visual Studio 17 2022"})
    for name, argv in (("cmake", [cmake, "--version"]), ("nvcc", [nvcc, "--version"]), ("cl", [str(msvc[0])])):
        result["evidence"][name] = probe(name, argv)
    output = result["evidence"]["nvcc"].get("output", "")
    if not re.search(r"release 12\.6(?:,|\s)", output):
        result["reason"] = "cuda_12_6_toolkit_not_confirmed"
        return result
    if result["evidence"]["cmake"].get("exit_code") != 0:
        result["reason"] = "cmake_unavailable"
        return result
    result.update({"available": True, "cuda_version": "12.6"})
    return result


def _pin(config, engine_id):
    if engine_id not in _PINS:
        raise ValueError("only reviewed pinned CUDA source engines are eligible")
    repo, commit = _PINS[engine_id]
    discovery = json.loads((Path(config["paths"]["artifacts"]) / "engine-discovery.json").read_text(encoding="utf-8-sig"))
    record = next(item for item in discovery["engines"] if item["id"] == engine_id)
    if record["pin"].get("commit") != commit:
        raise ValueError("source pin changed; review discovery before setup")
    return repo, commit


def _setup(config, state, engine_id, commit):
    if not state.get_control("harness_verified"):
        raise RuntimeError("source setup requires verified harness")
    state.start_execution("runtime_source_setup:" + engine_id)
    key = "runtime_setup:" + engine_id + "@" + commit
    setup = state.get_control(key)
    if setup is None:
        candidate = next(item for item in config["runtime_candidates"] if item["id"] == engine_id)
        now = state.clock()
        setup = {"started": now, "deadline": now + candidate["probe_minutes"] * 60, "phase": "source"}
        state.control(key, setup)
    state.check_budget()
    if state.clock() >= setup["deadline"]:
        raise RuntimeError("engine_setup_budget_exhausted")
    return key, setup


def _run(config, state, setup, runner, label, argv, cwd=None):
    remaining = setup["deadline"] - state.clock()
    if remaining <= 0:
        raise RuntimeError("engine_setup_budget_exhausted")
    result = runner(config, state, label, argv, cwd=cwd, timeout=remaining, setup_deadline=setup["deadline"])
    if result["exit_code"] != 0:
        raise RuntimeError("source_setup_command_failed:" + label + ":" + result.get("log", ""))
    return result


def _scripts(source):
    files = [p for p in source.rglob("*") if p.is_file() and ".git" not in p.relative_to(source).parts and
             (p.name == "CMakeLists.txt" or p.suffix.lower() in (".cmake", ".bat", ".ps1", ".sh"))]
    if sum(p.stat().st_size for p in files) > 16 * 1024 * 1024:
        raise RuntimeError("source_script_review_size_limit")
    hashes = {str(p.relative_to(source)).replace("\\", "/"): file_hash(p) for p in sorted(files)}
    if "CMakeLists.txt" not in hashes:
        raise RuntimeError("pinned_source_missing_cmake_project")
    return hashes


def prepare_pinned_source(config, state, engine_id, *, runner=run_owned):
    """Acquire/verify an owned detached checkout and return its script bundle."""
    repo, commit = _pin(config, engine_id)
    key, setup = _setup(config, state, engine_id, commit)
    _disk_headroom(config, 4 * 1024 ** 3)
    _runtime_headroom(config, 4 * 1024 ** 3)
    root = Path(config["paths"]["runtime_root"]) / (engine_id + "-" + commit)
    if not root.resolve().is_relative_to(Path(config["paths"]["runtime_root"]).resolve()):
        raise ValueError("source root escapes private runtime directory")
    marker = root / "localbench-owned.json"
    ownership = {"owner": "localbench", "run_id": state.run["id"], "engine": engine_id, "commit": commit, "repo": repo}
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise RuntimeError("preexisting_unowned_source_directory")
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) != ownership:
        raise RuntimeError("source_ownership_identity_mismatch")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, ownership)
    hooks = root / "disabled-hooks"
    hooks.mkdir(exist_ok=True)
    if any(hooks.iterdir()):
        raise RuntimeError("owned_disabled_hooks_directory_not_empty")
    source = root / "source"
    git = config["installed_tools"].get("git", "git")
    prefix = [git, "-c", "core.hooksPath=" + str(hooks), "-c", "core.fsmonitor=false", "-c", "credential.helper=",
              "-c", "protocol.file.allow=never", "-c", "submodule.recurse=false"]
    evidence = []
    if not (source / ".git").exists():
        if source.exists() and any(source.iterdir()):
            raise RuntimeError("preexisting_unowned_checkout")
        evidence.append(_run(config, state, setup, runner, engine_id + "-git-init", prefix + ["init", str(source)]))
        evidence.append(_run(config, state, setup, runner, engine_id + "-git-remote", prefix + ["-C", str(source), "remote", "add", "origin", repo]))
    else:
        remote = _run(config, state, setup, runner, engine_id + "-git-remote-check", prefix + ["-C", str(source), "remote", "get-url", "origin"])
        if remote["output"].strip() != repo:
            raise RuntimeError("pinned_source_remote_mismatch")
    present = _run(config, state, setup, runner, engine_id + "-git-objects", prefix + ["-C", str(source), "rev-parse", "--all"])
    if commit not in present["output"].split():
        evidence.append(_run(config, state, setup, runner, engine_id + "-git-fetch", prefix + ["-C", str(source), "fetch", "--depth", "1", "origin", commit]))
    evidence.append(_run(config, state, setup, runner, engine_id + "-git-checkout", prefix + ["-C", str(source), "checkout", "--detach", commit]))
    actual = _run(config, state, setup, runner, engine_id + "-git-sha", prefix + ["-C", str(source), "rev-parse", "HEAD"])
    status = _run(config, state, setup, runner, engine_id + "-git-clean", prefix + ["-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"])
    if actual["output"].strip() != commit or status["output"].strip():
        raise RuntimeError("source_checkout_not_exact_clean_pin")
    hashes = _scripts(source)
    review = Path(config["paths"]["artifacts"]) / ("source-review-" + engine_id + "-" + commit + ".txt")
    with review.open("w", encoding="utf-8") as output:
        for name, sha in hashes.items():
            output.write("\nFILE " + name + " SHA256 " + sha + "\n")
            output.write((source / name).read_text(encoding="utf-8", errors="replace"))
            output.write("\n")
    prepared = {**ownership, "root": str(root), "source": str(source), "script_hashes": hashes,
                "review_fingerprint": digest({"commit": commit, "scripts": hashes}), "review_bundle": str(review),
                "source_commands": evidence, "setup_key": key, "setup_deadline": setup["deadline"]}
    atomic_json(root / "source-manifest.json", prepared)
    state.entity("runtime_source", engine_id, prepared)
    return prepared


def plan_build(config, prepared, toolchain=None, docker_image=None):
    engine = prepared["engine"]
    if engine not in _PINS or prepared["commit"] != _PINS[engine][1]:
        raise ValueError("unreviewed engine pin")
    root, source = Path(prepared["root"]), Path(prepared["source"])
    flags = ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=89", "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON",
             "-DGGML_AVX512=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=ON"]
    flags += ["-DLLAMA_BUILD_UI=OFF","-DLLAMA_USE_PREBUILT_UI=OFF","-DGGML_CCACHE=OFF","-DLLAMA_LLGUIDANCE=OFF","-DLLAMA_BUILD_BORINGSSL=OFF","-DLLAMA_BUILD_LIBRESSL=OFF","-DGGML_CPU_KLEIDIAI=OFF","-DGGML_CUDA_CUB_3DOT2=OFF"]
    flags += ["-DGGML_CUDA_FA=ON", "-DGGML_CUDA_FA_ALL_QUANTS=OFF"] if engine == "turboquant-cuda" else ["-DGGML_IQK_FA_ALL_QUANTS=ON"]
    parallel = config.get("build_parallel_jobs", 2)
    if type(parallel) is not int or not 1 <= parallel <= 4:
        raise ValueError("source build parallelism must be between one and four")
    if toolchain and toolchain.get("available"):
        if toolchain.get("cuda_version") != "12.6" or toolchain.get("generator") != "Visual Studio 17 2022":
            raise ValueError("native source build requires confirmed CUDA12.6 and VS2022")
        build = root / "build-sm89-windows"
        configure = [toolchain["cmake"], "-S", str(source), "-B", str(build), "-G", toolchain["generator"],
                     "-A", "x64", "-T", "cuda=" + toolchain["cuda_root"], *flags]
        if toolchain.get('visual_studio'):
            instance = toolchain['visual_studio']
            if toolchain.get('visual_studio_version'):
                instance += ',version=' + toolchain['visual_studio_version']
            configure += ['-DCMAKE_GENERATOR_INSTANCE=' + instance]
        compile_cmd = [toolchain["cmake"], "--build", str(build), "--config", "Release", "--target", "llama-server", "--parallel", str(parallel)]
        return {"kind": "native_windows_cuda", "build": str(build), "configure": configure, "compile": compile_cmd,
                "flags": flags, "parallel": parallel, "toolchain": toolchain}
    if not docker_image or not re.fullmatch(r"[A-Za-z0-9./_:-]+@sha256:[0-9a-f]{64}", docker_image):
        raise RuntimeError("missing_compiler_toolkit_or_pinned_cuda_devel_image")
    build = root / "build-sm89-linux"
    return {"kind": "isolated_linux_cuda", "build": str(build), "base_image": docker_image, "flags": flags,
            "parallel": parallel, "configure_args": ["-S", "/source", "-B", "/build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", *flags],
            "compile_args": ["--build", "/build", "--target", "llama-server", "--parallel", str(parallel)]}


def _docker_command(config, state, prepared, image, name, args):
    return [config["installed_tools"]["docker"], "run", "--name", name, "--label", "localbench.owner=localbench",
            "--label", "localbench.run=" + state.run["id"], "--label", "localbench.job=" + prepared["engine"],
            "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "2000:2000", "--cpus", "4", "--memory", "20g", "--shm-size", "1g", "--pids-limit", "512",
            "--tmpfs", "/tmp:rw,size=2g", "--mount", "type=bind,source=" + prepared["source"] + ",target=/source,readonly",
            "--mount", "type=bind,source=" + str(Path(prepared["root"]) / "build-sm89-linux") + ",target=/build",
            "--entrypoint", "cmake", image, *args]



def _build_cache(build, plan):
    path = Path(build) / "CMakeCache.txt"
    if not path.is_file():
        raise RuntimeError("source_build_missing_cmake_cache")
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"([^#/:=]+):[^=]+=(.*)$", line)
        if match:
            values[match[1]] = match[2]
    if values.get("CMAKE_CUDA_ARCHITECTURES") != "89" or values.get("GGML_CUDA", "").upper() not in ("ON", "TRUE", "YES", "1"):
        raise RuntimeError("source_build_effective_cuda_sm89_unverified")
    return {arg.split("=", 1)[0][2:]: values.get(arg.split("=", 1)[0][2:]) for arg in plan["flags"]}


def _copy_cuda_dlls(config, state, toolchain, binary):
    copied = []
    root = Path(toolchain["cuda_root"]) / "bin"
    candidates = set()
    for pattern in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll",
                    "nvrtc64_*.dll", "nvrtc-builtins64_*.dll"):
        candidates.update(root.glob(pattern))
    for source in sorted(candidates):
        target = binary.parent / source.name
        source_sha = file_hash(source)
        if target.exists():
            if file_hash(target) != source_sha:
                raise RuntimeError("conflicting_adjacent_cuda_dll")
        else:
            _disk_headroom(config, source.stat().st_size)
            _runtime_headroom(config, source.stat().st_size)
            temporary = target.with_name(target.name + ".copying")
            if temporary.is_symlink() or not temporary.resolve().is_relative_to(binary.parent.resolve()):
                raise RuntimeError("cuda_dll_copy_path_unsafe")
            with source.open("rb") as incoming, temporary.open("wb") as outgoing:
                while chunk := incoming.read(1024 * 1024):
                    state.check_budget()
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if file_hash(temporary) != source_sha:
                raise RuntimeError("cuda_dll_copy_hash_mismatch")
            temporary.replace(target)
        copied.append({"source": str(source), "path": str(target), "sha256": source_sha, "bytes": source.stat().st_size})
    return copied

def build_pinned_source(config, state, prepared, review_fingerprint, *, toolchain=None,
                        docker_image=None, runner=run_owned):
    """Build only after the controller has read the exact source review bundle."""
    repo, commit = _pin(config, prepared["engine"])
    key, setup = _setup(config, state, prepared["engine"], commit)
    source, root = Path(prepared["source"]), Path(prepared["root"])
    git = config['installed_tools'].get('git', 'git')
    proof = _run(config, state, setup, runner, prepared['engine'] + '-build-source-sha', [git, '-C', str(source), 'rev-parse', 'HEAD'])
    clean = _run(config, state, setup, runner, prepared['engine'] + '-build-source-clean', [git, '-C', str(source), 'status', '--porcelain=v1', '--untracked-files=all'])
    if proof['output'].strip() != commit or clean['output'].strip():
        raise RuntimeError('source_checkout_not_exact_clean_pin')
    expected_root = Path(config["paths"]["runtime_root"]) / (prepared["engine"] + "-" + commit)
    if root.resolve() != expected_root.resolve() or source.resolve() != (root / "source").resolve():
        raise ValueError("source build paths differ from private pinned checkout")
    hashes = _scripts(source)
    current = digest({"commit": commit, "scripts": hashes})
    if current != prepared["review_fingerprint"] or current != review_fingerprint:
        raise RuntimeError("source_script_review_required_or_changed")
    marker = json.loads((root / "localbench-owned.json").read_text(encoding="utf-8"))
    if marker != {"owner": "localbench", "run_id": state.run["id"], "engine": prepared["engine"], "commit": commit, "repo": repo}:
        raise RuntimeError("source_ownership_identity_mismatch")
    manifest_path = root / "engine-manifest.json"
    plan = plan_build(config, prepared, toolchain, docker_image)
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("commit") == commit and previous.get("review_fingerprint") == current and previous.get("build_plan") == plan and all(Path(f["path"]).resolve().is_relative_to(root.resolve()) and Path(f["path"]).is_file() and file_hash(f["path"]) == f["sha256"] for f in previous["files"]):
            return {"binary_path": previous["binary_path"], "engine_manifest": previous}
    _disk_headroom(config, 8 * 1024 ** 3)
    _runtime_headroom(config, 8 * 1024 ** 3)
    Path(plan["build"]).mkdir(parents=True, exist_ok=True)
    setup["phase"] = "build"
    state.control(key, setup)
    commands, image_id = [], None
    if plan["kind"] == "native_windows_cuda":
        if os.name == "nt" and ctypes.windll.shell32.IsUserAnAdmin():
            raise RuntimeError("administrator_runtime_forbidden")
        commands.append(_run(config, state, setup, runner, prepared["engine"] + "-configure", plan["configure"]))
        _build_cache(plan["build"], plan)
        commands.append(_run(config, state, setup, runner, prepared["engine"] + "-build", plan["compile"]))
        binaries = list(Path(plan["build"]).rglob("llama-server.exe"))
    else:
        docker = config["installed_tools"]["docker"]
        recipe = root / "builder-context"
        recipe.mkdir(exist_ok=True)
        (recipe / "Dockerfile").write_text("FROM " + plan["base_image"] + "\nRUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build g++ ca-certificates && rm -rf /var/lib/apt/lists/*\n", encoding="utf-8")
        (recipe / ".dockerignore").write_text("*\n!Dockerfile\n", encoding="utf-8")
        tag = "localbench-builder-" + prepared["engine"] + ":" + commit[:12]
        commands.append(_run(config, state, setup, runner, prepared["engine"] + "-builder-image", [docker, "build", "--label", "localbench.owner=localbench", "--label", "localbench.run=" + state.run["id"], "--tag", tag, str(recipe)]))
        inspected = _run(config, state, setup, runner, prepared["engine"] + "-builder-inspect", [docker, "image", "inspect", tag])
        image = json.loads(inspected["output"])[0]
        image_id = image["Id"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) or image["Config"].get("Labels", {}).get("localbench.run") != state.run["id"]:
            raise RuntimeError("builder_image_identity_unverified")
        cuda = next((value.split("=", 1)[1] for value in image["Config"].get("Env", []) if value.startswith("CUDA_VERSION=")), "")
        if not cuda.startswith("12.6."):
            raise RuntimeError("builder_image_cuda12_6_unverified")
        _runtime_headroom(config, image["Size"])
        for step, args in (("configure", plan["configure_args"]), ("build", plan["compile_args"])):
            name = "localbench-" + prepared["engine"] + "-" + state.run["id"].rsplit("-", 1)[-1] + "-" + step
            argv = _docker_command(config, state, prepared, image_id, name, args)
            try:
                commands.append(_run(config, state, setup, runner, prepared["engine"] + "-docker-" + step, argv))
            finally:
                info = runner(config, state, prepared["engine"] + "-owned-container-inspect", [docker, "container", "inspect", name], timeout=30, cleanup=True)
                if info['exit_code'] != 0:
                    raise RuntimeError('container_cleanup_inspect_failed')
                container = json.loads(info["output"])[0]
                labels = container["Config"].get("Labels", {})
                if labels.get("localbench.owner") == "localbench" and labels.get("localbench.run") == state.run["id"] and labels.get("localbench.job") == prepared["engine"] and re.fullmatch(r"[0-9a-f]{64}", container["Id"]):
                    removed = runner(config, state, prepared["engine"] + "-owned-container-remove", [docker, "container", "rm", "--force", container["Id"]], timeout=30, cleanup=True)
                    if removed['exit_code'] != 0:
                        raise RuntimeError('owned_container_cleanup_failed')
                else:
                    raise RuntimeError("container_cleanup_ownership_unverified")
        binaries = [p for p in Path(plan["build"]).rglob("llama-server") if p.is_file()]
    if len(binaries) != 1:
        raise RuntimeError("source_build_missing_or_ambiguous_server")
    binary = binaries[0]
    cuda_dlls = _copy_cuda_dlls(config, state, toolchain, binary) if plan["kind"] == "native_windows_cuda" else []
    effective_flags = _build_cache(plan["build"], plan)
    files = [{"path": str(p), "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sorted(binary.parent.iterdir()) if p.is_file() and (p == binary or p.suffix.lower() == ".dll" or ".so" in p.name)]
    cache = Path(plan["build"]) / "CMakeCache.txt"
    manifest = {"id": prepared["engine"], "commit": commit, "source": repo, "runtime_kind": plan["kind"],
                "binary_path": str(binary), "binary_sha256": file_hash(binary), "files": files,
                "review_fingerprint": current, "review_bundle": prepared["review_bundle"], "build_plan": plan,
                "commands": commands, "builder_image_id": image_id, "cmake_cache_sha256": file_hash(cache) if cache.is_file() else None,
                "effective_build_flags": effective_flags, "cuda_runtime_dlls": cuda_dlls,
                "setup_started": setup["started"], "setup_finished": state.clock(), "setup_deadline": setup["deadline"],
                "executed_model": False}
    atomic_json(manifest_path, manifest)
    state.entity("engine", prepared["engine"], manifest)
    setup["phase"] = "complete"
    state.control(key, setup)
    return {"binary_path": str(binary), "engine_manifest": manifest}
