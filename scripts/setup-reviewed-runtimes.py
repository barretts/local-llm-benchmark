"""Execute reviewed CPU-only setup with existing deadlines and private writes.

This helper does not acquire the GPU lease, launch a runtime, import an engine,
load a model, convert weights, alter installed tools, or reset benchmark state.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
from localbench.config import atomic_json, load, file_hash
from localbench.state import State
from localbench.runtime_build import build_pinned_source, detect_windows_toolchain, plan_build, run_owned
from localbench.tabby_setup import install_tabby

parser = argparse.ArgumentParser()
parser.add_argument("--only", choices=("both", "turboquant-cuda", "exllamav3-tabby"), default="both")
selection = parser.parse_args().only

class SkipEngine(Exception):
    pass

config = load(BENCH / "benchmark-spec.json")
config["build_parallel_jobs"] = 2
state = State(config)
original_run = state.run
original_weights = state.db.execute("SELECT COALESCE(SUM(bytes),0) FROM weights").fetchone()[0]
execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
record = {"id": execution_id, "kind": "cpu_only_runtime_setup", "pid": os.getpid(),
          "started": time.time(), "build_jobs": 2, "original_run": original_run,
          "original_weight_bytes": original_weights, "engines": {}, "gpu_lease": False,
          "executed_model": False, "executed_runtime_probe": False}
receipt = Path(config["paths"]["artifacts"]) / ("cpu-runtime-setup-" + execution_id + ".json")


def persist():
    atomic_json(receipt, record)
    state.entity("runtime_setup_execution", execution_id, record)


def event(message):
    print(message, flush=True)
    persist()


def prepared_for(directory, engine):
    root = Path(config["paths"]["runtime_root"]) / directory
    manifest = root / "source-manifest.json"
    prepared = json.loads(manifest.read_text(encoding="utf-8"))
    if Path(prepared["root"]).resolve() != root.resolve() or not root.resolve().is_relative_to(Path(config["paths"]["runtime_root"]).resolve()):
        raise RuntimeError("setup path does not match private runtime root")
    review = state.get_control("source_review:" + engine)
    if not review or review.get("fingerprint") != prepared["review_fingerprint"]:
        raise RuntimeError("existing exact source-review receipt required")
    setup = state.get_control(prepared["setup_key"])
    if setup is None or setup["deadline"] != prepared["setup_deadline"]:
        raise RuntimeError("existing setup deadline mismatch; do not reset")
    if time.time() >= setup["deadline"]:
        raise RuntimeError("engine_setup_budget_exhausted")
    return prepared, root


def runner_for(root):
    for name in ("tmp", "pip-cache", "triton-cache", "torch-cache", "hf-cache"):
        (root / name).mkdir(exist_ok=True)
    programdata = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
    if not programdata.is_absolute() or not programdata.is_dir():
        raise RuntimeError("existing Windows CommonAppData path unavailable")

    def spawn(argv, **kwargs):
        env = dict(kwargs["env"])
        for key in ("PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432"):
            if key in os.environ:
                env[key] = os.environ[key]
        env.update({"TEMP": str(root / "tmp"), "TMP": str(root / "tmp"),
                    "PROGRAMDATA": str(programdata), "ALLUSERSPROFILE": str(programdata),
                    "SYSTEMDRIVE": os.environ.get("SYSTEMDRIVE") or Path(env.get("SYSTEMROOT", "C:/Windows")).drive,
                    "MSBUILDDISABLENODEREUSE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "PIP_CONFIG_FILE": os.devnull,
                    "PIP_CACHE_DIR": str(root / "pip-cache"), "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "PIP_NO_INPUT": "1", "TRITON_CACHE_DIR": str(root / "triton-cache"),
                    "TORCH_HOME": str(root / "torch-cache"), "HF_HOME": str(root / "hf-cache"),
                    "HUGGINGFACE_HUB_CACHE": str(root / "hf-cache"), "CUDA_VISIBLE_DEVICES": ""})
        kwargs["env"] = env
        return subprocess.Popen(argv, **kwargs)

    def runner(cfg, active_state, label, argv, **kwargs):
        kwargs.setdefault("cwd", str(root))
        return run_owned(cfg, active_state, label, argv, popen=spawn, **kwargs)

    return runner


persist()
try:
    if selection == "exllamav3-tabby":
        raise SkipEngine
    prepared, root = prepared_for("turboquant-cuda-8ed935a092ee66ac87fdeb5cb8d5f383edb28f90", "turboquant-cuda")
    runner = runner_for(root)
    toolchain = detect_windows_toolchain(config)
    if not toolchain.get("available"):
        raise RuntimeError("existing_supported_windows_toolchain_unavailable")
    plan = plan_build(config, prepared, toolchain=toolchain)
    required = {"GGML_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": "89", "GGML_NATIVE": "OFF",
                "GGML_AVX2": "ON", "GGML_AVX512": "OFF", "GGML_CCACHE": "OFF", "GGML_CUDA_FA": "ON",
                "GGML_CUDA_FA_ALL_QUANTS": "OFF", "LLAMA_BUILD_UI": "OFF", "LLAMA_USE_PREBUILT_UI": "OFF",
                "LLAMA_LLGUIDANCE": "OFF", "LLAMA_BUILD_BORINGSSL": "OFF", "LLAMA_BUILD_LIBRESSL": "OFF",
                "GGML_CPU_KLEIDIAI": "OFF", "GGML_CUDA_CUB_3DOT2": "OFF"}
    observed = {argument[2:].split("=", 1)[0]: argument.split("=", 1)[1] for argument in plan["flags"]}
    if plan["kind"] != "native_windows_cuda" or plan["parallel"] != 2 or any(observed.get(key) != value for key, value in required.items()):
        raise RuntimeError("reviewed TurboQuant build controls not enforced")
    record["engines"]["turboquant-cuda"] = {"started": time.time(), "status": "building", "plan": plan,
        "setup_deadline": prepared["setup_deadline"], "review_fingerprint": prepared["review_fingerprint"]}
    event("TurboQuant: reviewed native CPU build started; sm89, two jobs, no UI/network dependency build branches.")
    built = build_pinned_source(config, state, prepared, prepared["review_fingerprint"], toolchain=toolchain, runner=runner)
    effective = built["engine_manifest"]["effective_build_flags"]
    if any(str(effective.get(key)).upper() != value for key, value in required.items()):
        raise RuntimeError("actual TurboQuant CMake cache differs from reviewed controls")
    catalog = {"engine": "turboquant-cuda", "kind": "native", **built}
    state.entity("runtime_catalog", "turboquant-cuda", catalog)
    record["engines"]["turboquant-cuda"].update({"status": "complete", "finished": time.time(), "binary_path": built["binary_path"], "binary_sha256": file_hash(built["binary_path"]), "manifest": str(root / "engine-manifest.json")})
    event("TurboQuant: build complete and pinned catalog saved; runtime not launched.")
except SkipEngine:
    pass
except Exception as exc:
    record["engines"].setdefault("turboquant-cuda", {"started": time.time()}).update({"status": "failed", "finished": time.time(), "reason": str(exc)})
    event("TurboQuant: setup stopped: " + str(exc))

try:
    if selection == "turboquant-cuda":
        raise SkipEngine
    prepared, root = prepared_for("tabby-53da7919d4e45c63f4acbcbbc00cbe0f60a1ce65", "exllamav3-tabby")
    runner = runner_for(root)
    record["engines"]["exllamav3-tabby"] = {"started": time.time(), "status": "installing", "setup_deadline": prepared["setup_deadline"], "review_fingerprint": prepared["review_fingerprint"]}
    event("Tabby: reviewed public-wheel-only setup started in private F: venv.")
    built = install_tabby(config, state, prepared, prepared["review_fingerprint"], runner=runner)
    state.entity("runtime_catalog", "exllamav3-tabby", {"engine": "exllamav3-tabby", "kind": "tabby", **built})
    record["engines"]["exllamav3-tabby"].update({"status": "complete", "finished": time.time(), "manifest": built["manifest_path"], "dependency_lock_sha256": built["dependency_lock_sha256"], "package_count": len(built["packages"])})
    event("Tabby: private pinned package setup complete; engine imports/runtime launch not performed.")
except SkipEngine:
    pass
except Exception as exc:
    record["engines"].setdefault("exllamav3-tabby", {"started": time.time()}).update({"status": "failed", "finished": time.time(), "reason": str(exc)})
    event("Tabby: setup stopped: " + str(exc))

current = state.run
record.update({"finished": time.time(), "final_weight_bytes": state.db.execute("SELECT COALESCE(SUM(bytes),0) FROM weights").fetchone()[0],
               "benchmark_clock_unchanged": all(current[key] == original_run[key] for key in ("id", "started", "deadline"))})
persist()
state.close()
print(json.dumps({"receipt": str(receipt), "engines": {name: value["status"] for name, value in record["engines"].items()}, "benchmark_clock_unchanged": record["benchmark_clock_unchanged"], "final_weight_bytes": record["final_weight_bytes"]}), flush=True)
raise SystemExit(0 if all(value["status"] == "complete" for value in record["engines"].values()) and record["benchmark_clock_unchanged"] else 1)
