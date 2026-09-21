"""Inspect selected PE imports and pin required existing OpenSSL DLLs privately.

No selected binary/DLL is loaded, imported, or launched. The only executable
invoked is the already installed MSVC dumpbin metadata tool. Global files are
read only; writes stay in the exact owned TurboQuant runtime and bench logs.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
from localbench.config import atomic_json, file_hash, load
from localbench.runtime_build import _environment
from localbench.state import State

PIN = "8ed935a092ee66ac87fdeb5cb8d5f383edb28f90"
OPENSSL_ROOT = Path("C:/Program Files/OpenSSL-Win64/bin")
ALLOWED = {"libcrypto-3-x64.dll", "libssl-3-x64.dll"}


def main():
    config = load(BENCH / "benchmark-spec.json")
    state = State(config)
    before_run = state.run
    root = Path(config["paths"]["runtime_root"]) / ("turboquant-cuda-" + PIN)
    source = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
    owned = json.loads((root / "localbench-owned.json").read_text(encoding="utf-8"))
    if owned.get("owner") != "localbench" or owned.get("run_id") != state.run["id"] or owned.get("engine") != "turboquant-cuda" or owned.get("commit") != PIN:
        raise RuntimeError("private runtime ownership mismatch")
    setup = state.get_control(source["setup_key"])
    if setup is None or setup["deadline"] != source["setup_deadline"]:
        raise RuntimeError("original setup deadline mismatch")
    deadline = setup["deadline"]
    review = state.get_control("source_review:turboquant-cuda")
    if not review or review.get("fingerprint") != source["review_fingerprint"]:
        raise RuntimeError("exact source review control required")

    def budget():
        state.check_budget()
        if time.time() >= deadline:
            raise RuntimeError("engine_setup_budget_exhausted")

    budget()
    path = root / "engine-manifest.json"
    previous_sha = file_hash(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("commit") != PIN or manifest.get("review_fingerprint") != source["review_fingerprint"]:
        raise RuntimeError("pinned completed build manifest required")
    binary = Path(manifest["binary_path"])
    target_dir = binary.parent.resolve()
    build_dir = (root / "build-sm89-windows").resolve()
    if Path(manifest["build_plan"]["build"]).resolve() != build_dir or binary.name != "llama-server.exe" or not binary.is_file() or not target_dir.is_relative_to(build_dir):
        raise RuntimeError("selected binary directory escapes exact private root")
    for item in manifest["files"]:
        budget()
        selected = Path(item["path"])
        if selected.parent.resolve() != target_dir or not selected.is_file() or file_hash(selected) != item["sha256"]:
            raise RuntimeError("completed build file identity changed")
    toolchain = manifest["build_plan"]["toolchain"]
    dumpbin = Path(toolchain["cl"]).with_name("dumpbin.exe")
    if not dumpbin.is_file() or not dumpbin.resolve().is_relative_to(Path(toolchain["visual_studio"]).resolve()):
        raise RuntimeError("existing pinned MSVC metadata tool unavailable")
    execution = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_log = Path(config["paths"]["logs"]) / ("setup-turboquant-PE-dependencies-" + execution + ".log")
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    evidence, required, checked = [], set(), set()

    def inspect(selected):
        budget()
        resolved = selected.resolve()
        if resolved in checked:
            return
        checked.add(resolved)
        argv = [str(dumpbin), "/nologo", "/dependents", str(selected)]
        result = subprocess.run(argv, cwd=str(root), env=_environment(), capture_output=True,
                                text=True, errors="replace", timeout=min(30, deadline - time.time()),
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = (result.stdout or "") + (result.stderr or "")
        if len(output) > 1024 * 1024 or result.returncode:
            raise RuntimeError("bounded PE dependency inspection failed")
        with raw_log.open("a", encoding="utf-8") as target:
            target.write("FILE " + str(selected) + "\n" + output + "\n")
        imports = sorted(set(re.findall(r"(?im)^\s+([A-Za-z0-9_.+-]+\.dll)\s*$", output)))
        evidence.append({"path": str(selected), "argv": argv, "sha256": file_hash(selected), "imports": imports})
        required.update(name.lower() for name in imports if name.lower() in ALLOWED)

    for selected in sorted(target_dir.iterdir()):
        if selected == binary or selected.suffix.lower() == ".dll":
            inspect(selected)
    # Inspect only explicitly permitted existing DLLs to close OpenSSL imports.
    inspected_ssl = set()
    while required - inspected_ssl:
        for name in sorted(required - inspected_ssl):
            inspect(OPENSSL_ROOT / name)
            inspected_ssl.add(name)
    added = []
    for name in sorted(required):
        budget()
        selected = OPENSSL_ROOT / name
        if not selected.is_file() or selected.parent.resolve() != OPENSSL_ROOT.resolve():
            raise RuntimeError("required existing OpenSSL file unavailable")
        target = target_dir / name
        sha = file_hash(selected)
        if target.exists() and (not target.is_file() or file_hash(target) != sha):
            raise RuntimeError("refuse runtime dependency collision")
        if not target.exists():
            shutil.copy2(selected, target)
        if file_hash(target) != sha:
            raise RuntimeError("runtime dependency copy hash mismatch")
        added.append({"source": str(selected), "path": str(target), "bytes": target.stat().st_size, "sha256": sha})
    budget()
    manifest["files"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": file_hash(p)}
                         for p in sorted(target_dir.iterdir()) if p.is_file() and (p == binary or p.suffix.lower() == ".dll")]
    manifest["additional_existing_dependencies"] = added
    manifest["pe_dependency_inspection"] = {"dumpbin": str(dumpbin), "dumpbin_sha256": file_hash(dumpbin),
                                            "log": str(raw_log), "files": evidence,
                                            "selected_binaries_executed": False}
    manifest["dependency_packaging_finished"] = time.time()
    manifest["setup_finished"] = time.time()
    budget()
    atomic_json(path, manifest)
    state.entity("engine", "turboquant-cuda", manifest)
    state.entity("runtime_catalog", "turboquant-cuda", {"engine": "turboquant-cuda", "kind": "native",
                                                         "binary_path": str(binary), "engine_manifest": manifest})
    receipt = {"kind": "existing_native_dependency_packaging", "engine": "turboquant-cuda", "commit": PIN,
               "original_setup_deadline": deadline, "engine_manifest_before_sha256": previous_sha,
               "engine_manifest_after_sha256": file_hash(path), "added_dependency_dlls": added,
               "import_evidence_log": str(raw_log), "selected_binaries_executed": False,
               "global_configuration_changed": False, "finished": time.time(),
               "benchmark_clock_unchanged": all(state.run[k] == before_run[k] for k in ("id", "started", "deadline"))}
    artifact = Path(config["paths"]["artifacts"]) / ("turboquant-runtime-dependencies-" + execution + ".json")
    atomic_json(artifact, receipt)
    state.entity("runtime_dependency_packaging", "turboquant-cuda", receipt)
    state.close()
    print(json.dumps({"receipt": str(artifact), "added_dependency_dlls": added,
                      "selected_binaries_executed": False}), flush=True)


if __name__ == "__main__":
    main()
