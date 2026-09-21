"""Read and fingerprint the reviewed pinned source; never import or execute it."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


PIN = "8ed935a092ee66ac87fdeb5cb8d5f383edb28f90"
EXPECTED = "faff27c7e661aef3502330cc4eaead9aa4e58e042a805ce3ba440263ff783f93"
ROOT = Path("F:/local-llm-benchmark-runtime") / ("turboquant-cuda-" + PIN)
BENCH = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "source"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((ROOT / "source-manifest.json").read_text(encoding="utf-8"))
if manifest["commit"] != PIN or Path(manifest["source"]).resolve() != SOURCE.resolve():
    raise SystemExit("source manifest pin/path mismatch")
scripts = {
    str(path.relative_to(SOURCE)).replace("\\", "/"): sha(path)
    for path in sorted(SOURCE.rglob("*"))
    if path.is_file()
    and ".git" not in path.relative_to(SOURCE).parts
    and (path.name == "CMakeLists.txt" or path.suffix.lower() in (".cmake", ".bat", ".ps1", ".sh"))
}
fingerprint = hashlib.sha256(
    json.dumps({"commit": PIN, "scripts": scripts}, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if scripts != manifest["script_hashes"] or fingerprint != EXPECTED or manifest["review_fingerprint"] != EXPECTED:
    raise SystemExit("reviewed script fingerprint mismatch")
reviewed_files = [
    "CMakeLists.txt", "ggml/CMakeLists.txt", "ggml/src/CMakeLists.txt",
    "ggml/src/ggml-cuda/CMakeLists.txt", "ggml/src/ggml-cpu/CMakeLists.txt",
    "ggml/cmake/common.cmake", "ggml/cmake/FindNCCL.cmake",
    "cmake/common.cmake", "cmake/git-vars.cmake", "cmake/build-info.cmake", "cmake/license.cmake",
    "common/CMakeLists.txt", "src/CMakeLists.txt", "vendor/CMakeLists.txt",
    "vendor/cpp-httplib/CMakeLists.txt", "vendor/hash/CMakeLists.txt",
    "vendor/miniaudio/CMakeLists.txt", "vendor/nlohmann/CMakeLists.txt",
    "vendor/sheredom/CMakeLists.txt", "vendor/stb/CMakeLists.txt",
    "tools/CMakeLists.txt", "tools/server/CMakeLists.txt", "tools/mtmd/CMakeLists.txt",
    "tools/ui/CMakeLists.txt", "scripts/ui-assets.cmake", "tools/ui/embed.cpp",
    "compile-cuda-rtx3090.bat", "README.md", "common/arg.cpp",
    "src/llama-context.cpp", "src/llama-kv-cache.cpp", "ggml/src/ggml-cuda/common.cuh",
    "ggml/src/ggml-cuda/fattn.cu",
    "ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbqp3_0-tbq3_0.cu",
]
receipt = {
    "reviewer": "/root/verify_fixtures", "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    "engine": "turboquant-cuda", "commit": PIN, "source": str(SOURCE),
    "review_fingerprint": fingerprint, "scripts_verified": len(scripts),
    "bundle_sha256": sha(Path(manifest["review_bundle"])),
    "reviewed_files_sha256": {name: sha(SOURCE / name) for name in reviewed_files},
    "review_scope": "Reachable native Windows server CMake/custom build scripts and selected cache/architecture dispatch sections; not a full CUDA kernel audit.",
    "verdict": "conditional_build_review_complete",
    "required_cmake_controls": {
        "GGML_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": "89", "GGML_NATIVE": "OFF",
        "GGML_AVX2": "ON", "GGML_AVX512": "OFF", "GGML_CCACHE": "OFF",
        "GGML_CUDA_FA": "ON", "LLAMA_BUILD_UI": "OFF", "LLAMA_USE_PREBUILT_UI": "OFF",
        "LLAMA_LLGUIDANCE": "OFF", "LLAMA_BUILD_BORINGSSL": "OFF", "LLAMA_BUILD_LIBRESSL": "OFF",
        "GGML_CPU_KLEIDIAI": "OFF", "GGML_CUDA_CUB_3DOT2": "OFF",
    },
    "recommended_cmake_controls": {"LLAMA_BUILD_EXAMPLES": "OFF", "LLAMA_BUILD_APP": "OFF", "GGML_CUDA_NCCL": "OFF", "LLAMA_OPENSSL": "OFF"},
    "network": "UI npm/HF branches enabled by defaults must be disabled; opt-in dependency fetch branches stay OFF. Git/compiler version probes do not fetch.",
    "write_scope": "Generated package/license/build-info/UI output paths are inside the selected build directory. No install target is permitted. Disable compiler caches; compiler TEMP/TMP should be task-local if selected-write-only isolation is required.",
    "source_ui_dist_present": (SOURCE / "tools/ui/dist/index.html").exists(),
    "accepted_cache_cli": {"k": "tbqp3", "v": "tbq3", "internal_enum_names_rejected": ["tbqp3_0", "tbq3_0"]},
    "installed_qwen35_head_dimension": 256,
    "source_compatibility": "256 K/V resolves to TBQP3_0/TBQ3_0 and has CUDA Flash Attention vector dispatch/instantiations. Explicit architecture 89 skips sm86 batch defaults and CUDA12.8+ architecture branches.",
    "remaining_gates": ["Actual build and effective CMake cache", "Exact built binary help", "Observed all-layer/CUDA KV offload and effective cache", "Exact tokenization/usage", "Coding/tool/64K/stability gates"],
    "executed_build": False, "executed_installer": False, "executed_gpu_or_model": False,
    "source_review_control_modified": False,
}
target = BENCH / "artifacts" / ("source-review-receipt-turboquant-cuda-" + PIN + ".json")
if target.exists():
    raise SystemExit("review receipt already exists; preserve it")
target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"receipt": str(target), "scripts_verified": len(scripts), "review_fingerprint": fingerprint, "bundle_sha256": receipt["bundle_sha256"], "receipt_sha256": sha(target)}))
