from pathlib import Path
from unittest.mock import patch
import io
import json
import os
import tempfile
import unittest

from localbench.config import atomic_json
from localbench.runtime_build import (build_pinned_source, detect_windows_toolchain,
                                      plan_build, prepare_pinned_source, run_owned)
from localbench.state import State

TURBO = "8ed935a092ee66ac87fdeb5cb8d5f383edb28f90"
IK = "2ae132fa601ea06818ed3584f50f7eb4f72d4967"


class FakeRunner:
    def __init__(self, fixture):
        self.fixture = fixture
        self.commands = []
        self.fetched = False
        self.dirty = False
        self.failure = None
        self.commit = TURBO
        self.container_foreign = False

    def __call__(self, config, state, label, argv, **kwargs):
        self.commands.append((label, argv, kwargs))
        if self.failure and label.endswith(self.failure):
            self.fixture.now += 7201
            raise RuntimeError("owned_build_command_timeout")
        output = ""
        if label.endswith("-git-init"):
            source = Path(argv[-1])
            (source / ".git").mkdir(parents=True)
            (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.21)\nproject(fake LANGUAGES CXX CUDA)\n", encoding="utf-8")
            (source / "build_cuda.bat").write_text("rem old unexecuted build script sm86\n", encoding="utf-8")
            (source / "model.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
        elif label.endswith("-git-objects"):
            output = self.commit if self.fetched else ""
        elif label.endswith("-git-fetch"):
            self.fetched = True
        elif label.endswith("-git-sha") or label.endswith("-build-source-sha"):
            output = self.commit + "\n"
        elif label.endswith("-git-clean") or label.endswith("-build-source-clean"):
            output = " M model.cpp\n" if self.dirty else ""
        elif label.endswith("-git-remote-check"):
            output = "https://github.com/jarkevithwlad/llama.cpp-turboquant-cuda\n" if self.commit == TURBO else "https://github.com/ikawrakow/ik_llama.cpp\n"
        elif label.endswith("-builder-inspect"):
            output = json.dumps([{"Id": "sha256:" + "b" * 64, "Size": 1000,
                                  "Config": {"Labels": {"localbench.run": state.run["id"]}, "Env": ["CUDA_VERSION=12.6.3"]}}])
        elif label.endswith("-owned-container-inspect"):
            output = json.dumps([{"Id": "c" * 64, "Config": {"Labels": {
                "localbench.owner": "foreign" if self.container_foreign else "localbench",
                "localbench.run": state.run["id"], "localbench.job": "turboquant-cuda"}}}])
        elif label.endswith("-configure"):
            root = self.fixture.runtime / ("turboquant-cuda-" + TURBO if self.commit == TURBO else "ik-llama-" + IK)
            build = root / ("build-sm89-linux" if "-docker-" in label else "build-sm89-windows")
            build.mkdir(parents=True, exist_ok=True)
            (build / "CMakeCache.txt").write_text("CMAKE_CUDA_ARCHITECTURES:STRING=89\nGGML_CUDA:BOOL=ON\n", encoding="utf-8")
        elif label.endswith("-build") and not label.endswith("-builder-image"):
            root = self.fixture.runtime / ("turboquant-cuda-" + TURBO if self.commit == TURBO else "ik-llama-" + IK)
            build = root / ("build-sm89-linux" if "-docker-" in label else "build-sm89-windows")
            target = build / "bin" / ("llama-server" if "-docker-" in label else "llama-server.exe")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake compiled binary; never executed")
        return {"exit_code": 0, "output": output, "log": "mock-" + label + ".log", "argv": argv, "seconds": 0}


class RuntimeBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runtime = self.root / "runtime"
        self.now = 1000
        self.config = {"_spec_hash": "temporary-source-build-test",
                       "paths": {"state": str(self.root / "state"), "artifacts": str(self.root / "artifacts"),
                                 "runtime_root": str(self.runtime), "logs": str(self.root / "logs")},
                       "installed_tools": {"git": "mock-git", "docker": "mock-docker"},
                       "limits": {"benchmark_elapsed_seconds": 604800, "new_model_weight_bytes": 300000000000,
                                  "minimum_free_c_bytes": 50, "minimum_free_f_bytes": 64,
                                  "runtime_and_build_soft_cap_bytes": 100 * 1024 ** 3},
                       "runtime_candidates": [{"id": "turboquant-cuda", "probe_minutes": 120}, {"id": "ik-llama", "probe_minutes": 120}],
                       "acquisition_disk_roots": {"C": str(self.root), "F": str(self.root)}}
        self.state = State(self.config, clock=lambda: self.now)
        self.state.control("harness_verified", True)
        atomic_json(self.root / "artifacts" / "engine-discovery.json", {"engines": [
            {"id": "turboquant-cuda", "pin": {"commit": TURBO}}, {"id": "ik-llama", "pin": {"commit": IK}}]})
        self.runner = FakeRunner(self)
        self.toolchain = {"available": True, "cmake": "mock-cmake", "cuda_version": "12.6",
                          "cuda_root": "mock-CUDA12.6", "generator": "Visual Studio 17 2022"}
        self.image = "nvidia/cuda:12.6.3-devel-ubuntu22.04@sha256:" + "d" * 64

    def tearDown(self):
        self.state.close()
        self.tmp.cleanup()

    def prepare(self):
        return prepare_pinned_source(self.config, self.state, "turboquant-cuda", runner=self.runner)

    def test_native_plan_is_sm89_avx2_bounded_and_uses_cuda12_6(self):
        prepared = self.prepare()
        plan = plan_build(self.config, prepared, self.toolchain)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=89", plan["configure"])
        self.assertIn("-DGGML_AVX512=OFF", plan["configure"])
        self.assertIn("-DGGML_NATIVE=OFF", plan["configure"])
        self.assertIn("-DGGML_CUDA_FA_ALL_QUANTS=OFF", plan["configure"])
        for flag in ('-DLLAMA_BUILD_UI=OFF','-DLLAMA_USE_PREBUILT_UI=OFF','-DGGML_CCACHE=OFF'):
            self.assertIn(flag,plan['configure'])
        self.assertEqual(plan["compile"][-1], "2")
        self.assertFalse(any("86" in arg for arg in plan["configure"]))
        self.assertFalse(any(arg.endswith("build_cuda.bat") for arg in plan["configure"]))

    def test_ik_plan_uses_pinned_fork_specific_cuda_option(self):
        self.runner.commit = IK
        prepared = prepare_pinned_source(self.config, self.state, "ik-llama", runner=self.runner)
        plan = plan_build(self.config, prepared, self.toolchain)
        self.assertIn("-DGGML_IQK_FA_ALL_QUANTS=ON", plan["flags"])
        self.assertNotIn("-DGGML_CUDA_FA_ALL_QUANTS=ON", plan["flags"])

    def test_source_checkout_is_exact_detached_and_hooks_disabled(self):
        prepared = self.prepare()
        fetch = next(argv for label, argv, _ in self.runner.commands if label.endswith("-git-fetch"))
        self.assertEqual(fetch[-1], TURBO)
        self.assertIn("--depth", fetch)
        self.assertIn("credential.helper=", fetch)
        checkout = next(argv for label, argv, _ in self.runner.commands if label.endswith("-git-checkout"))
        self.assertIn("--detach", checkout)
        self.assertIn("core.fsmonitor=false", checkout)
        self.assertIn("submodule.recurse=false", checkout)
        bundle = Path(prepared["review_bundle"]).read_text(encoding="utf-8")
        self.assertIn("CMakeLists.txt SHA256", bundle)
        self.assertIn("old unexecuted build script sm86", bundle)

    def test_source_resume_preserves_deadline_and_avoids_repeat_fetch(self):
        prepared = self.prepare()
        self.runner.commands.clear()
        self.now += 30
        resumed = self.prepare()
        self.assertEqual(resumed["setup_deadline"], prepared["setup_deadline"])
        self.assertFalse(any(label.endswith("-git-fetch") for label, _, _ in self.runner.commands))
        self.now = prepared["setup_deadline"]
        self.runner.commands.clear()
        with self.assertRaisesRegex(RuntimeError, "setup_budget_exhausted"):
            self.prepare()
        self.assertEqual(self.runner.commands, [])

    def test_unowned_directory_never_modified(self):
        root = self.runtime / ("turboquant-cuda-" + TURBO)
        root.mkdir(parents=True)
        personal = root / "personal.txt"
        personal.write_text("leave intact", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "unowned_source_directory"):
            self.prepare()
        self.assertEqual(personal.read_text(encoding="utf-8"), "leave intact")
        self.assertEqual(self.runner.commands, [])

    def test_changed_pin_and_dirty_source_fail_before_build(self):
        prepared = self.prepare()
        self.runner.dirty = True
        with self.assertRaisesRegex(RuntimeError, "exact_clean_pin"):
            build_pinned_source(self.config, self.state, prepared, prepared["review_fingerprint"],
                                toolchain=self.toolchain, runner=self.runner)
        self.assertFalse(any(label.endswith("-configure") for label, _, _ in self.runner.commands))

    def test_review_fingerprint_required_and_source_edits_invalidate_it(self):
        prepared = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "script_review_required"):
            build_pinned_source(self.config, self.state, prepared, "unreviewed", toolchain=self.toolchain, runner=self.runner)
        Path(prepared["source"], "CMakeLists.txt").write_text("changed build script", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "script_review_required"):
            build_pinned_source(self.config, self.state, prepared, prepared["review_fingerprint"], toolchain=self.toolchain, runner=self.runner)

    def test_native_mock_build_manifest_pins_binary_and_resumes(self):
        prepared = self.prepare()
        admin = patch("localbench.runtime_build.ctypes.windll.shell32.IsUserAnAdmin", return_value=0) if os.name == "nt" else patch("localbench.runtime_build.time.time", return_value=0)
        with admin:
            result = build_pinned_source(self.config, self.state, prepared, prepared["review_fingerprint"],
                                         toolchain=self.toolchain, runner=self.runner)
            self.runner.commands.clear()
            resumed = build_pinned_source(self.config, self.state, prepared, prepared["review_fingerprint"],
                                          toolchain=self.toolchain, runner=self.runner)
        self.assertEqual(result, resumed)
        self.assertEqual(result["engine_manifest"]["commit"], TURBO)
        self.assertFalse(result["engine_manifest"]["executed_model"])
        self.assertTrue(Path(result["binary_path"]).is_file())
        self.assertFalse(any(label.endswith("-build") for label, _, _ in self.runner.commands))

    def test_docker_build_is_nonroot_without_gpu_socket_hostipc_or_network(self):
        prepared = self.prepare()
        result = build_pinned_source(self.config, self.state, prepared, prepared["review_fingerprint"],
                                     docker_image=self.image, runner=self.runner)
        runs = [argv for label, argv, _ in self.runner.commands if "-docker-" in label]
        self.assertEqual(len(runs), 2)
        for argv in runs:
            self.assertIn("2000:2000", argv)
            self.assertIn("no-new-privileges", argv)
            self.assertIn("none", argv)
            self.assertIn("ALL", argv)
            self.assertIn("20g", argv)
            self.assertNotIn("--privileged", argv)
            self.assertNotIn("--gpus", argv)
            self.assertFalse(any("docker.sock" in arg or "ipc=host" in arg for arg in argv))
            self.assertIn("sha256:" + "b" * 64, argv)
        self.assertEqual(result["engine_manifest"]["runtime_kind"], "isolated_linux_cuda")

    def test_docker_owned_cleanup_still_runs_after_setup_timeout(self):
        prepared = self.prepare()
        self.runner.failure = "-docker-configure"
        with self.assertRaisesRegex(RuntimeError, "owned_build_command_timeout"):
            build_pinned_source(self.config, self.state, prepared, prepared["review_fingerprint"],
                                docker_image=self.image, runner=self.runner)
        cleanup = [(label, argv, kwargs) for label, argv, kwargs in self.runner.commands if "-owned-container-" in label]
        self.assertEqual(len(cleanup), 2)
        self.assertTrue(all(kwargs["cleanup"] for _, _, kwargs in cleanup))
        self.assertEqual(cleanup[1][1][-1], "c" * 64)

    def test_docker_foreign_labels_never_terminated(self):
        prepared = self.prepare()
        self.runner.container_foreign = True
        with self.assertRaisesRegex(RuntimeError, "cleanup_ownership_unverified"):
            build_pinned_source(self.config, self.state, prepared, prepared["review_fingerprint"],
                                docker_image=self.image, runner=self.runner)
        self.assertFalse(any(label.endswith("-owned-container-remove") for label, _, _ in self.runner.commands))

    def test_docker_mutable_image_and_excessive_parallelism_rejected(self):
        prepared = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "pinned_cuda_devel_image"):
            plan_build(self.config, prepared, docker_image="nvidia/cuda:latest")
        self.config["build_parallel_jobs"] = 32
        with self.assertRaisesRegex(ValueError, "parallelism"):
            plan_build(self.config, prepared, self.toolchain)

    def test_existing_vswhere_and_cuda_are_detected_without_install(self):
        tools = self.root / "existing-tools"
        tools.mkdir()
        vswhere, cmake, nvcc = [tools / name for name in ("vswhere.exe", "cmake.exe", "nvcc.exe")]
        for item in (vswhere, cmake, nvcc):
            item.write_bytes(b"mock")
        vsroot = tools / "VS2022"
        cl = vsroot / "VC" / "Tools" / "MSVC" / "14.38" / "bin" / "Hostx64" / "x64" / "cl.exe"
        cl.parent.mkdir(parents=True)
        cl.write_bytes(b"mock")
        self.config["installed_tools"].update(vswhere=str(vswhere), cmake=str(cmake), nvcc=str(nvcc))
        calls = []
        def probe(label, argv):
            calls.append(argv)
            output = json.dumps([{"installationPath": str(vsroot), "installationVersion": "17.8"}]) if label == "vswhere" else "Cuda compilation tools, release 12.6, V12.6.77" if label == "nvcc" else "existing tool version"
            return {"exit_code": 0, "output": output, "argv": argv}
        result = detect_windows_toolchain(self.config, runner=probe)
        self.assertTrue(result["available"])
        self.assertEqual(result["cl"], str(cl))
        self.assertIn("[17.0,18.0)", calls[0])
        self.assertEqual(len(calls), 4)

    @unittest.skipUnless(os.name == "nt", "Windows owned job planner")
    def test_owned_runner_timeout_cleans_tree_and_redacts_logs(self):
        class Process:
            pid = 123
            returncode = None
            stdout = io.BytesIO(b"Authorization: Bearer secret\nhttps://cdn.hf.co/file?signature=private\n")
            def poll(self): return self.returncode
            def wait(self, timeout): return self.returncode
        process = Process()
        class Job:
            attached = False
            closed = False
            def attach_resume(self, p): self.attached = p is process
            def close(self):
                self.closed = True
                process.returncode = -9
        job = Job()
        elapsed = [0]
        captured = {}
        def create(argv, **kwargs):
            captured.update(kwargs)
            return process
        def sleep(seconds): elapsed[0] += 1
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "environment-secret", "HTTPS_PROXY": "https://user:secret@proxy", "GIT_SSH_COMMAND": "arbitrary"}):
            with self.assertRaisesRegex(RuntimeError, "owned_build_command_timeout"):
                run_owned(self.config, self.state, "mock-timeout", ["mock-build"], popen=create,
                          monotonic=lambda: elapsed[0], sleeper=sleep, timeout=2, job_factory=lambda: job)
        self.assertTrue(job.attached and job.closed)
        self.assertTrue(captured["creationflags"] & 4)
        self.assertNotIn("LM_BENCH_TOKEN", captured["env"])
        self.assertNotIn("HTTPS_PROXY", captured["env"])
        self.assertNotIn("GIT_SSH_COMMAND", captured["env"])
        saved = next((self.root / "logs").glob("*.log")).read_text(encoding="utf-8")
        self.assertNotIn("secret", saved)
        self.assertNotIn("signature=private", saved)



    def test_effective_wrong_cuda_architecture_stops_before_compile(self):
        prepared = self.prepare()
        def wrong_cache(config, state, label, argv, **kwargs):
            result = self.runner(config, state, label, argv, **kwargs)
            if label.endswith("-configure"):
                Path(prepared["root"], "build-sm89-windows", "CMakeCache.txt").write_text("CMAKE_CUDA_ARCHITECTURES:STRING=86\nGGML_CUDA:BOOL=ON\n", encoding="utf-8")
            return result
        admin = patch("localbench.runtime_build.ctypes.windll.shell32.IsUserAnAdmin", return_value=0) if os.name == "nt" else patch("localbench.runtime_build.time.time", return_value=0)
        with admin:
            with self.assertRaisesRegex(RuntimeError, "effective_cuda_sm89_unverified"):
                build_pinned_source(self.config, self.state, prepared, prepared["review_fingerprint"],
                                    toolchain=self.toolchain, runner=wrong_cache)
        self.assertFalse(any(label.endswith("-build") for label, _, _ in self.runner.commands))

    def test_adjacent_installed_cuda_dlls_are_copied_and_hashed(self):
        prepared = self.prepare()
        cuda = self.root / "fake-cuda"
        (cuda / "bin").mkdir(parents=True)
        (cuda / "bin" / "cudart64_12.dll").write_bytes(b"fake local CUDA DLL")
        toolchain = dict(self.toolchain, cuda_root=str(cuda))
        admin = patch("localbench.runtime_build.ctypes.windll.shell32.IsUserAnAdmin", return_value=0) if os.name == "nt" else patch("localbench.runtime_build.time.time", return_value=0)
        with admin:
            result = build_pinned_source(self.config, self.state, prepared, prepared["review_fingerprint"],
                                         toolchain=toolchain, runner=self.runner)
        adjacent = Path(result["binary_path"]).parent / "cudart64_12.dll"
        self.assertEqual(adjacent.read_bytes(), b"fake local CUDA DLL")
        self.assertEqual(len(result["engine_manifest"]["cuda_runtime_dlls"]), 1)
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], 0)

if __name__ == "__main__":
    unittest.main()
