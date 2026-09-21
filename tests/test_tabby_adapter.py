from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import tempfile
import unittest

from localbench.adapters.tabby import TabbyAdapter, owned_config, validate_exl3
from localbench.config import atomic_json, file_hash
from localbench.state import State
from localbench.tabby_setup import (TABBY_COMMIT, EXL3_COMMIT, WHEEL_SHA, WHEEL_NAME, PIP_HELPER,
                                   _freeze_report, install_tabby, plan_tabby_setup, prepare_tabby_source)


class FakeSetupRunner:
    def __init__(self, fixture):
        self.fixture = fixture
        self.calls = []
        self.fetched = False
        self.packages = {"exllamav3": "1.5.0+cu128.torch2.9.0", "torch": "2.9.0+cu128", "triton-windows": "3.5.1"}

    def __call__(self, config, state, label, argv, **kwargs):
        self.calls.append((label, argv, kwargs))
        output = ""
        if label.endswith("-git-init"):
            source = Path(argv[-1])
            (source / ".git").mkdir(parents=True)
            (source / "main.py").write_text("# dummy source, never launched\n", encoding="utf-8")
            (source / "pyproject.toml").write_text('[project]\nname="tabbyAPI"\nversion="0.0.1"\ndependencies=["fastapi"]\n[project.optional-dependencies]\ncu12=["torch==2.9.0+cu128", "triton-windows", "exllamav3 @ https://github.com/example/pinned.whl"]\n', encoding="utf-8")
        elif label.endswith("-git-fetch"):
            self.fetched = True
        elif label.endswith("-git-objects"):
            output = TABBY_COMMIT if self.fetched else ""
        elif label.endswith("-git-sha"):
            output = TABBY_COMMIT
        elif label.endswith("-git-remote-check"):
            output = "https://github.com/theroyallab/tabbyAPI"
        elif label.endswith("-venv"):
            python = Path(argv[-1]) / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"dummy interpreter, never executed")
        elif label.endswith("-resolve"):
            root = Path(plan_tabby_setup(config)["root"])
            wheel = root / "packages" / WHEEL_NAME
            output = json.dumps({"install": [
                {"metadata": {"name": name, "version": version}, "download_info": {
                    "url": wheel.as_uri() if name == "exllamav3" else "https://files.pythonhosted.org/packages/" + name + ".whl",
                    "archive_info": {"hashes": {"sha256": WHEEL_SHA if name == "exllamav3" else "a" * 64}}}}
                for name, version in self.packages.items()]})
        elif label.endswith("-versions"):
            output = json.dumps(self.packages)
        return {"exit_code": 0, "output": output, "log": "mock-" + label + ".log", "argv": argv, "seconds": 0}


class FakeAPI:
    def __init__(self, model):
        self.model = model
        self.requests = []
        self.parameters = {"max_seq_len": 65536, "cache_size": 65536, "cache_mode": "FP16", "max_batch_size": 1, "chunk_size": 2048}
        self.props = {"model_path": model["path"], "total_slots": 1, "default_generation_settings": {"n_ctx": 65536}}
        self.fields = {"messages", "tools", "top_k", "min_p", "add_bos_token", "repetition_penalty", "template_vars", "reasoning_budget_tokens"}

    def json(self, path, body=None):
        self.requests.append((path, body))
        if path == "/props": return self.props
        if path == "/v1/models": return {"data": [{"id": Path(self.model["path"]).name, "parameters": self.parameters}]}
        if path == "/openapi.json": return {"components": {"schemas": {"ChatCompletionRequest": {"properties": {key: {} for key in self.fields}}}}}
        if path == "/apply-template": return {"prompt": "fixture tool alpha <think>"}
        if path == "/v1/token/encode": return {"tokens": [11, 22, 33], "length": 3}
        if path == "/health": return {"status": "healthy", "issues": []}
        raise AssertionError(path)

    def measure(self, body, **kwargs):
        self.measured = body
        return {"valid_stream": True, "metrics": {"prompt_tokens": 3}, "calls": []}


class TabbyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.now = 1000
        self.config = {"_spec_hash": "temporary-tabby-test", "paths": {"state": str(self.root / "state"),
                       "artifacts": str(self.root / "artifacts"), "runtime_root": str(self.root / "runtime"), "logs": str(self.root / "logs")},
                       "installed_tools": {"python": "mock-existing-python313", "python_observed_version": "3.13.15", "git": "mock-git"},
                       "limits": {"benchmark_elapsed_seconds": 604800, "new_model_weight_bytes": 300000000000,
                                  "minimum_free_c_bytes": 50, "minimum_free_f_bytes": 64, "runtime_and_build_soft_cap_bytes": 100 * 1024 ** 3,
                                  "server_start_timeout_seconds": 300, "per_64k_request_timeout_seconds": 900},
                       "runtime_candidates": [{"id": "exllamav3-tabby", "probe_minutes": 120}], "private_ports": {"tabby": 38204},
                       "acquisition_disk_roots": {"C": str(self.root), "F": str(self.root)}}
        self.state = State(self.config, clock=lambda: self.now)
        self.state.control("harness_verified", True)
        atomic_json(self.root / "artifacts" / "engine-discovery.json", {"engines": [{"id": "exllamav3-tabby",
                    "pin": {"tabby_commit": TABBY_COMMIT, "exllamav3_commit": EXL3_COMMIT}, "paths": [{"kind": "native_windows_local_venv",
                    "wheel": WHEEL_NAME, "sha256": WHEEL_SHA, "bytes": 364860083, "url": "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.0/" + WHEEL_NAME}]}]})
        self.runner = FakeSetupRunner(self)
        self.prepared = prepare_tabby_source(self.config, self.state, runner=self.runner)
        self.downloads = []
        def download(config, state, artifact, path):
            self.downloads.append(artifact)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"dummy pinned-wheel fixture; never installed")
            return path
        self.downloader = download
        self.runtime = install_tabby(self.config, self.state, self.prepared, self.prepared["review_fingerprint"],
                                     runner=self.runner, downloader=download)
        modelroot = self.root / "model" / "pinned-revision"
        modelroot.mkdir(parents=True)
        atomic_json(modelroot / "config.json", {"max_position_embeddings": 262144, "quantization_config": {"quant_method": "exl3", "bits": 4}})
        weights = b"dummy EXL3 tensor fixture"
        (modelroot / "model.safetensors").write_bytes(weights)
        self.model = {"id": "fake-exl3", "path": str(modelroot), "revision": "a" * 40,
                      "files": [{"filename": "model.safetensors", "bytes": len(weights), "sha256": hashlib.sha256(weights).hexdigest(), "weight": True}]}
        self.api = FakeAPI(self.model)
        self.adapter = TabbyAdapter(self.config, self.state, self.model, self.runtime)
        self.adapter.http = self.api

    def tearDown(self):
        if self.adapter.process is not None:
            with patch("localbench.adapters.tabby.creation_time", return_value=123456):
                self.adapter.unload_owned()
        self.state.close()
        self.tmp.cleanup()

    def test_setup_pins_public_native_wheel_and_isolated_venv(self):
        self.assertEqual(self.downloads[0]["sha256"], WHEEL_SHA)
        self.assertIn(TABBY_COMMIT, self.runtime["root"])
        self.assertEqual(self.runtime["packages"]["torch"], "2.9.0+cu128")
        installs = [argv for label, argv, _ in self.runner.calls if label.endswith("-install")]
        self.assertEqual(len(installs), 1)
        self.assertIn("--require-hashes", installs[0])
        self.assertIn("--no-index", installs[0])
        self.assertIn("--no-deps", installs[0])
        self.assertFalse(self.runtime["executed_model"])
        compile(PIP_HELPER, "owned-pip-helper", "exec")
        self.assertIn('PIP_CONFIG_FILE', PIP_HELPER)

    def test_completed_setup_reuses_after_original_elapsed_deadline(self):
        self.now = self.runtime["setup_deadline"] + 3600
        self.runner.calls.clear()
        self.downloads.clear()
        reused = install_tabby(self.config, self.state, self.prepared, self.prepared["review_fingerprint"],
                               runner=self.runner, downloader=self.downloader)
        self.assertEqual(reused, self.runtime)
        self.assertEqual(self.downloads, [])
        self.assertEqual([label for label, _, _ in self.runner.calls], ["tabby-git-sha", "tabby-git-clean", "tabby-reuse-versions"])

    def test_source_review_change_rejected_before_install(self):
        Path(self.prepared["source"], "main.py").write_text("unreviewed change", encoding="utf-8")
        self.runner.calls.clear()
        with self.assertRaisesRegex(RuntimeError, "review_required_or_changed"):
            install_tabby(self.config, self.state, self.prepared, self.prepared["review_fingerprint"], runner=self.runner, downloader=self.downloader)
        self.assertEqual(self.runner.calls, [])

    def test_dependency_report_rejects_secret_urls_and_wrong_stack(self):
        root = Path(self.runtime["root"])
        wheel = root / "packages" / WHEEL_NAME
        record = {"metadata": {"name": "torch", "version": "2.9.0+cu128"}, "download_info": {"url": "https://files.pythonhosted.org/torch.whl?token=secret",
                  "archive_info": {"hashes": {"sha256": "a" * 64}}}}
        with self.assertRaisesRegex(RuntimeError, "secret_free"):
            _freeze_report(json.dumps({"install": [record]}), root, wheel)
        record["download_info"]["url"] = "https://files.pythonhosted.org/torch.whl"
        with self.assertRaisesRegex(RuntimeError, "stack_not_expected"):
            _freeze_report(json.dumps({"install": [record]}), root, wheel)

    def test_owned_config_blocks_browser_origins_fetches_auth_files_and_priority_changes(self):
        config = owned_config(self.model["path"], 38204, self.adapter.requested)
        self.assertEqual(config["network"]["host"], "127.0.0.1")
        self.assertEqual(config["network"]["allowed_origins"], [])
        self.assertTrue(config["network"]["disable_auth"])
        self.assertTrue(config["network"]["disable_fetch_requests"])
        self.assertFalse(config["developer"]["realtime_process_priority"])
        self.assertFalse(config["model"]["vision"])
        self.assertEqual(config["draft_model"]["draft_mode"], "disabled")

    def test_cache_pairs_and_reasoning_profiles_are_explicit(self):
        adapter = TabbyAdapter(self.config, self.state, self.model, self.runtime, {"cache_k": "q8_0", "cache_v": "q4_0", "reasoning": "reduced"})
        config = owned_config(self.model["path"], 38204, adapter.requested)
        self.assertEqual(config["model"]["cache_mode"], "8,4")
        self.assertEqual(config["model"]["reasoning_budget_tokens"], 256)
        disabled = owned_config(self.model["path"], 38204, dict(adapter.requested, reasoning="disabled"))
        self.assertEqual(disabled["model"]["template_vars_force"], {"enable_thinking": False})
        with self.assertRaises(ValueError):
            TabbyAdapter(self.config, self.state, self.model, self.runtime, {"cache_k": "f16", "cache_v": "q4_0"})

    def test_gguf_or_corrupt_or_short_context_checkpoint_rejected(self):
        config = Path(self.model["path"]) / "config.json"
        atomic_json(config, {"max_position_embeddings": 262144, "quantization_config": {"quant_method": "awq"}})
        with self.assertRaisesRegex(RuntimeError, "not_exl3"):
            validate_exl3(self.model)
        atomic_json(config, {"max_position_embeddings": 32768, "quantization_config": {"quant_method": "exl3"}})
        with self.assertRaisesRegex(RuntimeError, "below_64k"):
            validate_exl3(self.model)
        atomic_json(config, {"max_position_embeddings": 262144, "quantization_config": {"quant_method": "exl3"}})
        Path(self.model["path"], "model.safetensors").write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "hash_or_size"):
            validate_exl3(self.model)

    def test_effective_context_cache_model_and_slots_must_be_observed(self):
        self.adapter._verify_effective(self.api.props, self.api.json("/v1/models"))
        self.assertEqual(self.adapter.effective["effective_context"], 65536)
        self.api.props["total_slots"] = 4
        with self.assertRaisesRegex(RuntimeError, "per_slot_context"):
            self.adapter._verify_effective(self.api.props, self.api.json("/v1/models"))
        self.api.props["total_slots"] = 1
        self.api.parameters["cache_mode"] = "Q4"
        with self.assertRaisesRegex(RuntimeError, "cache_or_chunk"):
            self.adapter._verify_effective(self.api.props, self.api.json("/v1/models"))

    def test_exact_tokenization_includes_tools_generation_prefix_and_bos_setting(self):
        self.adapter._metadata = {"runtime_manifest_sha256": "f" * 64}
        tools = [{"function": {"name": "alpha"}}]
        encoded = self.adapter.tokenize([{"role": "system", "content": "fixture"}], tools)
        self.assertEqual(encoded["count"], 3)
        self.assertEqual(self.api.requests[-1][0], "/v1/token/encode")
        self.assertTrue(self.api.requests[-1][1]["add_bos_token"])
        self.assertTrue(self.api.requests[-2][1]["add_generation_prompt"])
        with self.assertRaisesRegex(RuntimeError, "omits_prompt_or_tools"):
            self.adapter.tokenize([], [{"function": {"name": "absent"}}])

    def test_stream_translation_records_seed_limit_and_reasoning_budget(self):
        self.adapter.chat_fields = self.api.fields
        self.adapter.requested["reasoning"] = "reduced"
        self.adapter.stream_chat({"model": "friendly-id", "repeat_penalty": 1, "seed": 42, "stream": True})
        self.assertEqual(self.api.measured["model"], Path(self.model["path"]).name)
        self.assertEqual(self.api.measured["repetition_penalty"], 1)
        self.assertNotIn("seed", self.api.measured)
        self.assertTrue(self.api.measured["add_bos_token"])
        self.assertEqual(self.api.measured["reasoning_budget_tokens"], 256)
        self.assertEqual(self.adapter.reset_cache()["status"], "owned_server_restart_required")

    @unittest.skipUnless(os.name == "nt", "Windows adapter launch planner")
    def test_mock_launch_owns_suspended_tree_filters_secrets_and_verifies_schema(self):
        adapter = self.adapter
        api = self.api
        captured = {}
        class Process:
            pid = 12345
            returncode = None
            def poll(self): return self.returncode
            def wait(self, timeout): self.returncode = 0
        process = Process()
        class Job:
            def attach_resume(self, p): captured["attached"] = p is process
            def close(self): captured["closed"] = True; process.returncode = 0
        def launch(argv, **kwargs):
            captured["argv"] = argv
            captured.update(kwargs)
            return process
        with patch("localbench.adapters.tabby.run_owned", return_value={"exit_code": 0, "log": "dummy-help.log", "output": "--config"}), \
             patch("localbench.adapters.tabby.subprocess.Popen", side_effect=launch), \
             patch("localbench.adapters.tabby._WindowsJob", return_value=Job()), \
             patch("localbench.adapters.tabby.creation_time", return_value=123456), \
             patch("localbench.adapters.tabby.check_port"), \
             patch("localbench.adapters.tabby.ctypes.windll.shell32.IsUserAnAdmin", return_value=0), \
             patch.object(adapter, "health", return_value=True), \
             patch.dict(os.environ, {"LM_BENCH_TOKEN": "test-secret", "TABBY_NETWORK_HOST": "0.0.0.0"}):
            result = adapter.launch()
            adapter.unload_owned()
        self.assertTrue(captured["attached"] and captured["closed"])
        self.assertTrue(captured["creationflags"] & 4)
        self.assertNotIn("LM_BENCH_TOKEN", captured["env"])
        self.assertNotIn("TABBY_NETWORK_HOST", captured["env"])
        self.assertEqual(captured["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(result["effective"]["effective_context"], 65536)
        self.assertFalse(result["effective"]["seed_supported"])
        owned = json.loads(Path(result["handle"]["config_path"]).read_text(encoding="utf-8"))
        self.assertEqual(owned["network"]["allowed_origins"], [])
        self.assertFalse(Path(captured["cwd"], "api_tokens.yml").exists())


if __name__ == "__main__":
    unittest.main()
