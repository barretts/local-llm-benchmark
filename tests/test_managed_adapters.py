import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from localbench.adapters.lms import LMStudioAdapter
from localbench.adapters.ollama import OllamaAdapter
from localbench.config import file_hash, load
from localbench.state import State


class ManagedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ("project", "state", "logs", "artifacts", "runtime_root", "new_model_root"):
            self.config["paths"][name] = str(Path(self.temp.name) / name)
        self.config["limits"]["minimum_free_f_bytes"] = 0
        self.headroom = patch("localbench.adapters.ollama._disk_headroom")
        self.headroom.start()
        self.addCleanup(self.headroom.stop)
        self.source = Path(self.temp.name) / "installed.gguf"
        self.source.write_bytes(b"owned test bytes without actual model weights" * 8)
        self.binary = Path(self.temp.name) / "fake-cli.exe"
        self.binary.write_bytes(b"unexecuted test executable")
        self.config["installed_tools"].update(lms=str(self.binary), ollama=str(self.binary))
        self.model = {"id": "installed-model", "path": str(self.source), "sha256": file_hash(self.source), "lms_key": "test-model"}
        self.state = State(self.config)
        self.state.control("harness_verified", True)

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def lms(self):
        adapter = LMStudioAdapter(self.config, self.state, self.model)
        adapter._metadata = {"model_sha256": self.model["sha256"], "engine_id": "lm-studio"}
        return adapter

    def ollama(self, **changes):
        adapter = OllamaAdapter(self.config, self.state, self.model, **changes)
        adapter._metadata = {"model_sha256": self.model["sha256"], "engine_id": "ollama"}
        adapter.local_name = "localbench-test-model"
        return adapter

    def inventory(self, instances=None):
        return {"models": [{"key": "test-model", "loaded_instances": instances or []}]}

    def load_lms(self, adapter):
        inventory = self.inventory()

        def http_json(path, body=None):
            if path == "/api/v1/models":
                return copy.deepcopy(inventory)
            if path == "/api/v1/models/unload":
                inventory["models"][0]["loaded_instances"] = [instance for instance in inventory["models"][0]["loaded_instances"]
                    if instance["id"] != body["instance_id"]]
                return {"instance_id": body["instance_id"]}
            raise AssertionError("unexpected LM Studio HTTP path:" + path)

        def command(config, label, argv, **kwargs):
            self.assertIn("--identifier", argv)
            identifier = argv[argv.index("--identifier") + 1]
            inventory["models"][0]["loaded_instances"].append({"id": identifier,
                "config": {"context_length": 65536, "parallel": 1, "eval_batch_size": 512}})
            return {"exit_code": 0, "output": "loaded", "log": "mock-load.log"}

        adapter.http = MagicMock()
        adapter.http.json.side_effect = http_json
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "memory-only-test-secret"}), \
                patch("localbench.adapters.lms._logged_command", side_effect=command):
            launched = adapter.launch()
        return inventory, launched

    def test_lm_studio_missing_auth_is_explicit_and_starts_no_clock(self):
        adapter = LMStudioAdapter(self.config, self.state, self.model)
        with patch.dict(os.environ, {}, clear=True), patch("localbench.adapters.lms._logged_command") as command:
            with self.assertRaisesRegex(RuntimeError, "lm_studio_auth_unavailable"):
                adapter.discover_capabilities()
            self.assertFalse(adapter.health())
        command.assert_not_called()
        self.assertIsNone(self.state.run["started"])

    def test_lm_studio_owned_instance_public_context_and_only_owned_unload(self):
        adapter = self.lms()
        inventory, launched = self.load_lms(adapter)
        identifier = adapter.instance_id
        self.assertTrue(identifier.startswith("localbench-"))
        self.assertEqual(launched["effective"]["effective_context"], 65536)
        self.assertIsNone(adapter.process)
        inventory["models"][0]["loaded_instances"].append({"id": "foreign-user-model", "config": {"context_length": 4096}})
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "memory-only-test-secret"}):
            adapter.unload_owned()
        self.assertEqual(inventory["models"][0]["loaded_instances"], [{"id": "foreign-user-model", "config": {"context_length": 4096}}])
        calls = [call.args for call in adapter.http.json.call_args_list if call.args[0] == "/api/v1/models/unload"]
        self.assertEqual(calls, [("/api/v1/models/unload", {"instance_id": identifier})])
        for path in Path(self.config["paths"]["state"]).glob("*.json"):
            self.assertNotIn("memory-only-test-secret", path.read_text(encoding="utf-8"))

    def test_lm_studio_foreign_loaded_instance_never_evicted(self):
        adapter = self.lms()
        adapter.http = MagicMock()
        adapter.http.json.return_value = self.inventory([{"id": "user-instance", "config": {"context_length": 4096}}])
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "memory-only-test-secret"}), \
                patch("localbench.adapters.lms._logged_command") as command:
            with self.assertRaisesRegex(RuntimeError, "foreign_loaded_instances"):
                adapter.launch()
        command.assert_not_called()
        self.assertIsNone(adapter.instance_id)

    def test_lm_studio_replaced_instance_and_new_foreign_workload_rejected(self):
        adapter = self.lms()
        inventory, _ = self.load_lms(adapter)
        inventory["models"][0]["loaded_instances"][0]["config"]["context_length"] = 4096
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "memory-only-test-secret"}):
            with self.assertRaisesRegex(RuntimeError, "identity_changed"):
                adapter.unload_owned()
        self.assertFalse(any(call.args[0] == "/api/v1/models/unload" for call in adapter.http.json.call_args_list))
        inventory["models"][0]["loaded_instances"].append({"id": "foreign", "config": {"context_length": 4096}})
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "memory-only-test-secret"}):
            with self.assertRaisesRegex(RuntimeError, "foreign_loaded_instances"):
                adapter.stream_chat({"messages": []})
        adapter.http.measure.assert_not_called()

    def test_managed_unsupported_controls_and_unverified_tokenization_fail_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported_public"):
            LMStudioAdapter(self.config, self.state, self.model, {"ubatch": 1024})
        with self.assertRaisesRegex(RuntimeError, "unsupported_asymmetric"):
            OllamaAdapter(self.config, self.state, self.model, {"cache_k": "q4_0", "cache_v": "q8_0"})
        for adapter in (self.lms(), self.ollama()):
            with self.assertRaisesRegex(RuntimeError, "exact_offline_template_tokenizer_unavailable"):
                adapter.tokenize([{"role": "user", "content": "hello"}], [])
            with self.assertRaisesRegex(RuntimeError, "exact_offline_template_tokenizer_unavailable"):
                adapter.tokenize_text("hello")

    def test_verified_offline_tokenizer_must_match_engine_and_model(self):
        helper = SimpleNamespace(offline=True, verified=True, model_sha256=self.model["sha256"],
            verified_engines={"lm-studio", "ollama"}, tokenize=lambda messages, tools: {"count": 19, "provenance": "verified-mock-template"},
            tokenize_text=lambda text: [1, 2, 3])
        for adapter in (self.lms(), self.ollama()):
            adapter.exact_tokenizer = helper
            self.assertEqual(adapter.tokenize([], [])["count"], 19)
            self.assertEqual(adapter.tokenize_text("hello"), [1, 2, 3])
        helper.model_sha256 = "different-weight"
        with self.assertRaisesRegex(RuntimeError, "exact_offline"):
            adapter.tokenize([], [])

    def test_ollama_scoped_environment_never_changes_user_settings_or_leaks_token(self):
        adapter = self.ollama()
        original = {"OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_MODELS": "personal-models",
            "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_API_KEY": "private-cloud-credential", "LM_BENCH_TOKEN": "memory-only-secret"}
        with patch.dict(os.environ, original):
            environment = adapter._environment()
            self.assertEqual(os.environ["OLLAMA_MODELS"], "personal-models")
            self.assertEqual(os.environ["OLLAMA_HOST"], "127.0.0.1:11434")
        self.assertEqual(environment["OLLAMA_HOST"], "127.0.0.1:38207")
        self.assertEqual(environment["OLLAMA_KV_CACHE_TYPE"], "f16")
        self.assertEqual(environment["OLLAMA_NO_CLOUD"], "1")
        self.assertNotIn("LM_BENCH_TOKEN", environment)
        self.assertNotIn("OLLAMA_API_KEY", environment)

    def test_ollama_hardlink_physical_reuse_is_verified_and_budget_zero(self):
        adapter = self.ollama()
        blob = adapter._prepare_blob()
        self.assertTrue(os.path.samefile(self.source, blob))
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], 0)
        self.assertEqual(adapter.metadata()["weight_import"]["additional_physical_weight_bytes"], 0)
        self.assertEqual(adapter._prepare_blob(), blob)
        self.assertEqual(self.source.read_bytes(), b"owned test bytes without actual model weights" * 8)

    def test_ollama_copy_fallback_resumes_reserved_artifact_and_counts_actual_bytes(self):
        adapter = self.ollama()
        target = adapter.models_root / "blobs" / ("sha256-" + self.model["sha256"])
        target.parent.mkdir(parents=True)
        target.with_name(target.name + ".part").write_bytes(self.source.read_bytes()[:23])
        with patch("localbench.adapters.ollama.os.link", side_effect=OSError("cross-volume")):
            blob = adapter._prepare_blob()
        self.assertEqual(file_hash(blob), self.model["sha256"])
        self.assertFalse(os.path.samefile(self.source, blob))
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], self.source.stat().st_size)
        self.assertEqual(self.state.db.execute("SELECT acquired FROM weights").fetchone()[0], 1)
        self.assertEqual(adapter.metadata()["weight_import"]["additional_physical_weight_bytes"], self.source.stat().st_size)
        adapter._prepare_blob()
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], self.source.stat().st_size)

    def test_ollama_copy_budget_enforced_before_copy_write(self):
        self.config["limits"]["new_model_weight_bytes"] = 1
        adapter = self.ollama()
        with patch("localbench.adapters.ollama.os.link", side_effect=OSError("cross-volume")):
            with self.assertRaisesRegex(RuntimeError, "weight_budget_exhausted"):
                adapter._prepare_blob()
        self.assertEqual(list((adapter.models_root / "blobs").iterdir()), [])

    def test_ollama_native_request_preserves_tool_arguments_thinking_and_actual_options(self):
        adapter = self.ollama(verified_controls={"truncate": "pinned-source-proof", "shift": "pinned-source-proof"})
        request = {"model": "local-model", "messages": [
            {"role": "system", "content": "nonce:unique"},
            {"role": "assistant", "content": "", "reasoning_content": "saved thought",
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"src/cache.py"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "public file text"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
            "stream": True, "stream_options": {"include_usage": True}, "tool_choice": "auto", "max_tokens": 256,
            "seed": 42, **self.config["default_sampling"]}
        native = adapter.native_request(request)
        self.assertEqual(native["model"], adapter.local_name)
        self.assertEqual(native["messages"][1]["tool_calls"][0]["function"]["arguments"], {"path": "src/cache.py"})
        self.assertEqual(native["messages"][1]["thinking"], "saved thought")
        self.assertEqual(native["messages"][2]["tool_name"], "read_file")
        self.assertEqual(native["options"]["num_predict"], 256)
        self.assertEqual(native["options"]["num_ctx"], 65536)
        self.assertEqual(native["options"]["seed"], 42)
        self.assertFalse(native["truncate"])
        self.assertFalse(native["shift"])
        self.assertNotIn("cached_tokens", native)
        with self.assertRaisesRegex(RuntimeError, "unsupported_ollama_request"):
            adapter.native_request({**request, "reasoning_budget": 256})

    def test_ollama_stale_pid_refuses_termination(self):
        adapter = self.ollama()
        process = MagicMock(pid=9001)
        process.poll.return_value = None
        adapter.process = process
        adapter.saved = {"owner": "localbench", "created": 100, "pid": 9001}
        with patch("localbench.adapters.ollama.creation_time", return_value=101):
            with self.assertRaisesRegex(RuntimeError, "stale_ollama_pid_identity"):
                adapter.unload_owned()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_ollama_listener_pid_must_be_owned_before_streaming(self):
        adapter = self.ollama()
        adapter.process = MagicMock(pid=9001)
        adapter.process.poll.return_value = None
        adapter.http = MagicMock()
        with patch("localbench.adapters.ollama._listener_pid", return_value=9002):
            if os.name == "nt":
                with self.assertRaisesRegex(RuntimeError, "listener_identity_unverified"):
                    adapter.stream_chat({"messages": []})
                adapter.http.measure.assert_not_called()

    def test_ollama_manifest_requires_exact_imported_weight_digest_and_size(self):
        adapter = self.ollama()
        manifest = adapter.models_root / "manifests" / "registry.ollama.ai" / "library" / adapter.local_name / "latest"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:" + "0" * 64, "size": self.source.stat().st_size}]}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "did_not_reuse_exact"):
            adapter._manifest()
        manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:" + self.model["sha256"], "size": self.source.stat().st_size}]}), encoding="utf-8")
        self.assertEqual(adapter._manifest()["sha256"], file_hash(manifest))

    def test_lm_studio_unsupported_sampler_cannot_be_silently_ignored(self):
        adapter = self.lms()
        self.load_lms(adapter)
        with self.assertRaisesRegex(RuntimeError, "unsupported_public_lm_studio_request_controls:min_p"):
            adapter.stream_chat({"messages": [], "min_p": .05})
        adapter.http.measure.assert_not_called()

    def test_ollama_probe_and_model_load_are_mocked_private_and_effective_context_checked(self):
        adapter = self.ollama()
        process = MagicMock(pid=9001)
        process.poll.return_value = None
        process.wait.return_value = 0
        adapter.http = MagicMock()
        def http_json(path, body=None):
            if path == "/api/version":
                return {"version": "0.mock"}
            if path == "/api/show":
                return {"capabilities": ["completion", "tools"], "details": {"format": "gguf"}, "template": "mock-native-template"}
            if path == "/api/chat":
                return {"done": True}
            if path == "/api/ps":
                return {"models": [{"name": adapter.local_name + ":latest", "context_length": 65536, "size": 1234, "size_vram": 1000}]}
            raise AssertionError("unexpected Ollama API path:" + path)
        adapter.http.json.side_effect = http_json
        manifest = adapter.models_root / "manifests" / "registry.ollama.ai" / "library" / adapter.local_name / "latest"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:" + self.model["sha256"], "size": self.source.stat().st_size}]}), encoding="utf-8")
        with patch("localbench.adapters.ollama.check_port") as port, \
                patch("localbench.adapters.ollama.subprocess.Popen", return_value=process) as launch, \
                patch("localbench.adapters.ollama.creation_time", return_value=100), \
                patch("localbench.adapters.ollama._listener_pid", return_value=9001), \
                patch("localbench.adapters.ollama._descendants", return_value={}), \
                patch("localbench.adapters.ollama._logged_command", return_value={"exit_code": 0, "output": "mock", "log": "mock.log"}), \
                patch.object(adapter, "health", return_value=True):
            result = adapter.launch()
            self.assertEqual(result["effective"]["effective_context"], 65536)
            self.assertFalse(result["effective"]["context_shift_disabled"])
            self.assertEqual(launch.call_args.kwargs["env"]["OLLAMA_MODELS"], str(adapter.models_root))
            self.assertEqual(launch.call_args.args[0], [str(self.binary), "serve"])
            port.assert_called_once_with(38207)
            adapter.unload_owned()
        process.terminate.assert_called_once()
        self.assertIsNone(adapter.process)


if __name__ == "__main__":
    unittest.main()
