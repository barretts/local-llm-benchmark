import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from localbench.config import atomic_json, canonical, file_hash, load
from localbench.state import State
from localbench.tokenizer_helper import NativeTokenizerHelper, _manifest_record, cpu_launch_plan


HELP = " " .join(["--model", "--ctx-size", "--parallel", "--n-gpu-layers", "--device", "--batch-size",
    "--ubatch-size", "--threads", "--threads-batch", "--host", "--port", "--cors-origins",
    "--no-kv-offload", "--no-op-offload", "--no-context-shift", "--jinja", "--no-warmup", "--no-webui",
    "--spec-type", "--cors-methods", "--cors-headers", "--no-cors-credentials"])


class TokenizerHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ("project", "state", "logs", "artifacts", "runtime_root"):
            self.config["paths"][name] = str(Path(self.temp.name) / name)
        self.state = State(self.config)
        self.state.control("harness_verified", True)
        self.runtime = Path(self.config["paths"]["runtime_root"]) / ("upstream-" + "a" * 40)
        self.binary = self.runtime / "bin" / "llama-server.exe"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b"unexecuted owned test binary")
        self.dll = self.binary.parent / "backend.dll"
        self.dll.write_bytes(b"unexecuted pinned DLL")
        self.manifest = {"id": "upstream-llama-nightly", "source": "https://github.com/ggml-org/llama.cpp",
            "commit": "a" * 40, "release_tag": "b-test", "binary_relative_path": "llama-server.exe",
            "binary_sha256": file_hash(self.binary), "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)}
                for path in (self.binary, self.dll)]}
        self.state.entity("runtime_catalog", "upstream-llama-nightly", {"kind": "native",
            "binary_path": str(self.binary), "engine_manifest": self.manifest})
        self.source = Path(self.temp.name) / "installed.gguf"
        self.source.write_bytes(b"GGUF mock bytes; never loaded by a real process")
        self.model = {"id": "installed-test-model", "path": str(self.source), "sha256": file_hash(self.source)}
        self.calls = []
        self.client = MagicMock()
        self.client.json.side_effect = self.http_json
        self.process = MagicMock()
        self.process.pid = 12345
        self.process.poll.return_value = None
        self.process.terminate.side_effect = lambda: setattr(self.process.poll, "return_value", 0)

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def http_json(self, path, body=None):
        self.calls.append((path, copy.deepcopy(body)))
        if path == "/health":
            return {"status": "ok"}
        if path == "/props":
            return {"default_generation_settings": {"n_ctx": 65536}, "total_slots": 1}
        if path == "/apply-template":
            return {"prompt": canonical(body)}
        if path == "/tokenize":
            # Mock a deterministic endpoint, not an implementation estimate.
            return {"tokens": list(range(1 + len(body["content"].encode("utf-8")) // 8))}
        raise AssertionError("inference endpoint called:" + path)

    def command(self, config, label, argv, **kwargs):
        self.assertNotIn("LM_BENCH_TOKEN", kwargs["environment"])
        return {"exit_code": 0, "output": HELP if argv[-1] == "--help" else "pinned-test-version",
            "log": label + ".log"}

    def launched(self, **patches):
        helper = NativeTokenizerHelper(self.config, self.state, self.model)
        listeners = [None]
        def listener(port):
            return listeners.pop(0) if listeners else self.process.pid
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "memory-only-test-secret"}), \
                patch("localbench.tokenizer_helper._logged_command", side_effect=self.command), \
                patch("localbench.tokenizer_helper._ephemeral_port", return_value=41023), \
                patch("localbench.tokenizer_helper._listener_pid", side_effect=listener), \
                patch("localbench.tokenizer_helper.creation_time", return_value=7654321), \
                patch("localbench.tokenizer_helper.LocalHTTP", return_value=self.client), \
                patch("localbench.tokenizer_helper.subprocess.Popen", return_value=self.process) as popen:
            helper.launch()
            argv = popen.call_args.args[0]
        return helper, argv

    def live_identity(self, helper):
        return patch.multiple("localbench.tokenizer_helper", creation_time=MagicMock(return_value=helper.saved["created"]),
            _listener_pid=MagicMock(return_value=helper.process.pid))

    def tools(self):
        return [{"type": "function", "function": {"name": name, "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}}, "required": ["path"]}}} for name in ("read_file", "write_file")]

    def test_constructor_and_manifest_resolution_do_not_execute_or_start_clock(self):
        with patch("localbench.tokenizer_helper.subprocess.Popen") as popen, \
                patch("localbench.tokenizer_helper._logged_command") as command:
            helper = NativeTokenizerHelper(self.config, self.state, self.model)
            binary, manifest = _manifest_record(self.config, self.state)
        self.assertEqual(binary, self.binary.resolve())
        self.assertEqual(manifest["binary_sha256"], file_hash(self.binary))
        self.assertFalse(helper.verified)
        self.assertEqual(helper.verified_engines, set())
        self.assertIsNone(self.state.run["started"])
        popen.assert_not_called()
        command.assert_not_called()

    def test_manifest_fallback_is_exact_and_ambiguous_fallback_rejected(self):
        with self.state.db:
            self.state.db.execute("DELETE FROM entities WHERE kind='runtime_catalog'")
        atomic_json(self.runtime / "engine-manifest.json", self.manifest)
        binary, _ = _manifest_record(self.config, self.state)
        self.assertEqual(binary, self.binary.resolve())
        second = Path(self.config["paths"]["runtime_root"]) / "upstream-second"
        (second / "bin").mkdir(parents=True)
        for path in (self.binary, self.dll):
            (second / "bin" / path.name).write_bytes(path.read_bytes())
        atomic_json(second / "engine-manifest.json", self.manifest)
        with self.assertRaisesRegex(RuntimeError, "missing_or_ambiguous"):
            _manifest_record(self.config, self.state)
        self.assertEqual(_manifest_record(self.config, self.state, self.binary)[0], self.binary.resolve())

    def test_tampered_binary_dll_extra_dll_and_model_fail_before_execution(self):
        self.dll.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "file_hash_mismatch"):
            _manifest_record(self.config, self.state)
        self.dll.write_bytes(b"unexecuted pinned DLL")
        added = self.binary.parent / "unreviewed.dll"
        added.write_bytes(b"new library")
        with self.assertRaisesRegex(RuntimeError, "file_hash_mismatch"):
            _manifest_record(self.config, self.state)
        added.unlink()
        helper = NativeTokenizerHelper(self.config, self.state, self.model)
        self.source.write_bytes(b"different weight bytes")
        with patch("localbench.tokenizer_helper._logged_command") as command:
            with self.assertRaisesRegex(RuntimeError, "model_hash_or_format"):
                helper.discover_capabilities()
        command.assert_not_called()
        self.assertIsNone(self.state.run["started"])

    def test_cpu_plan_requires_every_isolation_flag_and_never_silently_drops(self):
        argv = cpu_launch_plan(self.binary, self.source, 41111, HELP)
        for flag, value in (("--device", "none"), ("--n-gpu-layers", "0"), ("--ctx-size", "65536"),
                ("--parallel", "1"), ("--batch-size", "512"), ("--ubatch-size", "128"), ("--threads", "2")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        for flag in ("--no-kv-offload", "--no-op-offload", "--no-warmup", "--no-webui", "--no-context-shift"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--cors-origins") + 1], "http://127.0.0.1:41111")
        self.assertEqual(argv[argv.index("--spec-type") + 1], "none")
        self.assertNotIn("--spec-type", cpu_launch_plan(self.binary, self.source, 41111, HELP.replace("--spec-type", "")))
        for missing in ("--device", "--no-op-offload", "--no-kv-offload", "--cors-origins", "--no-warmup"):
            with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, "unsupported_tokenizer_helper_flags"):
                cpu_launch_plan(self.binary, self.source, 41111, HELP.replace(missing, missing + "-unsupported"))

    def test_environment_is_scoped_cpu_only_and_credentials_are_not_inherited(self):
        helper = NativeTokenizerHelper(self.config, self.state, self.model)
        original = {"CUDA_VISIBLE_DEVICES": "0", "LLAMA_ARG_N_GPU_LAYERS": "99", "LM_BENCH_TOKEN": "secret",
            "SOME_API_KEY": "secret-api", "PATH": "unchanged-path"}
        with patch.dict(os.environ, original):
            environment = helper.process_environment()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(os.environ["LLAMA_ARG_N_GPU_LAYERS"], "99")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("LLAMA_ARG_N_GPU_LAYERS", environment)
        self.assertNotIn("LM_BENCH_TOKEN", environment)
        self.assertNotIn("SOME_API_KEY", environment)
        self.assertEqual(environment["PATH"], "unchanged-path")

    def test_launch_verifies_intrinsic_512_and_has_no_managed_engine_or_inference_claim(self):
        helper, argv = self.launched()
        self.assertTrue(helper.offline)
        self.assertTrue(helper.verified)
        self.assertEqual(helper.verified_engines, set())
        self.assertEqual(helper._metadata["intrinsic_verification"]["exact_prompt_tokens"], 512)
        self.assertFalse(helper._metadata["intrinsic_verification"]["managed_usage_counter_verified"])
        self.assertEqual(helper._metadata["inference_calls"], 0)
        self.assertEqual(helper.port, 41023)
        self.assertEqual(helper.saved["argv"], argv)
        self.assertEqual(helper.saved["created"], 7654321)
        self.assertTrue(all(path in {"/health", "/props", "/apply-template", "/tokenize"} for path, _ in self.calls))
        self.assertIsNotNone(self.state.run["started"])
        text = json.dumps(helper.metadata())
        self.assertNotIn("memory-only-test-secret", text)
        with self.live_identity(helper):
            helper.close()

    def test_exact_tokenization_includes_every_tool_and_special_tokens(self):
        helper, _ = self.launched()
        messages = [{"role": "system", "content": "Only synthetic code."}, {"role": "user", "content": "café"}]
        tools = self.tools()
        with self.live_identity(helper):
            observed = helper.tokenize(messages, tools)
            raw = helper.tokenize_text("café")
            with self.assertRaisesRegex(RuntimeError, "inference_endpoint_forbidden"):
                helper._json("/v1/chat/completions", {})
            helper.close()
        template = canonical({"messages": messages, "tools": tools, "add_generation_prompt": True})
        self.assertEqual(observed["count"], 1 + len(template.encode()) // 8)
        self.assertEqual(observed["model_sha256"], self.model["sha256"])
        token_call = [body for path, body in self.calls if path == "/tokenize" and body["content"] == template][-1]
        self.assertTrue(token_call["add_special"])
        self.assertTrue(token_call["parse_special"])
        self.assertEqual(raw, list(range(1 + len("café".encode()) // 8)))

    def test_missing_second_tool_or_noninteger_token_response_cannot_verify(self):
        helper, _ = self.launched()
        normal = self.client.json.side_effect
        def omitted(path, body=None):
            if path == "/apply-template":
                return {"prompt": "read_file only"}
            return normal(path, body)
        self.client.json.side_effect = omitted
        with self.live_identity(helper):
            with self.assertRaisesRegex(RuntimeError, "omits_tools"):
                helper.tokenize([], self.tools())
            self.client.json.side_effect = lambda path, body=None: {"tokens": [True]} if path == "/tokenize" else normal(path, body)
            with self.assertRaisesRegex(RuntimeError, "invalid_exact_token"):
                helper.tokenize_text("input")
            helper.close()

    def test_managed_engine_requires_actual_counter_and_mismatch_revokes_prior_proof(self):
        helper, _ = self.launched()
        messages, tools = [{"role": "user", "content": "repair code"}], self.tools()
        with self.live_identity(helper):
            expected = helper.tokenize(messages, tools)["count"]
            proof = helper.verify_against("ollama", messages, tools, expected)
            self.assertTrue(proof["passed"])
            self.assertEqual(helper.verified_engines, {"ollama"})
            self.assertNotIn("lm-studio", helper.verified_engines)
            with self.assertRaisesRegex(RuntimeError, "prompt_count_mismatch"):
                helper.verify_against("ollama", messages, tools, expected + 65)
            self.assertNotIn("ollama", helper.verified_engines)
            self.assertFalse(helper.proofs["ollama"]["passed"])
            with self.assertRaisesRegex(ValueError, "unknown_managed_engine"):
                helper.verify_against("unobserved-engine", messages, tools, expected)
            helper.close()
        rows = self.state.db.execute("SELECT data FROM entities WHERE kind='tokenizer_verification'").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertFalse(json.loads(rows[0][0])["passed"])

    def test_managed_counter_tolerance_does_not_allow_above_full_target_or_missing_counts(self):
        helper, _ = self.launched()
        with self.live_identity(helper):
            expected = helper.tokenize([], [])["count"]
            self.assertTrue(helper.verify_against("lm-studio", [], [], expected + 64)["passed"])
            for actual in (None, "512", True, 0, 61441):
                with self.subTest(actual=actual), self.assertRaisesRegex(RuntimeError, "prompt_count_mismatch"):
                    helper.verify_against("lm-studio", [], [], actual)
                self.assertNotIn("lm-studio", helper.verified_engines)
            with patch.object(helper, "tokenize", return_value={"count": 61441, "template_sha256": "fake"}):
                with self.assertRaisesRegex(RuntimeError, "prompt_count_mismatch"):
                    helper.verify_against("ollama", [], [], 61440)
            helper.close()

    def test_listener_race_and_replaced_pid_do_not_contact_or_kill_unrelated_process(self):
        helper, _ = self.launched()
        previous = len(self.calls)
        with patch("localbench.tokenizer_helper.creation_time", return_value=helper.saved["created"]), \
                patch("localbench.tokenizer_helper._listener_pid", return_value=99999):
            with self.assertRaisesRegex(RuntimeError, "listener_not_owned"):
                helper.tokenize([], [])
        self.assertEqual(len(self.calls), previous)
        with patch("localbench.tokenizer_helper.creation_time", return_value=helper.saved["created"] + 1):
            with self.assertRaisesRegex(RuntimeError, "stale_tokenizer_helper_pid"):
                helper.close()
        self.process.terminate.assert_not_called()
        self.process.kill.assert_not_called()
        with self.live_identity(helper):
            helper.close()

    def test_spawn_port_race_refuses_before_popen(self):
        helper = NativeTokenizerHelper(self.config, self.state, self.model)
        with patch("localbench.tokenizer_helper._logged_command", side_effect=self.command), \
                patch("localbench.tokenizer_helper._ephemeral_port", return_value=41023), \
                patch("localbench.tokenizer_helper._listener_pid", return_value=99999), \
                patch("localbench.tokenizer_helper.subprocess.Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "port_race_before_spawn"):
                helper.launch()
        popen.assert_not_called()

    def test_owned_popen_handle_is_only_termination_target_even_if_force_stop_needed(self):
        helper, _ = self.launched()
        self.process.wait.side_effect = [subprocess.TimeoutExpired("owned", 15), 0]
        with self.live_identity(helper):
            helper.close()
        self.process.terminate.assert_called_once_with()
        self.process.kill.assert_called_once_with()
        self.assertFalse(helper.verified)
        self.assertIsNone(helper.process)

    def test_observed_gpu_buffer_or_insufficient_context_unloads_only_owned_helper(self):
        for bad in ("CUDA0 compute buffer size = 1.00 MiB\n", "offloaded 1/24 layers to GPU\n"):
            with self.subTest(log=bad):
                helper = NativeTokenizerHelper(self.config, self.state, self.model)
                process = MagicMock(pid=12345)
                process.poll.return_value = None
                process.terminate.side_effect = lambda: setattr(process.poll, "return_value", 0)
                def spawn(argv, **kwargs):
                    kwargs["stderr"].write(bad.encode())
                    kwargs["stderr"].flush()
                    return process
                listeners = [None]
                with patch("localbench.tokenizer_helper._logged_command", side_effect=self.command), \
                        patch("localbench.tokenizer_helper._ephemeral_port", return_value=41023), \
                        patch("localbench.tokenizer_helper._listener_pid", side_effect=lambda port: listeners.pop(0) if listeners else 12345), \
                        patch("localbench.tokenizer_helper.creation_time", return_value=7654321), \
                        patch("localbench.tokenizer_helper.LocalHTTP", return_value=self.client), \
                        patch("localbench.tokenizer_helper.subprocess.Popen", side_effect=spawn):
                    with self.assertRaisesRegex(RuntimeError, "observed_gpu_allocation"):
                        helper.launch()
                process.terminate.assert_called_once_with()
                self.assertFalse(helper.verified)
                self.assertEqual(helper.verified_engines, set())


if __name__ == "__main__":
    unittest.main()
