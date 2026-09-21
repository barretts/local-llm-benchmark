"""Mock-only container contract tests: never call Docker or load weights."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from localbench.adapters.container import (
    CONTAINER_FORMAT, IMAGE_FORMAT, TOKENIZER_HELPER,
    ContainerAdapter, ContainerHandle, classify_failure, _mount_path,
)
from localbench.config import file_hash

PIN = "example.invalid/engine@sha256:" + "a" * 64
IMAGE = "sha256:" + "b" * 64
CID = "c" * 64
FLAGS = """
--model --ctx-size --parallel --n-gpu-layers --flash-attn --cache-type-k
--cache-type-v --batch-size --ubatch-size --threads --threads-batch --spec-type
--reasoning --reasoning-budget --jinja --no-context-shift --peg --host --port
--max-model-len --max-num-seqs --gpu-memory-utilization --max-num-batched-tokens
--tool-call-parser --reasoning-parser --served-model-name --kv-cache-dtype
--enable-auto-tool-choice --enable-prefix-caching --no-enable-prefix-caching
--language-model-only
--model-path --context-length --max-running-requests --mem-fraction-static
--chunked-prefill-size --max-total-tokens --disable-radix-cache --attention-backend
"""


class FakeState:
    def __init__(self):
        self.run = {"id": "mock-run"}
        self.now = 1000.0
        self.clock = lambda: self.now
        self.controls = {key: True for key in
                         ("doctor_verified", "harness_verified", "fixtures_verified")}
        self.starts, self.entities, self.budgets = [], [], []

    def get_control(self, key):
        return copy.deepcopy(self.controls.get(key))

    def control(self, key, value):
        self.controls[key] = copy.deepcopy(value)

    def check_budget(self, **kwargs):
        self.budgets.append(kwargs)

    def start_execution(self, reason):
        self.starts.append(reason)

    def entity(self, kind, identity, metadata):
        self.entities.append((kind, identity, copy.deepcopy(metadata)))


class FakeLogProcess:
    def __init__(self, *args, **kwargs):
        self.running = True
        self.terminated = False

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.running = False

    def wait(self, **kwargs):
        self.running = False
        return 0


class FakeDocker:
    def __init__(self, adapter):
        self.adapter, self.calls = adapter, []
        self.current, self.stop_code = None, 0
        self.logs = "n_ctx_per_seq = 65536\nn_seq_max = 1"
        self.tokenizer_output = "warning from tokenizer\nLOCALBENCH_TOKENIZER_JSON:" + json.dumps({
            "count": 3, "tokens": [10, 11, 12], "provenance": "mock exact tokenizer"})
        self.tokenizer_code = 0

    def __call__(self, label, args, timeout=30, stdin_json=None):
        self.calls.append((label, list(args), timeout, copy.deepcopy(stdin_json)))
        output, code = "", 0
        if label in ("image-inspect", "recovery-image-inspect"):
            output = json.dumps({"id": IMAGE, "digests": [PIN],
                "os": "linux", "architecture": "amd64",
                "labels": {"org.opencontainers.image.revision": "mock-source-revision"},
                "entrypoint": ["/app/llama-server"]})
        elif label == "help":
            self.current.update(started="help-started", running=False)
            output = FLAGS
        elif label in ("create", "help-create"):
            labels = {args[index + 1].split("=", 1)[0]:
                      args[index + 1].split("=", 1)[1]
                      for index, arg in enumerate(args) if arg == "--label"}
            self.current = {"id": CID, "created": "created-1", "started": "not-started",
                "running": False, "exit_code": 0, "oom_killed": False,
                "image": IMAGE, "labels": labels}
            output = CID
        elif label in ("container-inspect", "recovery-container-inspect"):
            if self.current is None:
                code = 1
                output = "Error: No such object: " + args[-1]
            else:
                output = json.dumps(self.current)
        elif label == "start":
            self.current.update(started="started-1", running=True)
            output = CID
        elif label in ("stop-owned", "help-stop-owned", "recovery-stop-owned"):
            code = self.stop_code
            if code == 0:
                self.current["running"] = False
        elif label in ("remove-owned", "help-remove-owned", "recovery-remove-owned"):
            self.current = None
        elif label in ("startup-logs", "startup-failure-logs"):
            output = self.logs
        elif label == "exact-tokenize":
            output, code = self.tokenizer_output, self.tokenizer_code
        else:
            raise AssertionError("unexpected mock Docker operation: " + label)
        return {"argv": ["docker"] + list(args), "output": output,
                "exit_code": code, "log": str(Path(self.adapter.config["paths"]["logs"])/(label + ".log"))}


class ContainerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ("installed", "new", "runtime", "artifacts", "logs", "state"):
            (self.root/name).mkdir()
        self.config = {
            "paths": {"project": str(self.root),
                "installed_model_root": str(self.root/"installed"),
                "new_model_root": str(self.root/"new"),
                "runtime_root": str(self.root/"runtime"),
                "artifacts": str(self.root/"artifacts"),
                "logs": str(self.root/"logs"), "state": str(self.root/"state")},
            "installed_tools": {"docker": "DO-NOT-EXECUTE-docker"},
            "private_ports": {"ik": 38202, "vllm": 38205, "sglang": 38206},
            "limits": {"per_64k_request_timeout_seconds": 30,
                       "server_start_timeout_seconds": 1},
        }
        self.state = FakeState()
        gguf = self.root/"installed"/"mock-Q4_K_M.gguf"
        gguf.write_bytes(b"inert unit-test artifact; no actual model")
        self.gguf = {"id": "mock-gguf", "path": str(gguf), "sha256": file_hash(gguf)}
        snapshot = self.root/"new"/"mock-snapshot"
        snapshot.mkdir()
        (snapshot/"config.json").write_text("{}", encoding="utf-8")
        (snapshot/"tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot/"model.safetensors").write_bytes(b"inert artifact")
        self.snapshot = {"id": "mock-snapshot", "path": str(snapshot),
            "repo": "mock/repo", "revision": "f"*40,
            "files": [{"filename": path.name, "bytes": path.stat().st_size,
                       "sha256": file_hash(path)} for path in sorted(snapshot.iterdir())]}

    def tearDown(self):
        self.temp.cleanup()

    def adapter(self, engine="ik-llama", settings=None, image=PIN):
        return ContainerAdapter(self.config, self.state, engine,
            self.gguf if engine == "ik-llama" else self.snapshot,
            image_digest=image, settings=settings)

    def prepared(self, engine="ik-llama", settings=None):
        adapter = self.adapter(engine, settings)
        docker = FakeDocker(adapter)
        adapter._docker = docker
        adapter.discover_capabilities()
        adapter.health = Mock(return_value=True)
        adapter.http = Mock()
        if engine == "ik-llama":
            adapter.http.json.return_value = {"total_slots": 1, "default_generation_settings": {"n_ctx": 65536}}
        elif engine == "vllm":
            adapter.http.json.return_value = {"data": [{"id": self.snapshot["id"], "max_model_len": 65536}]}
            docker.logs = "max_num_seqs=1 max_model_len=65536"
        else:
            adapter.http.json.return_value = {"context_length": 65536, "max_running_requests": 1}
        return adapter, docker

    def launch_mock(self, adapter):
        with patch("localbench.adapters.container.check_port"), patch(
                "localbench.adapters.container.subprocess.Popen", FakeLogProcess):
            return adapter.launch()

    def owned(self, adapter, docker):
        labels = {"localbench.owner": "localbench",
                  "localbench.run": self.state.run["id"], "localbench.job": "owned-job"}
        docker.current = {"id": CID, "created": "created-1", "started": "started-1",
            "running": True, "exit_code": 0, "oom_killed": False, "image": IMAGE, "labels": labels}
        adapter.saved = {"owner": "localbench", "run_id": self.state.run["id"],
            "job_id": "owned-job", "container_id": CID,
            **{key: docker.current[key] for key in ("id", "created", "started", "image")}}
        adapter.process = ContainerHandle(adapter, CID)

    def test_launch_argv_is_private_bounded_unprivileged_and_model_readonly(self):
        for engine in ("ik-llama", "vllm", "sglang"):
            with self.subTest(engine=engine):
                adapter, docker = self.prepared(engine)
                runtime = self.root/"runtime"/("job-"+engine)
                runtime.mkdir()
                helper = adapter._tokenizer_script() if engine != "ik-llama" else None
                args = adapter.build_launch_argv("localbench-mock-"+engine, "job-one", runtime, helper)
                values = lambda option: [args[i+1] for i, item in enumerate(args) if item == option]
                self.assertEqual(values("--publish"), [f"127.0.0.1:{adapter.port}:{adapter.profile['container_port']}"])
                self.assertEqual(values("--gpus"), ["device=0"])
                self.assertEqual(values("--memory"), ["28g"])
                self.assertEqual(values("--shm-size"), ["8g"])
                self.assertEqual(values("--user"), ["1000:1000"])
                self.assertEqual(values("--cap-drop"), ["ALL"])
                self.assertEqual(values("--security-opt"), ["no-new-privileges"])
                self.assertEqual(values("--pull"), ["never"])
                self.assertIn("--read-only", args)
                mounts = values("--mount")
                self.assertTrue(any(f"source={adapter.model_path}," in value and value.endswith(",readonly") for value in mounts))
                self.assertTrue(any("target=/runtime" in value and not value.endswith("readonly") for value in mounts))
                if helper:
                    self.assertTrue(any("target=/bench-tokenizer/helper.py,readonly" in value for value in mounts))
                self.assertFalse(any("docker.sock" in value or "auth" in value.lower() for value in mounts))
                self.assertNotIn("--privileged", args)
                self.assertNotIn("--ipc", args)
                self.assertNotIn("--network=host", args)
                self.assertIn(PIN, args)
                self.assertNotIn("--trust-remote-code", args)

    def test_help_discovery_pins_identity_and_uses_cpu_without_model_or_credentials(self):
        adapter, docker = self.prepared("vllm")
        help_args = next(call[1] for call in docker.calls if call[0] == "help-create")
        self.assertNotIn("--gpus", help_args)
        self.assertNotIn("--mount", help_args)
        self.assertIn("none", help_args)
        self.assertIn("--help=all", help_args)
        self.assertEqual(next(call[1] for call in docker.calls if call[0] == "help-remove-owned"), ["rm", CID])
        self.assertEqual(self.state.starts, ["runtime_probe:vllm"])
        metadata = adapter.metadata()
        self.assertEqual(metadata["image_digest"], PIN)
        self.assertEqual(metadata["image_id"], IMAGE)
        self.assertEqual(metadata["model_identity_kind"], "pinned_checkpoint_file_manifest")
        self.assertNotEqual(metadata["model_sha256"], self.snapshot["files"][0]["sha256"])
        self.assertTrue(metadata["tokenizer_file_pins"])
        self.assertNotIn("binary_sha256", metadata)
        self.assertNotIn(".Config.Env", IMAGE_FORMAT + CONTAINER_FORMAT)

    def test_discovery_requires_all_acceptance_gates_before_any_docker_call(self):
        for gate in self.state.controls:
            with self.subTest(gate=gate):
                state = FakeState()
                state.controls[gate] = False
                adapter = ContainerAdapter(self.config, state, "ik-llama", self.gguf, PIN)
                adapter._docker = Mock()
                with self.assertRaisesRegex(RuntimeError, "requires_verified"):
                    adapter.discover_capabilities()
                adapter._docker.assert_not_called()
                self.assertEqual(state.starts, [])

    def test_installed_tag_resolves_to_repo_digest_without_pull(self):
        adapter = self.adapter(image=None)
        docker = FakeDocker(adapter)
        adapter._docker = docker
        adapter.discover_capabilities()
        self.assertEqual(adapter.image_digest, PIN)
        self.assertEqual(docker.calls[0][1][-1], adapter.profile["tag"])
        self.assertFalse(any("pull" == call[0] for call in docker.calls))

    def test_unsupported_settings_are_explicit_and_ik_xl_is_rejected(self):
        cases = [
            ("ik-llama", {"imaginary_flag": True}, RuntimeError),
            ("ik-llama", {"cache_k": "q3_0"}, ValueError),
            ("ik-llama", {"prefix_cache": False}, RuntimeError),
            ("ik-llama", {"slots": True}, ValueError),
            ("vllm", {"attention_backend": "FLASH_ATTN_3"}, RuntimeError),
            ("vllm", {"reasoning": "disabled"}, RuntimeError),
            ("sglang", {"memory_gib": 32}, ValueError),
        ]
        for engine, settings, error in cases:
            with self.subTest(engine=engine, settings=settings), self.assertRaises(error):
                self.adapter(engine, settings)
        xl = self.root/"installed"/"mock-Q4_K_M_XL.gguf"
        xl.write_bytes(b"inert")
        with self.assertRaisesRegex(RuntimeError, "plain"):
            ContainerAdapter(self.config, self.state, "ik-llama", {"id": "xl", "path": str(xl)}, PIN)

    def test_missing_or_nonstring_indexed_shard_is_rejected(self):
        index = Path(self.snapshot["path"])/"model.safetensors.index.json"
        for filename in ("missing.safetensors", ["invalid"]):
            with self.subTest(filename=filename):
                index.write_text(json.dumps({"weight_map": {"weight": filename}}), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "incomplete_local"):
                    self.adapter("vllm")

    def test_flags_are_checked_against_exact_help(self):
        adapter, _ = self.prepared()
        adapter.help = adapter.help.replace("--no-context-shift", "")
        with self.assertRaisesRegex(RuntimeError, "unsupported_exact_image_flag:--no-context-shift"):
            adapter._engine_args()
        adapter, _ = self.prepared("sglang", {"prefix_cache": False})
        self.assertIn("--disable-radix-cache", adapter._engine_args())
        adapter.help = adapter.help.replace("--disable-radix-cache", "")
        with self.assertRaisesRegex(RuntimeError, "unsupported_exact_image_flag"):
            adapter._engine_args()

    def test_occupied_port_prevents_creation_and_does_not_claim_listener(self):
        adapter, docker = self.prepared()
        docker.calls.clear()
        with patch("localbench.adapters.container.check_port", side_effect=RuntimeError("occupied_port")):
            with self.assertRaisesRegex(RuntimeError, "occupied_port"):
                adapter.launch()
        self.assertEqual(docker.calls, [])
        self.assertIsNone(adapter.saved)
        self.assertFalse((self.root/"runtime"/"containers").exists())

    def test_owned_container_handle_is_full_identity_and_no_configuration_entity_is_written(self):
        adapter, docker = self.prepared()
        result = self.launch_mock(adapter)
        self.assertEqual(adapter.process.pid, CID)
        self.assertIsNone(adapter.process.poll())
        self.assertEqual(result["effective"]["effective_context"], 65536)
        self.assertEqual(result["effective"]["effective_slots"], 1)
        self.assertEqual(result["effective"]["container_gpu_ownership"]["windows_pid_attribution"], "unverified")
        self.assertFalse(any(kind == "configuration" for kind, _, _ in self.state.entities))
        adapter.unload_owned()
        mutations = [call for call in docker.calls if call[0] in ("stop-owned", "remove-owned")]
        self.assertEqual([call[1][-1] for call in mutations], [CID, CID])
        self.assertIsNone(adapter.saved)

    def test_foreign_labels_or_identity_changes_refuse_all_mutations(self):
        changes = [
            {"id": "d"*64}, {"created": "created-other"}, {"started": "restarted-other"},
            {"image": "sha256:"+"e"*64},
            {"labels": {"localbench.owner": "unrelated", "localbench.run": "mock-run", "localbench.job": "owned-job"}},
        ]
        for change in changes:
            with self.subTest(change=change):
                adapter, docker = self.prepared()
                self.owned(adapter, docker)
                docker.current.update(change)
                docker.calls.clear()
                with self.assertRaisesRegex(RuntimeError, "stale_container_identity"):
                    adapter.unload_owned()
                self.assertFalse(any(call[0] in ("stop-owned", "remove-owned") for call in docker.calls))
                self.assertIsNotNone(adapter.saved)

    def test_owned_stop_failure_retains_identity_and_never_removes(self):
        adapter, docker = self.prepared()
        self.owned(adapter, docker)
        docker.stop_code = 1
        with self.assertRaisesRegex(RuntimeError, "identity_retained"):
            adapter.unload_owned()
        self.assertIsNotNone(adapter.saved)
        self.assertFalse(any(call[0] == "remove-owned" for call in docker.calls))

    def test_insufficient_context_and_malformed_startup_api_clean_up_owned_container(self):
        for response in ({"total_slots": 1, "n_ctx": 32768}, None):
            with self.subTest(response=response):
                adapter, docker = self.prepared()
                adapter.http.json.return_value = response
                docker.logs = ""
                with self.assertRaises((RuntimeError, AttributeError)):
                    self.launch_mock(adapter)
                self.assertIsNone(adapter.saved)
                self.assertTrue(any(call[0] == "remove-owned" for call in docker.calls))

    def test_effective_window_and_slots_must_be_observed(self):
        for engine in ("vllm", "sglang"):
            with self.subTest(engine=engine):
                adapter, docker = self.prepared(engine)
                if engine == "vllm":
                    docker.logs = ""
                else:
                    adapter.http.json.return_value = {"context_length": 65536}
                with self.assertRaisesRegex(RuntimeError, "unverified_or_insufficient"):
                    self.launch_mock(adapter)
                self.assertIsNone(adapter.saved)

    def test_container_tokenizer_is_cpu_stdin_json_and_preserves_warning_logs(self):
        adapter, docker = self.prepared("vllm")
        self.owned(adapter, docker)
        messages = [{"role": "user", "content": "write a function"}]
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
        result = adapter.tokenize(messages, tools)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["tokens"], [10, 11, 12])
        call = next(call for call in docker.calls if call[0] == "exact-tokenize")
        self.assertIn("CUDA_VISIBLE_DEVICES=", call[1])
        self.assertIn("--interactive", call[1])
        self.assertIn(CID, call[1])
        self.assertEqual(call[3], {"operation": "chat", "messages": messages, "tools": tools})
        self.assertEqual(result["tokenizer_image_digest"], PIN)
        self.assertEqual(result["tokenizer_helper_sha256"], hashlib.sha256(TOKENIZER_HELPER.encode()).hexdigest())
        self.assertIn("actual engine input usage", result["count_cross_check"])
        self.assertIn("trust_remote_code=False", TOKENIZER_HELPER)
        self.assertIn("local_files_only=True", TOKENIZER_HELPER)
        self.assertEqual(adapter.tokenize_text("abc"), [10, 11, 12])

    def test_container_tokenizer_rejects_missing_duplicate_marker_and_invalid_tokens(self):
        for output in ("{}", "LOCALBENCH_TOKENIZER_JSON:{}\nLOCALBENCH_TOKENIZER_JSON:{}",
                       'LOCALBENCH_TOKENIZER_JSON:{"count":1,"tokens":[true]}',
                       'LOCALBENCH_TOKENIZER_JSON:{"count":2,"tokens":[10]}'):
            with self.subTest(output=output):
                adapter, docker = self.prepared("sglang")
                self.owned(adapter, docker)
                docker.tokenizer_output = output
                with self.assertRaises(RuntimeError):
                    adapter.tokenize([], [])

    def test_tokenizer_helper_changed_on_disk_is_rejected(self):
        adapter, _ = self.prepared("vllm")
        path = adapter._tokenizer_script()
        path.write_text("changed trusted helper", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "helper_changed"):
            adapter._tokenizer_script()

    def test_ik_template_requires_all_tool_schemas_and_exact_integer_token_ids(self):
        adapter, docker = self.prepared()
        self.owned(adapter, docker)
        tools = [{"function": {"name": "read_file"}}, {"function": {"name": "apply_patch"}}]
        adapter.http.json.side_effect = [{"prompt": "read_file only"}]
        with self.assertRaisesRegex(RuntimeError, "exploratory_only"):
            adapter.tokenize([], tools)
        adapter.http.json.side_effect = [{"prompt": "read_file apply_patch"},
                                        {"tokens": [1, 2, 3]}]
        result = adapter.tokenize([], tools)
        self.assertEqual(result["count"], 3)
        self.assertEqual(adapter.http.json.call_args[0][1]["parse_special"], True)

    def test_request_controls_are_explicit_and_sampler_alias_is_recorded(self):
        adapter, docker = self.prepared("vllm")
        self.owned(adapter, docker)
        with self.assertRaisesRegex(RuntimeError, "unsupported_container_request_controls"):
            adapter.stream_chat({"messages": [], "invented_sampler": 1})
        adapter.http.measure.assert_not_called()
        adapter.stream_chat({"messages": [], "model": "wrong", "repeat_penalty": 1.05})
        payload = adapter.http.measure.call_args[0][0]
        self.assertEqual(payload["model"], self.snapshot["id"])
        self.assertEqual(payload["repetition_penalty"], 1.05)
        self.assertNotIn("repeat_penalty", payload)
        controls = adapter.metadata()["last_request_controls"]
        self.assertEqual(controls["requested"]["repeat_penalty"], 1.05)
        self.assertEqual(controls["emitted"]["repetition_penalty"], 1.05)
        self.assertIn("unverified", controls["effective"])

    def test_mount_path_rejects_root_personal_escape_and_literal_commas(self):
        for path in (self.root/"installed", self.root/"personal"/"file",
                     self.root/"installed"/"unsafe,name.gguf"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                _mount_path(path, [self.root/"installed"])

    def test_linux_health_requires_expected_served_model(self):
        adapter = self.adapter("vllm")
        probe = Mock()
        with patch("localbench.adapters.container.LocalHTTP", return_value=probe):
            probe.json.return_value = {"data": [{"id": "unrelated-model"}]}
            self.assertFalse(adapter.health())
            probe.json.return_value = {"data": [{"id": self.snapshot["id"]}]}
            self.assertTrue(adapter.health())

    def test_known_oom_and_kernel_failures_are_nontransient(self):
        for output, expected in [
            ("CUDA out of memory", "model_oom"),
            ("no kernel image is available", "unsupported_kernel_or_architecture"),
            ("Illegal instruction", "unsupported_cpu_instruction_set"),
            ("unrecognized arguments: --invented", "unsupported_engine_option"),
        ]:
            with self.subTest(output=output):
                self.assertEqual(classify_failure(output), {"reason": expected, "transient": False})
        self.assertEqual(classify_failure("", oom_killed=True)["reason"], "model_oom")

    def test_trusted_tokenizer_helper_is_valid_python_without_importing_checkpoint_code(self):
        compile(TOKENIZER_HELPER, "trusted-tokenizer-helper.py", "exec")

    def test_help_timeout_stops_only_original_owned_cpu_container(self):
        adapter = self.adapter()
        docker = FakeDocker(adapter)
        original = docker.__call__
        def timeout_help(label, args, **kwargs):
            if label == "help":
                docker.calls.append((label, list(args), kwargs.get("timeout", 30), None))
                docker.current.update(started="help-started", running=True)
                return {"output": "TimeoutExpired", "exit_code": None, "log": "help.log"}
            return original(label, args, **kwargs)
        adapter._docker = timeout_help
        with self.assertRaisesRegex(RuntimeError, "container_help_failed"):
            adapter.discover_capabilities()
        self.assertEqual(next(call[1] for call in docker.calls if call[0] == "help-stop-owned"), ["stop", "--time", "5", CID])
        self.assertEqual(next(call[1] for call in docker.calls if call[0] == "help-remove-owned"), ["rm", CID])
        self.assertIsNone(docker.current)

    def orphan(self):
        adapter, docker = self.prepared()
        self.launch_mock(adapter)
        adapter.close_logs()
        adapter.saved = adapter.process = None
        resumed = self.adapter()
        resumed._docker = docker
        docker.calls.clear()
        return resumed, docker

    def test_explicit_recovery_removes_only_original_full_container_identity_without_probe(self):
        adapter, docker = self.orphan()
        starts = list(self.state.starts)
        result = adapter.recover_owned()
        self.assertEqual(result["status"], "recovered")
        self.assertEqual(result["containers"][0]["container_id"], CID)
        self.assertEqual([call[1][-1] for call in docker.calls if call[0] in (
            "recovery-stop-owned", "recovery-remove-owned")], [CID, CID])
        self.assertEqual(self.state.starts, starts)
        self.assertIsNone(adapter.saved)
        record = json.loads((self.root/"state"/"owned-container-38202.json").read_text(encoding="utf-8"))
        self.assertTrue(record["recovered"])
        self.assertEqual(adapter.recover_owned()["status"], "no_owned_orphans")

    def test_explicit_recovery_refuses_reused_ids_labels_and_changed_startup_timestamps(self):
        for change in ({"created": "changed"}, {"started": "restart-after-crash"},
                       {"id": "d"*64}, {"labels": {"localbench.owner": "other"}},
                       {"image": "sha256:"+"e"*64}):
            with self.subTest(change=change):
                adapter, docker = self.orphan()
                docker.current.update(change)
                with self.assertRaisesRegex(RuntimeError, "stale_recovery"):
                    adapter.recover_owned()
                self.assertFalse(any(call[0] in ("recovery-stop-owned", "recovery-remove-owned") for call in docker.calls))
                # Reset this inert record for the next independent case.
                (self.root/"state"/"owned-container-38202.json").unlink()

    def test_explicit_recovery_refuses_configuration_or_handle_command_tampering(self):
        adapter, docker = self.orphan()
        adapter.requested["batch"] = 4096
        with self.assertRaisesRegex(RuntimeError, "configuration_mismatch"):
            adapter.recover_owned()
        self.assertFalse(any(call[0] == "recovery-stop-owned" for call in docker.calls))
        adapter.requested["batch"] = 2048
        path = self.root/"state"/"owned-container-38202.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["argv"].append("--unrelated")
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "unverified_recovery_handle"):
            adapter.recover_owned()
        self.assertFalse(any(call[0] == "recovery-stop-owned" for call in docker.calls))

    def test_explicit_recovery_cleans_bounded_original_help_orphan(self):
        adapter, docker = self.prepared()
        record_path = next((self.root/"state").glob("owned-help-help-*.json"))
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.pop("removed_at", None)
        record.pop("stopped", None)
        record_path.write_text(json.dumps(record), encoding="utf-8")
        docker.current = {"id": CID, "created": record["created"], "started": record["started"],
                          "running": True, "exit_code": 0, "oom_killed": False, "image": IMAGE,
                          "labels": {"localbench.owner": "localbench", "localbench.run": self.state.run["id"],
                                     "localbench.job": record["job_id"], "localbench.engine": "ik-llama",
                                     "localbench.config": record["configuration_fingerprint"]}}
        docker.calls.clear()
        self.assertEqual(adapter.recover_owned()["status"], "recovered")
        self.assertEqual(next(call[1] for call in docker.calls if call[0] == "recovery-stop-owned"), ["stop", "--time", "5", CID])
        self.assertEqual(next(call[2] for call in docker.calls if call[0] == "recovery-remove-owned"), 15)

    def test_explicit_recovery_refuses_unknown_image_identity_before_mutation(self):
        adapter, docker = self.orphan()
        def changed_image(label, args, **kwargs):
            result = docker(label, args, **kwargs)
            if label == "recovery-image-inspect":
                value = json.loads(result["output"])
                value["id"] = "sha256:"+"e"*64
                result["output"] = json.dumps(value)
            return result
        adapter._docker = changed_image
        with self.assertRaisesRegex(RuntimeError, "image_identity_mismatch"):
            adapter.recover_owned()
        self.assertFalse(any(call[0] in ("recovery-stop-owned", "recovery-remove-owned") for call in docker.calls))


if __name__ == "__main__":
    unittest.main()
