"""Mock-only image storage and Ada control tests: never invoke Docker or GPUs."""

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from localbench.acquisition import _disk_headroom, _runtime_headroom
from localbench.adapters.container import ContainerAdapter
from localbench.config import atomic_json, load
from localbench.runtime_build import _environment
from localbench.runtime_catalog import TAGS, acquire_image, select_manifest
from localbench.state import State


GIB = 1024 ** 3
PIN = "sha256:" + "a" * 64
ENGINE_FLAGS = """--max-model-len --max-num-seqs --gpu-memory-utilization --max-num-batched-tokens
--tool-call-parser --reasoning-parser --served-model-name --kv-cache-dtype --enable-auto-tool-choice
--enable-prefix-caching --no-enable-prefix-caching --host --port --attention-backend --moe-backend
--dtype --generation-config --enforce-eager --model-path --context-length --max-running-requests
--mem-fraction-static --chunked-prefill-size --max-total-tokens --disable-radix-cache
--moe-runner-backend --disable-cuda-graph
"""


class RuntimeCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ("project", "state", "logs", "artifacts", "runtime_root", "installed_model_root", "new_model_root"):
            self.config["paths"][name] = str(Path(self.temp.name) / name)
            Path(self.config["paths"][name]).mkdir(parents=True)
        self.config["acquisition_disk_roots"] = {"C": "mock-C", "F": "mock-F"}
        self.state = State(self.config, clock=lambda: 1000.)
        self.state.control("harness_verified", True)
        self.ledger = Path(self.config["paths"]["artifacts"]) / "runtime-container-storage.json"
        self.calls, self.pulls, self.disk_checks = [], [], []
        self.present, self.sizes = set(), {}
        self.pull_code = 0
        self.free = {"mock-C": 1024 * GIB, "mock-F": 1024 * GIB}
        self.size = 20 * GIB

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def image(self, engine="vllm"):
        return TAGS[engine].rsplit(":", 1)[0] + "@" + PIN

    def command(self, config, label, argv, **kwargs):
        self.calls.append((label, list(argv)))
        output, code = "", 0
        if label.endswith("-manifest"):
            output = json.dumps([{ "Descriptor": {"platform": {"os": "windows", "architecture": "amd64"}, "digest": "sha256:" + "b" * 64}},
                {"Descriptor": {"platform": {"os": "linux", "architecture": "amd64"}, "digest": PIN}}])
        elif label.endswith("-existing-image"):
            code = 0 if argv[-1] in self.present else 1
            output = "sha256:" + "c" * 64 if code == 0 else "not installed"
        elif label.endswith("-image-size"):
            output = str(self.sizes.get(argv[-1], self.size))
        else:
            raise AssertionError("unexpected tool request:" + label)
        return {"exit_code": code, "output": output, "log": label + ".log"}

    def pull(self, config, state, label, argv, **kwargs):
        self.pulls.append((label, list(argv), dict(kwargs)))
        storage = json.loads(self.ledger.read_text(encoding="utf-8"))
        self.assertGreater(storage["images"][argv[-1]]["accounted_bytes"], 0)
        self.assertEqual(storage["images"][argv[-1]]["status"], "reserved_pull")
        if self.pull_code == 0:
            self.present.add(argv[-1])
        return {"exit_code": self.pull_code, "output": "mock pull", "log": "mock-pull.log"}

    def disk_usage(self, path):
        return SimpleNamespace(free=self.free[path])

    def disk_headroom(self, config, remaining):
        self.disk_checks.append(remaining)
        return _disk_headroom(config, remaining, disk_usage=self.disk_usage)

    def mocks(self):
        return patch.multiple("localbench.runtime_catalog", command=self.command, run_owned=self.pull,
            _disk_headroom=self.disk_headroom, **{"shutil": SimpleNamespace(disk_usage=self.disk_usage)})

    def test_selects_only_valid_immutable_linux_amd64_manifest(self):
        wrong = {"platform": {"os": "linux", "architecture": "arm64"}, "digest": "sha256:" + "b" * 64}
        right = {"platform": {"os": "linux", "architecture": "amd64"}, "digest": PIN}
        self.assertEqual(select_manifest({"manifests": [wrong, right]}), PIN)
        self.assertEqual(select_manifest([{"Descriptor": wrong}, {"Descriptor": right}]), PIN)
        self.assertEqual(select_manifest({"descriptor": right}), PIN)
        for payload in ({"manifests": [wrong]}, {"Descriptor": {**right, "digest": "latest"}},
                {"manifests": [{**right, "digest": "sha256:" + "z" * 64}]}):
            with self.subTest(payload=payload), self.assertRaisesRegex(RuntimeError, "linux_amd64_manifest_not_established"):
                select_manifest(payload)

    def test_reuses_installed_pinned_image_without_pull_and_accounts_full_size(self):
        image = self.image()
        self.present.add(image)
        with self.mocks():
            result = acquire_image(self.config, self.state, "vllm")
        self.assertEqual(self.pulls, [])
        self.assertEqual(result["image_digest"], image)
        self.assertIsNone(self.state.run["started"])
        storage = json.loads(self.ledger.read_text(encoding="utf-8"))
        self.assertEqual(storage["images"][image]["accounted_bytes"], self.size)
        self.assertEqual(storage["images"][image]["status"], "acquired")

    def test_pull_reserves_before_bytes_uses_digest_and_records_full_uncompressed_size(self):
        with self.mocks():
            result = acquire_image(self.config, self.state, "vllm")
        self.assertEqual(len(self.pulls), 1)
        label, argv, kwargs = self.pulls[0]
        self.assertEqual(argv[1:4], ["pull", "--platform", "linux/amd64"])
        self.assertEqual(argv[-1], self.image())
        self.assertEqual(kwargs["setup_deadline"], 1000. + 120 * 60)
        self.assertIn(24 * GIB, self.disk_checks)
        storage = json.loads(self.ledger.read_text(encoding="utf-8"))
        self.assertEqual(storage["images"][self.image()]["accounted_bytes"], self.size)
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], 0)
        self.assertEqual(result["platform"], "linux/amd64")

    def test_failed_partial_pull_keeps_reservation_and_resume_does_not_double_count(self):
        self.pull_code = 1
        with self.mocks():
            with self.assertRaisesRegex(RuntimeError, "pinned_image_pull_failed"):
                acquire_image(self.config, self.state, "vllm")
        first_run = self.state.run
        storage = json.loads(self.ledger.read_text(encoding="utf-8"))
        self.assertEqual(storage["images"][self.image()]["accounted_bytes"], 24 * GIB)
        self.assertEqual(storage["images"][self.image()]["status"], "reserved_pull")
        self.pull_code = 0
        self.calls.clear()
        self.disk_checks.clear()
        with self.mocks():
            acquire_image(self.config, self.state, "vllm")
        self.assertNotIn(24 * GIB, self.disk_checks)
        self.assertEqual(self.state.run["started"], first_run["started"])
        self.assertEqual(self.state.run["deadline"], first_run["deadline"])
        self.assertFalse(any(label.endswith("-manifest") for label, _ in self.calls))
        self.assertEqual(json.loads(self.ledger.read_text())["images"][self.image()]["accounted_bytes"], self.size)

    def test_shared_layers_are_conservatively_counted_as_full_image_sizes(self):
        self.present.update({self.image("vllm"), self.image("sglang")})
        atomic_json(self.ledger, {"images": {image: {"accounted_bytes": 24 * GIB, "status": "reserved_pull"}
            for image in self.present}})
        with self.mocks():
            acquire_image(self.config, self.state, "vllm")
            acquire_image(self.config, self.state, "sglang")
        storage = json.loads(self.ledger.read_text())
        self.assertEqual(sum(record["accounted_bytes"] for record in storage["images"].values()), 40 * GIB)
        self.assertEqual(self.pulls, [])

    def test_full_acquired_size_remains_accounted_if_post_pull_runtime_cap_fails(self):
        self.size = 30 * GIB
        self.config["limits"]["runtime_and_build_soft_cap_bytes"] = 26 * GIB
        with self.mocks():
            with self.assertRaisesRegex(RuntimeError, "runtime_soft_cap"):
                acquire_image(self.config, self.state, "vllm")
        self.assertEqual(len(self.pulls), 1)
        self.assertEqual(json.loads(self.ledger.read_text())["images"][self.image()]["accounted_bytes"], 30 * GIB)

    def test_c_and_f_headroom_and_runtime_ledger_block_before_pull(self):
        scenarios = [("mock-C", self.config["limits"]["minimum_free_c_bytes"] + 23 * GIB, "docker_storage_disk_headroom_c"),
            ("mock-F", self.config["limits"]["minimum_free_f_bytes"] + 23 * GIB, "disk_headroom_f")]
        for path, free, reason in scenarios:
            with self.subTest(path=path):
                self.free[path] = free
                with self.mocks():
                    with self.assertRaisesRegex(RuntimeError, reason):
                        acquire_image(self.config, self.state, "vllm")
                self.assertEqual(self.pulls, [])
                self.free[path] = 1024 * GIB
        atomic_json(self.ledger, {"images": {"another-pin": {"accounted_bytes": 90 * GIB, "status": "reserved_pull"}}})
        with self.mocks():
            with self.assertRaisesRegex(RuntimeError, "runtime_soft_cap"):
                acquire_image(self.config, self.state, "vllm")
        self.assertEqual(self.pulls, [])
        self.assertIsNone(self.state.run["started"])

    def test_runtime_headroom_includes_reserved_container_ledger_and_local_build_bytes(self):
        self.config["limits"]["runtime_and_build_soft_cap_bytes"] = 100
        (Path(self.config["paths"]["runtime_root"]) / "owned-build.bin").write_bytes(b"x" * 10)
        atomic_json(self.ledger, {"images": {"first": {"accounted_bytes": 20, "status": "reserved_pull"}}})
        _runtime_headroom(self.config, 70)
        with self.assertRaisesRegex(RuntimeError, "runtime_soft_cap"):
            _runtime_headroom(self.config, 71)

    def test_source_and_pull_argv_never_include_process_credentials(self):
        with patch.dict(os.environ, {"LM_BENCH_TOKEN": "private-memory-only-test-key", "HF_TOKEN": "private-hf-key"}):
            environment = _environment()
            self.assertNotIn("LM_BENCH_TOKEN", environment)
            self.assertNotIn("HF_TOKEN", environment)
            with self.mocks():
                acquire_image(self.config, self.state, "vllm")
        text = json.dumps(self.calls + [(label, argv) for label, argv, _ in self.pulls])
        self.assertNotIn("private-memory-only-test-key", text)
        self.assertNotIn("private-hf-key", text)
        for _, argv in self.calls:
            self.assertTrue(argv[-1] == self.image() or argv[-1] == TAGS["vllm"])

    def gptoss(self, engine="vllm", settings=None):
        root = Path(self.config["paths"]["installed_model_root"]) / "gptoss-owned-mock"
        root.mkdir(exist_ok=True)
        atomic_json(root / "config.json", {"model_type": "gpt_oss", "max_position_embeddings": 131072,
            "quantization_config": {"quant_method": "mxfp4"}})
        (root / "model.safetensors").write_bytes(b"mock; no real weights")
        model = {"id": "gptoss20b-mxfp4", "path": str(root), "revision": "a" * 40}
        return ContainerAdapter(self.config, self.state, engine, model,
            image_digest="example.invalid/engine@" + PIN, settings=settings)

    def test_gptoss_ada_defaults_and_exact_required_help_flags(self):
        for engine, attention, moe in (("vllm", "TRITON_ATTN", "marlin"), ("sglang", "triton", "triton_kernel")):
            with self.subTest(engine=engine):
                adapter = self.gptoss(engine)
                adapter.help = ENGINE_FLAGS
                argv = adapter._engine_args()
                self.assertEqual(adapter.requested["attention_backend"], attention)
                self.assertEqual(adapter.requested["moe_backend"], moe)
                self.assertEqual(adapter.requested["dtype"], "bfloat16")
                self.assertTrue(adapter.requested["eager"])
                self.assertEqual(argv[argv.index("--attention-backend") + 1], attention)
                backend_flag = "--moe-backend" if engine == "vllm" else "--moe-runner-backend"
                self.assertEqual(argv[argv.index(backend_flag) + 1], moe)
                self.assertIn("--enforce-eager" if engine == "vllm" else "--disable-cuda-graph", argv)
                adapter.help = ENGINE_FLAGS.replace(backend_flag, backend_flag + "-unsupported")
                with self.assertRaisesRegex(RuntimeError, "unsupported_exact_image_flag"):
                    adapter._engine_args()
        self.assertIsNone(self.state.run["started"])

    def test_sglang_sm90_marlin_and_falsely_disabled_harmony_are_explicitly_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "mxfp4_marlin_requires_sm90"):
            self.gptoss("sglang", {"moe_backend": "marlin"})
        for engine in ("vllm", "sglang"):
            for profile in ("reduced", "disabled"):
                with self.subTest(engine=engine, profile=profile), self.assertRaisesRegex(RuntimeError, "unsupported_linux_numeric_or_disable"):
                    self.gptoss(engine, {"reasoning": profile})
        with self.assertRaisesRegex(RuntimeError, "requires_separately_verified_Ada"):
            self.gptoss("vllm", {"attention_backend": "FLASH_ATTN_3"})

    def test_harmony_low_effort_is_an_actual_effort_value_not_numeric_budget_or_disable(self):
        adapter = self.gptoss("vllm", {"reasoning_effort": "low"})
        adapter.http = MagicMock()
        with patch.object(adapter, "_inspect_owned", return_value={}):
            adapter.stream_chat({"messages": [{"role": "user", "content": "repair synthetic fixture"}],
                "model": "ignored", "repeat_penalty": 1.05}, log_prefix="unused-mock-log")
        payload = adapter.http.measure.call_args.args[0]
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(adapter.requested["reasoning"], "default")
        self.assertNotIn("reasoning_budget_tokens", payload)
        self.assertNotIn("enable_thinking", payload)
        self.assertEqual(payload["model"], "gptoss20b-mxfp4")
        self.assertEqual(payload["repetition_penalty"], 1.05)


if __name__ == "__main__":
    unittest.main()
