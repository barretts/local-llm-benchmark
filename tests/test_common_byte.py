import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from localbench.common_byte import common_byte, code_corpus, workload
from localbench.config import load
from localbench.process_attribution import classify_processes, parse_gpu_process_table
from localbench.report import Report
from localbench.state import State


class FakeAdapter:
    def __init__(self, factor=1, cached=0, owned=True, prompt_error=0, generated=128, decode=2., context=65536):
        self.factor, self.cached, self.owned = factor, cached, owned
        self.prompt_error, self.generated, self.decode = prompt_error, generated, decode
        self.process = SimpleNamespace(pid=100)
        self.saved = {"owner": "localbench" if owned else "unrelated", "created": 1000}
        self.effective = {"effective_context": context}
        self.http = SimpleNamespace(timeout=1)
        self.requests, self.restarts = [], 0

    def fresh_restart(self):
        self.restarts += 1
        if self.owned:
            self.process = SimpleNamespace(pid=100 + self.restarts)
            self.saved = {"owner": "localbench", "created": 1000 + self.restarts}
        return {"handle": self.saved}

    def tokenize(self, messages, tools):
        count = 20 + self.factor * sum(len(message["content"].split()) for message in messages) + len(tools) * 10
        return {"count": count, "provenance": "mock exact whole-template tokenizer factor=" + str(self.factor)}

    def stream_chat(self, request, **kwargs):
        self.requests.append(copy.deepcopy(request))
        warmup = "disjoint-warmup:" in request["messages"][0]["content"]
        tool_probe = bool(request.get("tools")) and not warmup
        count = self.tokenize(request["messages"], request.get("tools", []))["count"]
        content = "READY" if warmup else "" if tool_probe else "async def worker_pool(items):\n    return list(items)\n"
        calls = [{"type": "tool_call", "name": "read_file", "arguments": {"path": "src/cache.py"}}] if tool_probe else []
        if tool_probe:
            kwargs["validator"]("read_file", {"path": "src/cache.py"})
        return {"valid_stream": True, "content": content, "calls": calls,
            "events": [{"type": "content", "text": "mock chunk"}] * 500,
            "metrics": {"prompt_tokens": count if warmup else count + self.prompt_error,
                "generated_tokens": 8 if warmup else self.generated,
                "cached_tokens": 0 if warmup else self.cached,
                "decode_seconds": .1 if warmup else self.decode,
                "wall_seconds": 100. if warmup else 3., "first_stream_token_seconds": 50. if warmup else 1.,
                "first_action_valid": tool_probe, "first_action_seconds": 2. if tool_probe else None}}


class CommonByteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = load()
        for name in ("project", "state", "logs", "artifacts"):
            self.config["paths"][name] = str(Path(self.temp.name) / name)
        self.state = State(self.config)

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def run_sample(self, adapter, **changes):
        return common_byte(self.config, self.state, adapter, "configuration-test", line_count=32, **changes)

    def test_same_bytes_across_different_tokenizers_actual_counts_are_separate(self):
        first, second = FakeAdapter(1), FakeAdapter(2)
        one, two = self.run_sample(first), self.run_sample(second)
        self.assertTrue(one["passed"])
        self.assertTrue(two["passed"])
        self.assertEqual(one["common_byte_payload_sha256"], two["common_byte_payload_sha256"])
        self.assertEqual(one["common_byte_payload_bytes"], two["common_byte_payload_bytes"])
        self.assertNotEqual(one["actual_prompt_tokens"], two["actual_prompt_tokens"])
        self.assertEqual(first.requests[-1]["messages"], second.requests[-1]["messages"])
        self.assertEqual(first.requests[-1]["tools"], second.requests[-1]["tools"])
        self.assertEqual(one["actual_prompt_tokens"], one["expected_prompt_tokens"])
        self.assertEqual(two["actual_prompt_tokens"], two["expected_prompt_tokens"])
        self.assertFalse(one["qualifying_equal_token_measurement"])
        self.assertEqual(one["target_label"], "common_byte")
        self.assertTrue(Path(one["workload_artifact"]).is_file())
        self.assertIsNone(self.state.run["started"])

    def test_fresh_restart_disjoint_warmup_and_warm_timing_excluded(self):
        adapter = FakeAdapter()
        result = self.run_sample(adapter)
        self.assertEqual(adapter.restarts, 1)
        self.assertEqual(len(adapter.requests), 2)
        self.assertTrue(adapter.requests[0]["messages"][0]["content"].startswith("disjoint-warmup:"))
        self.assertTrue(adapter.requests[1]["messages"][0]["content"].startswith("common-byte-nonce:"))
        self.assertNotEqual(adapter.requests[0]["messages"][0]["content"], adapter.requests[1]["messages"][0]["content"])
        self.assertEqual(result["metrics"]["wall_seconds"], 3.)
        self.assertEqual(result["warmup_metrics"]["wall_seconds"], 100.)
        self.assertEqual(result["metrics"]["first_stream_token_seconds"], 1.)
        self.assertTrue(result["cache_isolation_verified"])
        self.assertTrue(result["warmup_excluded_from_metrics"])

    def test_cached_prefix_reuse_invalid_unknown_cache_requires_owned_restart(self):
        reused = self.run_sample(FakeAdapter(cached=65))
        self.assertFalse(reused["passed"])
        self.assertIn("fresh_prefix_cache_reuse", reused["reason"])
        self.assertTrue(self.run_sample(FakeAdapter(cached=None))["valid"])
        unknown = self.run_sample(FakeAdapter(cached=None, owned=False))
        self.assertFalse(unknown["valid"])
        self.assertIn("fresh_cache_isolation_unverified", unknown["reason"])

    def test_server_counters_drive_rates_not_stream_chunk_count(self):
        result = self.run_sample(FakeAdapter(generated=128, decode=2.))
        self.assertEqual(result["exact_sustained_decode_tokens_per_second"], 64.)
        self.assertTrue(result["sustained_throughput_qualified"])
        approximate = self.run_sample(FakeAdapter(decode=None))
        self.assertIsNone(approximate["exact_sustained_decode_tokens_per_second"])
        self.assertFalse(approximate["sustained_throughput_qualified"])
        short = self.run_sample(FakeAdapter(generated=32))
        self.assertTrue(short["valid"])
        self.assertFalse(short["sustained_throughput_qualified"])

    def test_actual_counts_and_output_budget_are_verified(self):
        mismatch = self.run_sample(FakeAdapter(prompt_error=65))
        self.assertFalse(mismatch["passed"])
        self.assertIn("actual_prompt_count_mismatch", mismatch["reason"])
        overflow = self.run_sample(FakeAdapter(generated=257))
        self.assertFalse(overflow["passed"])
        self.assertIn("generation_counter_or_budget", overflow["reason"])

    def test_fit_failure_preserves_same_bytes_and_never_sends_main_request(self):
        adapter = FakeAdapter(context=1024)
        observed = []
        result = common_byte(self.config, self.state, adapter, "configuration-test", line_count=256, on_packed=observed.append)
        self.assertFalse(result["request_sent"])
        self.assertEqual(result["reason"], "common_byte_does_not_fit_effective_context")
        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["prompt_hash"], result["common_byte_payload_sha256"])
        same = workload(line_count=256, run_id=self.state.run["id"])
        self.assertEqual(same["payload_sha256"], result["common_byte_payload_sha256"])

    def test_optional_native_tool_action_is_schema_checked_and_not_executed(self):
        result = self.run_sample(FakeAdapter(), request_kind="tool")
        self.assertTrue(result["first_tool_action_valid"])
        self.assertTrue(result["passed"])
        self.assertIn("no tool executed", result["tool_execution"])
        self.assertFalse(result["generated_source_executed"])

    def test_deterministic_line_count_utf8_and_replicate_identity(self):
        corpus = code_corpus(42, 73)
        self.assertEqual(corpus.count("\n"), 73)
        self.assertEqual(corpus, code_corpus(42, 73))
        self.assertNotEqual(corpus, code_corpus(43, 73))
        self.assertGreater(len(corpus.encode("utf-8")), len(corpus))
        self.assertNotEqual(workload(replicate=0)["payload_sha256"], workload(replicate=1)["payload_sha256"])

    def test_common_byte_pass_cannot_qualify_a_64k_winner(self):
        self.state.entity("configuration", "configuration-test", {"effective_settings": {"context": 65536}})
        result = self.run_sample(FakeAdapter())
        job = self.state.enqueue({"engine": "mock-engine", "model": "mock-weight", "effective_settings": {"context": 65536},
            "profile": "default", "fixture_prompt_hash": result["prompt_hash"], "tokenizer": "mock-exact",
            "seed": 42, "mode": "fresh", "replicate": 0})
        self.state.finish(self.state.begin(job), "passed", result)
        summary = Report(self.config, self.state).summarize()
        self.assertIsNone(summary["winner_id"])
        self.assertFalse(summary["configurations"][0]["qualified"])
        self.assertEqual(summary["configurations"][0]["latency"]["fresh"]["first_action"]["n"], 0)

    def test_typed_gpu_table_and_namespace_collision_never_imply_docker_ownership(self):
        table = """|    0   N/A  N/A       200      C+G   C:\\Apps\\desktop app.exe               N/A |
|    0   N/A  N/A       201        C   engine-worker                       512MiB |
|    0                 202        G   drawing-app                           1GiB |
|    0   N/A  N/A       203        C   C:\\Windows\\vmmemWSL.exe               N/A |
"""
        windows = parse_gpu_process_table(table)
        self.assertEqual([row["type"] for row in windows], ["C+G", "C", "G", "C"])
        self.assertIsNone(windows[0]["used_gpu_mib"])
        self.assertEqual(windows[2]["used_gpu_mib"], 1024.)
        classified = classify_processes(windows, owned={("linux_wsl_host", 201), ("windows", 203)})
        self.assertEqual(classified["owned"], [])
        self.assertEqual([row["pid"] for row in classified["aggregate_scope_ambiguous"]], [203])
        self.assertEqual(classified["active_foreign"], [])
        self.assertEqual([row["pid"] for row in classified["new_mixed_contexts"]], [200])
        active = classify_processes(windows, activity={("windows", 200): 25.})
        self.assertEqual([row["pid"] for row in active["active_foreign"]], [200])
        with self.assertRaisesRegex(ValueError, "never a bare integer"):
            classify_processes(windows, owned={201})


if __name__ == "__main__":
    unittest.main()
