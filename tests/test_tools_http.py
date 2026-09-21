"""Permission, patch, exact-output and loopback HTTP controller-only tests."""

import http.server
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from localbench.adapters.base import LocalHTTP
from localbench.schema import ValidationError, validate_tool
from localbench.tools import ToolExecutor, apply_unified_patch


def token_ids(text):
    """Mock tokenizer with a defined exact UTF-8-byte vocabulary, not an estimate."""
    return list(text.encode("utf-8"))


def canonical(response):
    return json.dumps(response, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        (self.workspace / "src").mkdir()
        (self.workspace / "src/main.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
        (self.workspace / "src/errors.ts").write_text("readonly dependency\n", encoding="utf-8")
        self.fixture = {"source": {"src/main.py": "one\ntwo\nthree\n", "src/errors.ts": "readonly dependency\n"},
                        "writable": ["src/main.py"]}
        self.grader_calls = []
        def grader(workspace, phase):
            self.grader_calls.append((workspace, phase))
            return {"passed": True, "test_count": 2, "feedback": "public tests passed"}
        self.executor = ToolExecutor(self.workspace, self.fixture, grader, token_ids)

    def tearDown(self):
        self.temporary.cleanup()

    def test_source_only_read_and_list(self):
        self.assertEqual(self.executor.execute("list_files", {})["result"]["files"],
                         ["src/errors.ts", "src/main.py"])
        self.assertEqual(self.executor.execute("read_file", {"path": "src/main.py"})["result"]["content"],
                         "one\ntwo\nthree\n")
        for path in ("../secret.py", "src/../../secret.py", "C:/personal.py", "//server/share.py",
                     "src\\main.py", "tests/public.py", "grader/hidden.py", "src/unknown.py"):
            with self.subTest(path=path), self.assertRaises(ValidationError):
                self.executor.validate("read_file", {"path": path})
        for directory in (".", "..", "/src", "C:/src", "src/../grader", "tests", "grader"):
            with self.subTest(directory=directory), self.assertRaises(ValidationError):
                self.executor.validate("list_files", {"path": directory})

    def test_write_known_source_and_readonly_dependency(self):
        response = self.executor.execute("write_file", {"path": "src/main.py", "content": "replacement\n"})
        self.assertTrue(response["ok"])
        self.assertEqual((self.workspace / "src/main.py").read_text(), "replacement\n")
        with self.assertRaises(ValidationError):
            self.executor.validate("write_file", {"path": "src/errors.ts", "content": "changed"})
        self.assertEqual((self.workspace / "src/errors.ts").read_text(), "readonly dependency\n")
        dependency = ToolExecutor(self.workspace, {"source": self.fixture["source"]}, None)
        with self.assertRaises(ValidationError):
            dependency.validate("write_file", {"path": "src/errors.ts", "content": "changed"})

    def test_arbitrary_shell_rejected_and_only_public_callback_used(self):
        for name, args in (("shell", {"command": "bad"}), ("run_tests", {"command": "bad"}),
                           ("run_tests", {"phase": "held_out"}), ("run_tests", {"executable": "python"})):
            with self.subTest(name=name, args=args), self.assertRaises(ValidationError):
                self.executor.validate(name, args)
        response = self.executor.execute("run_tests", {})
        self.assertTrue(response["ok"])
        self.assertEqual(self.grader_calls, [(self.workspace.resolve(), "public")])
        self.assertEqual(response["result"]["test_count"], 2)
        unavailable = ToolExecutor(self.workspace, self.fixture, None)
        with self.assertRaises(ValidationError):
            unavailable.validate("run_tests", {})

    def test_patch_validates_context_before_permission_action(self):
        patch = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1,3 +1,3 @@\n one\n-two\n+updated\n three\n"
        self.executor.validate("apply_patch", {"path": "src/main.py", "patch": patch})
        self.assertEqual((self.workspace / "src/main.py").read_text(), "one\ntwo\nthree\n")
        response = self.executor.execute("apply_patch", {"path": "src/main.py", "patch": patch})
        self.assertTrue(response["ok"])
        self.assertEqual((self.workspace / "src/main.py").read_text(), "one\nupdated\nthree\n")
        with self.assertRaises(ValidationError):
            self.executor.validate("apply_patch", {"path": "src/main.py", "patch": patch})

    def test_patch_escape_multi_file_counts_and_drift_rejected(self):
        valid = "--- a/src/main.py\n+++ b/src/main.py\n@@ -2 +2 @@\n-two\n+ok\n"
        bad = [valid.replace("a/src/main.py", "a/../../personal.py"),
               valid + "--- a/src/errors.ts\n+++ b/src/errors.ts\n@@ -1 +1 @@\n-a\n+b\n",
               valid.replace("-two", "-different"), valid.replace("@@ -2 +2 @@", "@@ -2,2 +2 @@"),
               valid.replace("@@ -2 +2 @@", "@@ -2 +1 @@"),
               "*** Begin Patch\n*** Update File: src/main.py\n*** End Patch\n"]
        for patch in bad:
            with self.subTest(patch=patch), self.assertRaises(ValidationError):
                self.executor.validate("apply_patch", {"path": "src/main.py", "patch": patch})
        self.assertEqual((self.workspace / "src/main.py").read_text(), "one\ntwo\nthree\n")

    def test_patch_insert_delete_multiple_hunks_and_no_newline(self):
        self.assertEqual(apply_unified_patch("a\nb\nc\nd\n",
            "--- src/main.py\n+++ src/main.py\n@@ -1 +1 @@\n-a\n+A\n@@ -4 +4 @@\n-d\n+D\n", "src/main.py"),
            "A\nb\nc\nD\n")
        self.assertEqual(apply_unified_patch("a\n", "--- src/main.py\n+++ src/main.py\n@@ -0,0 +1 @@\n+start\n", "src/main.py"),
                         "start\na\n")
        self.assertEqual(apply_unified_patch("a\n", "--- src/main.py\n+++ src/main.py\n@@ -1 +0,0 @@\n-a\n", "src/main.py"), "")
        self.assertEqual(apply_unified_patch("a", "--- src/main.py\n+++ src/main.py\n@@ -1 +1 @@\n-a\n\\ No newline at end of file\n+b\n\\ No newline at end of file\n", "src/main.py"), "b")

    def test_long_final_action_only_once_and_exact_answer_schema(self):
        fixture = dict(self.fixture, permitted_final_action="write_file", writable=["src/main.py"])
        executor = ToolExecutor(self.workspace, fixture, None, token_ids)
        with self.assertRaises(ValidationError):
            executor.validate("read_file", {"path": "src/main.py"})
        self.assertTrue(executor.execute("write_file", {"path": "src/main.py", "content": "fixed"})["ok"])
        with self.assertRaises(ValidationError):
            executor.validate("write_file", {"path": "src/main.py", "content": "again"})
        answer_fixture = {"source": {}, "permitted_final_action": "submit_answer", "answer_schema": {
            "type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"],
            "additionalProperties": False}}
        answer = ToolExecutor(self.workspace, answer_fixture, None, token_ids)
        with self.assertRaises(ValidationError):
            answer.validate("submit_answer", {"answer": {"count": True}, "evidence_ids": ["id"]})
        self.assertTrue(answer.execute("submit_answer", {"answer": {"count": 1}, "evidence_ids": ["id"]})["ok"])
        self.assertEqual(answer.submitted_answer, {"answer": {"count": 1}, "evidence_ids": ["id"]})

    def test_exact_token_cap_truncates_with_explicit_notice(self):
        (self.workspace / "src/main.py").write_text("very large 🧪 output " * 1000, encoding="utf-8")
        executor = ToolExecutor(self.workspace, self.fixture, None, token_ids, maximum_output_tokens=512)
        response = executor.execute("read_file", {"path": "src/main.py"})
        self.assertTrue(response["ok"])
        self.assertTrue(response["truncated"])
        self.assertTrue(response["token_count_verified"])
        self.assertIn("truncated", response["feedback"])
        self.assertEqual(response["output_token_count"], len(token_ids(canonical(response))))
        self.assertLessEqual(response["output_token_count"], 512)

    def test_absent_exact_tokenizer_reports_unverified_and_withholds_large_output(self):
        executor = ToolExecutor(self.workspace, self.fixture, None)
        response = executor.execute("read_file", {"path": "src/main.py"})
        self.assertFalse(response["token_count_verified"])
        self.assertIsNone(response["output_token_count"])
        self.assertIn("unverified", response["feedback"])
        (self.workspace / "src/main.py").write_text("x" * 70000)
        response = executor.execute("read_file", {"path": "src/main.py"})
        self.assertFalse(response["ok"])
        self.assertIsNone(response["result"])

    def test_source_symlink_escape_rejected(self):
        outside = self.workspace / "outside.py"
        outside.write_text("private", encoding="utf-8")
        target = self.workspace / "src/main.py"
        target.unlink()
        try:
            os.symlink(outside, target)
        except OSError:
            # Windows may forbid creating links without extra authority. Exercise
            # the same link guard with filesystem metadata mocked instead of
            # requesting that authority or leaving the safety assertion skipped.
            target.write_text("safe fixture", encoding="utf-8")
            original = Path.is_symlink
            with mock.patch.object(Path, "is_symlink", lambda item: item == target or original(item)):
                with self.assertRaises(ValidationError):
                    self.executor.validate("read_file", {"path": "src/main.py"})
            self.assertEqual(outside.read_text(), "private")
            return
        with self.assertRaises(ValidationError):
            self.executor.validate("read_file", {"path": "src/main.py"})
        self.assertEqual(outside.read_text(), "private")

    def test_grader_error_feedback_does_not_echo_secrets(self):
        def failing_grader(workspace, phase):
            raise RuntimeError("pretend-credential do not echo")
        executor = ToolExecutor(self.workspace, self.fixture, failing_grader, token_ids)
        response = executor.execute("run_tests", {})
        self.assertFalse(response["ok"])
        self.assertNotIn("pretend-credential", canonical(response))


class FakeHTTPTests(unittest.TestCase):
    def _measure(self, raw, validator=validate_tool):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.received = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                # Real loopback HTTP writes deliberately split JSON/UTF-8/SSE.
                for index in range(0, len(raw), 3):
                    self.wfile.write(raw[index:index + 3])
                    self.wfile.flush()

            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                prefix = Path(temporary) / "fake"
                request = {"model": "mock", "messages": [{"role": "user", "content": "read source"}],
                           "stream": True}
                result = LocalHTTP("http://127.0.0.1:" + str(server.server_port), timeout=5).measure(
                    request, prefix, validator)
                self.assertEqual(server.received, request)
                self.assertEqual(Path(str(prefix) + "-raw.bin").read_bytes(), raw)
                stored = json.loads(Path(str(prefix) + "-result.json").read_text())
                self.assertEqual(stored["valid_stream"], result["valid_stream"])
                return result
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_real_streamed_http_tools_usage_and_timing(self):
        records = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "inspect 🧪"}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {
                "name": "read_file", "arguments": '{"path":'}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {
                "arguments": '"src/main.py"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 61440, "completion_tokens": 100,
             "prompt_tokens_details": {"cached_tokens": 0}},
             "timings": {"predicted_n": 100, "predicted_ms": 2500}},
        ]
        raw = ("".join("data: " + json.dumps(record, ensure_ascii=False) + "\n\n" for record in records)
               + "data: [DONE]\n\n").encode("utf-8")
        result = self._measure(raw)
        self.assertTrue(result["valid_stream"])
        self.assertEqual(result["calls"][0]["arguments"], {"path": "src/main.py"})
        metrics = result["metrics"]
        self.assertTrue(metrics["first_action_valid"])
        self.assertGreaterEqual(metrics["first_action_seconds"], metrics["first_stream_token_seconds"])
        self.assertGreaterEqual(metrics["wall_seconds"], metrics["first_action_seconds"])
        self.assertEqual(metrics["decode_tokens_per_second"], 40)
        self.assertEqual(metrics["prompt_tokens"], 61440)

    def test_real_http_partial_tool_closure_not_valid_zero_action(self):
        raw = b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file","arguments":"{\\\"path\\\":"}}]}}]}\n\n'
        result = self._measure(raw)
        self.assertFalse(result["valid_stream"])
        self.assertEqual(result["calls"], [])
        self.assertIsNone(result["metrics"]["first_action_seconds"])
        self.assertFalse(result["metrics"]["stream_complete"])

    def test_real_http_tail_after_done_invalidates_action(self):
        record = {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
            "name": "read_file", "arguments": '{"path":"src/main.py"}'}}]}}]}
        raw = ("data: " + json.dumps(record) + "\n\ndata: [DONE]\n\ndata: {}\n\n").encode()
        result = self._measure(raw)
        self.assertFalse(result["valid_stream"])
        self.assertEqual(result["calls"], [])
        self.assertFalse(result["metrics"]["first_action_valid"])


if __name__ == "__main__":
    unittest.main()
