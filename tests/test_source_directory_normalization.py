"""Controller-only source-directory and native-call regressions; no candidate execution."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from localbench.schema import ValidationError
from localbench.streams import StreamError, normalize_openai
from localbench.tools import ToolExecutor


def byte_vocabulary(text):
    """Defined fake exact tokenizer: each UTF-8 byte is one vocabulary token."""
    return list(text.encode("utf-8"))


def native_stream(name="list_files", arguments='{"path":"src/"}'):
    chunks = []
    for index, fragment in enumerate([arguments[:1], arguments[1:8], arguments[8:]]):
        function = {"arguments": fragment}
        tool = {"index": 0, "function": function}
        if index == 0:
            function["name"] = name
            tool.update({"id": "saved-native-call", "type": "function"})
        record = {"choices": [{"index": 0, "delta": {"tool_calls": [tool]}}]}
        chunks.append(("data: " + json.dumps(record) + "\n\n").encode())
    chunks.append(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n')
    chunks.append(b"data: [DONE]\n\n")
    return chunks


class SourceDirectoryNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.source = {"src/main.py": "fixture source\n", "src/pkg/tool.py": "nested source\n",
                       "src/pkg2/tool.ts": "other source\n"}
        for relative, text in self.source.items():
            target = self.workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        # On-disk contents outside the source map must not become visible.
        (self.workspace / "src/pkg/hidden.py").write_text("private\n", encoding="utf-8")
        self.executor = ToolExecutor(self.workspace, {"source": self.source}, None, byte_vocabulary)

    def test_single_root_separator_lists_identical_known_sources_and_preserves_arguments(self):
        args = {"path": "src/"}
        self.assertEqual(self.executor.validate("list_files", args), args)
        self.assertEqual(args, {"path": "src/"})
        results = [self.executor.execute("list_files", value) for value in ({}, {"path": "src"}, args)]
        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual([result["result"]["files"] for result in results], [sorted(self.source)] * 3)

    def test_nested_separator_preserves_source_map_and_directory_boundary(self):
        for path in ("src/pkg", "src/pkg/"):
            with self.subTest(path=path):
                response = self.executor.execute("list_files", {"path": path})
                self.assertTrue(response["ok"])
                self.assertEqual(response["result"]["files"], ["src/pkg/tool.py"])

    def test_unsafe_and_unknown_directories_are_rejected_without_repair(self):
        paths = ["/src/", "//server/share/", "C:/src/", "src\\", "src\\pkg/", "src:/",
                 "../src/", "./src/", "src/../src/", "src/./", "src//", "src//pkg/", "src/pkg//",
                 "src/pkg/../pkg/", "src/\0/", "src/unknown/", "tests/", "grader/", "src/main.py/"]
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(ValidationError):
                    self.executor.validate("list_files", {"path": path})
                self.assertFalse(self.executor.execute("list_files", {"path": path})["ok"])

    def test_source_symlink_metadata_is_rechecked_after_construction(self):
        target = self.workspace / "src/pkg"
        original = Path.is_symlink
        with mock.patch.object(Path, "is_symlink", lambda path: path == target or original(path)):
            for path in ("src/pkg", "src/pkg/", "src/"):
                with self.subTest(path=path), self.assertRaises(ValidationError):
                    self.executor.validate("list_files", {"path": path})

    def test_source_junction_metadata_is_rechecked_after_construction(self):
        target = self.workspace / "src/pkg"
        original = getattr(Path, "is_junction", lambda path: False)
        with mock.patch.object(Path, "is_junction", lambda path: path == target or original(path), create=True):
            with self.assertRaises(ValidationError):
                self.executor.validate("list_files", {"path": "src/pkg/"})

    def test_file_tools_keep_exact_source_path_permissions(self):
        calls = [("read_file", {"path": "src/main.py/"}),
                 ("write_file", {"path": "src/main.py/", "content": "changed"}),
                 ("apply_patch", {"path": "src/main.py/", "patch": "invalid patch"})]
        for name, args in calls:
            with self.subTest(name=name), self.assertRaises(ValidationError):
                self.executor.validate(name, args)
        self.assertEqual((self.workspace / "src/main.py").read_text(encoding="utf-8"), self.source["src/main.py"])

    def test_fragmented_native_call_is_validated_and_executes_same_source_listing(self):
        events = list(normalize_openai(native_stream(), validator=self.executor.validate))
        calls = [event for event in events if event["type"] == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "list_files")
        self.assertEqual(calls[0]["arguments"], {"path": "src/"})
        response = self.executor.execute(calls[0]["name"], calls[0]["arguments"])
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["files"], sorted(self.source))
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["finish_reason"], "tool_calls")

    def test_unpermitted_native_call_still_fails_and_plaintext_is_not_a_call(self):
        with self.assertRaises(StreamError):
            list(normalize_openai(native_stream("shell"), validator=self.executor.validate))
        text = 'list_files({"path":"src/"})'
        record = {"choices": [{"index": 0, "delta": {"content": text}}]}
        raw = ("data: " + json.dumps(record) + "\n\ndata: [DONE]\n\n").encode()
        events = list(normalize_openai([raw], validator=self.executor.validate))
        self.assertFalse(any(event["type"] == "tool_call" for event in events))


if __name__ == "__main__":
    unittest.main()
