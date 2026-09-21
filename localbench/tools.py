"""Controller-owned source tools; generated code is never executed here.

ToolExecutor accepts fixture dictionaries (or objects) with source, optional
writable, answer_schema, and permitted_final_action.  Its grader callback must
run public tests inside the separately configured CPU-only isolation container.
The exact tokenizer callback accepts text and returns a token count or token ids.
Token budgets are measured over the canonical JSON returned to the agent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

from .paths import safe_source
from .schema import ValidationError, validate_tool


def _field(fixture: Any, name: str, default=None):
    return fixture.get(name, default) if isinstance(fixture, dict) else getattr(fixture, name, default)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def serialize_tool_response(response: dict) -> str:
    """Use this exact serialization when packing tool output into chat context."""
    return _canonical(response)


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")


def apply_unified_patch(original: str, patch: str, path: str) -> str:
    """Apply one unified diff with exact context, positions and line counts.

    Only existing-file headers for the supplied relative path are permitted.
    There is no fuzzy matching, external command, or multi-file interpretation.
    """
    if not isinstance(patch, str) or not patch:
        raise ValidationError("patch must be a nonempty unified diff")
    raw_lines = patch.splitlines(keepends=True)
    lines = []
    for line in raw_lines:
        if line.rstrip("\r\n") == "\\ No newline at end of file":
            if not lines or not lines[-1].startswith((" ", "+", "-")) or lines[-1].startswith(("--- ", "+++ ")):
                raise ValidationError("misplaced no-newline marker")
            if not lines[-1].endswith(("\n", "\r")):
                raise ValidationError("duplicate no-newline marker")
            lines[-1] = lines[-1].rstrip("\r\n")
        else:
            # HTTP JSON strings normally contain LF. Normalizing patch line
            # endings does not authorize context drift in source text.
            lines.append(line.replace("\r\n", "\n"))
    position = 0
    if lines and lines[0].startswith("diff --git "):
        if lines[0].rstrip("\n") != f"diff --git a/{path} b/{path}":
            raise ValidationError("patch git header does not match target")
        position += 1
    if len(lines) < position + 3:
        raise ValidationError("patch requires two file headers and a hunk")
    before = lines[position].rstrip("\n")
    after = lines[position + 1].rstrip("\n")
    if before not in (f"--- {path}", f"--- a/{path}") or after not in (f"+++ {path}", f"+++ b/{path}"):
        raise ValidationError("patch file headers do not match target")
    position += 2
    original_lines = original.splitlines(keepends=True)
    output = []
    cursor = 0
    hunk_count = 0
    previous_start = None
    while position < len(lines):
        match = _HUNK.fullmatch(lines[position].rstrip("\n"))
        if match is None:
            raise ValidationError("unexpected patch data or multiple files")
        old_start, old_count, new_start, new_count = match.groups()
        old_start, new_start = int(old_start), int(new_start)
        old_count = 1 if old_count is None else int(old_count)
        new_count = 1 if new_count is None else int(new_count)
        if (old_count and old_start == 0) or (new_count and new_start == 0):
            raise ValidationError("nonempty hunk positions start at one")
        old_index = old_start if old_count == 0 else old_start - 1
        new_index = new_start if new_count == 0 else new_start - 1
        if old_index < cursor or old_index > len(original_lines) or (previous_start is not None and old_index <= previous_start):
            raise ValidationError("overlapping, unordered or out-of-range hunks")
        output.extend(original_lines[cursor:old_index])
        cursor = old_index
        if new_index != len(output):
            raise ValidationError("new hunk position disagrees with preceding patch")
        previous_start = old_index
        position += 1
        old_seen = new_seen = 0
        while position < len(lines) and not lines[position].startswith("@@ "):
            line = lines[position]
            if not line or line[0] not in " +-":
                raise ValidationError("invalid unified diff line")
            prefix, payload = line[0], line[1:]
            if prefix in " -":
                if cursor >= len(original_lines) or original_lines[cursor] != payload:
                    raise ValidationError("patch context does not match source")
                old_seen += 1
                cursor += 1
            if prefix in " +":
                output.append(payload)
                new_seen += 1
            if old_seen > old_count or new_seen > new_count:
                raise ValidationError("hunk contains extra data or multiple files")
            position += 1
        if old_seen != old_count or new_seen != new_count:
            raise ValidationError("hunk line counts do not match header")
        hunk_count += 1
    if not hunk_count:
        raise ValidationError("patch has no hunks")
    output.extend(original_lines[cursor:])
    return "".join(output)


class ToolExecutor:
    def __init__(self, workspace, fixture, grader: Callable | None,
                 tokenize_text: Callable | None = None, maximum_output_tokens: int = 2048):
        self.workspace = Path(workspace).resolve()
        self.fixture = fixture
        self.grader = grader
        self.tokenize_text = tokenize_text
        self.maximum_output_tokens = maximum_output_tokens
        if type(maximum_output_tokens) is not int or maximum_output_tokens < 1:
            raise ValueError("tool output cap must be a positive integer")
        source = _field(fixture, "source", {})
        if not isinstance(source, dict):
            raise ValueError("fixture requires an explicit source map")
        self.source = set(source)
        writable = _field(fixture, "writable", None)
        self.writable = set(source) if writable is None else set(writable)
        if not self.writable.issubset(self.source):
            raise ValueError("writable paths must belong to fixture source")
        # The long retry fixture dependency is always immutable, even if a
        # caller forgot to supply its narrower writable list.
        self.writable.discard("src/errors.ts")
        for path in self.source:
            safe_source(self.workspace, path)
        self.permitted_final_action = _field(fixture, "permitted_final_action", None)
        self.answer_schema = _field(fixture, "answer_schema", None)
        self.final_action_used = False
        self.submitted_answer = None
        self.last_grade_result = None

    def _path(self, relative: str, write: bool = False) -> Path:
        if relative not in self.source:
            raise ValidationError("path is not an agent-visible fixture source")
        try:
            return safe_source(self.workspace, relative, self.writable if write else self.source)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def _read(self, target: Path) -> str:
        if not target.is_file():
            raise ValidationError("source file does not exist")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise ValidationError("source file is not UTF-8 text") from exc

    def _directory(self, arguments: dict) -> str:
        directory = arguments.get("path", "src")
        components = directory.split("/")
        # A single final separator denotes the same relative directory. Check
        # raw components before normalization; repeated separators and escapes
        # must never be repaired into an authorized source path.
        if components[-1] == "":
            components = components[:-1]
        if (directory.startswith("/") or not components
                or any(part in ("", ".", "..") for part in components)
                or "\\" in directory or ":" in directory or "\0" in directory):
            raise ValidationError("invalid source directory")
        directory = "/".join(components)
        source_paths = [path for path in self.source if path.startswith(directory + "/")]
        if directory != "src" and not source_paths:
            raise ValidationError("only source directories may be listed")
        # Recheck selected source metadata at use time, including directory
        # symlinks/junctions created after the executor was constructed.
        for path in source_paths:
            self._path(path)
        return directory

    def validate(self, name: str, arguments: dict) -> dict:
        validated = validate_tool(name, arguments, answer_schema=self.answer_schema)
        allowed = _field(self.fixture, "allowed_tools", None)
        if name == "submit_answer" and self.answer_schema is None and self.permitted_final_action != "submit_answer" and (allowed is None or name not in allowed):
            raise ValidationError("submit_answer is not permitted for this fixture")
        if allowed is not None and name not in allowed:
            raise ValidationError("tool is not permitted for this fixture")
        if self.permitted_final_action is not None:
            if name != self.permitted_final_action:
                raise ValidationError("long fixture permits only its specified final action")
            if self.final_action_used:
                raise ValidationError("long fixture final action has already executed")
        if name in {"read_file", "write_file", "apply_patch"}:
            target = self._path(arguments["path"], name != "read_file")
            if name in {"read_file", "apply_patch"}:
                original = self._read(target)
            if name == "apply_patch":
                apply_unified_patch(original, arguments["patch"], arguments["path"])
        elif name == "list_files":
            self._directory(arguments)
        elif name == "run_tests" and not callable(self.grader):
            raise ValidationError("isolated public grader is unavailable")
        return validated

    def _write(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Existing source paths were checked immediately before this operation.
        descriptor, temporary = tempfile.mkstemp(prefix=".localbench-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()

    def _count(self, value: str) -> int:
        counted = self.tokenize_text(value)
        if type(counted) is int:
            if counted < 0:
                raise ValueError("tokenizer returned negative count")
            return counted
        if isinstance(counted, (list, tuple)):
            return len(counted)
        raise ValueError("exact tokenizer must return count or token ids")

    def _finish_count(self, response: dict) -> tuple[dict, int]:
        # Include token-count metadata itself in the agent-visible budget.
        previous = None
        for _ in range(12):
            count = self._count(_canonical(response))
            if count == response.get("output_token_count"):
                return response, count
            if previous == count:
                break
            previous = response.get("output_token_count")
            response["output_token_count"] = count
        # A tokenizer with a nonconvergent count metadata interaction is not a
        # valid proof of the cap. Fail closed rather than approximating it.
        raise ValueError("exact output count did not converge")

    def _cap(self, response: dict) -> dict:
        response.update({"truncated": False, "token_count_verified": self.tokenize_text is not None,
                         "output_token_count": None})
        if self.tokenize_text is None:
            response["feedback"] = "Exact tool-output token count unavailable; output budget unverified."
            if len(_canonical(response).encode("utf-8")) > 65536:
                response.update({"ok": False, "result": None, "error": "unverified_output_too_large",
                                 "feedback": "Output withheld: exact tokenizer unavailable and byte safety cap exceeded."})
            return response
        response, count = self._finish_count(response)
        if count <= self.maximum_output_tokens:
            return response
        original = _canonical(response["result"])
        notice = "[Tool output truncated at the verified token limit.]"
        response.update({"result": "", "truncated": True, "feedback": notice})
        response, count = self._finish_count(response)
        if count > self.maximum_output_tokens:
            raise ValidationError("tool output token limit cannot fit response metadata")
        best = dict(response)
        lower, upper = 0, len(original)
        while lower <= upper:
            middle = (lower + upper) // 2
            candidate = dict(response, result=original[:middle])
            candidate, count = self._finish_count(candidate)
            if count <= self.maximum_output_tokens:
                best = candidate
                lower = middle + 1
            else:
                upper = middle - 1
        # Tokenization need not be monotone in character length: accept only an
        # individually measured candidate, never a length-derived token count.
        if self._count(_canonical(best)) > self.maximum_output_tokens:
            raise ValidationError("truncated output exceeds exact token cap")
        return best

    def execute(self, name: str, arguments: dict) -> dict:
        try:
            self.validate(name, arguments)
            if name == "list_files":
                prefix = self._directory(arguments) + "/"
                result = {"files": sorted(path for path in self.source if path.startswith(prefix))}
            elif name == "read_file":
                result = {"path": arguments["path"], "content": self._read(self._path(arguments["path"]))}
            elif name == "write_file":
                target = self._path(arguments["path"], write=True)
                self._write(target, arguments["content"])
                result = {"path": arguments["path"], "bytes_written": len(arguments["content"].encode("utf-8"))}
            elif name == "apply_patch":
                target = self._path(arguments["path"], write=True)
                updated = apply_unified_patch(self._read(target), arguments["patch"], arguments["path"])
                self._write(target, updated)
                result = {"path": arguments["path"], "patched": True}
            elif name == "run_tests":
                result = self.grader(self.workspace, phase="public")
                self.last_grade_result = result
            else:
                self.submitted_answer = json.loads(_canonical(arguments))
                result = {"submitted": True}
            if self.permitted_final_action is not None:
                self.final_action_used = True
            return self._cap({"ok": True, "result": result, "feedback": ""})
        except Exception as exc:
            # Do not echo controller paths, arbitrary grader errors, or request
            # contents. Deterministic validation messages contain no credentials.
            feedback = str(exc) if isinstance(exc, ValidationError) else "Tool failed; inspect the controller-owned sanitized log."
            response = {"ok": False, "result": None, "error": type(exc).__name__, "feedback": feedback}
            try:
                return self._cap(response)
            except Exception:
                return {"ok": False, "result": None, "error": "output_count_unavailable",
                        "feedback": "Exact tool-output budget could not be verified.",
                        "output_token_count": None, "token_count_verified": False, "truncated": False}
