"""Incremental SSE/NDJSON and coding-tool normalization, standard library only.

Input is an iterable of bytes from the HTTP response.  iter_sse yields decoded
object records and a {type: done} sentinel for [DONE].  Normalizers yield:
reasoning/content(text), tool_fragment(arguments text), tool_call(name and parsed
arguments), usage(normalized counters plus raw), and done.  Partial tool JSON is
never an action.  A tool may complete before the finish chunk; later conflicting
fragments invalidate the whole stream.  Consumers must require terminal done and
catch StreamError before accepting the resulting measurement or executing tools.
"""

from __future__ import annotations

import codecs
import json
import math
from typing import Any, Callable, Iterable, Iterator

from .schema import ValidationError, loads_json, validate_tool


class StreamError(ValueError):
    """Malformed response or a response closed before its terminal marker."""


def _lines(byte_chunks: Iterable[bytes]) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pending = ""
    try:
        for chunk in byte_chunks:
            if not isinstance(chunk, bytes):
                raise StreamError("HTTP stream chunks must be bytes")
            pending += decoder.decode(chunk, final=False)
            start = 0
            position = 0
            while position < len(pending):
                if pending[position] not in "\r\n":
                    position += 1
                    continue
                if pending[position] == "\r" and position + 1 == len(pending):
                    break  # the next byte may complete CRLF
                yield pending[start:position]
                position += 2 if pending[position:position + 2] == "\r\n" else 1
                start = position
            pending = pending[start:]
        pending += decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise StreamError("invalid or truncated UTF-8") from exc
    # EOF can complete a bare CR line; an unterminated final line is also useful
    # for NDJSON, while SSE still requires a blank event dispatch line.
    if pending.endswith("\r"):
        yield pending[:-1]
    elif pending:
        yield pending


def _record(data: str) -> dict[str, Any]:
    try:
        result = loads_json(data)
    except (ValueError, TypeError) as exc:
        raise StreamError("malformed JSON stream record") from exc
    if not isinstance(result, dict):
        raise StreamError("stream record must be an object")
    return result


def iter_sse(byte_chunks: Iterable[bytes], require_done: bool = True) -> Iterator[dict[str, Any]]:
    data = []
    done = False
    for line in _lines(byte_chunks):
        if line.startswith("\ufeff"):
            line = line[1:]
        if not line:
            if not data:
                continue
            payload = "\n".join(data)
            data = []
            if not payload.strip():
                continue
            if done:
                raise StreamError("data after SSE DONE")
            if payload.strip() == "[DONE]":
                done = True
                yield {"type": "done"}
            else:
                yield _record(payload)
            continue
        if line.startswith(":"):
            continue
        key, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if key == "data":
            data.append(value)
    if data:
        raise StreamError("SSE closed in an undispatched event")
    if require_done and not done:
        raise StreamError("SSE closed before DONE")


def iter_ndjson(byte_chunks: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    for line in _lines(byte_chunks):
        if line.strip():
            yield _record(line)


class _ThinkSplitter:
    def __init__(self):
        self.reasoning = False
        self.pending = ""

    def feed(self, text: str, final: bool = False) -> Iterator[dict[str, Any]]:
        self.pending += text
        while self.pending:
            tag = "</think>" if self.reasoning else "<think>"
            index = self.pending.find(tag)
            if index >= 0:
                if index:
                    yield {"type": "reasoning" if self.reasoning else "content",
                           "text": self.pending[:index]}
                self.pending = self.pending[index + len(tag):]
                self.reasoning = not self.reasoning
                continue
            keep = 0
            if not final:
                for size in range(1, min(len(tag) - 1, len(self.pending)) + 1):
                    if self.pending.endswith(tag[:size]):
                        keep = size
            emit = self.pending[:-keep] if keep else self.pending
            self.pending = self.pending[-keep:] if keep else ""
            if emit:
                yield {"type": "reasoning" if self.reasoning else "content", "text": emit}
            break


def _integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise StreamError(f"invalid counter: {field}")
    return value


def _duration(value: Any, divisor: float, field: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise StreamError(f"invalid duration: {field}")
    return value / divisor


def _openai_usage(record: dict[str, Any]) -> dict[str, Any] | None:
    usage = record.get("usage")
    timings = record.get("timings")
    if usage is None and timings is None:
        return None
    if usage is not None and not isinstance(usage, dict):
        raise StreamError("usage must be an object")
    if timings is not None and not isinstance(timings, dict):
        raise StreamError("timings must be an object")
    result = {"type": "usage", "raw": {"usage": usage, "timings": timings}}
    if usage:
        for source, destination in (("prompt_tokens", "prompt_tokens"),
                                    ("completion_tokens", "generated_tokens"),
                                    ("cached_tokens", "cached_tokens"),
                                    ("evaluated_tokens", "evaluated_tokens"),
                                    ("reasoning_tokens", "reasoning_tokens"),
                                    ("visible_tokens", "visible_tokens")):
            if source in usage:
                result[destination] = _integer(usage[source], source)
        for key, destination, field in (("prompt_tokens_details", "cached_tokens", "cached_tokens"),
                                        ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens")):
            details = usage.get(key)
            if details is not None:
                if not isinstance(details, dict):
                    raise StreamError(f"{key} must be an object")
                if field in details:
                    result[destination] = _integer(details[field], f"{key}.{field}")
        if "decode_seconds" in usage:
            result["decode_seconds"] = _duration(usage["decode_seconds"], 1, "decode_seconds")
    if timings:
        if "predicted_n" in timings:
            count = _integer(timings["predicted_n"], "timings.predicted_n")
            if result.get("generated_tokens") is not None and count != result["generated_tokens"]:
                raise StreamError("completion usage and predicted_n disagree")
            result["generated_tokens"] = count
        if "predicted_ms" in timings:
            result["decode_seconds"] = _duration(timings["predicted_ms"], 1e3, "predicted_ms")
        if "prompt_ms" in timings:
            result["prompt_seconds"] = _duration(timings["prompt_ms"], 1e3, "prompt_ms")
        # A timing prompt_n is not silently assumed to be the total formatted
        # prompt. Preserve it under a diagnostic field until adapter mapping is
        # verified against the pinned server's actual semantics.
        if "prompt_n" in timings:
            result["timing_prompt_tokens"] = _integer(timings["prompt_n"], "prompt_n")
    return result


class _Tools:
    def __init__(self, validator: Callable):
        self.validator = validator
        self.calls = {}
        self.ids = {}

    def feed(self, fragment: dict[str, Any], choice_index: int,
             fallback_index: int) -> Iterator[dict[str, Any]]:
        if not isinstance(fragment, dict):
            raise StreamError("tool fragment must be an object")
        identifier = fragment.get("id")
        if identifier is not None and (not isinstance(identifier, str) or not identifier):
            raise StreamError("tool id must be nonempty text")
        index = fragment.get("index")
        if index is None:
            index = self.ids.get((choice_index, identifier), fallback_index)
        if type(index) is not int or index < 0:
            raise StreamError("tool index must be a nonnegative integer")
        key = (choice_index, index)
        if identifier is not None:
            existing = self.ids.get((choice_index, identifier))
            if existing is not None and existing != index:
                raise StreamError("tool id assigned to multiple indices")
            self.ids[(choice_index, identifier)] = index
        call = self.calls.setdefault(key, {"id": None, "name": "", "arguments": "", "emitted": None})
        if identifier is not None:
            if call["id"] is not None and call["id"] != identifier:
                raise StreamError("tool index changed id")
            call["id"] = identifier
        if fragment.get("type", "function") != "function":
            raise StreamError("unsupported tool type")
        function = fragment.get("function", {})
        if not isinstance(function, dict):
            raise StreamError("tool function must be an object")
        name = function.get("name", "")
        arguments = function.get("arguments", "")
        if not isinstance(name, str):
            raise StreamError("tool name fragment must be text")
        if isinstance(arguments, dict):
            if call["arguments"]:
                raise StreamError("complete tool arguments conflict with prior fragments")
            arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if not isinstance(arguments, str):
            raise StreamError("tool arguments must be JSON text or object")
        call["name"] += name
        call["arguments"] += arguments
        if name or arguments:
            yield {"type": "tool_fragment", "choice_index": choice_index, "index": index,
                   "id": call["id"], "name": name, "arguments": arguments}
        yield from self._complete(key, final=False)

    def _complete(self, key: tuple[int, int], final: bool) -> Iterator[dict[str, Any]]:
        call = self.calls[key]
        try:
            arguments = loads_json(call["arguments"])
            validated = self.validator(call["name"], arguments)
            if validated is False:
                raise ValidationError("tool permission denied")
        except (ValueError, TypeError) as exc:
            if final or call["emitted"] is not None:
                raise StreamError("incomplete, invalid or unpermitted tool call") from exc
            return
        current = (call["name"], arguments)
        if call["emitted"] is not None:
            if call["emitted"] != current:
                raise StreamError("tool call changed after completing")
            return
        call["emitted"] = current
        yield {"type": "tool_call", "choice_index": key[0], "index": key[1],
               "id": call["id"], "name": call["name"], "arguments": arguments,
               "schema_valid": True}

    def finish(self, choice_index: int | None = None) -> Iterator[dict[str, Any]]:
        for key in self.calls:
            if choice_index is None or key[0] == choice_index:
                yield from self._complete(key, final=True)


def normalize_openai(byte_chunks: Iterable[bytes], validator: Callable = validate_tool) \
        -> Iterator[dict[str, Any]]:
    tools = _Tools(validator)
    splitters = {}
    finished = set()
    reasons = {}
    terminal_event = None
    for record in iter_sse(byte_chunks):
        if record == {"type": "done"}:
            yield from tools.finish()
            for index, splitter in splitters.items():
                for event in splitter.feed("", final=True):
                    yield {**event, "choice_index": index}
            event = {"type": "done"}
            if reasons:
                event["finish_reasons"] = reasons
                if len(reasons) == 1:
                    event["finish_reason"] = next(iter(reasons.values()))
            terminal_event = event
            continue
        if "error" in record:
            raise StreamError("engine returned an error record")
        usage = _openai_usage(record)
        if usage is not None:
            yield usage
        choices = record.get("choices", [])
        if not isinstance(choices, list):
            raise StreamError("choices must be an array")
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise StreamError("choice must be an object")
            index = choice.get("index", position)
            if type(index) is not int or index < 0:
                raise StreamError("choice index must be a nonnegative integer")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise StreamError("delta must be an object")
            if index in finished and any(delta.get(key) for key in
                                         ("content", "reasoning", "reasoning_content", "tool_calls", "function_call")):
                raise StreamError("choice changed after finish")
            for field in ("reasoning_content", "reasoning"):
                text = delta.get(field)
                if text is not None:
                    if not isinstance(text, str):
                        raise StreamError("reasoning delta must be text")
                    if text:
                        yield {"type": "reasoning", "text": text, "choice_index": index}
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise StreamError("content delta must be text")
                splitter = splitters.setdefault(index, _ThinkSplitter())
                for event in splitter.feed(content):
                    yield {**event, "choice_index": index}
            fragments = delta.get("tool_calls", [])
            if not isinstance(fragments, list):
                raise StreamError("tool_calls delta must be an array")
            if delta.get("function_call") is not None:
                if fragments:
                    raise StreamError("legacy and native tool calls mixed")
                fragments = [{"index": 0, "function": delta["function_call"]}]
            for tool_position, fragment in enumerate(fragments):
                yield from tools.feed(fragment, index, tool_position)
            reason = choice.get("finish_reason")
            if reason is not None:
                if not isinstance(reason, str) or not reason:
                    raise StreamError("finish_reason must be nonempty text")
                yield from tools.finish(index)
                splitter = splitters.get(index)
                if splitter is not None:
                    for event in splitter.feed("", final=True):
                        yield {**event, "choice_index": index}
                finished.add(index)
                reasons[index] = reason
    if terminal_event is not None:
        yield terminal_event


def _ollama_usage(record: dict[str, Any]) -> dict[str, Any]:
    result = {"type": "usage", "raw": {key: value for key, value in record.items()
                                         if key != "message"}}
    for source, destination in (("eval_count", "generated_tokens"),
                                ("prompt_eval_count", "prompt_tokens"),
                                ("cached_tokens", "cached_tokens"),
                                ("evaluated_tokens", "evaluated_tokens"),
                                ("reasoning_tokens", "reasoning_tokens"),
                                ("visible_tokens", "visible_tokens")):
        if source in record:
            result[destination] = _integer(record[source], source)
    if "prompt_eval_cached_count" in record:
        cached = _integer(record["prompt_eval_cached_count"], "prompt_eval_cached_count")
        if result.get("cached_tokens") is not None and cached != result["cached_tokens"]:
            raise StreamError("Ollama cached counters disagree")
        result["cached_tokens"] = cached
        total = result.get("prompt_tokens")
        if total is not None and cached is not None:
            if cached > total:
                raise StreamError("Ollama cached input exceeds total")
            result["evaluated_tokens"] = total - cached
    for source, destination in (("eval_duration", "decode_seconds"),
                                ("prompt_eval_duration", "prompt_seconds"),
                                ("load_duration", "load_seconds"),
                                ("total_duration", "server_total_seconds")):
        if source in record:
            result[destination] = _duration(record[source], 1e9, source)
    return result


def normalize_ollama(byte_chunks: Iterable[bytes], validator: Callable = validate_tool) \
        -> Iterator[dict[str, Any]]:
    tools = _Tools(validator)
    splitter = _ThinkSplitter()
    done = False
    terminal_event = None
    for record in iter_ndjson(byte_chunks):
        if done:
            raise StreamError("data after Ollama done")
        if "error" in record:
            raise StreamError("engine returned an error record")
        message = record.get("message", {})
        if not isinstance(message, dict):
            raise StreamError("Ollama message must be an object")
        thinking = message.get("thinking")
        if thinking is not None:
            if not isinstance(thinking, str):
                raise StreamError("Ollama thinking must be text")
            if thinking:
                yield {"type": "reasoning", "text": thinking, "choice_index": 0}
        content = message.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise StreamError("Ollama content must be text")
            for event in splitter.feed(content):
                yield {**event, "choice_index": 0}
        fragments = message.get("tool_calls", [])
        if not isinstance(fragments, list):
            raise StreamError("Ollama tool_calls must be an array")
        for position, fragment in enumerate(fragments):
            yield from tools.feed(fragment, 0, position)
        if "done" in record and type(record["done"]) is not bool:
            raise StreamError("Ollama done must be boolean")
        if record.get("done", False):
            yield from tools.finish()
            for event in splitter.feed("", final=True):
                yield {**event, "choice_index": 0}
            yield _ollama_usage(record)
            done = True
            reason = record.get("done_reason")
            terminal_event = {"type": "done", "finish_reason": reason}
    if not done:
        raise StreamError("Ollama closed before final counters/done")
    yield terminal_event
