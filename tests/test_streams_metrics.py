"""Controller-only tests: byte streams, fake clocks and data, no generated code."""

import json
import unittest

from localbench.metrics import Measurement, decode_throughput, nearest_rank, sample_summary
from localbench.schema import ValidationError, loads_json, tool_definitions, validate_tool
from localbench.streams import StreamError, iter_ndjson, iter_sse, normalize_ollama, normalize_openai


def sse(*records, done=True):
    output = b"".join(("data: " + json.dumps(record, ensure_ascii=False) + "\n\n").encode("utf-8")
                      for record in records)
    return output + (b"data: [DONE]\n\n" if done else b"")


def delta(value, index=0, finish_reason=None):
    return {"choices": [{"index": index, "delta": value, "finish_reason": finish_reason}]}


def fragment(index, name="", arguments="", identifier=None):
    result = {"index": index, "function": {"name": name, "arguments": arguments}}
    if identifier is not None:
        result["id"] = identifier
    return result


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value

    def at(self, seconds):
        self.value = int(seconds * 1e9)


class SSETests(unittest.TestCase):
    def test_utf8_and_crlf_fragmented_at_every_byte(self):
        raw = sse(delta({"content": "café 🧪"})).replace(b"\n", b"\r\n")
        events = list(normalize_openai(bytes([value]) for value in raw))
        self.assertEqual("".join(event["text"] for event in events if event["type"] == "content"),
                         "café 🧪")
        self.assertEqual(events[-1], {"type": "done"})

    def test_multiline_data_comments_and_empty_events(self):
        raw = b':heartbeat\n\nevent: message\ndata: {"choices":\ndata: []}\n\ndata: \n\ndata: [DONE]\n\n'
        self.assertEqual(list(iter_sse([raw])), [{"choices": []}, {"type": "done"}])

    def test_bare_cr_lines_and_empty_byte_chunks(self):
        raw = b'data: {"choices": []}\r\rdata: [DONE]\r\r'
        self.assertEqual(list(iter_sse([b"", raw[:5], raw[5:]])),
                         [{"choices": []}, {"type": "done"}])

    def test_role_only_and_usage_tail(self):
        events = list(normalize_openai([sse(
            delta({"role": "assistant", "content": ""}),
            delta({"content": "code"}, finish_reason="stop"),
            {"choices": [], "usage": {"prompt_tokens": 61440, "completion_tokens": 128,
             "prompt_tokens_details": {"cached_tokens": 0},
             "completion_tokens_details": {"reasoning_tokens": 12}}},
            {"choices": [], "timings": {"predicted_n": 128, "predicted_ms": 2000}},
        )]))
        content = [event for event in events if event["type"] == "content"]
        usage = [event for event in events if event["type"] == "usage"]
        self.assertEqual(content, [{"type": "content", "text": "code", "choice_index": 0}])
        self.assertEqual(usage[0]["cached_tokens"], 0)
        self.assertEqual(usage[0]["reasoning_tokens"], 12)
        self.assertEqual(usage[1]["decode_seconds"], 2)
        self.assertEqual(events[-1]["type"], "done")

    def test_explicit_reasoning_and_split_think_regions(self):
        events = list(normalize_openai([sse(
            delta({"reasoning_content": "explicit"}),
            delta({"content": "<thi"}),
            delta({"content": "nk>internal</th"}),
            delta({"content": "ink>print(1)"}),
        )]))
        self.assertEqual("".join(event["text"] for event in events if event["type"] == "reasoning"),
                         "explicitinternal")
        self.assertEqual("".join(event["text"] for event in events if event["type"] == "content"),
                         "print(1)")

    def test_non_tag_less_than_text_is_not_lost(self):
        events = list(normalize_openai([sse(delta({"content": "x <t"}),
                                             delta({"content": " = 3"}),
                                             delta({"content": "<"}))]))
        self.assertEqual("".join(event["text"] for event in events if event["type"] == "content"),
                         "x <t = 3<")

    def test_missing_done_and_undispatched_event_fail(self):
        with self.assertRaisesRegex(StreamError, "before DONE"):
            list(normalize_openai([sse(delta({"content": "partial"}), done=False)]))
        with self.assertRaisesRegex(StreamError, "undispatched"):
            list(iter_sse([b'data: [DONE]\n']))

    def test_invalid_utf8_json_and_tail_after_done(self):
        for raw in (b'data: {"x":"\xf0\x9f"}\n\n', b'data: nope\n\n',
                    b'data: {"x":NaN}\n\n', b'data: {"x":1,"x":2}\n\n',
                    b'data: [DONE]\n\ndata: {}\n\n'):
            with self.subTest(raw=raw), self.assertRaises(StreamError):
                list(iter_sse([raw]))


class ToolStreamTests(unittest.TestCase):
    def test_parallel_split_arguments_do_not_cross(self):
        records = [
            delta({"tool_calls": [fragment(0, "read_", '{"path":', "a"),
                                   fragment(1, "write_file", '{"path":"src/b.py",', "b")]}),
            delta({"tool_calls": [fragment(1, arguments='"content":"ok"}'),
                                   fragment(0, "file", '"src/a.py"}')]}),
            delta({}, finish_reason="tool_calls"),
        ]
        events = list(normalize_openai([sse(*records)]))
        calls = [event for event in events if event["type"] == "tool_call"]
        self.assertEqual([call["index"] for call in calls], [1, 0])
        self.assertEqual(calls[0]["arguments"], {"path": "src/b.py", "content": "ok"})
        self.assertEqual(calls[1]["arguments"], {"path": "src/a.py"})
        self.assertEqual([call["id"] for call in calls], ["b", "a"])

    def test_no_call_from_partial_json(self):
        first = sse(delta({"tool_calls": [fragment(0, "read_file", '{"path":"src/a.py"')]}),
                    done=False)
        iterator = normalize_openai([first, sse(delta({"tool_calls": [fragment(0, arguments="}")]}))])
        event = next(iterator)
        self.assertEqual(event["type"], "tool_fragment")
        event = next(iterator)
        self.assertEqual(event["type"], "tool_fragment")
        self.assertEqual(next(iterator)["type"], "tool_call")
        self.assertEqual(next(iterator)["type"], "done")
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_invalid_tools_missing_fields_and_premature_argument_closure(self):
        cases = [("read_file", '{}'), ("execute_shell", '{"command":"bad"}'),
                 ("read_file", '{"path":'), ("read_file", '{"path":"a","extra":1}'),
                 ("read_file", '{"path":false}'), ("read_file", '{"path":}'),
                 ("read_file", '{"path":"a","path":"b"}')]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments), self.assertRaises(StreamError):
                list(normalize_openai([sse(delta({"tool_calls": [fragment(0, name, arguments)]}),
                                           delta({}, finish_reason="tool_calls"))]))

    def test_complete_call_followed_by_extra_json_invalidates_stream(self):
        iterator = normalize_openai([sse(
            delta({"tool_calls": [fragment(0, "read_file", '{"path":"src/a.py"}')]}),
            delta({"tool_calls": [fragment(0, arguments='{}')]}),
        )])
        self.assertEqual(next(iterator)["type"], "tool_fragment")
        self.assertEqual(next(iterator)["type"], "tool_call")
        self.assertEqual(next(iterator)["type"], "tool_fragment")
        with self.assertRaises(StreamError):
            next(iterator)

    def test_permission_callback_rejects_schema_valid_unsafe_path(self):
        def permission(name, arguments):
            validate_tool(name, arguments)
            if arguments.get("path") == "../secret":
                raise ValidationError("path forbidden")
            return arguments
        with self.assertRaises(StreamError):
            list(normalize_openai([sse(delta({"tool_calls": [
                fragment(0, "read_file", '{"path":"../secret"}')]}))], validator=permission))

    def test_id_index_conflicts_and_choice_change_after_finish(self):
        for records in (
            [delta({"tool_calls": [fragment(0, "read_file", '{}', "same")]}),
             delta({"tool_calls": [fragment(1, "read_file", '{}', "same")]})],
            [delta({"tool_calls": [fragment(0, "read_file", '{', "a")]}),
             delta({"tool_calls": [fragment(0, arguments='}', identifier="b")]})],
            [delta({}, finish_reason="stop"), delta({"content": "late"})],
        ):
            with self.subTest(records=records), self.assertRaises(StreamError):
                list(normalize_openai([sse(*records)]))


class OllamaTests(unittest.TestCase):
    def test_fragmented_ndjson_unicode_final_line_and_ns_counters(self):
        records = [
            {"message": {"thinking": "reason 🧪", "content": ""}, "done": False},
            {"message": {"tool_calls": [{"function": {
                "name": "read_file", "arguments": {"path": "src/a.py"}}}]}, "done": False},
            {"message": {"content": ""}, "done": True, "done_reason": "stop",
             "eval_count": 100, "prompt_eval_count": 61440, "eval_duration": 2500000000,
             "prompt_eval_duration": 2000000000, "total_duration": 5000000000},
        ]
        raw = "\n".join(json.dumps(record, ensure_ascii=False) for record in records).encode()
        events = list(normalize_ollama(bytes([value]) for value in raw))
        usage = next(event for event in events if event["type"] == "usage")
        self.assertEqual(usage["generated_tokens"], 100)
        self.assertEqual(usage["decode_seconds"], 2.5)
        self.assertEqual(usage["prompt_seconds"], 2)
        self.assertNotIn("cached_tokens", usage)
        self.assertNotIn("evaluated_tokens", usage)
        self.assertEqual(len([event for event in events if event["type"] == "tool_call"]), 1)
        self.assertEqual(events[-1], {"type": "done", "finish_reason": "stop"})

    def test_absent_optional_counters_are_unknown(self):
        events = list(normalize_ollama([b'{"done":true}\n']))
        usage = next(event for event in events if event["type"] == "usage")
        self.assertNotIn("generated_tokens", usage)
        self.assertNotIn("decode_seconds", usage)

    def test_ollama_closure_invalid_counter_and_extra_final_record(self):
        for raw in (b'{"done":false}\n', b'{"done":true,"eval_count":true}\n',
                    b'{"done":true,"eval_duration":-1}\n', b'{"done":true}\n{}\n'):
            with self.subTest(raw=raw), self.assertRaises(StreamError):
                list(normalize_ollama([raw]))

    def test_ndjson_malformed_and_non_object(self):
        for raw in (b'[]\n', b'{\n', b'{"a":Infinity}\n'):
            with self.subTest(raw=raw), self.assertRaises(StreamError):
                list(iter_ndjson([raw]))


class SchemaTests(unittest.TestCase):
    def test_strict_tool_types_fields_and_json(self):
        self.assertEqual(validate_tool("run_tests", {}), {})
        self.assertEqual(validate_tool("list_files", {}), {})
        for name, arguments in (("read_file", {}), ("read_file", {"path": 1}),
                                ("read_file", {"path": "a", "command": "bad"}),
                                ("run_tests", {"command": "python x"}),
                                ("write_file", {"path": "a", "content": None}),
                                ("bad", {})):
            with self.subTest(name=name, arguments=arguments), self.assertRaises(ValidationError):
                validate_tool(name, arguments)
        for value in ('{"x":1,"x":2}', 'NaN', '{"x":1e999}'):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                loads_json(value)

    def test_answer_fixture_schema_and_independent_definitions(self):
        contract = {"type": "object", "properties": {"timeout_ms": {"type": "integer"}},
                    "required": ["timeout_ms"], "additionalProperties": False}
        validate_tool("submit_answer", {"answer": {"timeout_ms": 7500}, "evidence_ids": ["id"]},
                      answer_schema=contract)
        with self.assertRaises(ValidationError):
            validate_tool("submit_answer", {"answer": {"timeout_ms": True}, "evidence_ids": ["id"]},
                          answer_schema=contract)
        tools = tool_definitions(include_answer=False)
        self.assertEqual(len(tools), 5)
        tools[0]["function"]["parameters"]["properties"]["oops"] = {}
        self.assertNotIn("oops", tool_definitions()[0]["function"]["parameters"]["properties"])


class MeasurementTests(unittest.TestCase):
    def test_clock_starts_before_send_and_action_after_complete_permitted_json(self):
        clock = Clock()
        clock.at(1)
        measurement = Measurement(clock_ns=clock).start()
        clock.at(2)
        measurement.observe({"type": "role", "role": "assistant"})
        clock.at(3)
        measurement.observe({"type": "reasoning", "text": "consider"})
        clock.at(4)
        measurement.observe({"type": "tool_fragment", "name": "read_file", "arguments": '{"path":'})
        clock.at(5)
        measurement.observe({"type": "content", "text": "I will read it"})
        clock.at(6)
        measurement.observe({"type": "tool_call", "name": "read_file", "arguments": {"path": "src/a.py"}})
        clock.at(7)
        measurement.observe({"type": "usage", "generated_tokens": 128, "decode_seconds": 2})
        measurement.observe({"type": "done", "finish_reason": "tool_calls"})
        clock.at(8)
        result = measurement.finish()
        self.assertEqual(result["first_stream_token_seconds"], 2)
        self.assertEqual(result["first_action_seconds"], 5)
        self.assertEqual(result["reasoning_to_action_seconds"], 3)
        self.assertEqual(result["wall_seconds"], 7)
        self.assertEqual(result["decode_tokens_per_second"], 64)
        self.assertTrue(result["sustained_throughput_qualified"])
        self.assertEqual(result["completion_reason"], "tool_calls")

    def test_whitespace_intent_and_invalid_call_never_count_action(self):
        clock = Clock()
        measurement = Measurement(clock_ns=clock).start()
        clock.at(1)
        measurement.observe({"type": "content", "text": "   "})
        measurement.observe({"type": "tool_call", "name": "read_file", "arguments": {}})
        measurement.observe({"type": "tool_call", "name": "execute_shell", "arguments": {}})
        measurement.observe({"type": "done"})
        result = measurement.finish()
        self.assertEqual(result["first_stream_token_seconds"], 1)
        self.assertIsNone(result["first_action_seconds"])
        self.assertTrue(result["action_censored"])
        self.assertEqual(result["invalid_tool_calls"], 2)

    def test_executor_permission_validator_rejection(self):
        def validator(name, arguments):
            validate_tool(name, arguments)
            return arguments.get("path") == "src/allowed.py"
        clock = Clock()
        measurement = Measurement(clock_ns=clock, tool_validator=validator).start()
        clock.at(1)
        measurement.observe({"type": "tool_call", "name": "read_file", "arguments": {"path": "../bad"}})
        clock.at(2)
        measurement.observe({"type": "tool_call", "name": "read_file", "arguments": {"path": "src/allowed.py"}})
        measurement.observe({"type": "done"})
        result = measurement.finish()
        self.assertEqual(result["invalid_tool_calls"], 1)
        self.assertEqual(result["first_action_seconds"], 2)

    def test_no_chunk_token_conversion_and_modes_remain_separate(self):
        clock = Clock()
        fresh = Measurement(clock_ns=clock, mode="fresh").start()
        cached = Measurement(clock_ns=clock, mode="cached").start()
        for index in range(1, 6):
            clock.at(index)
            fresh.observe({"type": "content", "text": "fragment"})
        fresh.observe({"type": "done"})
        cached.observe({"type": "usage", "usage": {"generated_tokens": 32, "decode_seconds": 1}})
        cached.observe({"type": "done"})
        fresh_result = fresh.finish()
        cached_result = cached.finish()
        self.assertIsNone(fresh_result["decode_tokens_per_second"])
        self.assertNotIn("generated_tokens", fresh_result)
        self.assertEqual(cached_result["decode_tokens_per_second"], 32)
        self.assertFalse(cached_result["sustained_throughput_qualified"])
        self.assertIsNone(cached_result["sustained_decode_tokens_per_second"])
        self.assertEqual((fresh_result["mode"], cached_result["mode"]), ("fresh", "cached"))

    def test_measurement_lifecycle_and_backwards_clock(self):
        clock = Clock()
        measurement = Measurement(clock_ns=clock)
        with self.assertRaises(RuntimeError):
            measurement.observe({"type": "done"})
        measurement.start()
        with self.assertRaises(RuntimeError):
            measurement.start()
        clock.at(2)
        measurement.observe({"type": "content", "text": "a"})
        clock.at(1)
        with self.assertRaises(ValueError):
            measurement.observe({"type": "done"})

    def test_nearest_rank_20_samples_and_counter_unknowns(self):
        values = list(range(20, 0, -1))
        self.assertEqual(nearest_rank(values, .5), 10)
        self.assertEqual(nearest_rank(values, .95), 19)
        self.assertEqual(nearest_rank(values, 95), 19)
        self.assertEqual(sample_summary(values), {"n": 20, "min": 1, "max": 20, "p50": 10, "p95": 19})
        self.assertIsNone(nearest_rank([], .95))
        self.assertEqual(decode_throughput(100, 2.5), 40)
        for tokens, duration in ((None, 1), (100, None), (100, 0), (True, 2),
                                 (100, float("nan")), (100, -1)):
            self.assertIsNone(decode_throughput(tokens, duration))
        with self.assertRaises(ValueError):
            nearest_rank([1, None], .95)


if __name__ == "__main__":
    unittest.main()
