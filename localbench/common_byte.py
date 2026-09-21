"""Separate identical-byte experiment; never a substitute for equal-token gates."""

from __future__ import annotations

import hashlib
import random
import re
import time
from pathlib import Path

from .config import atomic_json, canonical, digest
from .measurement import warm
from .metrics import decode_throughput
from .quality import request_for
from .schema import tool_definitions, validate_tool


VARIANTS = {"medium": 256, "large": 1024}
VERSION = "common-byte-v1"


def code_corpus(seed=42, line_count=1024):
    """Exactly line_count deterministic UTF-8 code lines, without token estimates."""
    if type(seed) is not int or type(line_count) is not int or not 1 <= line_count <= 8192:
        raise ValueError("common-byte corpus requires integer seed and 1..8192 lines")
    generator = random.Random(seed)
    lines = ["export const archive_%05d = { revision: %d, timeout_ms: %d, obsolete: true, tag: 'café-%08x' };\n" %
        (index, generator.randrange(1000000), generator.randrange(100, 20000), generator.getrandbits(32))
        for index in range(line_count)]
    return "".join(lines)


def workload(seed=42, replicate=0, variant="large", line_count=None, request_kind="code", run_id="standalone"):
    if variant not in VARIANTS or request_kind not in {"code", "tool"} or type(replicate) is not int or replicate < 0:
        raise ValueError("invalid common-byte variant, request kind or replicate")
    count = VARIANTS[variant] if line_count is None else line_count
    corpus = code_corpus(seed, count)
    # Same nonce for the same cross-model replicate. It never depends on model,
    # tokenizer, configuration or padding; owned restart gives fresh isolation.
    nonce = digest({"version": VERSION, "run": run_id, "seed": seed, "replicate": replicate,
                    "variant": variant, "line_count": count, "request_kind": request_kind})
    system = "common-byte-nonce:" + nonce + "\nYou are a local coding assistant. Work only on this synthetic corpus."
    tools = []
    if request_kind == "code":
        query = "\nFINAL REQUEST: Write complete Python source for an ordered bounded asynchronous worker pool. " \
            "Include validation, cancellation cleanup and complete usage examples. Produce at least 150 substantive lines. " \
            "Output only source; do not call tools."
    else:
        tools = [tool for tool in tool_definitions(include_answer=False) if tool["function"]["name"] == "read_file"]
        query = "\nFINAL REQUEST: Your first action must be one native read_file call with path src/cache.py. " \
            "Do not edit source or run tests in this latency probe."
    messages = [{"role": "system", "content": system}, {"role": "user", "content":
        "SYNTHETIC REPOSITORY ARCHIVE (all records below are obsolete examples):\n" + corpus + query}]
    payload = canonical({"messages": messages, "tools": tools}).encode("utf-8")
    return {"version": VERSION, "seed": seed, "replicate": replicate, "variant": variant,
        "line_count": count, "request_kind": request_kind, "messages": messages, "tools": tools,
        "payload_bytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "corpus_bytes": len(corpus.encode("utf-8")), "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "target_label": "common_byte", "qualifying_equal_token_measurement": False}


def _identity(adapter):
    process = getattr(adapter, "process", None)
    saved = getattr(adapter, "saved", {}) or {}
    return (getattr(process, "pid", None), saved.get("created"))


def common_byte(config, state, adapter, configuration_id, seed=42, replicate=0, variant="large",
                line_count=None, request_kind="code", on_packed=None):
    """Measure common source bytes with actual model-specific template token counts.

    Both variants have fixed line counts, not promised token sizes. If a model
    cannot fit the same bytes, record that limit; never trim only its workload.
    A code response is observed only and is never executed on the host.
    """
    previous = _identity(adapter)
    restart = adapter.fresh_restart()
    current = _identity(adapter)
    saved = getattr(adapter, "saved", {}) or (restart.get("handle", {}) if isinstance(restart, dict) else {})
    owned_restart = current[0] is not None and current != previous and saved.get("owner") == "localbench"
    warm_metrics = warm(config, adapter, seed + replicate * 7919 + 710101)
    sample = workload(seed, replicate, variant, line_count, request_kind, state.run["id"])
    tokenization = adapter.tokenize(sample["messages"], sample["tools"])
    expected = tokenization.get("count")
    if type(expected) is not int or expected < 0 or not tokenization.get("provenance"):
        raise RuntimeError("common_byte_exact_template_tokenization_unverified")
    packed = {"messages": sample["messages"], "expected_tokens": expected,
        "tokenization": tokenization, "prompt_hash": sample["payload_sha256"],
        "target_tokens": None, "target_label": "common_byte", "common_byte_workload": sample}
    if on_packed:
        on_packed(packed)
    artifact = Path(config["paths"]["artifacts"]) / ("common-byte-" + sample["payload_sha256"] + ".json")
    atomic_json(artifact, sample)
    maximum = config["measurement"]["throughput_max_output_tokens"]
    context = adapter.effective.get("effective_context")
    result = {"kind": "common_byte", "configuration_id": configuration_id, "seed": seed,
        "replicate": replicate, "mode": "fresh", "target_label": "common_byte", "target_tokens": None,
        "qualifying_equal_token_measurement": False, "workload_version": VERSION,
        "variant": variant, "line_count": sample["line_count"], "request_kind": request_kind,
        "prompt_hash": sample["payload_sha256"], "common_byte_payload_sha256": sample["payload_sha256"],
        "common_byte_payload_bytes": sample["payload_bytes"], "corpus_sha256": sample["corpus_sha256"],
        "corpus_bytes": sample["corpus_bytes"], "workload_artifact": str(artifact),
        "expected_prompt_tokens": expected, "effective_context_tokens": context,
        "tokenization": tokenization, "warmup_metrics": warm_metrics,
        "owned_process_restart_verified": owned_restart, "warmup_excluded_from_metrics": True,
        "generated_source_executed": False, "tool_execution": "schema-only measurement;no tool executed",
        "note": "Identical pre-template UTF-8 messages/tool schemas; each model's exact template determines actual tokens. "
                "This secondary workload cannot satisfy equal-token 64K capacity/quality/final timing gates."}
    if type(context) is not int or expected + maximum > context:
        return {**result, "passed": False, "valid": False, "reason": "common_byte_does_not_fit_effective_context",
                "metrics": {}, "request_sent": False}
    timeout = config["limits"]["per_64k_request_timeout_seconds"] if expected > 16384 \
        else config["limits"]["per_16k_request_timeout_seconds"]
    state.check_budget(timeout)
    adapter.http.timeout = timeout
    request = request_for(config, sample["messages"], sample["tools"], seed, maximum)
    if request_kind == "code":
        request.pop("tool_choice", None)

    def validator(name, arguments):
        if request_kind != "tool" or name != "read_file" or arguments != {"path": "src/cache.py"}:
            raise ValueError("common-byte probe permits only requested read_file action")
        return validate_tool(name, arguments)

    response = adapter.stream_chat(request, log_prefix=Path(config["paths"]["logs"]) /
        ("common-byte-" + configuration_id + "-" + sample["payload_sha256"][:16] + "-" + str(time.time_ns())),
        validator=validator, mode="fresh")
    metrics = response["metrics"]
    actual, generated, cached = metrics.get("prompt_tokens"), metrics.get("generated_tokens"), metrics.get("cached_tokens")
    reasons = []
    if not response.get("valid_stream"):
        reasons.append("common_byte_stream_failure")
    if type(actual) is not int or abs(actual - expected) > config["measurement"]["prompt_target_absolute_tolerance_tokens"]:
        reasons.append("common_byte_actual_prompt_count_mismatch")
    if type(generated) is not int or not 0 <= generated <= maximum:
        reasons.append("common_byte_generation_counter_or_budget_unverified")
    if type(actual) is int and type(generated) is int and actual + generated > context:
        reasons.append("common_byte_context_budget_violation")
    cache_verified = owned_restart or type(cached) is int and 0 <= cached <= config["measurement"]["cold_prefix_expected_cached_tokens_max"]
    if cached is not None and (type(cached) is not int or cached < 0 or cached >
            config["measurement"]["cold_prefix_expected_cached_tokens_max"]):
        reasons.append("common_byte_fresh_prefix_cache_reuse")
    if not cache_verified:
        reasons.append("common_byte_fresh_cache_isolation_unverified")
    action = metrics.get("first_action_valid") is True and any(call.get("name") == "read_file" and
        call.get("arguments") == {"path": "src/cache.py"} for call in response.get("calls", []))
    if request_kind == "tool" and not action:
        reasons.append("common_byte_missing_or_incorrect_first_tool_action")
    if request_kind == "code" and not response.get("content", "").strip():
        reasons.append("common_byte_missing_code_content")
    rate = decode_throughput(generated, metrics.get("decode_seconds"))
    sustained = type(generated) is int and generated >= 64 and rate is not None
    return {**result, "passed": not reasons, "valid": not reasons, "reason": ";".join(reasons) or "passed",
        "metrics": metrics, "request_sent": True, "actual_prompt_tokens": actual,
        "cached_tokens": cached, "cache_isolation_verified": cache_verified,
        "exact_sustained_decode_tokens_per_second": rate if sustained else None,
        "sustained_throughput_qualified": sustained, "output_content_bytes": len(response.get("content", "").encode("utf-8")),
        "source_like_output_observed": bool(re.search(r"(^|\n)(?:async\s+)?(?:def|class|import|from)\s", response.get("content", ""))),
        "first_tool_action_valid": action if request_kind == "tool" else None}


common_byte_workload = common_byte
