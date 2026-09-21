"""Client arrival clocks and exact server counters, never stream-chunk tokens."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable

from .schema import ValidationError, validate_tool


def nearest_rank(samples: Iterable[float], p: float) -> float | None:
    """Nearest-rank quantile; p is a fraction, or a percentage greater than 1."""
    if type(p) not in (int, float) or not math.isfinite(p):
        raise ValueError("percentile must be finite")
    if p > 1:
        p /= 100
    if not 0 < p <= 1:
        raise ValueError("percentile must be in (0, 1] or (1, 100]")
    values = list(samples)
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in values):
        raise ValueError("samples must be finite numbers; censored values are separate")
    values.sort()
    return values[math.ceil(p * len(values)) - 1] if values else None


def sample_summary(samples: Iterable[float]) -> dict[str, Any]:
    values = list(samples)
    p50 = nearest_rank(values, .50)
    return {"n": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None, "p50": p50,
            "p95": nearest_rank(values, .95)}


def decode_throughput(generated_tokens: int | None,
                      decode_seconds: float | None) -> float | None:
    """Exact server-generated tokens / server decode-only seconds, if observed."""
    if type(generated_tokens) is not int or generated_tokens < 0:
        return None
    if type(decode_seconds) not in (int, float) or not math.isfinite(decode_seconds) \
            or decode_seconds <= 0:
        return None
    return generated_tokens / decode_seconds


decode_tokens_per_second = decode_throughput


class Measurement:
    """One request; call start immediately before sending, then observe arrivals.

    The mode is carried into output so fresh/cached requests cannot be silently
    conflated.  Missing action remains None/censored.  The optional validator is
    the executor's permission gate; a normalizer's schema_valid flag is not trust.
    """

    def __init__(self, clock_ns: Callable[[], int] = time.perf_counter_ns,
                 mode: str = "fresh", tool_validator: Callable = validate_tool):
        if mode not in {"fresh", "cached", "fixture_cached", "exploratory"}:
            raise ValueError("unknown measurement mode")
        self.clock_ns = clock_ns
        self.mode = mode
        self.tool_validator = tool_validator
        self._start_ns = None
        self._finish_ns = None
        self._first_ns = None
        self._last_ns = None
        self._action_ns = None
        self._reasoning_ns = None
        self._last_clock_ns = None
        self._arrivals = []
        self._usage = {}
        self._done = False
        self._invalid_tools = 0
        self._completion_reason = None

    def _now(self) -> int:
        value = self.clock_ns()
        if type(value) is not int:
            raise ValueError("clock must return integer nanoseconds")
        if self._last_clock_ns is not None and value < self._last_clock_ns:
            raise ValueError("clock moved backwards")
        self._last_clock_ns = value
        return value

    def start(self) -> "Measurement":
        if self._start_ns is not None:
            raise RuntimeError("measurement already started")
        self._start_ns = self._now()
        return self

    def observe(self, event: dict[str, Any]) -> None:
        if self._start_ns is None or self._finish_ns is not None:
            raise RuntimeError("measurement is not active")
        now = self._now()
        kind = event.get("type")
        text = event.get("text") if kind in {"reasoning", "content"} else \
            event.get("arguments") if kind == "tool_fragment" else None
        if isinstance(text, str) and text:
            if self._first_ns is None:
                self._first_ns = now
            self._last_ns = now
            self._arrivals.append(now)
            if kind == "reasoning" and self._reasoning_ns is None:
                self._reasoning_ns = now
        if kind == "tool_call":
            try:
                if self.tool_validator(event.get("name"), event.get("arguments")) is False:
                    raise ValidationError("tool permission denied")
            except (ValidationError, ValueError, TypeError):
                self._invalid_tools += 1
            else:
                if self._action_ns is None:
                    self._action_ns = now
        if kind == "usage":
            payload = event.get("usage", event)
            for key in ("generated_tokens", "prompt_tokens", "cached_tokens", "evaluated_tokens",
                        "reasoning_tokens", "visible_tokens", "decode_seconds", "prompt_seconds",
                        "load_seconds", "server_total_seconds"):
                if key in payload and payload[key] is not None:
                    self._usage[key] = payload[key]
        if kind == "done":
            self._done = True
        if event.get("finish_reason") is not None:
            self._completion_reason = event["finish_reason"]

    def finish(self) -> dict[str, Any]:
        if self._start_ns is None:
            raise RuntimeError("measurement was not started")
        if self._finish_ns is None:
            self._finish_ns = self._now()
        seconds = lambda value: None if value is None else (value - self._start_ns) / 1e9
        generated = self._usage.get("generated_tokens")
        duration = self._usage.get("decode_seconds")
        arrival_seconds = None if self._first_ns is None or self._last_ns == self._first_ns \
            else (self._last_ns - self._first_ns) / 1e9
        arrival_rate = decode_throughput(generated, arrival_seconds)
        throughput = decode_throughput(generated, duration)
        intervals = [(b - a) / 1e9 for a, b in zip(self._arrivals, self._arrivals[1:])]
        sustained = type(generated) is int and generated >= 64 and throughput is not None
        result = {
            "mode": self.mode,
            "first_stream_token_seconds": seconds(self._first_ns),
            "first_reasoning_seconds": seconds(self._reasoning_ns),
            "first_action_seconds": seconds(self._action_ns),
            "reasoning_to_action_seconds": None if self._reasoning_ns is None or self._action_ns is None
                else (self._action_ns - self._reasoning_ns) / 1e9,
            "wall_seconds": seconds(self._finish_ns),
            "action_censored": self._action_ns is None,
            "first_action_valid": self._action_ns is not None and self._done and self._invalid_tools == 0,
            "stream_complete": self._done,
            "invalid_tool_calls": self._invalid_tools,
            "completion_reason": self._completion_reason,
            "decode_tokens_per_second": throughput,
            "sustained_throughput_qualified": sustained,
            "sustained_decode_tokens_per_second": throughput if sustained else None,
            "client_arrival_tokens_per_second": arrival_rate,
            "client_arrival_rate_is_auxiliary": True,
            "inter_event_seconds": intervals,
            "inter_event_p95_seconds": nearest_rank(intervals, .95),
        }
        result.update(self._usage)
        return result
