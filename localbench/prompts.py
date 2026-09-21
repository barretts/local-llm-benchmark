import copy
import random
from .config import digest

AGENT_SYSTEM = "You are a local coding agent repairing only the provided synthetic fixture. Follow the contract. Use native tools. Only permitted source files may be changed. Tests and dependencies are controller-owned. Run public tests when useful, then finish. Never invoke arbitrary shell commands."


def corpus(seed, lines=18000):
    rng = random.Random(seed)
    result = []
    for i in range(lines):
        n = rng.randrange(1_000_000)
        result.append(f"// OBSOLETE EXAMPLE unrelated tenant archive_{i:05d}.ts\nexport const sample_{i:05d} = {{revision: {n}, timeout_ms: {rng.randrange(100,20000)}, obsolete: true}};\n")
    return "".join(result)


def pack(tokenize, make_messages, seed, target, tolerance=64):
    """Binary trim deterministic filler, retokenizing the entire template each time.

    make_messages(filler) must retain every authoritative fact and final query.
    No character/token conversion is used. Returns exact endpoint provenance.
    """
    filler = corpus(seed)
    empty = tokenize(make_messages(""))
    if empty["count"] > target:
        raise ValueError("required prompt exceeds target")
    high_count = tokenize(make_messages(filler))
    if high_count["count"] < target - tolerance:
        raise ValueError("deterministic corpus too small")
    low, high, best = 0, len(filler), ("", empty)
    while low <= high:
        mid = (low + high) // 2
        count = tokenize(make_messages(filler[:mid]))
        if count["count"] <= target:
            best = (filler[:mid], count)
            if count["count"] == target:
                break
            low = mid + 1
        else:
            high = mid - 1
    if not target - tolerance <= best[1]["count"] <= target:
        raise ValueError("exact token packing failed")
    messages = make_messages(best[0])
    return {"messages":messages,"expected_tokens":best[1]["count"],"tokenization":best[1],"prompt_hash":digest(messages),"target_tokens":target}


def validate_context(expected, usage, measurement, effective_context, truncated=False):
    actual = usage.get("prompt_tokens")
    target = measurement["full_prompt_tokens"]
    tolerance = measurement["prompt_target_absolute_tolerance_tokens"]
    reasons = []
    if type(effective_context) is not int or effective_context < measurement["context_window_tokens"]:
        reasons.append("insufficient_effective_context")
    if truncated:
        reasons.append("truncated_context")
    if type(expected) is not int or not target-tolerance <= expected <= target:
        reasons.append("incorrect_packed_length")
    if type(actual) is not int:
        reasons.append("missing_actual_prompt_count")
    elif actual > target or actual < target-tolerance or abs(actual-expected)>tolerance:
        reasons.append("actual_prompt_count_mismatch")
    if expected + measurement["reserved_output_tokens"] > effective_context:
        reasons.append("insufficient_output_reserve")
    return reasons
