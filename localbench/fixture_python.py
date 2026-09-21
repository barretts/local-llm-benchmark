"""Trusted deterministic Python fixture construction.

This module emits text and controller-owned oracle data. It never imports or
executes the source/gold/test text it emits; execution belongs to the isolated
grader. All source imports use the ``src`` namespace from the fixture root.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import random
from textwrap import dedent


SEEDS = (42, 314159, 271828)


def _text(value: str) -> str:
    return dedent(value).lstrip("\n")


def _tests(imports: str, body: str, cases: object) -> str:
    return (
        "import unittest\nimport copy\nimport math\nimport asyncio\n"
        "from decimal import Decimal, ROUND_HALF_UP\n"
        + imports + "\nCASES = " + repr(cases) + "\n\n"
        + _text(body) + "\n\nif __name__ == '__main__':\n    unittest.main()\n"
    )


def _result(source, gold, public, hidden, public_count, hidden_count, contract, cases):
    return {
        "language": "python", "source": source, "gold": gold,
        "public_tests": {"test_public.py": public},
        "hidden_tests": {"test_hidden.py": hidden},
        "public_count": public_count, "hidden_count": hidden_count,
        "contract": contract, "cases": cases,
    }


_CACHE_GOLD = _text('''
    from collections import OrderedDict
    from copy import deepcopy
    import math

    def _ttl(value):
        try:
            valid = not isinstance(value, bool) and math.isfinite(value) and value > 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("TTL must be finite and positive")
        return value

    class TTLCache:
        def __init__(self, capacity, default_ttl, clock):
            if type(capacity) is not int or capacity < 1:
                raise ValueError("capacity must be a positive integer")
            self.capacity = capacity
            self.default_ttl = _ttl(default_ttl)
            self.clock = clock
            self.entries = OrderedDict()

        def _purge(self, now):
            for key, (_, expiry) in list(self.entries.items()):
                if now >= expiry:
                    del self.entries[key]

        def put(self, key, value, ttl=None):
            duration = self.default_ttl if ttl is None else _ttl(ttl)
            now = self.clock()
            self._purge(now)
            self.entries[key] = (deepcopy(value), now + duration)
            self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)

        def get(self, key, default=None):
            self._purge(self.clock())
            if key not in self.entries:
                return default
            value, _ = self.entries[key]
            self.entries.move_to_end(key)
            return deepcopy(value)

        def __len__(self):
            self._purge(self.clock())
            return len(self.entries)
''')

_CACHE_BUG = _text('''
    from collections import OrderedDict
    import math

    def _ttl(value):
        try:
            valid = not isinstance(value, bool) and math.isfinite(value) and value > 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("TTL must be finite and positive")
        return value

    class TTLCache:
        def __init__(self, capacity, default_ttl, clock):
            if type(capacity) is not int or capacity < 1:
                raise ValueError("capacity must be a positive integer")
            self.capacity = capacity
            self.default_ttl = _ttl(default_ttl)
            self.clock = clock
            self.entries = OrderedDict()

        def _purge(self, now):
            for key, (_, expiry) in list(self.entries.items()):
                if now > expiry:
                    del self.entries[key]

        def put(self, key, value, ttl=None):
            duration = self.default_ttl if ttl is None else _ttl(ttl)
            self.entries[key] = (value, self.clock() + duration)
            self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)

        def get(self, key, default=None):
            self._purge(self.clock())
            if key not in self.entries:
                return default
            self.entries.move_to_end(key)
            return self.entries[key][0]

        def __len__(self):
            self._purge(self.clock())
            return len(self.entries)
''')


def _cache(seed):
    rng = random.Random(seed)
    capacity, default_ttl = rng.randint(2, 4), rng.randint(2, 5)
    clock = 0
    reference = OrderedDict()
    trace = []
    for index in range(90):
        clock += rng.choice([0, 0, 1, 2])
        for key, (_, expiry) in list(reference.items()):
            if clock >= expiry:
                del reference[key]
        operation = rng.choice(["put", "put", "get", "len"])
        key = rng.choice(["a", "b", "c", "d", "e"])
        if operation == "put":
            value = {"n": [rng.randint(-99, 99), {"step": index}]}
            ttl = rng.choice([None, 1, 3, 7])
            reference[key] = (deepcopy(value), clock + (default_ttl if ttl is None else ttl))
            reference.move_to_end(key)
            while len(reference) > capacity:
                reference.popitem(last=False)
            trace.append({"op": operation, "now": clock, "key": key, "value": value, "ttl": ttl})
        elif operation == "get":
            expected = None
            if key in reference:
                expected = deepcopy(reference[key][0])
                reference.move_to_end(key)
            trace.append({"op": operation, "now": clock, "key": key, "expected": expected})
        else:
            trace.append({"op": operation, "now": clock, "expected": len(reference)})
    cases = {"capacity": capacity, "default_ttl": default_ttl, "trace": trace}
    imports = "from src.cache import TTLCache"
    public = _tests(imports, '''
        class Public(unittest.TestCase):
            def test_deadline_expiry(self):
                now = [10]
                cache = TTLCache(2, 3, lambda: now[0])
                cache.put("a", 7)
                now[0] = 13
                self.assertIsNone(cache.get("a"))
                self.assertEqual(len(cache), 0)

            def test_put_is_defensive(self):
                cache = TTLCache(2, 3, lambda: 0)
                value = {"items": [1]}
                cache.put("a", value)
                value["items"].append(2)
                self.assertEqual(cache.get("a"), {"items": [1]})
    ''', {})
    hidden = _tests(imports, '''
        class Hidden(unittest.TestCase):
            def test_get_is_defensive_nested(self):
                cache = TTLCache(2, 4, lambda: 0)
                original = {"outer": [{"inner": [1, 2]}]}
                cache.put("x", original)
                result = cache.get("x")
                result["outer"][0]["inner"].clear()
                self.assertEqual(cache.get("x"), original)

            def test_overwrite_and_get_recency(self):
                cache = TTLCache(2, 9, lambda: 0)
                cache.put("a", 1); cache.put("b", 2); cache.put("a", 3)
                self.assertEqual(len(cache), 2)
                cache.put("c", 4)
                self.assertEqual(cache.get("a"), 3)
                self.assertIsNone(cache.get("b"))
                cache.get("a"); cache.put("d", 5)
                self.assertIsNone(cache.get("c"))

            def test_expired_before_eviction(self):
                now = [0]
                cache = TTLCache(2, 20, lambda: now[0])
                cache.put("live", 1); cache.put("expired", 2, ttl=1)
                now[0] = 2
                cache.put("new", 3)
                self.assertEqual(cache.get("live"), 1)
                self.assertEqual(cache.get("new"), 3)
                self.assertEqual(len(cache), 2)

            def test_miss_default_identity_and_recency(self):
                cache = TTLCache(2, 9, lambda: 0)
                cache.put("a", 1); cache.put("b", 2)
                default = []
                for _ in range(4):
                    self.assertIs(cache.get("missing", default), default)
                cache.put("c", 3)
                self.assertIsNone(cache.get("a"))

            def test_override_does_not_change_default(self):
                now = [0]
                cache = TTLCache(3, 5, lambda: now[0])
                cache.put("short", 1, ttl=2); cache.put("normal", 2)
                now[0] = 2
                self.assertIsNone(cache.get("short"))
                self.assertEqual(cache.get("normal"), 2)
                now[0] = 5
                self.assertEqual(len(cache), 0)

            def test_len_purges_all_live_and_empty(self):
                now = [0]
                cache = TTLCache(4, 1, lambda: now[0])
                self.assertEqual(len(cache), 0)
                cache.put("a", 1); cache.put("b", 2)
                self.assertEqual(len(cache), 2)
                now[0] = 1
                self.assertEqual(len(cache), 0)

            def test_invalid_inputs(self):
                for capacity in (0, -1, 1.5, True, "2"):
                    with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                        TTLCache(capacity, 1, lambda: 0)
                for ttl in (0, -1, float('inf'), float('nan'), True, "1"):
                    with self.subTest(ttl=ttl), self.assertRaises(ValueError):
                        TTLCache(2, ttl, lambda: 0)
                    cache = TTLCache(2, 1, lambda: 0)
                    with self.assertRaises(ValueError):
                        cache.put("a", 1, ttl=ttl)

            def test_seeded_operation_trace(self):
                now = [0]
                cache = TTLCache(CASES["capacity"], CASES["default_ttl"], lambda: now[0])
                for index, operation in enumerate(CASES["trace"]):
                    now[0] = operation["now"]
                    with self.subTest(index=index, operation=operation["op"]):
                        if operation["op"] == "put":
                            value = copy.deepcopy(operation["value"])
                            cache.put(operation["key"], value, ttl=operation["ttl"])
                            value["n"].append("caller mutation")
                        elif operation["op"] == "get":
                            self.assertEqual(cache.get(operation["key"]), operation["expected"])
                        else:
                            self.assertEqual(len(cache), operation["expected"])
    ''', cases)
    contract = "Repair src/cache.py: TTLCache(capacity, default_ttl, clock), put/get/__len__. Capacity is an integer >=1; TTL is finite and >0. Expire at now >= deadline. Deep-copy stored and returned values; misses return their default unchanged. Refresh LRU on put/get, overwrite without increasing size. Purge expired keys before capacity eviction and in len. A ttl override affects only that put."
    return _result({"src/cache.py": _CACHE_BUG}, {"src/cache.py": _CACHE_GOLD}, public, hidden, 2, 8, contract, cases)


_RANGES_GOLD = _text('''
    def merge_ranges(ranges):
        normalized = []
        for start, end in ranges:
            if type(start) is not int or type(end) is not int or end < start:
                raise ValueError("invalid range")
            if start != end:
                normalized.append((start, end))
        normalized.sort()
        result = []
        for start, end in normalized:
            if result and start < result[-1][1]:
                result[-1] = (result[-1][0], max(result[-1][1], end))
            else:
                result.append((start, end))
        return result
''')

_RANGES_BUG = _text('''
    def merge_ranges(ranges):
        for start, end in ranges:
            if type(start) is not int or type(end) is not int or end < start:
                raise ValueError("invalid range")
        ranges.sort()
        result = []
        for start, end in ranges:
            if start == end:
                continue
            if result and start <= result[-1][1]:
                result[-1] = (result[-1][0], max(result[-1][1], end))
            else:
                result.append((start, end))
        return result
''')


def _ranges(seed):
    rng = random.Random(seed)
    cases = []
    for _ in range(45):
        inputs = []
        for _ in range(rng.randint(0, 25)):
            start = rng.randint(-20, 20)
            inputs.append([start, start + rng.randint(0, 8)])
        rng.shuffle(inputs)
        expected = []
        for start, end in sorted((a, b) for a, b in inputs if a != b):
            if expected and start < expected[-1][1]:
                expected[-1][1] = max(expected[-1][1], end)
            else:
                expected.append([start, end])
        cases.append({"input": inputs, "expected": expected})
    imports = "from src.ranges import merge_ranges"
    public = _tests(imports, '''
        class Public(unittest.TestCase):
            def test_adjacent_remain_separate(self):
                self.assertEqual(merge_ranges([(1, 3), (3, 5)]), [(1, 3), (3, 5)])
            def test_input_unchanged(self):
                inputs = [[5, 8], [1, 4], [2, 6]]
                before = copy.deepcopy(inputs)
                self.assertEqual(merge_ranges(inputs), [(1, 8)])
                self.assertEqual(inputs, before)
    ''', {})
    hidden = _tests(imports, '''
        class Hidden(unittest.TestCase):
            def test_nested_duplicates(self):
                self.assertEqual(merge_ranges([(2, 3), (1, 9), (1, 9), (4, 6)]), [(1, 9)])
            def test_transitive_overlap(self):
                self.assertEqual(merge_ranges([(6, 10), (0, 4), (3, 7)]), [(0, 10)])
            def test_negative_adjacent_and_zero(self):
                self.assertEqual(merge_ranges([(-8, -5), (-5, -2), (-1, -1)]), [(-8, -5), (-5, -2)])
            def test_reversed_pair(self):
                with self.assertRaises(ValueError):
                    merge_ranges([(3, 2)])
            def test_empty_and_zero_only(self):
                self.assertEqual(merge_ranges([]), [])
                self.assertEqual(merge_ranges([(2, 2), (-4, -4)]), [])
            def test_sorted_tuple_results(self):
                result = merge_ranges([(10, 12), (-5, -3), (1, 2)])
                self.assertEqual(result, [(-5, -3), (1, 2), (10, 12)])
                self.assertTrue(all(type(pair) is tuple for pair in result))
            def test_immutability_nested_pairs(self):
                inputs = [[7, 8], [-3, 0], [0, 2]]
                before = copy.deepcopy(inputs)
                self.assertEqual(merge_ranges(inputs), [(-3, 0), (0, 2), (7, 8)])
                self.assertEqual(inputs, before)
            def test_seeded_overlap_properties(self):
                for index, case in enumerate(CASES):
                    inputs = copy.deepcopy(case["input"])
                    before = copy.deepcopy(inputs)
                    with self.subTest(index=index):
                        result = merge_ranges(inputs)
                        self.assertEqual(result, [tuple(pair) for pair in case["expected"]])
                        self.assertEqual(inputs, before)
                        self.assertTrue(all(a < b for a, b in result))
                        self.assertTrue(all(result[i][1] <= result[i + 1][0] for i in range(len(result)-1)))
    ''', cases)
    contract = "Repair src/ranges.py: merge_ranges(ranges) returns a sorted list of integer tuples for half-open ranges. Merge only actual overlap or duplicates; adjacent ranges remain separate. Drop zero-length ranges, reject end < start with ValueError, allow negative coordinates, and never mutate the input container or its pairs. Empty input returns []."
    return _result({"src/ranges.py": _RANGES_BUG}, {"src/ranges.py": _RANGES_GOLD}, public, hidden, 2, 8, contract, cases)


_CONFIG_GOLD = _text('''
    from copy import deepcopy

    def overlay_config(base: dict, overrides: dict) -> dict:
        result = deepcopy(base)
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                result[key] = overlay_config(base[key], value)
            else:
                result[key] = deepcopy(value)
        return result
''')

_CONFIG_BUG = _text('''
    def overlay_config(base: dict, overrides: dict) -> dict:
        result = base.copy()
        for key, value in overrides.items():
            if not value:
                continue
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key].update(overlay_config(result[key], value))
            elif isinstance(result.get(key), list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value
        return result
''')


def _overlay_oracle(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            result[key] = _overlay_oracle(base[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _config(seed):
    rng = random.Random(seed)
    def tree(depth):
        if depth == 0:
            return rng.choice([None, False, True, 0, 7, "", "value"])
        kind = rng.randrange(3)
        if kind == 0:
            return {key: tree(depth-1) for key in rng.sample(["a", "b", "c", "d"], rng.randint(0, 4))}
        if kind == 1:
            return [tree(depth-1) for _ in range(rng.randint(0, 3))]
        return tree(0)
    cases = []
    for _ in range(40):
        base = {key: tree(3) for key in rng.sample(["alpha", "beta", "gamma"], rng.randint(0, 3))}
        overrides = {key: tree(3) for key in rng.sample(["alpha", "beta", "delta"], rng.randint(0, 3))}
        cases.append({"base": base, "overrides": overrides, "expected": _overlay_oracle(base, overrides)})
    imports = "from src.config import overlay_config"
    public = _tests(imports, '''
        class Public(unittest.TestCase):
            def test_falsy_override(self):
                self.assertEqual(overlay_config({"flag": True, "n": 9}, {"flag": False, "n": 0}), {"flag": False, "n": 0})
            def test_nested_base_unchanged(self):
                base = {"a": {"x": 1, "y": 2}}
                before = copy.deepcopy(base)
                self.assertEqual(overlay_config(base, {"a": {"x": 3}}), {"a": {"x": 3, "y": 2}})
                self.assertEqual(base, before)
    ''', {})
    hidden = _tests(imports, '''
        def mutable_ids(value):
            found = set()
            if isinstance(value, (dict, list)):
                found.add(id(value))
                values = value.values() if isinstance(value, dict) else value
                for child in values:
                    found.update(mutable_ids(child))
            return found

        class Hidden(unittest.TestCase):
            def test_null_empty_string_and_list(self):
                self.assertEqual(overlay_config({"a": 1, "b": "x", "c": [1]}, {"a": None, "b": "", "c": []}), {"a": None, "b": "", "c": []})
            def test_lists_replace_never_concatenate(self):
                self.assertEqual(overlay_config({"a": [1, 2]}, {"a": [3]}), {"a": [3]})
            def test_dict_scalar_both_directions(self):
                self.assertEqual(overlay_config({"a": {"x": 1}, "b": 2}, {"a": 7, "b": {"y": [3]}}), {"a": 7, "b": {"y": [3]}})
            def test_nested_list_dict_isolation(self):
                base = {"a": [{"x": [1]}]}
                overrides = {"b": [{"y": [2]}]}
                result = overlay_config(base, overrides)
                result["a"][0]["x"].append(9)
                result["b"][0]["y"].append(9)
                self.assertEqual(base, {"a": [{"x": [1]}]})
                self.assertEqual(overrides, {"b": [{"y": [2]}]})
            def test_one_side_keys_survive(self):
                base = {"base": {"x": [1]}}
                overrides = {"new": [2]}
                result = overlay_config(base, overrides)
                self.assertEqual(result, {"base": {"x": [1]}, "new": [2]})
                self.assertFalse(mutable_ids(result) & (mutable_ids(base) | mutable_ids(overrides)))
            def test_empty_inputs_independent(self):
                base = {"a": []}
                result = overlay_config(base, {})
                result["a"].append(1)
                self.assertEqual(base, {"a": []})
                self.assertEqual(overlay_config({}, {}), {})
            def test_recursive_merge_preserves_siblings(self):
                self.assertEqual(overlay_config({"a": {"b": {"x": 1, "y": 2}, "c": [3]}}, {"a": {"b": {"x": 0}}}), {"a": {"b": {"x": 0, "y": 2}, "c": [3]}})
            def test_seeded_trees_and_no_shared_descendants(self):
                for index, case in enumerate(CASES):
                    base, overrides = copy.deepcopy(case["base"]), copy.deepcopy(case["overrides"])
                    before_base, before_overrides = copy.deepcopy(base), copy.deepcopy(overrides)
                    with self.subTest(index=index):
                        result = overlay_config(base, overrides)
                        self.assertEqual(result, case["expected"])
                        self.assertEqual(base, before_base)
                        self.assertEqual(overrides, before_overrides)
                        self.assertFalse(mutable_ids(result) & (mutable_ids(base) | mutable_ids(overrides)))
                        result["controller_output_mutation"] = [index]
                        self.assertEqual(base, before_base)
                        self.assertEqual(overrides, before_overrides)
    ''', cases)
    contract = "Repair src/config.py: overlay_config(base: dict, overrides: dict) -> dict for JSON-like trees. Recursively merge only dict/dict values. Override lists replace lists. Every other override replaces, including False, 0, empty string and None. Retain one-side keys. Return a fully independent deep structure sharing no mutable descendant with either unchanged input."
    return _result({"src/config.py": _CONFIG_BUG}, {"src/config.py": _CONFIG_GOLD}, public, hidden, 2, 8, contract, cases)


_RETRY_GOLD = _text('''
    import math

    def call_with_retry(fn, *, max_attempts, base_delay, retry_on, sleep):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("invalid max_attempts")
        try:
            valid = not isinstance(base_delay, bool) and math.isfinite(base_delay) and base_delay >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("invalid base_delay")
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as error:
                if not isinstance(error, retry_on):
                    raise
                if attempt + 1 == max_attempts:
                    raise
                sleep(base_delay * 2 ** attempt)
''')

_RETRY_BUG = _text('''
    import math

    def call_with_retry(fn, *, max_attempts, base_delay, retry_on, sleep):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("invalid max_attempts")
        try:
            valid = not isinstance(base_delay, bool) and math.isfinite(base_delay) and base_delay >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("invalid base_delay")
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception:
                sleep(base_delay * 2 ** attempt)
                if attempt + 1 == max_attempts:
                    raise
''')


def _retry(seed):
    rng = random.Random(seed)
    cases = []
    for index in range(35):
        attempts = rng.randint(1, 6)
        failures = rng.randint(0, attempts + 1)
        delay = rng.choice([0, 0.125, 0.5, 2])
        cases.append({"attempts": attempts, "failures": failures, "delay": delay,
                      "calls": min(failures+1, attempts), "sleeps": [delay * 2 ** i for i in range(min(failures, attempts-1))],
                      "result": rng.choice([None, False, 0, "ok", index]), "success": failures < attempts})
    imports = "from src.retry import call_with_retry"
    public = _tests(imports, '''
        class Public(unittest.TestCase):
            def test_nonretryable_propagates_immediately(self):
                calls, sleeps = [], []
                error = ValueError("do not retry")
                def fn():
                    calls.append(1)
                    raise error
                with self.assertRaises(ValueError) as captured:
                    call_with_retry(fn, max_attempts=3, base_delay=1, retry_on=(OSError,), sleep=sleeps.append)
                self.assertIs(captured.exception, error)
                self.assertEqual((len(calls), sleeps), (1, []))
            def test_one_attempt_does_not_sleep(self):
                sleeps = []
                def fn():
                    raise OSError("final")
                with self.assertRaises(OSError):
                    call_with_retry(fn, max_attempts=1, base_delay=2, retry_on=(OSError,), sleep=sleeps.append)
                self.assertEqual(sleeps, [])
    ''', {})
    hidden = _tests(imports, '''
        class Hidden(unittest.TestCase):
            def test_first_success_and_falsy_results(self):
                for result in (None, False, 0, ""):
                    calls, sleeps = [], []
                    def fn():
                        calls.append(1)
                        return result
                    self.assertIs(call_with_retry(fn, max_attempts=4, base_delay=1, retry_on=(OSError,), sleep=sleeps.append), result)
                    self.assertEqual((len(calls), sleeps), (1, []))
            def test_later_success_exact_delays(self):
                calls, sleeps = [], []
                def fn():
                    calls.append(1)
                    if len(calls) < 4:
                        raise OSError("transient")
                    return "recovered"
                self.assertEqual(call_with_retry(fn, max_attempts=5, base_delay=0.5, retry_on=(OSError,), sleep=sleeps.append), "recovered")
                self.assertEqual((len(calls), sleeps), (4, [0.5, 1.0, 2.0]))
            def test_final_exception_identity_no_final_sleep(self):
                error = OSError("same object")
                calls, sleeps = [], []
                def fn():
                    calls.append(1)
                    raise error
                with self.assertRaises(OSError) as captured:
                    call_with_retry(fn, max_attempts=3, base_delay=2, retry_on=(OSError,), sleep=sleeps.append)
                self.assertIs(captured.exception, error)
                self.assertEqual((len(calls), sleeps), (3, [2, 4]))
            def test_empty_policy_and_keyboard_interrupt(self):
                for error, policy in ((RuntimeError("ordinary"), ()), (KeyboardInterrupt(), (Exception,)), (KeyboardInterrupt(), (BaseException,))):
                    calls, sleeps = [], []
                    def fn():
                        calls.append(1)
                        raise error
                    with self.assertRaises(type(error)) as captured:
                        call_with_retry(fn, max_attempts=5, base_delay=1, retry_on=policy, sleep=sleeps.append)
                    self.assertIs(captured.exception, error)
                    self.assertEqual((len(calls), sleeps), (1, []))
            def test_zero_delay(self):
                calls, sleeps = [], []
                def fn():
                    calls.append(1)
                    if len(calls) < 3:
                        raise OSError()
                    return False
                self.assertIs(call_with_retry(fn, max_attempts=3, base_delay=0, retry_on=(OSError,), sleep=sleeps.append), False)
                self.assertEqual(sleeps, [0, 0])
            def test_invalid_bounds_before_call(self):
                for attempts in (0, -1, 2.5, True, "2"):
                    with self.assertRaises(ValueError):
                        call_with_retry(lambda: self.fail("called"), max_attempts=attempts, base_delay=1, retry_on=(OSError,), sleep=lambda _: None)
                for delay in (-1, float('inf'), float('nan'), True, "1"):
                    with self.assertRaises(ValueError):
                        call_with_retry(lambda: self.fail("called"), max_attempts=2, base_delay=delay, retry_on=(OSError,), sleep=lambda _: None)
            def test_sleeper_failure_propagates(self):
                calls, sleeps = [], []
                error = RuntimeError("sleep failure")
                def fn():
                    calls.append(1)
                    raise OSError()
                def sleeper(delay):
                    sleeps.append(delay)
                    raise error
                with self.assertRaises(RuntimeError) as captured:
                    call_with_retry(fn, max_attempts=4, base_delay=0.5, retry_on=(OSError,), sleep=sleeper)
                self.assertIs(captured.exception, error)
                self.assertEqual((len(calls), sleeps), (1, [0.5]))
            def test_seeded_schedules(self):
                for index, case in enumerate(CASES):
                    calls, sleeps = [], []
                    error = OSError("seeded failure")
                    def fn():
                        calls.append(1)
                        if len(calls) <= case["failures"]:
                            raise error
                        return case["result"]
                    with self.subTest(index=index):
                        if case["success"]:
                            result = call_with_retry(fn, max_attempts=case["attempts"], base_delay=case["delay"], retry_on=(OSError,), sleep=sleeps.append)
                            self.assertEqual(result, case["result"])
                        else:
                            with self.assertRaises(OSError) as captured:
                                call_with_retry(fn, max_attempts=case["attempts"], base_delay=case["delay"], retry_on=(OSError,), sleep=sleeps.append)
                            self.assertIs(captured.exception, error)
                        self.assertEqual(len(calls), case["calls"])
                        self.assertEqual(sleeps, case["sleeps"])
    ''', cases)
    contract = "Repair src/retry.py: call_with_retry(fn, *, max_attempts, base_delay, retry_on, sleep). max_attempts integer >=1, base_delay finite >=0 or ValueError. Call immediately and return any success including None/False. Retry only retry_on exceptions; preserve final exception identity. Before another attempt sleep base_delay*2**failure_index, first index 0. Never sleep after final attempt. Propagate BaseException and sleeper exceptions immediately."
    return _result({"src/retry.py": _RETRY_BUG}, {"src/retry.py": _RETRY_GOLD}, public, hidden, 2, 8, contract, cases)


_MODELS = _text('''
    from dataclasses import dataclass
    from decimal import Decimal

    @dataclass(frozen=True)
    class LineItem:
        unit_price: Decimal
        quantity: int
        discount: Decimal
''')

_MONEY_GOLD = _text('''
    from decimal import Decimal, ROUND_HALF_UP

    CENT = Decimal("0.01")

    def _decimal(value, minimum, maximum=None):
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError("finite Decimal required")
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError("out of range")

    def line_total(unit_price, quantity, discount) -> Decimal:
        _decimal(unit_price, Decimal(0))
        _decimal(discount, Decimal(0), Decimal(1))
        if type(quantity) is not int or quantity < 0:
            raise ValueError("invalid quantity")
        return (unit_price * quantity * (Decimal(1) - discount)).quantize(CENT, rounding=ROUND_HALF_UP)
''')

_MONEY_BUG = _MONEY_GOLD.replace("ROUND_HALF_UP", "ROUND_HALF_EVEN")

_INVOICE_GOLD = _text('''
    from decimal import Decimal, ROUND_HALF_UP
    from .models import LineItem
    from .money import line_total, _decimal

    def invoice_total(items, tax_rate) -> Decimal:
        _decimal(tax_rate, Decimal(0), Decimal(1))
        subtotal = sum((line_total(item.unit_price, item.quantity, item.discount) for item in items), Decimal("0.00"))
        tax = (subtotal * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return subtotal + tax
''')

_INVOICE_BUG = _text('''
    from decimal import Decimal, ROUND_HALF_UP
    from .models import LineItem
    from .money import line_total, _decimal

    def invoice_total(items, tax_rate) -> Decimal:
        _decimal(tax_rate, Decimal(0), Decimal(1))
        subtotal = Decimal("0.00")
        for item in items:
            line_total(item.unit_price, item.quantity, item.discount)
            subtotal += item.unit_price * item.quantity * (Decimal(1) - item.discount)
        return (subtotal * (Decimal(1) + tax_rate)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
''')

_SETTLEMENT_GOLD = _text('''
    from decimal import Decimal, ROUND_HALF_UP

    def _decimal(value, minimum, maximum=None):
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError("finite Decimal required")
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError("out of range")

    def settle(lines, tax_rate) -> Decimal:
        _decimal(tax_rate, Decimal(0), Decimal(1))
        cent = Decimal("0.01")
        subtotal = Decimal("0.00")
        for price, quantity, discount in lines:
            _decimal(price, Decimal(0))
            _decimal(discount, Decimal(0), Decimal(1))
            if type(quantity) is not int or quantity < 0:
                raise ValueError("invalid quantity")
            subtotal += (price * quantity * (Decimal(1) - discount)).quantize(cent, rounding=ROUND_HALF_UP)
        tax = (subtotal * tax_rate).quantize(cent, rounding=ROUND_HALF_UP)
        return subtotal + tax
''')

_SETTLEMENT_BUG = _text('''
    from decimal import Decimal

    def _decimal(value, minimum, maximum=None):
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError("finite Decimal required")
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError("out of range")

    def settle(lines, tax_rate) -> Decimal:
        _decimal(tax_rate, Decimal(0), Decimal(1))
        subtotal = 0.0
        for price, quantity, discount in lines:
            _decimal(price, Decimal(0))
            _decimal(discount, Decimal(0), Decimal(1))
            if type(quantity) is not int or quantity < 0:
                raise ValueError("invalid quantity")
            subtotal += float(price) * quantity * (1.0 - float(discount))
        return Decimal(str(round(subtotal * (1.0 + float(tax_rate)), 2)))
''')


def _money_cases(seed):
    rng = random.Random(seed)
    cases = []
    for index in range(45):
        lines = [["0.005", 1, "0"], ["0.005", 1, "0"]] if index < 3 else []
        for _ in range(rng.randint(0, 8)):
            price = Decimal(rng.randint(0, 500000)) / Decimal(1000)
            quantity = rng.randint(0, 12)
            discount = Decimal(rng.choice([0, 5, 10, 25, 33, 50, 100])) / Decimal(100)
            lines.append([str(price), quantity, str(discount)])
        rate = Decimal(rng.choice([0, 5, 10, 125, 200, 500, 1000])) / Decimal(1000)
        line_totals = [(Decimal(price) * quantity * (Decimal(1) - Decimal(discount))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) for price, quantity, discount in lines]
        subtotal = sum(line_totals, Decimal("0.00"))
        total = subtotal + (subtotal * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        cases.append({"lines": lines, "tax_rate": str(rate), "line_totals": [str(value) for value in line_totals], "total": str(total)})
    return cases


def _money(seed, settlement=False):
    cases = _money_cases(seed)
    if settlement:
        imports = "from src.settlement import settle\nimport inspect\ndef total(lines, rate):\n    return settle([(Decimal(p), q, Decimal(d)) for p, q, d in lines], Decimal(rate))"
        public_body = '''
            class Public(unittest.TestCase):
                def test_half_cent_rounds_up(self):
                    self.assertEqual(total([["0.005", 1, "0"]], "0"), Decimal("0.01"))
                def test_lines_round_before_aggregate(self):
                    self.assertEqual(total([["0.005", 1, "0"], ["0.005", 1, "0"]], "0"), Decimal("0.02"))
        '''
        helper_test = '''
    def test_preserved_signature(self):
        signature = inspect.signature(settle)
        self.assertEqual(list(signature.parameters), ["lines", "tax_rate"])
        self.assertEqual(total([["12.345", 2, "0.25"]], "0.125"), Decimal("20.84"))
'''
    else:
        imports = "from src.models import LineItem\nfrom src.money import line_total\nfrom src.invoice import invoice_total\nfrom dataclasses import FrozenInstanceError\ndef total(lines, rate):\n    return invoice_total([LineItem(Decimal(p), q, Decimal(d)) for p, q, d in lines], Decimal(rate))"
        public_body = '''
            class Public(unittest.TestCase):
                def test_half_cent_rounds_up(self):
                    self.assertEqual(line_total(Decimal("0.005"), 1, Decimal("0")), Decimal("0.01"))
                def test_lines_round_before_aggregate(self):
                    self.assertEqual(total([["0.005", 1, "0"], ["0.005", 1, "0"]], "0"), Decimal("0.02"))
        '''
        helper_test = '''
    def test_individual_helper_and_integrated_consistency(self):
        immutable = LineItem(Decimal("1.00"), 1, Decimal("0"))
        with self.assertRaises(FrozenInstanceError):
            immutable.unit_price = Decimal("2.00")
        for index, case in enumerate(CASES):
            with self.subTest(index=index):
                lines = [LineItem(Decimal(p), q, Decimal(d)) for p, q, d in case["lines"]]
                individual = [line_total(item.unit_price, item.quantity, item.discount) for item in lines]
                self.assertEqual(individual, [Decimal(v) for v in case["line_totals"]])
                subtotal = sum(individual, Decimal("0.00"))
                expected = subtotal + (subtotal * Decimal(case["tax_rate"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                self.assertEqual(invoice_total(lines, Decimal(case["tax_rate"])), expected)
'''
    common_hidden = _text('''
        class Hidden(unittest.TestCase):
            def test_tax_half_cent_separate_rounding(self):
                self.assertEqual(total([["0.05", 1, "0"]], "0.1"), Decimal("0.06"))
                self.assertEqual(total([["0.005", 1, "0"]], "0.5"), Decimal("0.02"))
            def test_quantity_discount_and_zero(self):
                self.assertEqual(total([["1.005", 3, "0.5"]], "0"), Decimal("1.51"))
                self.assertEqual(total([["999.99", 0, "0.2"], ["12.34", 7, "1"]], "0.9"), Decimal("0.00"))
            def test_empty_is_two_decimal_zero(self):
                result = total([], "0.9")
                self.assertIsInstance(result, Decimal)
                self.assertEqual(str(result), "0.00")
            def test_invalid_prices_and_discounts(self):
                for price in (Decimal("-0.01"), Decimal("NaN"), Decimal("Infinity"), 1.2):
                    with self.subTest(price=price), self.assertRaises(ValueError):
                        RAW_TOTAL([(price, 1, Decimal(0))], Decimal(0))
                for discount in (Decimal("-0.01"), Decimal("1.01"), Decimal("NaN"), Decimal("Infinity"), 0.5):
                    with self.subTest(discount=discount), self.assertRaises(ValueError):
                        RAW_TOTAL([(Decimal(1), 1, discount)], Decimal(0))
            def test_invalid_quantities_and_tax(self):
                for quantity in (-1, 1.5, True, "1"):
                    with self.subTest(quantity=quantity), self.assertRaises(ValueError):
                        RAW_TOTAL([(Decimal(1), quantity, Decimal(0))], Decimal(0))
                for rate in (Decimal("-0.01"), Decimal("1.01"), Decimal("NaN"), Decimal("Infinity"), 0.5):
                    with self.subTest(rate=rate), self.assertRaises(ValueError):
                        RAW_TOTAL([], rate)
            def test_input_unchanged_and_decimal_precision(self):
                lines = [(Decimal("123456789.125"), 3, Decimal("0.125")), (Decimal("0.005"), 1, Decimal(0))]
                before = copy.deepcopy(lines)
                result = RAW_TOTAL(lines, Decimal("0.125"))
                subtotal = Decimal("324074071.45") + Decimal("0.01")
                expected = subtotal + (subtotal * Decimal("0.125")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                self.assertEqual(result, expected)
                self.assertEqual(lines, before)
            def test_seeded_decimal_invoices(self):
                for index, case in enumerate(CASES):
                    with self.subTest(index=index):
                        result = total(case["lines"], case["tax_rate"])
                        self.assertIsInstance(result, Decimal)
                        self.assertEqual(result, Decimal(case["total"]))
                        self.assertEqual(str(result), case["total"])
    ''')
    if settlement:
        hidden_imports = imports + "\nRAW_TOTAL = settle"
    else:
        hidden_imports = imports + "\ndef RAW_TOTAL(lines, rate):\n    return invoice_total([LineItem(p, q, d) for p, q, d in lines], rate)"
    hidden = _tests(hidden_imports, common_hidden.rstrip() + "\n" + helper_test, cases)
    public = _tests(imports, public_body, {})
    contract = "Prices/rates are finite Decimals; price >=0, quantity integer >=0, discount and tax_rate in [0,1]; invalid values raise ValueError. Each line is unit_price * quantity * (1-discount), rounded to cents with ROUND_HALF_UP. Sum rounded lines, round subtotal*tax_rate separately to cents HALF_UP, then add. Empty total is Decimal('0.00'). Never use binary floats or mutate inputs."
    if settlement:
        contract = "Repair only src/settlement.py. Preserve settle(lines, tax_rate) -> Decimal; lines contain (Decimal unit_price, int quantity, Decimal discount). " + contract
        return _result({"src/settlement.py": _SETTLEMENT_BUG}, {"src/settlement.py": _SETTLEMENT_GOLD}, public, hidden, 2, 8, contract, cases)
    contract = "Repair src/money.py and src/invoice.py; preserve src/models.py frozen LineItem(unit_price: Decimal, quantity: int, discount: Decimal). Preserve money.line_total and invoice.invoice_total(items, tax_rate). " + contract
    source = {"src/models.py": _MODELS, "src/money.py": _MONEY_BUG, "src/invoice.py": _INVOICE_BUG}
    gold = {"src/models.py": _MODELS, "src/money.py": _MONEY_GOLD, "src/invoice.py": _INVOICE_GOLD}
    return _result(source, gold, public, hidden, 2, 8, contract, cases)


_WORKERS_GOLD = _text('''
    import asyncio

    async def map_limited(items, worker, limit) -> list:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        items = list(items)
        if not items:
            return []
        semaphore = asyncio.Semaphore(limit)
        async def run(item):
            async with semaphore:
                return await worker(item)
        tasks = [asyncio.create_task(run(item)) for item in items]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
''')

_WORKERS_BUG = _text('''
    import asyncio

    async def map_limited(items, worker, limit) -> list:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        tasks = [asyncio.create_task(worker(item)) for item in items]
        results = []
        for task in asyncio.as_completed(tasks):
            results.append(await task)
        return results
''')


def _workers(seed):
    rng = random.Random(seed)
    cases = [{"items": [rng.randint(-99, 99) for _ in range(rng.randint(1, 15))], "limit": rng.randint(1, 5)} for _ in range(15)]
    imports = "from src.workers import map_limited"
    public = _tests(imports, '''
        class Public(unittest.IsolatedAsyncioTestCase):
            async def test_reverse_completion_keeps_input_order(self):
                gates = [asyncio.Event() for _ in range(3)]
                started = set()
                async def worker(item):
                    started.add(item)
                    await gates[item].wait()
                    return item * 10
                task = asyncio.create_task(map_limited([0, 1, 2], worker, 3))
                try:
                    for _ in range(20):
                        await asyncio.sleep(0)
                    self.assertEqual(started, {0, 1, 2})
                    for index in (2, 1, 0):
                        gates[index].set()
                        for _ in range(5):
                            await asyncio.sleep(0)
                    self.assertEqual(await asyncio.wait_for(task, 3), [0, 10, 20])
                finally:
                    for gate in gates:
                        gate.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

            async def test_active_count_is_bounded(self):
                gate = asyncio.Event()
                active = 0
                peak = 0
                async def worker(item):
                    nonlocal active, peak
                    active += 1
                    peak = max(peak, active)
                    try:
                        await gate.wait()
                        return item
                    finally:
                        active -= 1
                task = asyncio.create_task(map_limited(list(range(7)), worker, 2))
                try:
                    for _ in range(30):
                        await asyncio.sleep(0)
                    self.assertLessEqual(peak, 2)
                    self.assertEqual(active, 2)
                finally:
                    gate.set()
                    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
    ''', {})
    hidden = _tests(imports, '''
        class Hidden(unittest.IsolatedAsyncioTestCase):
            async def test_limit_one_order_and_peak(self):
                active, peak = 0, 0
                async def worker(item):
                    nonlocal active, peak
                    active += 1; peak = max(peak, active)
                    try:
                        await asyncio.sleep(0)
                        return item + 1
                    finally:
                        active -= 1
                self.assertEqual(await map_limited([3, 2, 1], worker, 1), [4, 3, 2])
                self.assertEqual((peak, active), (1, 0))

            async def test_empty_never_calls_worker(self):
                async def worker(item):
                    self.fail("worker called on empty input")
                self.assertEqual(await map_limited([], worker, 4), [])

            async def test_multiple_limits_and_seeded_inputs(self):
                for index, case in enumerate(CASES):
                    active, peak = 0, 0
                    async def worker(item):
                        nonlocal active, peak
                        active += 1; peak = max(peak, active)
                        try:
                            for _ in range(abs(item) % 4 + 1):
                                await asyncio.sleep(0)
                            return {"value": item * item}
                        finally:
                            active -= 1
                    with self.subTest(index=index):
                        result = await asyncio.wait_for(map_limited(case["items"], worker, case["limit"]), 3)
                        self.assertEqual(result, [{"value": value * value} for value in case["items"]])
                        self.assertLessEqual(peak, case["limit"])
                        self.assertEqual(active, 0)

            async def test_worker_failure_cancels_and_awaits_cleanup(self):
                ready, blocked = asyncio.Event(), asyncio.Event()
                running, cleaned = set(), set()
                owned = set()
                error = RuntimeError("observed failure")
                async def worker(item):
                    owned.add(asyncio.current_task())
                    running.add(item)
                    if item == 1:
                        ready.set()
                    try:
                        if item == 0:
                            await ready.wait()
                            raise error
                        await blocked.wait()
                        return item
                    finally:
                        await asyncio.sleep(0)
                        running.discard(item)
                        cleaned.add(item)
                task = asyncio.create_task(map_limited([0, 1, 2, 3], worker, 2))
                try:
                    with self.assertRaises(RuntimeError) as captured:
                        await asyncio.wait_for(task, 3)
                    self.assertIs(captured.exception, error)
                    self.assertEqual(running, set())
                    self.assertTrue({0, 1}.issubset(cleaned))
                    self.assertTrue(all(t.done() for t in owned))
                finally:
                    blocked.set()
                    for owned_task in owned:
                        if not owned_task.done():
                            owned_task.cancel()
                    await asyncio.gather(*owned, return_exceptions=True)

            async def test_caller_cancellation_awaits_owned_workers(self):
                started, blocked = asyncio.Event(), asyncio.Event()
                running, owned = set(), set()
                async def worker(item):
                    owned.add(asyncio.current_task())
                    running.add(item)
                    started.set()
                    try:
                        await blocked.wait()
                        return item
                    finally:
                        await asyncio.sleep(0)
                        running.discard(item)
                task = asyncio.create_task(map_limited(list(range(5)), worker, 2))
                try:
                    await asyncio.wait_for(started.wait(), 3)
                    for _ in range(15):
                        await asyncio.sleep(0)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertEqual(running, set())
                    self.assertTrue(all(t.done() for t in owned))
                finally:
                    blocked.set()
                    for owned_task in owned:
                        if not owned_task.done():
                            owned_task.cancel()
                    await asyncio.gather(*owned, return_exceptions=True)

            async def test_blocked_release_controls_admission(self):
                gates = [asyncio.Event() for _ in range(4)]
                started = []
                async def worker(item):
                    started.append(item)
                    await gates[item].wait()
                    return item
                task = asyncio.create_task(map_limited([0, 1, 2, 3], worker, 2))
                try:
                    for _ in range(20):
                        await asyncio.sleep(0)
                    self.assertEqual(started, [0, 1])
                    gates[1].set()
                    for _ in range(20):
                        await asyncio.sleep(0)
                    self.assertEqual(started, [0, 1, 2])
                    gates[0].set()
                    for _ in range(20):
                        await asyncio.sleep(0)
                    self.assertEqual(started, [0, 1, 2, 3])
                finally:
                    for gate in gates:
                        gate.set()
                    result = await asyncio.wait_for(task, 3)
                self.assertEqual(result, [0, 1, 2, 3])

            async def test_invalid_limit(self):
                async def worker(item):
                    self.fail("worker called for invalid limit")
                for limit in (0, -1, 1.5, True, "2", None):
                    with self.subTest(limit=limit), self.assertRaises(ValueError):
                        await map_limited([], worker, limit)

            async def test_single_worker_exception_identity(self):
                error = LookupError("single worker")
                async def worker(item):
                    raise error
                with self.assertRaises(LookupError) as captured:
                    await map_limited([1], worker, 1)
                self.assertIs(captured.exception, error)
    ''', cases)
    contract = "Repair src/workers.py: async map_limited(items, worker, limit) -> list. limit integer >=1 or ValueError; empty returns [] without calls. At most limit workers execute at once; preserve input order. On observed worker failure or caller cancellation cancel pending/running owned tasks, await their cleanup, and propagate the exception or CancelledError. No owned worker may remain running after return/raise."
    return _result({"src/workers.py": _WORKERS_BUG}, {"src/workers.py": _WORKERS_GOLD}, public, hidden, 2, 8, contract, cases)


def build(fixture_id: str, seed: int) -> dict:
    """Return one immutable-by-convention deterministic source/test bundle."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    builders = {
        "py01": _cache, "py02": _ranges, "py03": _config,
        "py04": _retry, "py05": _money, "py06": _workers,
        "long03": lambda value: _money(value, settlement=True),
    }
    try:
        builder = builders[fixture_id]
    except KeyError:
        raise ValueError(f"unknown Python fixture: {fixture_id}") from None
    return builder(seed)
