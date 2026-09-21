import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.cache import TTLCache
CASES = {'capacity': 4, 'default_ttl': 3, 'trace': [{'op': 'len', 'now': 0, 'expected': 0}, {'op': 'put', 'now': 0, 'key': 'a', 'value': {'n': [-53, {'step': 1}]}, 'ttl': None}, {'op': 'get', 'now': 0, 'key': 'e', 'expected': None}, {'op': 'put', 'now': 2, 'key': 'e', 'value': {'n': [-5, {'step': 3}]}, 'ttl': 3}, {'op': 'len', 'now': 2, 'expected': 2}, {'op': 'len', 'now': 2, 'expected': 2}, {'op': 'get', 'now': 3, 'key': 'b', 'expected': None}, {'op': 'put', 'now': 3, 'key': 'd', 'value': {'n': [69, {'step': 7}]}, 'ttl': None}, {'op': 'put', 'now': 4, 'key': 'b', 'value': {'n': [35, {'step': 8}]}, 'ttl': 3}, {'op': 'put', 'now': 5, 'key': 'a', 'value': {'n': [-89, {'step': 9}]}, 'ttl': 3}, {'op': 'len', 'now': 5, 'expected': 3}, {'op': 'get', 'now': 5, 'key': 'd', 'expected': {'n': [69, {'step': 7}]}}, {'op': 'get', 'now': 6, 'key': 'c', 'expected': None}, {'op': 'put', 'now': 6, 'key': 'c', 'value': {'n': [-82, {'step': 13}]}, 'ttl': None}, {'op': 'get', 'now': 6, 'key': 'c', 'expected': {'n': [-82, {'step': 13}]}}, {'op': 'get', 'now': 7, 'key': 'e', 'expected': None}, {'op': 'get', 'now': 7, 'key': 'a', 'expected': {'n': [-89, {'step': 9}]}}, {'op': 'get', 'now': 7, 'key': 'e', 'expected': None}, {'op': 'len', 'now': 8, 'expected': 1}, {'op': 'put', 'now': 10, 'key': 'c', 'value': {'n': [41, {'step': 19}]}, 'ttl': 1}, {'op': 'put', 'now': 10, 'key': 'a', 'value': {'n': [37, {'step': 20}]}, 'ttl': 1}, {'op': 'len', 'now': 10, 'expected': 2}, {'op': 'len', 'now': 10, 'expected': 2}, {'op': 'get', 'now': 11, 'key': 'd', 'expected': None}, {'op': 'put', 'now': 11, 'key': 'd', 'value': {'n': [-92, {'step': 24}]}, 'ttl': None}, {'op': 'put', 'now': 12, 'key': 'a', 'value': {'n': [55, {'step': 25}]}, 'ttl': 3}, {'op': 'len', 'now': 12, 'expected': 2}, {'op': 'put', 'now': 13, 'key': 'a', 'value': {'n': [37, {'step': 27}]}, 'ttl': 1}, {'op': 'put', 'now': 14, 'key': 'e', 'value': {'n': [-33, {'step': 28}]}, 'ttl': 7}, {'op': 'get', 'now': 16, 'key': 'e', 'expected': {'n': [-33, {'step': 28}]}}, {'op': 'len', 'now': 17, 'expected': 1}, {'op': 'put', 'now': 17, 'key': 'e', 'value': {'n': [-64, {'step': 31}]}, 'ttl': 3}, {'op': 'put', 'now': 18, 'key': 'a', 'value': {'n': [-62, {'step': 32}]}, 'ttl': None}, {'op': 'len', 'now': 19, 'expected': 2}, {'op': 'get', 'now': 21, 'key': 'b', 'expected': None}, {'op': 'get', 'now': 21, 'key': 'c', 'expected': None}, {'op': 'get', 'now': 21, 'key': 'b', 'expected': None}, {'op': 'put', 'now': 23, 'key': 'c', 'value': {'n': [-32, {'step': 37}]}, 'ttl': None}, {'op': 'get', 'now': 25, 'key': 'b', 'expected': None}, {'op': 'len', 'now': 25, 'expected': 1}, {'op': 'put', 'now': 25, 'key': 'c', 'value': {'n': [75, {'step': 40}]}, 'ttl': 7}, {'op': 'get', 'now': 26, 'key': 'a', 'expected': None}, {'op': 'get', 'now': 27, 'key': 'a', 'expected': None}, {'op': 'len', 'now': 29, 'expected': 1}, {'op': 'len', 'now': 29, 'expected': 1}, {'op': 'put', 'now': 30, 'key': 'b', 'value': {'n': [3, {'step': 45}]}, 'ttl': 7}, {'op': 'get', 'now': 32, 'key': 'a', 'expected': None}, {'op': 'get', 'now': 33, 'key': 'b', 'expected': {'n': [3, {'step': 45}]}}, {'op': 'len', 'now': 35, 'expected': 1}, {'op': 'len', 'now': 35, 'expected': 1}, {'op': 'len', 'now': 35, 'expected': 1}, {'op': 'get', 'now': 37, 'key': 'e', 'expected': None}, {'op': 'put', 'now': 37, 'key': 'e', 'value': {'n': [44, {'step': 52}]}, 'ttl': 3}, {'op': 'put', 'now': 39, 'key': 'd', 'value': {'n': [-14, {'step': 53}]}, 'ttl': None}, {'op': 'len', 'now': 40, 'expected': 1}, {'op': 'len', 'now': 41, 'expected': 1}, {'op': 'put', 'now': 43, 'key': 'e', 'value': {'n': [87, {'step': 56}]}, 'ttl': None}, {'op': 'get', 'now': 43, 'key': 'c', 'expected': None}, {'op': 'put', 'now': 43, 'key': 'e', 'value': {'n': [-5, {'step': 58}]}, 'ttl': None}, {'op': 'put', 'now': 43, 'key': 'b', 'value': {'n': [21, {'step': 59}]}, 'ttl': 3}, {'op': 'get', 'now': 45, 'key': 'd', 'expected': None}, {'op': 'put', 'now': 45, 'key': 'a', 'value': {'n': [-71, {'step': 61}]}, 'ttl': None}, {'op': 'len', 'now': 45, 'expected': 3}, {'op': 'len', 'now': 45, 'expected': 3}, {'op': 'put', 'now': 47, 'key': 'a', 'value': {'n': [21, {'step': 64}]}, 'ttl': 7}, {'op': 'get', 'now': 47, 'key': 'd', 'expected': None}, {'op': 'get', 'now': 48, 'key': 'c', 'expected': None}, {'op': 'len', 'now': 48, 'expected': 1}, {'op': 'len', 'now': 48, 'expected': 1}, {'op': 'len', 'now': 48, 'expected': 1}, {'op': 'get', 'now': 48, 'key': 'a', 'expected': {'n': [21, {'step': 64}]}}, {'op': 'put', 'now': 50, 'key': 'e', 'value': {'n': [41, {'step': 71}]}, 'ttl': 1}, {'op': 'put', 'now': 50, 'key': 'b', 'value': {'n': [-22, {'step': 72}]}, 'ttl': 7}, {'op': 'put', 'now': 51, 'key': 'c', 'value': {'n': [72, {'step': 73}]}, 'ttl': 7}, {'op': 'put', 'now': 53, 'key': 'a', 'value': {'n': [73, {'step': 74}]}, 'ttl': 3}, {'op': 'put', 'now': 53, 'key': 'e', 'value': {'n': [-26, {'step': 75}]}, 'ttl': None}, {'op': 'put', 'now': 54, 'key': 'e', 'value': {'n': [-93, {'step': 76}]}, 'ttl': 3}, {'op': 'get', 'now': 54, 'key': 'c', 'expected': {'n': [72, {'step': 73}]}}, {'op': 'put', 'now': 54, 'key': 'd', 'value': {'n': [-60, {'step': 78}]}, 'ttl': None}, {'op': 'get', 'now': 56, 'key': 'e', 'expected': {'n': [-93, {'step': 76}]}}, {'op': 'put', 'now': 58, 'key': 'e', 'value': {'n': [32, {'step': 80}]}, 'ttl': 3}, {'op': 'len', 'now': 58, 'expected': 1}, {'op': 'len', 'now': 60, 'expected': 1}, {'op': 'put', 'now': 61, 'key': 'e', 'value': {'n': [3, {'step': 83}]}, 'ttl': 3}, {'op': 'put', 'now': 61, 'key': 'c', 'value': {'n': [92, {'step': 84}]}, 'ttl': 3}, {'op': 'put', 'now': 61, 'key': 'e', 'value': {'n': [-8, {'step': 85}]}, 'ttl': 3}, {'op': 'get', 'now': 63, 'key': 'd', 'expected': None}, {'op': 'put', 'now': 64, 'key': 'a', 'value': {'n': [50, {'step': 87}]}, 'ttl': None}, {'op': 'len', 'now': 65, 'expected': 1}, {'op': 'len', 'now': 67, 'expected': 0}]}

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


if __name__ == '__main__':
    unittest.main()
