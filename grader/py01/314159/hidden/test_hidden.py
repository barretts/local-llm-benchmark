import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.cache import TTLCache
CASES = {'capacity': 2, 'default_ttl': 4, 'trace': [{'op': 'put', 'now': 1, 'key': 'a', 'value': {'n': [47, {'step': 0}]}, 'ttl': 3}, {'op': 'put', 'now': 2, 'key': 'b', 'value': {'n': [46, {'step': 1}]}, 'ttl': None}, {'op': 'put', 'now': 2, 'key': 'b', 'value': {'n': [-18, {'step': 2}]}, 'ttl': None}, {'op': 'get', 'now': 2, 'key': 'c', 'expected': None}, {'op': 'len', 'now': 3, 'expected': 2}, {'op': 'put', 'now': 4, 'key': 'a', 'value': {'n': [-68, {'step': 5}]}, 'ttl': 3}, {'op': 'put', 'now': 5, 'key': 'b', 'value': {'n': [-22, {'step': 6}]}, 'ttl': None}, {'op': 'put', 'now': 5, 'key': 'c', 'value': {'n': [-22, {'step': 7}]}, 'ttl': 1}, {'op': 'put', 'now': 5, 'key': 'b', 'value': {'n': [89, {'step': 8}]}, 'ttl': None}, {'op': 'put', 'now': 7, 'key': 'e', 'value': {'n': [10, {'step': 9}]}, 'ttl': None}, {'op': 'len', 'now': 7, 'expected': 2}, {'op': 'put', 'now': 7, 'key': 'e', 'value': {'n': [5, {'step': 11}]}, 'ttl': None}, {'op': 'get', 'now': 7, 'key': 'e', 'expected': {'n': [5, {'step': 11}]}}, {'op': 'put', 'now': 9, 'key': 'e', 'value': {'n': [-18, {'step': 13}]}, 'ttl': None}, {'op': 'get', 'now': 10, 'key': 'e', 'expected': {'n': [-18, {'step': 13}]}}, {'op': 'put', 'now': 10, 'key': 'a', 'value': {'n': [-91, {'step': 15}]}, 'ttl': 3}, {'op': 'get', 'now': 12, 'key': 'a', 'expected': {'n': [-91, {'step': 15}]}}, {'op': 'put', 'now': 14, 'key': 'e', 'value': {'n': [-42, {'step': 17}]}, 'ttl': None}, {'op': 'len', 'now': 16, 'expected': 1}, {'op': 'put', 'now': 16, 'key': 'a', 'value': {'n': [-25, {'step': 19}]}, 'ttl': 1}, {'op': 'put', 'now': 16, 'key': 'c', 'value': {'n': [-97, {'step': 20}]}, 'ttl': 3}, {'op': 'len', 'now': 17, 'expected': 1}, {'op': 'len', 'now': 19, 'expected': 0}, {'op': 'put', 'now': 19, 'key': 'e', 'value': {'n': [50, {'step': 23}]}, 'ttl': 7}, {'op': 'get', 'now': 20, 'key': 'a', 'expected': None}, {'op': 'get', 'now': 20, 'key': 'd', 'expected': None}, {'op': 'get', 'now': 22, 'key': 'e', 'expected': {'n': [50, {'step': 23}]}}, {'op': 'get', 'now': 22, 'key': 'b', 'expected': None}, {'op': 'get', 'now': 22, 'key': 'b', 'expected': None}, {'op': 'put', 'now': 23, 'key': 'b', 'value': {'n': [-83, {'step': 29}]}, 'ttl': 7}, {'op': 'put', 'now': 24, 'key': 'c', 'value': {'n': [63, {'step': 30}]}, 'ttl': 7}, {'op': 'len', 'now': 24, 'expected': 2}, {'op': 'put', 'now': 26, 'key': 'd', 'value': {'n': [3, {'step': 32}]}, 'ttl': None}, {'op': 'put', 'now': 26, 'key': 'd', 'value': {'n': [3, {'step': 33}]}, 'ttl': 7}, {'op': 'put', 'now': 26, 'key': 'c', 'value': {'n': [-13, {'step': 34}]}, 'ttl': 7}, {'op': 'put', 'now': 28, 'key': 'a', 'value': {'n': [-9, {'step': 35}]}, 'ttl': None}, {'op': 'get', 'now': 28, 'key': 'b', 'expected': None}, {'op': 'len', 'now': 29, 'expected': 2}, {'op': 'put', 'now': 30, 'key': 'b', 'value': {'n': [-94, {'step': 38}]}, 'ttl': 3}, {'op': 'put', 'now': 32, 'key': 'c', 'value': {'n': [-44, {'step': 39}]}, 'ttl': None}, {'op': 'get', 'now': 33, 'key': 'a', 'expected': None}, {'op': 'put', 'now': 33, 'key': 'a', 'value': {'n': [2, {'step': 41}]}, 'ttl': 7}, {'op': 'len', 'now': 34, 'expected': 2}, {'op': 'len', 'now': 34, 'expected': 2}, {'op': 'get', 'now': 34, 'key': 'b', 'expected': None}, {'op': 'get', 'now': 36, 'key': 'e', 'expected': None}, {'op': 'len', 'now': 36, 'expected': 1}, {'op': 'get', 'now': 36, 'key': 'c', 'expected': None}, {'op': 'put', 'now': 37, 'key': 'd', 'value': {'n': [49, {'step': 48}]}, 'ttl': None}, {'op': 'len', 'now': 39, 'expected': 2}, {'op': 'len', 'now': 41, 'expected': 0}, {'op': 'get', 'now': 41, 'key': 'e', 'expected': None}, {'op': 'put', 'now': 41, 'key': 'd', 'value': {'n': [62, {'step': 52}]}, 'ttl': None}, {'op': 'put', 'now': 41, 'key': 'd', 'value': {'n': [57, {'step': 53}]}, 'ttl': None}, {'op': 'put', 'now': 42, 'key': 'b', 'value': {'n': [-86, {'step': 54}]}, 'ttl': 3}, {'op': 'put', 'now': 44, 'key': 'b', 'value': {'n': [-71, {'step': 55}]}, 'ttl': 1}, {'op': 'get', 'now': 46, 'key': 'b', 'expected': None}, {'op': 'len', 'now': 48, 'expected': 0}, {'op': 'get', 'now': 50, 'key': 'a', 'expected': None}, {'op': 'put', 'now': 51, 'key': 'd', 'value': {'n': [-83, {'step': 59}]}, 'ttl': 1}, {'op': 'len', 'now': 53, 'expected': 0}, {'op': 'put', 'now': 55, 'key': 'e', 'value': {'n': [-92, {'step': 61}]}, 'ttl': 7}, {'op': 'len', 'now': 57, 'expected': 1}, {'op': 'len', 'now': 57, 'expected': 1}, {'op': 'put', 'now': 58, 'key': 'e', 'value': {'n': [-12, {'step': 64}]}, 'ttl': 3}, {'op': 'put', 'now': 59, 'key': 'a', 'value': {'n': [-70, {'step': 65}]}, 'ttl': None}, {'op': 'put', 'now': 60, 'key': 'a', 'value': {'n': [68, {'step': 66}]}, 'ttl': None}, {'op': 'get', 'now': 60, 'key': 'b', 'expected': None}, {'op': 'put', 'now': 60, 'key': 'e', 'value': {'n': [42, {'step': 68}]}, 'ttl': 1}, {'op': 'get', 'now': 61, 'key': 'd', 'expected': None}, {'op': 'put', 'now': 61, 'key': 'c', 'value': {'n': [-20, {'step': 70}]}, 'ttl': 3}, {'op': 'get', 'now': 61, 'key': 'a', 'expected': {'n': [68, {'step': 66}]}}, {'op': 'get', 'now': 61, 'key': 'e', 'expected': None}, {'op': 'get', 'now': 61, 'key': 'b', 'expected': None}, {'op': 'len', 'now': 61, 'expected': 2}, {'op': 'get', 'now': 61, 'key': 'b', 'expected': None}, {'op': 'put', 'now': 63, 'key': 'b', 'value': {'n': [-57, {'step': 76}]}, 'ttl': None}, {'op': 'put', 'now': 63, 'key': 'a', 'value': {'n': [-26, {'step': 77}]}, 'ttl': None}, {'op': 'put', 'now': 63, 'key': 'e', 'value': {'n': [89, {'step': 78}]}, 'ttl': 7}, {'op': 'put', 'now': 63, 'key': 'c', 'value': {'n': [55, {'step': 79}]}, 'ttl': 3}, {'op': 'len', 'now': 63, 'expected': 2}, {'op': 'len', 'now': 63, 'expected': 2}, {'op': 'put', 'now': 63, 'key': 'c', 'value': {'n': [77, {'step': 82}]}, 'ttl': None}, {'op': 'len', 'now': 63, 'expected': 2}, {'op': 'len', 'now': 65, 'expected': 2}, {'op': 'get', 'now': 65, 'key': 'a', 'expected': None}, {'op': 'put', 'now': 65, 'key': 'd', 'value': {'n': [-27, {'step': 86}]}, 'ttl': 3}, {'op': 'put', 'now': 65, 'key': 'e', 'value': {'n': [-30, {'step': 87}]}, 'ttl': None}, {'op': 'get', 'now': 65, 'key': 'c', 'expected': None}, {'op': 'len', 'now': 65, 'expected': 2}]}

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
