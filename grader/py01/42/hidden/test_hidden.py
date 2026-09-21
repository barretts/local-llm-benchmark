import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.cache import TTLCache
CASES = {'capacity': 4, 'default_ttl': 2, 'trace': [{'op': 'get', 'now': 0, 'key': 'b', 'expected': None}, {'op': 'put', 'now': 0, 'key': 'a', 'value': {'n': [74, {'step': 1}]}, 'ttl': None}, {'op': 'put', 'now': 2, 'key': 'a', 'value': {'n': [-76, {'step': 2}]}, 'ttl': 1}, {'op': 'put', 'now': 2, 'key': 'e', 'value': {'n': [-49, {'step': 3}]}, 'ttl': 7}, {'op': 'len', 'now': 2, 'expected': 2}, {'op': 'put', 'now': 3, 'key': 'b', 'value': {'n': [79, {'step': 5}]}, 'ttl': 7}, {'op': 'get', 'now': 4, 'key': 'b', 'expected': {'n': [79, {'step': 5}]}}, {'op': 'get', 'now': 4, 'key': 'a', 'expected': None}, {'op': 'len', 'now': 4, 'expected': 2}, {'op': 'get', 'now': 5, 'key': 'e', 'expected': {'n': [-49, {'step': 3}]}}, {'op': 'put', 'now': 6, 'key': 'd', 'value': {'n': [38, {'step': 10}]}, 'ttl': None}, {'op': 'put', 'now': 8, 'key': 'e', 'value': {'n': [-24, {'step': 11}]}, 'ttl': 3}, {'op': 'put', 'now': 8, 'key': 'a', 'value': {'n': [70, {'step': 12}]}, 'ttl': 1}, {'op': 'put', 'now': 9, 'key': 'b', 'value': {'n': [-74, {'step': 13}]}, 'ttl': 7}, {'op': 'len', 'now': 10, 'expected': 2}, {'op': 'get', 'now': 10, 'key': 'c', 'expected': None}, {'op': 'get', 'now': 10, 'key': 'a', 'expected': None}, {'op': 'put', 'now': 10, 'key': 'b', 'value': {'n': [19, {'step': 17}]}, 'ttl': 7}, {'op': 'put', 'now': 11, 'key': 'c', 'value': {'n': [97, {'step': 18}]}, 'ttl': None}, {'op': 'put', 'now': 11, 'key': 'c', 'value': {'n': [3, {'step': 19}]}, 'ttl': 3}, {'op': 'put', 'now': 11, 'key': 'e', 'value': {'n': [84, {'step': 20}]}, 'ttl': 3}, {'op': 'len', 'now': 11, 'expected': 3}, {'op': 'put', 'now': 13, 'key': 'c', 'value': {'n': [-64, {'step': 22}]}, 'ttl': 1}, {'op': 'len', 'now': 14, 'expected': 1}, {'op': 'get', 'now': 16, 'key': 'b', 'expected': {'n': [19, {'step': 17}]}}, {'op': 'len', 'now': 16, 'expected': 1}, {'op': 'put', 'now': 16, 'key': 'b', 'value': {'n': [61, {'step': 26}]}, 'ttl': 1}, {'op': 'put', 'now': 18, 'key': 'd', 'value': {'n': [-2, {'step': 27}]}, 'ttl': 7}, {'op': 'put', 'now': 19, 'key': 'a', 'value': {'n': [75, {'step': 28}]}, 'ttl': 3}, {'op': 'put', 'now': 20, 'key': 'c', 'value': {'n': [12, {'step': 29}]}, 'ttl': 1}, {'op': 'put', 'now': 22, 'key': 'c', 'value': {'n': [29, {'step': 30}]}, 'ttl': 1}, {'op': 'get', 'now': 22, 'key': 'e', 'expected': None}, {'op': 'put', 'now': 22, 'key': 'c', 'value': {'n': [96, {'step': 32}]}, 'ttl': 1}, {'op': 'get', 'now': 22, 'key': 'd', 'expected': {'n': [-2, {'step': 27}]}}, {'op': 'put', 'now': 22, 'key': 'c', 'value': {'n': [-21, {'step': 34}]}, 'ttl': 1}, {'op': 'put', 'now': 22, 'key': 'e', 'value': {'n': [-79, {'step': 35}]}, 'ttl': None}, {'op': 'put', 'now': 24, 'key': 'e', 'value': {'n': [97, {'step': 36}]}, 'ttl': 1}, {'op': 'len', 'now': 24, 'expected': 2}, {'op': 'get', 'now': 24, 'key': 'e', 'expected': {'n': [97, {'step': 36}]}}, {'op': 'put', 'now': 26, 'key': 'e', 'value': {'n': [94, {'step': 39}]}, 'ttl': 1}, {'op': 'len', 'now': 27, 'expected': 0}, {'op': 'len', 'now': 29, 'expected': 0}, {'op': 'put', 'now': 29, 'key': 'a', 'value': {'n': [-13, {'step': 42}]}, 'ttl': None}, {'op': 'put', 'now': 29, 'key': 'a', 'value': {'n': [-81, {'step': 43}]}, 'ttl': None}, {'op': 'put', 'now': 29, 'key': 'a', 'value': {'n': [-15, {'step': 44}]}, 'ttl': None}, {'op': 'get', 'now': 29, 'key': 'd', 'expected': None}, {'op': 'put', 'now': 29, 'key': 'e', 'value': {'n': [48, {'step': 46}]}, 'ttl': 7}, {'op': 'len', 'now': 29, 'expected': 2}, {'op': 'put', 'now': 29, 'key': 'a', 'value': {'n': [69, {'step': 48}]}, 'ttl': 7}, {'op': 'len', 'now': 30, 'expected': 2}, {'op': 'put', 'now': 32, 'key': 'a', 'value': {'n': [-84, {'step': 50}]}, 'ttl': 7}, {'op': 'put', 'now': 33, 'key': 'b', 'value': {'n': [-50, {'step': 51}]}, 'ttl': 1}, {'op': 'put', 'now': 35, 'key': 'd', 'value': {'n': [-53, {'step': 52}]}, 'ttl': 3}, {'op': 'put', 'now': 37, 'key': 'a', 'value': {'n': [14, {'step': 53}]}, 'ttl': None}, {'op': 'put', 'now': 37, 'key': 'a', 'value': {'n': [93, {'step': 54}]}, 'ttl': 1}, {'op': 'len', 'now': 37, 'expected': 2}, {'op': 'put', 'now': 39, 'key': 'd', 'value': {'n': [-84, {'step': 56}]}, 'ttl': 1}, {'op': 'put', 'now': 41, 'key': 'd', 'value': {'n': [-32, {'step': 57}]}, 'ttl': 7}, {'op': 'len', 'now': 42, 'expected': 1}, {'op': 'put', 'now': 44, 'key': 'b', 'value': {'n': [-24, {'step': 59}]}, 'ttl': 1}, {'op': 'put', 'now': 44, 'key': 'c', 'value': {'n': [-85, {'step': 60}]}, 'ttl': None}, {'op': 'put', 'now': 46, 'key': 'a', 'value': {'n': [31, {'step': 61}]}, 'ttl': None}, {'op': 'put', 'now': 46, 'key': 'e', 'value': {'n': [-82, {'step': 62}]}, 'ttl': 1}, {'op': 'put', 'now': 48, 'key': 'e', 'value': {'n': [-36, {'step': 63}]}, 'ttl': None}, {'op': 'len', 'now': 48, 'expected': 1}, {'op': 'get', 'now': 49, 'key': 'b', 'expected': None}, {'op': 'put', 'now': 50, 'key': 'c', 'value': {'n': [2, {'step': 66}]}, 'ttl': 1}, {'op': 'len', 'now': 51, 'expected': 0}, {'op': 'put', 'now': 51, 'key': 'd', 'value': {'n': [60, {'step': 68}]}, 'ttl': None}, {'op': 'put', 'now': 51, 'key': 'e', 'value': {'n': [-32, {'step': 69}]}, 'ttl': 1}, {'op': 'put', 'now': 52, 'key': 'b', 'value': {'n': [-5, {'step': 70}]}, 'ttl': 3}, {'op': 'len', 'now': 52, 'expected': 2}, {'op': 'put', 'now': 53, 'key': 'e', 'value': {'n': [-23, {'step': 72}]}, 'ttl': None}, {'op': 'get', 'now': 53, 'key': 'a', 'expected': None}, {'op': 'put', 'now': 53, 'key': 'c', 'value': {'n': [-27, {'step': 74}]}, 'ttl': 1}, {'op': 'put', 'now': 54, 'key': 'c', 'value': {'n': [30, {'step': 75}]}, 'ttl': 7}, {'op': 'put', 'now': 55, 'key': 'a', 'value': {'n': [63, {'step': 76}]}, 'ttl': 7}, {'op': 'put', 'now': 56, 'key': 'a', 'value': {'n': [-14, {'step': 77}]}, 'ttl': 1}, {'op': 'put', 'now': 57, 'key': 'd', 'value': {'n': [42, {'step': 78}]}, 'ttl': 7}, {'op': 'put', 'now': 57, 'key': 'a', 'value': {'n': [77, {'step': 79}]}, 'ttl': 1}, {'op': 'get', 'now': 57, 'key': 'e', 'expected': None}, {'op': 'len', 'now': 57, 'expected': 3}, {'op': 'get', 'now': 57, 'key': 'c', 'expected': {'n': [30, {'step': 75}]}}, {'op': 'get', 'now': 57, 'key': 'b', 'expected': None}, {'op': 'put', 'now': 57, 'key': 'c', 'value': {'n': [44, {'step': 84}]}, 'ttl': 7}, {'op': 'put', 'now': 57, 'key': 'b', 'value': {'n': [-54, {'step': 85}]}, 'ttl': 7}, {'op': 'put', 'now': 57, 'key': 'c', 'value': {'n': [6, {'step': 86}]}, 'ttl': 1}, {'op': 'put', 'now': 58, 'key': 'a', 'value': {'n': [-2, {'step': 87}]}, 'ttl': None}, {'op': 'put', 'now': 60, 'key': 'b', 'value': {'n': [18, {'step': 88}]}, 'ttl': 3}, {'op': 'put', 'now': 61, 'key': 'b', 'value': {'n': [-93, {'step': 89}]}, 'ttl': 1}]}

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
