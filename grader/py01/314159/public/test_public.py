import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.cache import TTLCache
CASES = {}

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


if __name__ == '__main__':
    unittest.main()
