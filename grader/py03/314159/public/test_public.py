import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.config import overlay_config
CASES = {}

class Public(unittest.TestCase):
    def test_falsy_override(self):
        self.assertEqual(overlay_config({"flag": True, "n": 9}, {"flag": False, "n": 0}), {"flag": False, "n": 0})
    def test_nested_base_unchanged(self):
        base = {"a": {"x": 1, "y": 2}}
        before = copy.deepcopy(base)
        self.assertEqual(overlay_config(base, {"a": {"x": 3}}), {"a": {"x": 3, "y": 2}})
        self.assertEqual(base, before)


if __name__ == '__main__':
    unittest.main()
