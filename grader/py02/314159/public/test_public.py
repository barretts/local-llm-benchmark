import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.ranges import merge_ranges
CASES = {}

class Public(unittest.TestCase):
    def test_adjacent_remain_separate(self):
        self.assertEqual(merge_ranges([(1, 3), (3, 5)]), [(1, 3), (3, 5)])
    def test_input_unchanged(self):
        inputs = [[5, 8], [1, 4], [2, 6]]
        before = copy.deepcopy(inputs)
        self.assertEqual(merge_ranges(inputs), [(1, 8)])
        self.assertEqual(inputs, before)


if __name__ == '__main__':
    unittest.main()
