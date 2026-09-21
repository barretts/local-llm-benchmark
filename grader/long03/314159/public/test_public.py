import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.settlement import settle
import inspect
def total(lines, rate):
    return settle([(Decimal(p), q, Decimal(d)) for p, q, d in lines], Decimal(rate))
CASES = {}

class Public(unittest.TestCase):
    def test_half_cent_rounds_up(self):
        self.assertEqual(total([["0.005", 1, "0"]], "0"), Decimal("0.01"))
    def test_lines_round_before_aggregate(self):
        self.assertEqual(total([["0.005", 1, "0"], ["0.005", 1, "0"]], "0"), Decimal("0.02"))


if __name__ == '__main__':
    unittest.main()
