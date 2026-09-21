import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.models import LineItem
from src.money import line_total
from src.invoice import invoice_total
from dataclasses import FrozenInstanceError
def total(lines, rate):
    return invoice_total([LineItem(Decimal(p), q, Decimal(d)) for p, q, d in lines], Decimal(rate))
CASES = {}

class Public(unittest.TestCase):
    def test_half_cent_rounds_up(self):
        self.assertEqual(line_total(Decimal("0.005"), 1, Decimal("0")), Decimal("0.01"))
    def test_lines_round_before_aggregate(self):
        self.assertEqual(total([["0.005", 1, "0"], ["0.005", 1, "0"]], "0"), Decimal("0.02"))


if __name__ == '__main__':
    unittest.main()
