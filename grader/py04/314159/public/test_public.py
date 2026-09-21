import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.retry import call_with_retry
CASES = {}

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


if __name__ == '__main__':
    unittest.main()
