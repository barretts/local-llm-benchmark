import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.retry import call_with_retry
CASES = [{'attempts': 6, 'failures': 1, 'delay': 0, 'calls': 2, 'sleeps': [0], 'result': 0, 'success': True}, {'attempts': 2, 'failures': 1, 'delay': 0.125, 'calls': 2, 'sleeps': [0.125], 'result': None, 'success': True}, {'attempts': 6, 'failures': 1, 'delay': 2, 'calls': 2, 'sleeps': [2], 'result': None, 'success': True}, {'attempts': 1, 'failures': 0, 'delay': 0.125, 'calls': 1, 'sleeps': [], 'result': False, 'success': True}, {'attempts': 5, 'failures': 4, 'delay': 0, 'calls': 5, 'sleeps': [0, 0, 0, 0], 'result': 4, 'success': True}, {'attempts': 2, 'failures': 3, 'delay': 0.125, 'calls': 2, 'sleeps': [0.125], 'result': 'ok', 'success': False}, {'attempts': 5, 'failures': 2, 'delay': 0, 'calls': 3, 'sleeps': [0, 0], 'result': False, 'success': True}, {'attempts': 6, 'failures': 6, 'delay': 0.5, 'calls': 6, 'sleeps': [0.5, 1.0, 2.0, 4.0, 8.0], 'result': 0, 'success': False}, {'attempts': 2, 'failures': 1, 'delay': 0.5, 'calls': 2, 'sleeps': [0.5], 'result': None, 'success': True}, {'attempts': 1, 'failures': 1, 'delay': 0, 'calls': 1, 'sleeps': [], 'result': 0, 'success': False}, {'attempts': 3, 'failures': 4, 'delay': 0.5, 'calls': 3, 'sleeps': [0.5, 1.0], 'result': None, 'success': False}, {'attempts': 6, 'failures': 7, 'delay': 0, 'calls': 6, 'sleeps': [0, 0, 0, 0, 0], 'result': 'ok', 'success': False}, {'attempts': 1, 'failures': 2, 'delay': 0.5, 'calls': 1, 'sleeps': [], 'result': 12, 'success': False}, {'attempts': 3, 'failures': 4, 'delay': 0.125, 'calls': 3, 'sleeps': [0.125, 0.25], 'result': None, 'success': False}, {'attempts': 1, 'failures': 2, 'delay': 0.125, 'calls': 1, 'sleeps': [], 'result': 0, 'success': False}, {'attempts': 1, 'failures': 0, 'delay': 0, 'calls': 1, 'sleeps': [], 'result': 'ok', 'success': True}, {'attempts': 3, 'failures': 3, 'delay': 0.5, 'calls': 3, 'sleeps': [0.5, 1.0], 'result': False, 'success': False}, {'attempts': 3, 'failures': 2, 'delay': 0.125, 'calls': 3, 'sleeps': [0.125, 0.25], 'result': 0, 'success': True}, {'attempts': 6, 'failures': 1, 'delay': 0.125, 'calls': 2, 'sleeps': [0.125], 'result': 18, 'success': True}, {'attempts': 6, 'failures': 3, 'delay': 0.125, 'calls': 4, 'sleeps': [0.125, 0.25, 0.5], 'result': 'ok', 'success': True}, {'attempts': 4, 'failures': 2, 'delay': 0.125, 'calls': 3, 'sleeps': [0.125, 0.25], 'result': 0, 'success': True}, {'attempts': 1, 'failures': 0, 'delay': 0, 'calls': 1, 'sleeps': [], 'result': 0, 'success': True}, {'attempts': 4, 'failures': 2, 'delay': 0, 'calls': 3, 'sleeps': [0, 0], 'result': False, 'success': True}, {'attempts': 5, 'failures': 5, 'delay': 0.5, 'calls': 5, 'sleeps': [0.5, 1.0, 2.0, 4.0], 'result': False, 'success': False}, {'attempts': 6, 'failures': 7, 'delay': 2, 'calls': 6, 'sleeps': [2, 4, 8, 16, 32], 'result': 'ok', 'success': False}, {'attempts': 2, 'failures': 2, 'delay': 0.125, 'calls': 2, 'sleeps': [0.125], 'result': False, 'success': False}, {'attempts': 6, 'failures': 4, 'delay': 2, 'calls': 5, 'sleeps': [2, 4, 8, 16], 'result': 26, 'success': True}, {'attempts': 4, 'failures': 2, 'delay': 0.125, 'calls': 3, 'sleeps': [0.125, 0.25], 'result': False, 'success': True}, {'attempts': 5, 'failures': 3, 'delay': 0, 'calls': 4, 'sleeps': [0, 0, 0], 'result': None, 'success': True}, {'attempts': 1, 'failures': 0, 'delay': 0.125, 'calls': 1, 'sleeps': [], 'result': 'ok', 'success': True}, {'attempts': 5, 'failures': 0, 'delay': 2, 'calls': 1, 'sleeps': [], 'result': 'ok', 'success': True}, {'attempts': 5, 'failures': 3, 'delay': 0.5, 'calls': 4, 'sleeps': [0.5, 1.0, 2.0], 'result': 31, 'success': True}, {'attempts': 1, 'failures': 2, 'delay': 0, 'calls': 1, 'sleeps': [], 'result': 32, 'success': False}, {'attempts': 3, 'failures': 2, 'delay': 0, 'calls': 3, 'sleeps': [0, 0], 'result': 0, 'success': True}, {'attempts': 4, 'failures': 1, 'delay': 2, 'calls': 2, 'sleeps': [2], 'result': None, 'success': True}]

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


if __name__ == '__main__':
    unittest.main()
