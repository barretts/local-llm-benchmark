import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.workers import map_limited
CASES = [{'items': [-60, -74, 0, -88, 73, -67, -38, -98, -53, -96, 40], 'limit': 2}, {'items': [70, 47, -2, 94, -54], 'limit': 5}, {'items': [-24, -77, 27, 45, -66, 13], 'limit': 3}, {'items': [-23, -48, -53, 59, -92, -1], 'limit': 5}, {'items': [-23, 48], 'limit': 5}, {'items': [-37, 35], 'limit': 5}, {'items': [-10, 56, -44, -86, -89], 'limit': 3}, {'items': [84, 92, -2, 12], 'limit': 1}, {'items': [70, -12, 81, 81, 88, 26, -14, 31, -31, 77], 'limit': 3}, {'items': [54, -80], 'limit': 3}, {'items': [46, 47], 'limit': 1}, {'items': [-20, -25], 'limit': 3}, {'items': [33, -59, -4, -73, -84], 'limit': 3}, {'items': [-9, 24, -58, 25, 50, 73, -74, -6, 41, -63], 'limit': 2}, {'items': [75, 69, 57, 40, 78, -78, -69, 37, -54, 77, -63], 'limit': 4}]

class Hidden(unittest.IsolatedAsyncioTestCase):
    async def test_limit_one_order_and_peak(self):
        active, peak = 0, 0
        async def worker(item):
            nonlocal active, peak
            active += 1; peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                return item + 1
            finally:
                active -= 1
        self.assertEqual(await map_limited([3, 2, 1], worker, 1), [4, 3, 2])
        self.assertEqual((peak, active), (1, 0))

    async def test_empty_never_calls_worker(self):
        async def worker(item):
            self.fail("worker called on empty input")
        self.assertEqual(await map_limited([], worker, 4), [])

    async def test_multiple_limits_and_seeded_inputs(self):
        for index, case in enumerate(CASES):
            active, peak = 0, 0
            async def worker(item):
                nonlocal active, peak
                active += 1; peak = max(peak, active)
                try:
                    for _ in range(abs(item) % 4 + 1):
                        await asyncio.sleep(0)
                    return {"value": item * item}
                finally:
                    active -= 1
            with self.subTest(index=index):
                result = await asyncio.wait_for(map_limited(case["items"], worker, case["limit"]), 3)
                self.assertEqual(result, [{"value": value * value} for value in case["items"]])
                self.assertLessEqual(peak, case["limit"])
                self.assertEqual(active, 0)

    async def test_worker_failure_cancels_and_awaits_cleanup(self):
        ready, blocked = asyncio.Event(), asyncio.Event()
        running, cleaned = set(), set()
        owned = set()
        error = RuntimeError("observed failure")
        async def worker(item):
            owned.add(asyncio.current_task())
            running.add(item)
            if item == 1:
                ready.set()
            try:
                if item == 0:
                    await ready.wait()
                    raise error
                await blocked.wait()
                return item
            finally:
                await asyncio.sleep(0)
                running.discard(item)
                cleaned.add(item)
        task = asyncio.create_task(map_limited([0, 1, 2, 3], worker, 2))
        try:
            with self.assertRaises(RuntimeError) as captured:
                await asyncio.wait_for(task, 3)
            self.assertIs(captured.exception, error)
            self.assertEqual(running, set())
            self.assertTrue({0, 1}.issubset(cleaned))
            self.assertTrue(all(t.done() for t in owned))
        finally:
            blocked.set()
            for owned_task in owned:
                if not owned_task.done():
                    owned_task.cancel()
            await asyncio.gather(*owned, return_exceptions=True)

    async def test_caller_cancellation_awaits_owned_workers(self):
        started, blocked = asyncio.Event(), asyncio.Event()
        running, owned = set(), set()
        async def worker(item):
            owned.add(asyncio.current_task())
            running.add(item)
            started.set()
            try:
                await blocked.wait()
                return item
            finally:
                await asyncio.sleep(0)
                running.discard(item)
        task = asyncio.create_task(map_limited(list(range(5)), worker, 2))
        try:
            await asyncio.wait_for(started.wait(), 3)
            for _ in range(15):
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(running, set())
            self.assertTrue(all(t.done() for t in owned))
        finally:
            blocked.set()
            for owned_task in owned:
                if not owned_task.done():
                    owned_task.cancel()
            await asyncio.gather(*owned, return_exceptions=True)

    async def test_blocked_release_controls_admission(self):
        gates = [asyncio.Event() for _ in range(4)]
        started = []
        async def worker(item):
            started.append(item)
            await gates[item].wait()
            return item
        task = asyncio.create_task(map_limited([0, 1, 2, 3], worker, 2))
        try:
            for _ in range(20):
                await asyncio.sleep(0)
            self.assertEqual(started, [0, 1])
            gates[1].set()
            for _ in range(20):
                await asyncio.sleep(0)
            self.assertEqual(started, [0, 1, 2])
            gates[0].set()
            for _ in range(20):
                await asyncio.sleep(0)
            self.assertEqual(started, [0, 1, 2, 3])
        finally:
            for gate in gates:
                gate.set()
            result = await asyncio.wait_for(task, 3)
        self.assertEqual(result, [0, 1, 2, 3])

    async def test_invalid_limit(self):
        async def worker(item):
            self.fail("worker called for invalid limit")
        for limit in (0, -1, 1.5, True, "2", None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                await map_limited([], worker, limit)

    async def test_single_worker_exception_identity(self):
        error = LookupError("single worker")
        async def worker(item):
            raise error
        with self.assertRaises(LookupError) as captured:
            await map_limited([1], worker, 1)
        self.assertIs(captured.exception, error)


if __name__ == '__main__':
    unittest.main()
