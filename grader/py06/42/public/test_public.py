import unittest
import copy
import math
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from src.workers import map_limited
CASES = {}

class Public(unittest.IsolatedAsyncioTestCase):
    async def test_reverse_completion_keeps_input_order(self):
        gates = [asyncio.Event() for _ in range(3)]
        started = set()
        async def worker(item):
            started.add(item)
            await gates[item].wait()
            return item * 10
        task = asyncio.create_task(map_limited([0, 1, 2], worker, 3))
        try:
            for _ in range(20):
                await asyncio.sleep(0)
            self.assertEqual(started, {0, 1, 2})
            for index in (2, 1, 0):
                gates[index].set()
                for _ in range(5):
                    await asyncio.sleep(0)
            self.assertEqual(await asyncio.wait_for(task, 3), [0, 10, 20])
        finally:
            for gate in gates:
                gate.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_active_count_is_bounded(self):
        gate = asyncio.Event()
        active = 0
        peak = 0
        async def worker(item):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await gate.wait()
                return item
            finally:
                active -= 1
        task = asyncio.create_task(map_limited(list(range(7)), worker, 2))
        try:
            for _ in range(30):
                await asyncio.sleep(0)
            self.assertLessEqual(peak, 2)
            self.assertEqual(active, 2)
        finally:
            gate.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)


if __name__ == '__main__':
    unittest.main()
