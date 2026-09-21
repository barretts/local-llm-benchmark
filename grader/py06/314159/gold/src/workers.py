import asyncio

async def map_limited(items, worker, limit) -> list:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    items = list(items)
    if not items:
        return []
    semaphore = asyncio.Semaphore(limit)
    async def run(item):
        async with semaphore:
            return await worker(item)
    tasks = [asyncio.create_task(run(item)) for item in items]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
