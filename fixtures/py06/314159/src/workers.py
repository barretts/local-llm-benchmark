import asyncio

async def map_limited(items, worker, limit) -> list:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    tasks = [asyncio.create_task(worker(item)) for item in items]
    results = []
    for task in asyncio.as_completed(tasks):
        results.append(await task)
    return results
