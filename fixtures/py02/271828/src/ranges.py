def merge_ranges(ranges):
    for start, end in ranges:
        if type(start) is not int or type(end) is not int or end < start:
            raise ValueError("invalid range")
    ranges.sort()
    result = []
    for start, end in ranges:
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result
