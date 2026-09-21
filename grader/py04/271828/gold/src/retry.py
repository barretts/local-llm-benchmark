import math

def call_with_retry(fn, *, max_attempts, base_delay, retry_on, sleep):
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("invalid max_attempts")
    try:
        valid = not isinstance(base_delay, bool) and math.isfinite(base_delay) and base_delay >= 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("invalid base_delay")
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as error:
            if not isinstance(error, retry_on):
                raise
            if attempt + 1 == max_attempts:
                raise
            sleep(base_delay * 2 ** attempt)
