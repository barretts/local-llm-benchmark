from collections import OrderedDict
import math

def _ttl(value):
    try:
        valid = not isinstance(value, bool) and math.isfinite(value) and value > 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("TTL must be finite and positive")
    return value

class TTLCache:
    def __init__(self, capacity, default_ttl, clock):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.default_ttl = _ttl(default_ttl)
        self.clock = clock
        self.entries = OrderedDict()

    def _purge(self, now):
        for key, (_, expiry) in list(self.entries.items()):
            if now > expiry:
                del self.entries[key]

    def put(self, key, value, ttl=None):
        duration = self.default_ttl if ttl is None else _ttl(ttl)
        self.entries[key] = (value, self.clock() + duration)
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

    def get(self, key, default=None):
        self._purge(self.clock())
        if key not in self.entries:
            return default
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def __len__(self):
        self._purge(self.clock())
        return len(self.entries)
