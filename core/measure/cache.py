"""
Measurements of mask outlines, remembered by the outline itself.

The annotations table and the Sizes tab both show measurements of the
current image's masks and refresh after every edit. Measuring a mask costs
~35 µs, so a busy scan (5,000 masks) took ~0.2 s per refresh, twice. Keyed
by the outline's bytes (not the mask id: ids are renumbered by "reindex",
and an edited mask gets new bytes), a refresh only measures what changed.
"""
from collections import OrderedDict
from threading import Lock

import numpy as np

from core.measure.morphometrics import measure

_MAX_ENTRIES = 50_000          # ~10 busy images; oldest are dropped first
_cache: "OrderedDict[bytes, dict]" = OrderedDict()
_lock = Lock()
_MISSING = object()


def measure_cached(polygon):
    """morphometrics.measure(polygon), remembered by the outline's bytes."""
    key = np.ascontiguousarray(polygon, dtype=np.float32).tobytes()
    with _lock:
        hit = _cache.get(key, _MISSING)
        if hit is not _MISSING:
            _cache.move_to_end(key)
            return hit
    value = measure(polygon)
    with _lock:
        _cache[key] = value
        if len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)
    return value


def clear():
    with _lock:
        _cache.clear()


def size() -> int:
    return len(_cache)
