"""
locomp Kernel Cache — Disk-based MSL cache for compiled kernels.

Caches the generated MSL source + buffer_map to disk so that on subsequent
Python process launches the Python→IR→MSL compilation chain is skipped.
Metal's own driver-level shader cache handles the MSL→GPU-binary step.

Cache layout:
  ~/.cache/locomp/<func_name>_<hash>.json
  {"version": 1, "msl": "...", "buffer_map": {"1": 0, "2": 1}}

Cache key: SHA-256 of func source + func name + sorted constexpr values.
Stale entries are never loaded (different source = different key).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from typing import Any

_CACHE_VERSION = 1
_CACHE_DIR_ENV = "LOCOMP_CACHE_DIR"


def _default_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "locomp")


def _cache_dir() -> str:
    return os.environ.get(_CACHE_DIR_ENV, _default_cache_dir())


def _make_key(func, constexpr_values: dict) -> str:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = func.__name__
    payload = f"{func.__name__}:{source}:{sorted(constexpr_values.items())}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _cache_path(func_name: str, key: str) -> str:
    return os.path.join(_cache_dir(), f"{func_name}_{key}.json")


def get(func, constexpr_values: dict) -> tuple[str, dict] | None:
    """Return (msl_source, buffer_map) from disk cache, or None on miss."""
    key = _make_key(func, constexpr_values)
    path = _cache_path(func.__name__, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("version") != _CACHE_VERSION:
            return None
        msl = data["msl"]
        # JSON keys are always strings — restore int keys
        buffer_map = {int(k): v for k, v in data["buffer_map"].items()}
        return msl, buffer_map
    except Exception:
        return None


def put(func, constexpr_values: dict, msl: str, buffer_map: dict[int, int]) -> None:
    """Save (msl_source, buffer_map) to disk cache."""
    key = _make_key(func, constexpr_values)
    path = _cache_path(func.__name__, key)
    os.makedirs(_cache_dir(), exist_ok=True)
    data = {
        "version": _CACHE_VERSION,
        "msl": msl,
        "buffer_map": {str(k): v for k, v in buffer_map.items()},
    }
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)  # atomic write
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def clear(func=None) -> int:
    """Clear all cache entries (or entries for a specific func). Returns count deleted."""
    d = _cache_dir()
    if not os.path.isdir(d):
        return 0
    count = 0
    prefix = f"{func.__name__}_" if func is not None else ""
    for name in os.listdir(d):
        if name.endswith(".json") and name.startswith(prefix):
            try:
                os.unlink(os.path.join(d, name))
                count += 1
            except Exception:
                pass
    return count


def cache_dir() -> str:
    """Return the current cache directory path."""
    return _cache_dir()
