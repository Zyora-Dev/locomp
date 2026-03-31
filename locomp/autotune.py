"""
Locust Autotune — automatically find the fastest kernel configuration.

Usage:
    @locomp.autotune(
        configs=[
            locomp.Config(BLOCK_M=16, BLOCK_N=16, grid=lambda M, N, **kw: (N//16, M//16), tg=(32, 4)),
            locomp.Config(BLOCK_M=32, BLOCK_N=32, grid=lambda M, N, **kw: (N//32, M//32), tg=(32, 8)),
        ],
        key=["M", "N"],
    )
    @locomp.kernel
    def matmul(A, B, C, M: locomp.constexpr, N: locomp.constexpr,
               BLOCK_M: locomp.constexpr, BLOCK_N: locomp.constexpr):
        ...

    matmul(A_t, B_t, C_t, 256, 256)  # first call benchmarks configs, caches best
"""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

# --- Persistent cache ---

_CACHE_DIR = Path.home() / ".cache" / "locomp"
_CACHE_FILE = _CACHE_DIR / "autotune.json"
_disk_cache: dict | None = None


def _get_gpu_name() -> str:
    """Get the Metal GPU device name (e.g. 'Apple M1')."""
    try:
        from locomp.backends.metal_runtime import get_runtime
        return get_runtime().device_name
    except Exception:
        return "unknown"


def _load_disk_cache() -> dict:
    """Load persistent autotune cache from disk."""
    global _disk_cache
    if _disk_cache is not None:
        return _disk_cache
    if _CACHE_FILE.exists():
        try:
            _disk_cache = json.loads(_CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            _disk_cache = {}
    else:
        _disk_cache = {}
    return _disk_cache


def _save_disk_cache():
    """Write persistent autotune cache to disk."""
    if _disk_cache is None:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(_disk_cache, indent=2))


def clear_cache():
    """Clear the persistent autotune cache (all kernels, all GPUs)."""
    global _disk_cache
    _disk_cache = {}
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()


class Config:
    """A single autotuning configuration.

    Args:
        grid: callable(**constexprs) → grid tuple, or static tuple
        tg: threadgroup size tuple (e.g. (32, 4) for 128 threads)
        **constexprs: constexpr parameter overrides. Values can be:
            - int: static value
            - callable(**constexprs): derived from other constexprs
    """

    def __init__(self, grid, tg=(256,), **constexprs):
        self.static = {}
        self.derived = {}
        for k, v in constexprs.items():
            if callable(v):
                self.derived[k] = v
            else:
                self.static[k] = v
        self._grid = grid
        self.tg = tg

    def resolve(self, user_constexprs: dict) -> dict:
        """Merge user constexprs with config overrides, resolve derived values."""
        merged = {**user_constexprs, **self.static}
        for k, fn in self.derived.items():
            merged[k] = fn(**merged)
        return merged

    def get_grid(self, all_constexprs: dict) -> tuple:
        if callable(self._grid):
            return self._grid(**all_constexprs)
        return self._grid

    def __repr__(self):
        parts = [f"{k}={v}" for k, v in self.static.items()]
        parts.append(f"tg={self.tg}")
        return f"Config({', '.join(parts)})"


class AutotunedKernel:
    """Wraps a KernelLauncher with autotuning."""

    def __init__(self, launcher, configs: list[Config], key: list[str],
                 warmup: int = 2, rep: int = 5):
        self.launcher = launcher
        self.configs = configs
        self.key = key
        self.warmup = warmup
        self.rep = rep
        self._cache: dict[tuple, int] = {}
        self._gpu_name: str | None = None
        # Extract Python param names from the original function signature
        sig = inspect.signature(launcher.func)
        self._py_names = list(sig.parameters.keys())

    def _disk_key(self, cache_key: tuple) -> str:
        """Build a string key for the persistent cache: kernel:gpu:problem_dims."""
        if self._gpu_name is None:
            self._gpu_name = _get_gpu_name()
        key_str = ",".join(f"{k}={v}" for k, v in zip(self.key, cache_key))
        return f"{self.launcher.func_name}:{self._gpu_name}:{key_str}"

    def __call__(self, *args):
        # Map positional user args to Python param names
        # User may pass fewer args than the kernel signature — Config provides the rest
        user_values = {}
        py_constexprs = {}
        for i, arg in enumerate(args):
            name = self._py_names[i]
            user_values[name] = arg
            if isinstance(arg, (int, float)) and not hasattr(arg, '_metal_buffer'):
                py_constexprs[name] = int(arg)

        # Cache key from user-specified key params
        cache_key = tuple(py_constexprs.get(k, 0) for k in self.key)

        # Check in-memory cache first
        if cache_key in self._cache:
            return self._dispatch(user_values, py_constexprs,
                                  self.configs[self._cache[cache_key]])

        # Check persistent disk cache
        dk = self._disk_key(cache_key)
        disk = _load_disk_cache()
        if dk in disk:
            idx = disk[dk]
            if 0 <= idx < len(self.configs):
                self._cache[cache_key] = idx
                return self._dispatch(user_values, py_constexprs, self.configs[idx])

        # Benchmark all configs
        best_time = float('inf')
        best_idx = 0

        for i, config in enumerate(self.configs):
            try:
                # Warmup
                for _ in range(self.warmup):
                    self._dispatch(user_values, py_constexprs, config)

                # Benchmark
                times = []
                for _ in range(self.rep):
                    t0 = time.perf_counter()
                    self._dispatch(user_values, py_constexprs, config)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                median = sorted(times)[self.rep // 2]

                if median < best_time:
                    best_time = median
                    best_idx = i
            except Exception as e:
                # Config failed (e.g. grid size 0, too many threads) — skip it
                pass

        self._cache[cache_key] = best_idx
        cfg = self.configs[best_idx]
        key_desc = dict(zip(self.key, cache_key))
        print(f"[autotune] {key_desc} -> {cfg} ({best_time*1000:.3f}ms)")

        # Persist to disk
        disk[dk] = best_idx
        _save_disk_cache()

        return self._dispatch(user_values, py_constexprs, cfg)

    def _dispatch(self, user_values, py_constexprs, config):
        """Dispatch kernel with a specific config."""
        resolved = config.resolve(py_constexprs)
        grid = config.get_grid(resolved)
        tg = config.tg

        # Build full positional args in function signature order.
        # User-provided tensor args stay as-is; constexpr args come from resolved dict.
        full_args = []
        for name in self._py_names:
            if name in user_values and hasattr(user_values[name], '_metal_buffer'):
                # Tensor arg — pass through
                full_args.append(user_values[name])
            elif name in resolved:
                # Constexpr — from user or config
                full_args.append(resolved[name])
            elif name in user_values:
                # Tensor passed as ndarray or other
                full_args.append(user_values[name])
            else:
                raise ValueError(f"No value for parameter '{name}' — "
                                 f"not provided by user or Config")

        return self.launcher[grid, tg](*full_args)

    # Forward attribute access to underlying launcher
    def __getattr__(self, name):
        return getattr(self.launcher, name)


def autotune(configs: list[Config], key: list[str],
             warmup: int = 2, rep: int = 5):
    """Decorator for autotuning kernel configurations.

    Args:
        configs: List of Config objects to try
        key: List of constexpr param names that form the cache key
            (e.g. ["M", "N"] — different sizes get different best configs)
        warmup: Number of warmup runs per config (default 2)
        rep: Number of timed runs per config (default 5)
    """
    def decorator(kernel_launcher):
        return AutotunedKernel(kernel_launcher, configs, key, warmup, rep)
    return decorator
