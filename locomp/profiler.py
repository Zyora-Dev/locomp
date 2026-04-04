"""
locomp Kernel Profiler — measure wall-clock time per kernel call.

Usage:
    with locomp.profile() as p:
        kernel_a[(N,)](x, y, N=N)
        kernel_b[(N,)](y, z, N=N)

    print(p.report())

Or as a decorator:
    @locomp.profile()
    def my_fn():
        kernel[(N,)](x, out, N=N)
    my_fn()

Each kernel call inside the context is recorded with:
  - kernel function name
  - grid size
  - wall-clock time (ms)  — always available
  - call count (for aggregates)

Wall-clock time is measured around each individual kernel dispatch
(including the GPU sync for accurate timing). For production use where
you don't want per-call sync overhead, use the benchmark utilities instead.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable


class ProfileEntry:
    __slots__ = ("name", "grid", "ms", "calls")

    def __init__(self, name: str, grid: tuple, ms: float):
        self.name = name
        self.grid = grid
        self.ms = ms
        self.calls = 1


class ProfileResult:
    """Result of a profiling session."""

    def __init__(self, entries: list[ProfileEntry]):
        self._entries = entries

    @property
    def entries(self) -> list[ProfileEntry]:
        return list(self._entries)

    def total_ms(self) -> float:
        return sum(e.ms for e in self._entries)

    def by_kernel(self) -> dict[str, dict]:
        """Aggregate stats per kernel name."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            if e.name not in agg:
                agg[e.name] = {"calls": 0, "total_ms": 0.0, "min_ms": float("inf"), "max_ms": 0.0}
            d = agg[e.name]
            d["calls"] += 1
            d["total_ms"] += e.ms
            d["min_ms"] = min(d["min_ms"], e.ms)
            d["max_ms"] = max(d["max_ms"], e.ms)
        for d in agg.values():
            d["avg_ms"] = d["total_ms"] / d["calls"]
        return agg

    def report(self) -> str:
        """Human-readable profiling report."""
        lines = []
        lines.append(f"{'Kernel':<28} {'Calls':>6}  {'Total ms':>10}  {'Avg ms':>9}  {'Min ms':>9}  {'Max ms':>9}")
        lines.append("─" * 80)
        agg = self.by_kernel()
        # Sort by total time descending
        for name, d in sorted(agg.items(), key=lambda x: -x[1]["total_ms"]):
            lines.append(
                f"  {name:<26} {d['calls']:>6}  {d['total_ms']:>9.3f}  "
                f"{d['avg_ms']:>8.3f}  {d['min_ms']:>8.3f}  {d['max_ms']:>8.3f}"
            )
        lines.append("─" * 80)
        lines.append(f"  {'TOTAL':<26} {sum(d['calls'] for d in agg.values()):>6}  {self.total_ms():>9.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ProfileResult({len(self._entries)} calls, {self.total_ms():.3f}ms total)"


class Profiler:
    """Context manager that intercepts kernel dispatches and measures their time."""

    def __init__(self):
        self._entries: list[ProfileEntry] = []
        self._active = False
        self._orig_launch: dict = {}  # launcher_id → original _launch_metal

    def _wrap_launcher(self, launcher):
        """Monkey-patch a KernelLauncher to record timing."""
        orig = launcher._launch_metal_original = getattr(
            launcher, "_launch_metal_original", launcher._kc_orig_launch_metal if hasattr(launcher, "_kc_orig_launch_metal") else None
        )

        profiler = self
        func_name = launcher.func_name

        def _timed_launch(kc_self, *args, **kwargs):
            from locomp.backends.metal_runtime import get_runtime
            runtime = get_runtime()
            t0 = time.perf_counter()
            # Call original dispatch (synchronous so timing is accurate)
            result = profiler._orig_calls[id(launcher)](*args, **kwargs)
            runtime.sync()
            elapsed = (time.perf_counter() - t0) * 1000
            profiler._entries.append(ProfileEntry(func_name, kc_self.grid, elapsed))
            return result

        return _timed_launch

    def __enter__(self) -> "Profiler":
        self._entries.clear()
        self._orig_calls = {}
        self._patched = []
        _active_profiler.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _active_profiler.clear()
        return False

    def record(self, name: str, grid: tuple, ms: float):
        self._entries.append(ProfileEntry(name, grid, ms))

    @property
    def result(self) -> ProfileResult:
        return ProfileResult(self._entries)

    def report(self) -> str:
        return self.result.report()

    def __repr__(self) -> str:
        return f"Profiler({len(self._entries)} entries)"


# Global stack — allows nested profilers (last one wins)
_active_profiler: list[Profiler] = []


def get_active_profiler() -> Profiler | None:
    return _active_profiler[-1] if _active_profiler else None


def profile() -> Profiler:
    """Create a profiling context manager.

    Usage:
        with locomp.profile() as p:
            kernel[(N,)](x, out, N=N)
        print(p.report())
    """
    return Profiler()
