"""
Example 59 — Kernel caching benchmark.

Demonstrates the disk cache: first call compiles Python→IR→MSL,
subsequent calls in a new process load MSL from disk and skip that step.

Run this twice:
  python examples/59_kernel_cache.py          # cold (compiles + caches)
  python examples/59_kernel_cache.py          # warm (loads from disk cache)
"""

import time
import numpy as np
import locomp

print(f"locomp v{locomp.__version__}")
print(f"Cache dir: {locomp.kernel_cache_dir()}\n")


# ── Kernels ───────────────────────────────────────────────────────────────────

@locomp.kernel
def gelu(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    t = locomp.tanh(0.7978845608 * (x + 0.044715 * x * x * x))
    locomp.store(O + i, x * 0.5 * (1.0 + t))


@locomp.kernel
def rms_norm(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
             N: locomp.constexpr, eps: locomp.constexpr):
    i = locomp.program_id(0)
    acc = 0.0
    for j in range(N):
        v = locomp.load(X + j)
        acc = acc + v * v
    scale = locomp.rsqrt(acc / N + eps)
    v = locomp.load(X + i)
    locomp.store(O + i, v * scale * locomp.load(W + i))


@locomp.kernel
def vec_add(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(O + i, locomp.load(X + i) + locomp.load(Y + i))


# ── Helpers ───────────────────────────────────────────────────────────────────

def time_first_call(kernel_fn, args_fn, kwargs, grid, runs=3):
    """Time the FIRST call to a kernel (includes compilation if cold)."""
    times = []
    for _ in range(runs):
        # Clear in-memory cache so each measurement includes compile_msl
        kernel_fn._specialized.clear()
        kernel_fn._compiled = False
        kernel_fn._ir = None
        a, kw = args_fn()
        t0 = time.perf_counter()
        kernel_fn[grid](*a, **kw)
        times.append(time.perf_counter() - t0)
    return min(times)


# ── Prep data ─────────────────────────────────────────────────────────────────

N = 4096
x_np = np.random.randn(N).astype(np.float32)
y_np = np.random.randn(N).astype(np.float32)
w_np = np.ones(N, dtype=np.float32)

x = locomp.tensor(x_np)
y = locomp.tensor(y_np)
w = locomp.tensor(w_np)
o = locomp.empty(N)


# ── Warm disk cache (first run will be cold, second will be warm) ─────────────

kernels = [
    ("gelu",     gelu,     lambda: ([x, o], {"N": N}),     (N,)),
    ("vec_add",  vec_add,  lambda: ([x, y, o], {"N": N}),  (N,)),
    ("rms_norm", rms_norm, lambda: ([x, w, o], {"N": N, "eps": 1e-5}), (N,)),
]

import os
cache_dir = locomp.kernel_cache_dir()
any_cached = any(
    any(f.startswith(name) and f.endswith(".json") for f in (os.listdir(cache_dir) if os.path.isdir(cache_dir) else []))
    for name, *_ in kernels
)

print("─" * 52)
print(f"{'Kernel':<12} {'First-call (ms)':>16}  {'Status'}")
print("─" * 52)

for name, kfn, afn, grid in kernels:
    elapsed = time_first_call(kfn, afn, {}, grid) * 1000
    # Check if cache file exists now (was written during timing)
    cached = any(
        f.startswith(name) and f.endswith(".json")
        for f in (os.listdir(cache_dir) if os.path.isdir(cache_dir) else [])
    )
    status = "cache hit (fast)" if any_cached and cached else "compiled + cached"
    print(f"  {name:<10} {elapsed:>14.1f}  {status}")

print("─" * 52)

# ── Correctness check ─────────────────────────────────────────────────────────

x2 = locomp.tensor(x_np); o2 = locomp.empty(N)
gelu[(N,)](x2, o2, N=N)
result = o2.numpy()
expected = x_np * 0.5 * (1.0 + np.tanh(0.7978845608 * (x_np + 0.044715 * x_np**3)))
assert np.allclose(result, expected, atol=1e-4), "GELU correctness FAIL"
print("\nCorrectness: PASS (gelu output matches numpy reference)")
print(f"\nNext run will show 'cache hit (fast)' for all kernels.")
print(f"Cache files in: {cache_dir}")
