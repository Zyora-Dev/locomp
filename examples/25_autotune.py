"""
Example 25: Auto-tuned GELU kernel — locomp.autotune picks the fastest configuration.

Demonstrates the autotune API:
  - locomp.Config: defines constexpr overrides, grid function, threadgroup size
  - locomp.autotune: decorator that benchmarks configs and caches the best one
  - key: which constexpr params form the cache key (different input sizes may
    prefer different configs)

On first call for a given key, autotune benchmarks all configs (warmup + timed runs)
and prints the winner. Subsequent calls with the same key use the cached best config.
"""

import time
import numpy as np
import locomp


# --- Auto-tuned GELU kernel ---
# BLOCK_SIZE controls how many elements each threadgroup processes.
# Different BLOCK_SIZE values trade off occupancy vs. register pressure.
# autotune finds the best one for each input size.

@locomp.autotune(
    configs=[
        locomp.Config(BLOCK_SIZE=64,
                      grid=lambda N, BLOCK_SIZE, **kw: (N // BLOCK_SIZE,),
                      tg=(64,)),
        locomp.Config(BLOCK_SIZE=128,
                      grid=lambda N, BLOCK_SIZE, **kw: (N // BLOCK_SIZE,),
                      tg=(128,)),
        locomp.Config(BLOCK_SIZE=256,
                      grid=lambda N, BLOCK_SIZE, **kw: (N // BLOCK_SIZE,),
                      tg=(256,)),
        locomp.Config(BLOCK_SIZE=512,
                      grid=lambda N, BLOCK_SIZE, **kw: (N // BLOCK_SIZE,),
                      tg=(512,)),
        locomp.Config(BLOCK_SIZE=1024,
                      grid=lambda N, BLOCK_SIZE, **kw: (N // BLOCK_SIZE,),
                      tg=(1024,)),
    ],
    key=["N"],
)
@locomp.kernel
def gelu_auto(X: locomp.Tensor, O: locomp.Tensor,
              N: locomp.constexpr, BLOCK_SIZE: locomp.constexpr):
    pid = locomp.program_id(0)
    tid = locomp.local_id(0)
    idx = pid * BLOCK_SIZE + tid
    x = locomp.load(X + idx)
    # GELU approximation
    locomp.store(O + idx, 0.5 * x * (1.0 + locomp.tanh(
        0.7978845608 * (x + 0.044715 * x * x * x))))


# --- Fixed-config GELU for comparison ---
@locomp.kernel
def gelu_fixed(X: locomp.Tensor, O: locomp.Tensor,
               N: locomp.constexpr, BLOCK_SIZE: locomp.constexpr):
    pid = locomp.program_id(0)
    tid = locomp.local_id(0)
    idx = pid * BLOCK_SIZE + tid
    x = locomp.load(X + idx)
    locomp.store(O + idx, 0.5 * x * (1.0 + locomp.tanh(
        0.7978845608 * (x + 0.044715 * x * x * x))))


def gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


if __name__ == "__main__":
    sizes = [4096, 16384, 65536, 262144, 1048576]

    print("Auto-tuned GELU — locomp.autotune picks best BLOCK_SIZE per N")
    print("=" * 70)

    # Phase 1: Autotune runs (prints winner for each N)
    print("\n--- Phase 1: Autotuning ---")
    for N in sizes:
        np.random.seed(42)
        X_np = np.random.randn(N).astype(np.float32)
        X_t = locomp.tensor(X_np)
        O_t = locomp.empty(N)

        # This call triggers autotuning and prints the winner
        gelu_auto(X_t, O_t, N)

        # Verify correctness
        result = O_t.numpy()
        expected = gelu_np(X_np)
        err = np.max(np.abs(result - expected))
        print(f"    N={N:>8}: max error = {err:.2e}")

    # Phase 2: Benchmark autotuned vs fixed BLOCK_SIZE=256
    print(f"\n--- Phase 2: Autotuned vs Fixed (BLOCK_SIZE=256) ---")
    print(f"{'N':>10} | {'Autotuned':>10} | {'Fixed-256':>10} | {'Speedup':>8}")
    print("-" * 50)

    BLOCK_FIXED = 256
    WARMUP = 5
    RUNS = 15

    for N in sizes:
        np.random.seed(42)
        X_np = np.random.randn(N).astype(np.float32)
        X_t = locomp.tensor(X_np)
        O_auto = locomp.empty(N)
        O_fixed = locomp.empty(N)

        # Benchmark autotuned (uses cached best config)
        for _ in range(WARMUP):
            gelu_auto(X_t, O_auto, N)
        times_auto = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gelu_auto(X_t, O_auto, N)
            t1 = time.perf_counter()
            times_auto.append(t1 - t0)
        t_auto = sorted(times_auto)[RUNS // 2] * 1000

        # Benchmark fixed
        grid = (N // BLOCK_FIXED,)
        for _ in range(WARMUP):
            gelu_fixed[grid, (BLOCK_FIXED,)](X_t, O_fixed, N, BLOCK_FIXED)
        times_fixed = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gelu_fixed[grid, (BLOCK_FIXED,)](X_t, O_fixed, N, BLOCK_FIXED)
            t1 = time.perf_counter()
            times_fixed.append(t1 - t0)
        t_fixed = sorted(times_fixed)[RUNS // 2] * 1000

        speedup = t_fixed / t_auto
        print(f"{N:>10} | {t_auto:>8.3f}ms | {t_fixed:>8.3f}ms | {speedup:>6.2f}x")
