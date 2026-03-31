"""
Example: Parallel Softmax with shared memory reductions.

The naive softmax uses 1 thread per row — wastes >99% of GPU capacity.
This version uses T threads per row with tree-parallel reductions:
  - Phase 1: parallel max reduction (log2(T) steps)
  - Phase 2: parallel exp+sum reduction (log2(T) steps)
  - Phase 3: parallel normalize

For D=1024 with T=256: each thread handles 4 elements.
Tree reduction: 8 steps (log2(256)) vs 1024 sequential operations.
~128× faster than naive at large row widths.

Demonstrates: parallel reduction, shared_memory, shared_load/store,
              barrier, tree reduction pattern, interleaved access.
"""

import locomp
import numpy as np


@locomp.kernel
def parallel_softmax(X: locomp.Tensor, OUT: locomp.Tensor,
                     ROWS: locomp.constexpr, D: locomp.constexpr,
                     THREADS: locomp.constexpr, LOG_T: locomp.constexpr,
                     ELEMS: locomp.constexpr):
    # Each threadgroup handles one row
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS

    # Shared memory for tree reduction (one slot per thread)
    smem = locomp.shared_memory(256)

    # === Phase 1: Find row max (parallel) ===
    # Each thread finds local max over its interleaved elements
    local_max = locomp.load(X + (row * D + lid))
    for j in range(1, ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        local_max = locomp.where(val > local_max, val, local_max)

    # Store local max to shared memory
    locomp.shared_store(smem, lid, local_max)
    locomp.barrier()

    # Tree reduction for max (log2(THREADS) steps)
    stride = THREADS / 2
    for s in range(LOG_T):
        if lid < stride:
            a = locomp.shared_load(smem, lid)
            b = locomp.shared_load(smem, lid + stride)
            mx = locomp.where(b > a, b, a)
            locomp.shared_store(smem, lid, mx)
        locomp.barrier()
        stride = stride / 2

    # Broadcast global max
    row_max = locomp.shared_load(smem, 0)
    locomp.barrier()

    # === Phase 2: Sum of exp(x - max) (parallel) ===
    local_sum = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        local_sum = local_sum + e

    locomp.shared_store(smem, lid, local_sum)
    locomp.barrier()

    # Tree reduction for sum
    stride2 = THREADS / 2
    for s in range(LOG_T):
        if lid < stride2:
            a = locomp.shared_load(smem, lid)
            b = locomp.shared_load(smem, lid + stride2)
            locomp.shared_store(smem, lid, a + b)
        locomp.barrier()
        stride2 = stride2 / 2

    # Broadcast global sum
    total_sum = locomp.shared_load(smem, 0)
    locomp.barrier()

    # === Phase 3: Normalize and write (parallel) ===
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        result = e / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)


def main():
    import time

    print("=" * 60)
    print("PARALLEL SOFTMAX (tree reduction)")
    print("=" * 60)

    ROWS, D = 512, 1024
    THREADS = 256
    LOG_T = 8   # log2(256)
    ELEMS = D // THREADS  # 4

    x_np = np.random.randn(ROWS * D).astype(np.float32)
    x = locomp.tensor(x_np)
    out = locomp.empty(ROWS * D)

    grid = (ROWS,)
    tg = (THREADS,)
    parallel_softmax[grid, tg](x, out, ROWS, D, THREADS, LOG_T, ELEMS)

    # Verify against NumPy
    x_2d = x_np.reshape(ROWS, D)
    x_shifted = x_2d - x_2d.max(axis=1, keepdims=True)
    expected = np.exp(x_shifted) / np.exp(x_shifted).sum(axis=1, keepdims=True)
    expected = expected.flatten()
    result = out.numpy()

    max_error = np.max(np.abs(result - expected))
    print(f"Shape: {ROWS} rows × {D} cols, {THREADS} threads/row")
    print(f"Max error vs NumPy: {max_error}")
    print(f"Row sums (should be ~1.0): {result.reshape(ROWS, D).sum(axis=1)[:5]}")

    # Timing
    times = []
    for _ in range(3):
        out2 = locomp.empty(ROWS * D)
        parallel_softmax[grid, tg](x, out2, ROWS, D, THREADS, LOG_T, ELEMS)
    for _ in range(10):
        out2 = locomp.empty(ROWS * D)
        t0 = time.perf_counter()
        parallel_softmax[grid, tg](x, out2, ROWS, D, THREADS, LOG_T, ELEMS)
        times.append((time.perf_counter() - t0) * 1000)
    median = sorted(times)[5]
    print(f"Median time (10 runs): {median:.3f} ms")


if __name__ == "__main__":
    main()
