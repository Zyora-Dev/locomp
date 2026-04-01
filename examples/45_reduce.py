"""
Example 45: Standalone Reduce — sum/max/mean over arbitrary axis.

Building block for loss functions, normalization, pooling, argmax, etc.
Supports reduction over the last dimension with SIMD + shared memory.

Architecture:
  One threadgroup per row. Threads cooperatively reduce D elements.
  SIMD reduce within each simd group, then shared mem across groups.
"""

import time
import numpy as np
import locomp


# =============================================================================
# Reduce Sum: [ROWS, D] → [ROWS]
# =============================================================================

@locomp.kernel
def reduce_sum(X: locomp.Tensor, OUT: locomp.Tensor,
               D: locomp.constexpr, THREADS: locomp.constexpr,
               ELEMS: locomp.constexpr, NUM_SIMD: locomp.constexpr):
    row = locomp.program_id(0)
    tid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    smem = locomp.shared_memory(NUM_SIMD)
    base = row * D

    acc = 0.0
    for j in range(ELEMS):
        idx = tid + j * THREADS
        if idx < D:
            acc = acc + locomp.load(X + (base + idx))

    # SIMD reduce
    acc = locomp.simd_sum(acc)
    if lane == 0:
        locomp.shared_store(smem, sg, acc)
    locomp.barrier()

    if tid == 0:
        total = 0.0
        for g in range(NUM_SIMD):
            total = total + locomp.shared_load(smem, g)
        locomp.store(OUT + row, total)


# =============================================================================
# Reduce Max: [ROWS, D] → [ROWS]
# =============================================================================

@locomp.kernel
def reduce_max(X: locomp.Tensor, OUT: locomp.Tensor,
               D: locomp.constexpr, THREADS: locomp.constexpr,
               ELEMS: locomp.constexpr, NUM_SIMD: locomp.constexpr):
    row = locomp.program_id(0)
    tid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    smem = locomp.shared_memory(NUM_SIMD)
    base = row * D

    m = -1e30
    for j in range(ELEMS):
        idx = tid + j * THREADS
        if idx < D:
            m = locomp.max(m, locomp.load(X + (base + idx)))

    m = locomp.simd_max(m)
    if lane == 0:
        locomp.shared_store(smem, sg, m)
    locomp.barrier()

    if tid == 0:
        global_max = locomp.shared_load(smem, 0)
        for g in range(1, NUM_SIMD):
            global_max = locomp.max(global_max, locomp.shared_load(smem, g))
        locomp.store(OUT + row, global_max)


# =============================================================================
# Reduce Mean: [ROWS, D] → [ROWS]
# =============================================================================

@locomp.kernel
def reduce_mean(X: locomp.Tensor, OUT: locomp.Tensor,
                D: locomp.constexpr, THREADS: locomp.constexpr,
                ELEMS: locomp.constexpr, NUM_SIMD: locomp.constexpr):
    row = locomp.program_id(0)
    tid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    smem = locomp.shared_memory(NUM_SIMD)
    base = row * D

    acc = 0.0
    for j in range(ELEMS):
        idx = tid + j * THREADS
        if idx < D:
            acc = acc + locomp.load(X + (base + idx))

    acc = locomp.simd_sum(acc)
    if lane == 0:
        locomp.shared_store(smem, sg, acc)
    locomp.barrier()

    if tid == 0:
        total = 0.0
        for g in range(NUM_SIMD):
            total = total + locomp.shared_load(smem, g)
        locomp.store(OUT + row, total / D)


# =============================================================================
# Dispatch helpers
# =============================================================================

def _reduce_params(D):
    THREADS = min(128, D)
    NUM_SIMD = max(1, THREADS // 32)
    ELEMS = (D + THREADS - 1) // THREADS
    return THREADS, ELEMS, NUM_SIMD


def gpu_reduce_sum(x):
    """Reduce sum over last axis. x: [ROWS, D] → [ROWS]."""
    ROWS, D = x.shape
    THREADS, ELEMS, NUM_SIMD = _reduce_params(D)
    X_g = locomp.tensor(x.flatten())
    O_g = locomp.empty(ROWS)
    reduce_sum[(ROWS,), (THREADS,)](X_g, O_g, D, THREADS, ELEMS, NUM_SIMD)
    result = O_g.numpy()
    X_g.free(); O_g.free()
    return result


def gpu_reduce_max(x):
    ROWS, D = x.shape
    THREADS, ELEMS, NUM_SIMD = _reduce_params(D)
    X_g = locomp.tensor(x.flatten())
    O_g = locomp.empty(ROWS)
    reduce_max[(ROWS,), (THREADS,)](X_g, O_g, D, THREADS, ELEMS, NUM_SIMD)
    result = O_g.numpy()
    X_g.free(); O_g.free()
    return result


def gpu_reduce_mean(x):
    ROWS, D = x.shape
    THREADS, ELEMS, NUM_SIMD = _reduce_params(D)
    X_g = locomp.tensor(x.flatten())
    O_g = locomp.empty(ROWS)
    reduce_mean[(ROWS,), (THREADS,)](X_g, O_g, D, THREADS, ELEMS, NUM_SIMD)
    result = O_g.numpy()
    X_g.free(); O_g.free()
    return result


if __name__ == "__main__":
    shapes = [(32, 128), (64, 256), (128, 512), (32, 1024), (16, 4096)]

    print("=== Reduce Sum ===")
    for ROWS, D in shapes:
        x = np.random.randn(ROWS, D).astype(np.float32)
        out = gpu_reduce_sum(x)
        expected = x.sum(axis=-1)
        np.testing.assert_allclose(out, expected, rtol=1e-3)
        print(f"  [{ROWS}×{D}] ✓")

    print("\n=== Reduce Max ===")
    for ROWS, D in shapes:
        x = np.random.randn(ROWS, D).astype(np.float32)
        out = gpu_reduce_max(x)
        expected = x.max(axis=-1)
        np.testing.assert_allclose(out, expected, rtol=1e-5)
        print(f"  [{ROWS}×{D}] ✓")

    print("\n=== Reduce Mean ===")
    for ROWS, D in shapes:
        x = np.random.randn(ROWS, D).astype(np.float32)
        out = gpu_reduce_mean(x)
        expected = x.mean(axis=-1)
        np.testing.assert_allclose(out, expected, rtol=1e-3)
        print(f"  [{ROWS}×{D}] ✓")

    print("\nAll reduce tests passed.")
