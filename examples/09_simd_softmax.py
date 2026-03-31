"""
Example: SIMD Softmax — fastest possible softmax on Apple Silicon.

Uses Metal's SIMD group operations (warp-level on Apple, 32 threads) to
reduce across threads WITHOUT shared memory. This eliminates:
  - All tree reduction loops (log2(T) barrier-synchronized steps)
  - All shared memory reads/writes for reductions

Strategy:
  - Each SIMD group (32 threads) handles one row
  - Each thread processes D/32 elements
  - simd_sum / simd_max reduce across 32 lanes in 1 instruction
  - For rows wider than 32: multiple SIMD groups per row with shared memory

Demonstrates: simd_sum, simd_max, simd_lane_id, simd_group_id,
              mixed SIMD + shared_memory for multi-group rows.
"""

import locomp
import numpy as np


# === Version 1: Pure SIMD (1 SIMD group = 32 threads per row) ===
# For D that's a multiple of 32.

@locomp.kernel
def simd_softmax_32(X: locomp.Tensor, OUT: locomp.Tensor,
                    ROWS: locomp.constexpr, D: locomp.constexpr,
                    ELEMS: locomp.constexpr):
    # Each row handled by 1 SIMD group (32 threads). D must be multiple of 32.
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS

    # Phase 1: Each thread finds local max over its elements
    local_max = locomp.load(X + (row * D + lid))
    for j in range(1, ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        local_max = locomp.where(val > local_max, val, local_max)

    # SIMD max — single instruction, no shared memory needed
    row_max = locomp.simd_max(local_max)

    # Phase 2: Each thread sums exp(x - max) over its elements
    local_sum = 0.0
    for j in range(ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        local_sum = local_sum + e

    # SIMD sum — single instruction
    total_sum = locomp.simd_sum(local_sum)

    # Phase 3: Normalize and write
    for j in range(ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        result = e / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)


# === Version 2: SIMD + shared memory (256 threads per row, 8 SIMD groups) ===
# For D that's a multiple of 256.

@locomp.kernel
def simd_softmax_256(X: locomp.Tensor, OUT: locomp.Tensor,
                     ROWS: locomp.constexpr, D: locomp.constexpr,
                     ELEMS: locomp.constexpr, NUM_SIMD: locomp.constexpr):
    # Each row handled by 256 threads = 8 SIMD groups. D must be multiple of 256.
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS
    simd_gid = locomp.simd_group_id()

    # Shared memory for cross-SIMD-group reduction (8 entries, one per SIMD group)
    smem = locomp.shared_memory(256)

    # Phase 1: local max per thread → SIMD max per group → global max
    local_max = locomp.load(X + (row * D + lid))
    for j in range(1, ELEMS):
        idx = lid + j * 256
        val = locomp.load(X + (row * D + idx))
        local_max = locomp.where(val > local_max, val, local_max)

    # SIMD max within each 32-thread group
    group_max = locomp.simd_max(local_max)

    # Store each SIMD group's max to shared memory (lane 0 writes)
    lane = locomp.simd_lane_id()
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_max)
    locomp.barrier()

    # Cross-group max: read all maxes, find global max
    # Each thread reads all NUM_SIMD values (small, 8 values)
    global_max = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        gval = locomp.shared_load(smem, g)
        global_max = locomp.where(gval > global_max, gval, global_max)
    locomp.barrier()

    # Phase 2: exp sum
    local_sum = 0.0
    for j in range(ELEMS):
        idx = lid + j * 256
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - global_max)
        local_sum = local_sum + e

    # SIMD sum within each group
    group_sum = locomp.simd_sum(local_sum)

    # Store group sums
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_sum)
    locomp.barrier()

    # Cross-group sum
    total_sum = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        total_sum = total_sum + locomp.shared_load(smem, g)
    locomp.barrier()

    # Phase 3: Normalize
    for j in range(ELEMS):
        idx = lid + j * 256
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - global_max)
        result = e / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)


def main():
    import time

    print("=" * 60)
    print("SIMD SOFTMAX (Apple Silicon GPU)")
    print("=" * 60)

    # Test with 32 threads/row
    ROWS, D = 512, 1024
    ELEMS = D // 32  # 32

    x_np = np.random.randn(ROWS * D).astype(np.float32)
    x = locomp.tensor(x_np)
    out = locomp.empty(ROWS * D)

    simd_softmax_32[(ROWS,), (32,)](x, out, ROWS, D, ELEMS)

    x_2d = x_np.reshape(ROWS, D)
    x_shifted = x_2d - x_2d.max(axis=1, keepdims=True)
    expected = np.exp(x_shifted) / np.exp(x_shifted).sum(axis=1, keepdims=True)
    result = out.numpy()
    max_error = np.max(np.abs(result - expected.flatten()))
    print(f"\n[32 threads/row] {ROWS}×{D}")
    print(f"Max error: {max_error}")
    print(f"Row sums: {result.reshape(ROWS, D).sum(axis=1)[:5]}")

    # Timing for 32-thread version
    times = []
    for _ in range(3):
        out2 = locomp.empty(ROWS * D)
        simd_softmax_32[(ROWS,), (32,)](x, out2, ROWS, D, ELEMS)
    for _ in range(10):
        out2 = locomp.empty(ROWS * D)
        t0 = time.perf_counter()
        simd_softmax_32[(ROWS,), (32,)](x, out2, ROWS, D, ELEMS)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"Median: {sorted(times)[5]:.3f} ms")

    # Test with 256 threads/row
    ELEMS256 = D // 256  # 4
    NUM_SIMD = 256 // 32  # 8
    out3 = locomp.empty(ROWS * D)

    simd_softmax_256[(ROWS,), (256,)](x, out3, ROWS, D, ELEMS256, NUM_SIMD)
    result3 = out3.numpy()
    max_error3 = np.max(np.abs(result3 - expected.flatten()))
    print(f"\n[256 threads/row] {ROWS}×{D}")
    print(f"Max error: {max_error3}")
    print(f"Row sums: {result3.reshape(ROWS, D).sum(axis=1)[:5]}")

    # Timing for 256-thread version
    times256 = []
    for _ in range(3):
        out4 = locomp.empty(ROWS * D)
        simd_softmax_256[(ROWS,), (256,)](x, out4, ROWS, D, ELEMS256, NUM_SIMD)
    for _ in range(10):
        out4 = locomp.empty(ROWS * D)
        t0 = time.perf_counter()
        simd_softmax_256[(ROWS,), (256,)](x, out4, ROWS, D, ELEMS256, NUM_SIMD)
        times256.append((time.perf_counter() - t0) * 1000)
    print(f"Median: {sorted(times256)[5]:.3f} ms")


if __name__ == "__main__":
    main()
