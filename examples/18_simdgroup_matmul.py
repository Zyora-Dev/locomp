"""
Simdgroup Matrix Matmul — Hardware 8×8 matmul on Apple Silicon.

Uses Metal's simdgroup_matrix_multiply_accumulate to compute 8×8 tiles
in ~2 cycles instead of 64 FMAs.

Architecture:
  - Each SIMD group (32 threads) computes one 8×8 output block
  - TG = (32, SIMD_GROUPS_PER_TG) where each SIMD group handles a different output tile
  - Cooperative shared memory loading: all threads in threadgroup load A/B tiles
  - K-dimension tiled in steps of 8

For 64×64 @ 64×64 with TG=(32,8):
  - 8 SIMD groups per threadgroup = 8 output 8×8 tiles = 2 tile-rows × 4 tile-cols
  - Grid: (8,8) threadgroups = 64/8 × 64/8
  Actually simpler: grid = total_simd_groups / simd_groups_per_tg
"""

import time
import numpy as np
import locomp

TILE = 8  # simdgroup matrix is 8×8


@locomp.kernel
def matmul_simd(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    # Each threadgroup = 1 SIMD group = 32 threads = one 8×8 output tile
    sgid = locomp.simd_group_id()  # always 0 (1 SIMD group per TG)

    # 2D grid: program_id(0) = tile col, program_id(1) = tile row
    tile_col = locomp.program_id(0)
    tile_row = locomp.program_id(1)

    # Shared memory for A and B tiles (8×8 each)
    As = locomp.shared_memory(TILE * TILE)  # 64 floats
    Bs = locomp.shared_memory(TILE * TILE)  # 64 floats

    # Thread within SIMD group
    lane = locomp.simd_lane_id()

    # Initialize accumulator to zero
    acc = locomp.simdgroup_matrix(0.0)

    # K-dimension loop, 8 elements at a time
    for k_tile in range(K // TILE):
        # Cooperative loading: 32 threads load 64 elements (2 each)
        # A tile: rows [tile_row*8 .. tile_row*8+7], cols [k_tile*8 .. k_tile*8+7]
        # Flatten: element i is at row i//8, col i%8
        elem0 = lane
        elem1 = lane + 32
        # A[tile_row*8 + elem//8, k_tile*8 + elem%8]
        if elem0 < TILE * TILE:
            a_row = tile_row * TILE + elem0 // TILE
            a_col = k_tile * TILE + elem0 % TILE
            locomp.shared_store(As, elem0, locomp.load(A + (a_row * K + a_col)))
        if elem1 < TILE * TILE:
            a_row1 = tile_row * TILE + elem1 // TILE
            a_col1 = k_tile * TILE + elem1 % TILE
            locomp.shared_store(As, elem1, locomp.load(A + (a_row1 * K + a_col1)))

        # B tile: rows [k_tile*8 .. k_tile*8+7], cols [tile_col*8 .. tile_col*8+7]
        if elem0 < TILE * TILE:
            b_row = k_tile * TILE + elem0 // TILE
            b_col = tile_col * TILE + elem0 % TILE
            locomp.shared_store(Bs, elem0, locomp.load(B + (b_row * N + b_col)))
        if elem1 < TILE * TILE:
            b_row1 = k_tile * TILE + elem1 // TILE
            b_col1 = tile_col * TILE + elem1 % TILE
            locomp.shared_store(Bs, elem1, locomp.load(B + (b_row1 * N + b_col1)))

        locomp.barrier()

        # Load A and B tiles into simdgroup matrices and multiply-accumulate
        a_mat = locomp.simdgroup_matrix_load(As, 0, TILE)
        b_mat = locomp.simdgroup_matrix_load(Bs, 0, TILE)
        acc = locomp.simdgroup_mac(acc, a_mat, b_mat)

        locomp.barrier()

    # Store result: C[tile_row*8..+8, tile_col*8..+8]
    c_offset = tile_row * TILE * N + tile_col * TILE
    locomp.simdgroup_matrix_store_device(acc, C + c_offset, N)


if __name__ == "__main__":
    import mlx.core as mx

    sizes = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64),
             (128, 128, 128), (256, 256, 256), (512, 512, 512)]

    print(f"Simdgroup Matrix Matmul: 8×8 hardware tiles, 1 SIMD group/TG")
    print(f"{'Size':>12} | {'Locust':>8} | {'MLX':>8} | {'Ratio':>7} | {'Error':>8}")
    print("-" * 60)

    for M, N, K in sizes:
        np.random.seed(42)
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)
        expected = A_np @ B_np

        A_t = locomp.tensor(A_np.flatten())
        B_t = locomp.tensor(B_np.flatten())
        C_t = locomp.empty(M * N)

        grid_x = N // TILE  # tile columns
        grid_y = M // TILE  # tile rows

        # Run + check correctness
        matmul_simd[(grid_x, grid_y), (32,)](A_t, B_t, C_t, M, N, K)
        result = C_t.numpy().reshape(M, N)
        err = np.max(np.abs(result - expected))

        # Benchmark
        WARMUP = 3; RUNS = 10
        for _ in range(WARMUP):
            matmul_simd[(grid_x, grid_y), (32,)](A_t, B_t, C_t, M, N, K)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            matmul_simd[(grid_x, grid_y), (32,)](A_t, B_t, C_t, M, N, K)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[5]

        # MLX
        mA = mx.array(A_np); mB = mx.array(B_np)
        mx.eval(mA, mB)
        for _ in range(WARMUP):
            out = mA @ mB; mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mA @ mB; mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[5]

        ratio = t_loc / t_mlx
        print(f"{M:>4}×{K:>4}×{N:>2} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
