"""
Simdgroup Matrix Matmul v2 — Multi-SIMD tiled matmul.

Each threadgroup has 4 SIMD groups (128 threads) computing a 16×16 output tile
as 2×2 grid of 8×8 simdgroup tiles. Shared memory holds 16×8 A-tiles and 8×16
B-tiles, reused across all 4 SIMD groups.

For C[M,N] = A[M,K] × B[K,N]:
  - Threadgroup tile: 16×16 output
  - K-tiles: 8 columns at a time
  - 4 SIMD groups per TG, each owns one 8×8 sub-tile
  - Grid: (N/16, M/16) threadgroups
"""

import time
import numpy as np
import locomp

BM = 16  # threadgroup output rows
BN = 16  # threadgroup output cols
BK = 8   # K-tile size (matches 8×8 simdgroup)


@locomp.kernel
def matmul_simd_v2(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                   M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    # 4 SIMD groups per threadgroup: 2 rows × 2 cols of 8×8 tiles
    sgid = locomp.simd_group_id()   # 0..3
    lane = locomp.simd_lane_id()    # 0..31

    tile_n = locomp.program_id(0)   # output tile column
    tile_m = locomp.program_id(1)   # output tile row

    # Which 8×8 sub-tile does this SIMD group own?
    sg_row = sgid // 2   # 0 or 1
    sg_col = sgid % 2    # 0 or 1

    # Shared memory: A[16×8] + B[8×16]
    As = locomp.shared_memory(BM * BK)   # 128 floats
    Bs = locomp.shared_memory(BK * BN)   # 128 floats

    # Initialize 8×8 accumulator
    acc = locomp.simdgroup_matrix(0.0)

    for k_tile in range(K // BK):
        # === Cooperative loading: 128 threads load 128+128=256 elements ===
        # Each thread loads 1 element of A and 1 element of B
        # Flatten thread ID: sgid*32 + lane = 0..127
        tid_flat = sgid * 32 + lane

        # A[16×8] = 128 elements: tid_flat maps to row=tid//8, col=tid%8
        a_row = tile_m * BM + tid_flat // BK
        a_col = k_tile * BK + tid_flat % BK
        locomp.shared_store(As, tid_flat, locomp.load(A + (a_row * K + a_col)))

        # B[8×16] = 128 elements: tid_flat maps to row=tid//16, col=tid%16
        b_row = k_tile * BK + tid_flat // BN
        b_col = tile_n * BN + tid_flat % BN
        locomp.shared_store(Bs, tid_flat, locomp.load(B + (b_row * N + b_col)))

        locomp.barrier()

        # === Simdgroup matrix multiply ===
        # A sub-tile: As[sg_row*8 .. sg_row*8+7, 0..7] → offset=sg_row*8*BK, stride=BK
        a_mat = locomp.simdgroup_matrix_load(As, sg_row * 8 * BK, BK)
        # B sub-tile: Bs[0..7, sg_col*8 .. sg_col*8+7] → offset=sg_col*8, stride=BN
        b_mat = locomp.simdgroup_matrix_load(Bs, sg_col * 8, BN)
        acc = locomp.simdgroup_mac(acc, a_mat, b_mat)

        locomp.barrier()

    # === Store 8×8 result to C ===
    c_row = tile_m * BM + sg_row * 8
    c_col = tile_n * BN + sg_col * 8
    locomp.simdgroup_matrix_store_device(acc, C + (c_row * N + c_col), N)


if __name__ == "__main__":
    import mlx.core as mx

    sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64),
             (128, 128, 128), (256, 256, 256), (512, 512, 512)]

    print(f"Simdgroup Matmul v2: 16×16 TG tile, 4 SIMD groups (128 threads)")
    print(f"{'Size':>12} | {'Locust':>8} | {'MLX':>8} | {'Ratio':>7} | {'Error':>8}")
    print("-" * 60)

    for M_sz, N_sz, K_sz in sizes:
        np.random.seed(42)
        A_np = np.random.randn(M_sz, K_sz).astype(np.float32)
        B_np = np.random.randn(K_sz, N_sz).astype(np.float32)
        expected = A_np @ B_np

        A_t = locomp.tensor(A_np.flatten())
        B_t = locomp.tensor(B_np.flatten())
        C_t = locomp.empty(M_sz * N_sz)

        grid_x = N_sz // BN
        grid_y = M_sz // BM

        matmul_simd_v2[(grid_x, grid_y), (32, 4)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
        result = C_t.numpy().reshape(M_sz, N_sz)
        err = np.max(np.abs(result - expected))

        WARMUP = 3; RUNS = 10
        for _ in range(WARMUP):
            matmul_simd_v2[(grid_x, grid_y), (32, 4)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            matmul_simd_v2[(grid_x, grid_y), (32, 4)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[5]

        mA = mx.array(A_np); mB = mx.array(B_np); mx.eval(mA, mB)
        for _ in range(WARMUP):
            out = mA @ mB; mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mA @ mB; mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[5]

        ratio = t_loc / t_mlx
        print(f"{M_sz:>4}×{K_sz:>4}×{N_sz:>2} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
