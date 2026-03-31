"""
Simdgroup Matrix Matmul v3 — 32×32 output tiles, 8 SIMD groups per TG.

Each TG: 256 threads = 8 SIMD groups.
Output tile: 32×32 = 4×4 grid of 8×8 simdgroup tiles.
Each SIMD group computes a 8×32 strip (1×4 blocks of 8×8).
K swept in steps of 8.

Shared memory: A[32×8] = 256 + B[8×32] = 256 = 512 floats = 2KB.
"""

import time
import numpy as np
import locomp

BM = 32
BN = 32
BK = 8


@locomp.kernel
def matmul_simd_v3(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                   M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    sgid = locomp.simd_group_id()   # 0..7
    lane = locomp.simd_lane_id()    # 0..31

    tile_n = locomp.program_id(0)
    tile_m = locomp.program_id(1)

    # 8 SIMD groups: 4 rows × 2 cols of 8×8 tiles → each owns 8×16 output
    # Actually: 8 groups computing 32×32 = each group gets 1 row of 4 tiles
    # sg_row = sgid // 4, sg_col = sgid % 4 → but then each group does only 1 of 16 tiles
    # Better: each group owns 2 tiles (2×1 block = 16×8)
    # Layout: 4 rows × 2 cols, sgid → (sgid//2, sgid%2)
    sg_row = sgid // 2  # 0..3 (selects 8-row band within 32)
    sg_col = sgid % 2   # 0..1 (selects left or right 16-col half)

    As = locomp.shared_memory(BM * BK)   # 256 floats
    Bs = locomp.shared_memory(BK * BN)   # 256 floats

    # Each SIMD group owns 2 accumulators: [sg_row*8, sg_col*16+0:8] and [sg_row*8, sg_col*16+8:16]
    acc0 = locomp.simdgroup_matrix(0.0)  # left 8×8
    acc1 = locomp.simdgroup_matrix(0.0)  # right 8×8

    for k_tile in range(K // BK):
        # Cooperative loading: 256 threads load 256+256 elements
        tid_flat = sgid * 32 + lane  # 0..255

        # A[32×8] = 256 elements
        a_row = tile_m * BM + tid_flat // BK
        a_col = k_tile * BK + tid_flat % BK
        locomp.shared_store(As, tid_flat, locomp.load(A + (a_row * K + a_col)))

        # B[8×32] = 256 elements
        b_row = k_tile * BK + tid_flat // BN
        b_col = tile_n * BN + tid_flat % BN
        locomp.shared_store(Bs, tid_flat, locomp.load(B + (b_row * N + b_col)))

        locomp.barrier()

        # A sub-tile: As[sg_row*8:+8, 0:8]
        a_mat = locomp.simdgroup_matrix_load(As, sg_row * 8 * BK, BK)
        # B sub-tiles: two 8×8 blocks at columns sg_col*16+0 and sg_col*16+8
        b_mat0 = locomp.simdgroup_matrix_load(Bs, sg_col * 16, BN)
        b_mat1 = locomp.simdgroup_matrix_load(Bs, sg_col * 16 + 8, BN)

        acc0 = locomp.simdgroup_mac(acc0, a_mat, b_mat0)
        acc1 = locomp.simdgroup_mac(acc1, a_mat, b_mat1)

        locomp.barrier()

    # Store: acc0 at C[row, col], acc1 at C[row, col+8]
    c_row = tile_m * BM + sg_row * 8
    c_col0 = tile_n * BN + sg_col * 16
    c_col1 = c_col0 + 8
    locomp.simdgroup_matrix_store_device(acc0, C + (c_row * N + c_col0), N)
    locomp.simdgroup_matrix_store_device(acc1, C + (c_row * N + c_col1), N)


if __name__ == "__main__":
    import mlx.core as mx

    sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128),
             (256, 256, 256), (512, 512, 512)]

    print(f"Simdgroup Matmul v3: 32×32 TG tile, 8 SIMD groups (256 threads)")
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

        matmul_simd_v3[(grid_x, grid_y), (32, 8)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
        result = C_t.numpy().reshape(M_sz, N_sz)
        err = np.max(np.abs(result - expected))

        WARMUP = 3; RUNS = 10
        for _ in range(WARMUP):
            matmul_simd_v3[(grid_x, grid_y), (32, 8)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            matmul_simd_v3[(grid_x, grid_y), (32, 8)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
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
