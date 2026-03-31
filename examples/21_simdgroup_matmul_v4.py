"""
Simdgroup Matrix Matmul v4 — 64×64 tiles, 16 SIMD groups, 4-accumulator register blocking.

Each TG: 512 threads = 16 SIMD groups.
Output tile: 64×64 = 8×8 grid of 8×8 tiles.
Group layout: 4 rows × 4 cols → each group owns 16×16 output (2×2 of 8×8).
4 accumulators per group: A loaded twice, B loaded twice, each used for 2 MACs.

Shared memory: A[64×BK] + B[BK×64], BK=8 → 512+512 = 4KB.
Arithmetic intensity: 2× better than v3 (64×64 vs 32×32 tiles).
"""

import time
import numpy as np
import locomp

BM = 64
BN = 64
BK = 8


@locomp.kernel
def matmul_simd_v4(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                   M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    sgid = locomp.simd_group_id()   # 0..15
    lane = locomp.simd_lane_id()    # 0..31

    tile_n = locomp.program_id(0)
    tile_m = locomp.program_id(1)

    # 16 SIMD groups in 4×4 layout
    sg_row = sgid // 4  # 0..3 → row band of 16 within 64
    sg_col = sgid % 4   # 0..3 → col band of 16 within 64

    As = locomp.shared_memory(BM * BK)   # 64×8 = 512 floats
    Bs = locomp.shared_memory(BK * BN)   # 8×64 = 512 floats

    # 4 accumulators: 2×2 grid of 8×8 within this group's 16×16 region
    acc00 = locomp.simdgroup_matrix(0.0)
    acc01 = locomp.simdgroup_matrix(0.0)
    acc10 = locomp.simdgroup_matrix(0.0)
    acc11 = locomp.simdgroup_matrix(0.0)

    for k_tile in range(K // BK):
        # Cooperative loading: 512 threads load 512+512=1024 elements (2 per thread)
        tid = sgid * 32 + lane  # 0..511

        # A[64×8]: tid maps to (tid//8, tid%8) → 64 rows, 8 cols = perfect
        a_row = tile_m * BM + tid // BK
        a_col = k_tile * BK + tid % BK
        locomp.shared_store(As, tid, locomp.load(A + (a_row * K + a_col)))

        # B[8×64]: tid maps to (tid//64, tid%64) → 8 rows, 64 cols = perfect
        b_row = k_tile * BK + tid // BN
        b_col = tile_n * BN + tid % BN
        locomp.shared_store(Bs, tid, locomp.load(B + (b_row * N + b_col)))

        locomp.barrier()

        # Load A: two 8×8 sub-tiles vertically (rows sg_row*16+{0,8})
        a0 = locomp.simdgroup_matrix_load(As, (sg_row * 16) * BK, BK)
        a1 = locomp.simdgroup_matrix_load(As, (sg_row * 16 + 8) * BK, BK)

        # Load B: two 8×8 sub-tiles horizontally (cols sg_col*16+{0,8})
        b0 = locomp.simdgroup_matrix_load(Bs, sg_col * 16, BN)
        b1 = locomp.simdgroup_matrix_load(Bs, sg_col * 16 + 8, BN)

        # 4 MACs: every A-B combination in the 2×2 grid
        acc00 = locomp.simdgroup_mac(acc00, a0, b0)
        acc01 = locomp.simdgroup_mac(acc01, a0, b1)
        acc10 = locomp.simdgroup_mac(acc10, a1, b0)
        acc11 = locomp.simdgroup_mac(acc11, a1, b1)

        locomp.barrier()

    # Store 4 blocks to device memory
    c_base_r = tile_m * BM + sg_row * 16
    c_base_c = tile_n * BN + sg_col * 16
    locomp.simdgroup_matrix_store_device(acc00, C + (c_base_r * N + c_base_c), N)
    locomp.simdgroup_matrix_store_device(acc01, C + (c_base_r * N + (c_base_c + 8)), N)
    locomp.simdgroup_matrix_store_device(acc10, C + ((c_base_r + 8) * N + c_base_c), N)
    locomp.simdgroup_matrix_store_device(acc11, C + ((c_base_r + 8) * N + (c_base_c + 8)), N)


if __name__ == "__main__":
    import mlx.core as mx

    sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256),
             (512, 512, 512), (1024, 1024, 1024)]

    print(f"Simdgroup Matmul v4: 64×64 TG tile, 16 SIMD groups (512 threads), 4 accumulators")
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

        matmul_simd_v4[(grid_x, grid_y), (32, 16)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
        result = C_t.numpy().reshape(M_sz, N_sz)
        err = np.max(np.abs(result - expected))

        WARMUP = 5; RUNS = 15
        for _ in range(WARMUP):
            matmul_simd_v4[(grid_x, grid_y), (32, 16)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            matmul_simd_v4[(grid_x, grid_y), (32, 16)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[RUNS // 2]

        mA = mx.array(A_np); mB = mx.array(B_np); mx.eval(mA, mB)
        for _ in range(WARMUP):
            out = mA @ mB; mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mA @ mB; mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[RUNS // 2]

        ratio = t_loc / t_mlx
        print(f"{M_sz:>4}×{K_sz:>4}×{N_sz:>2} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
