"""
Float16 Simdgroup Matrix Matmul — half precision, 64×64 tiles, 16 SIMD groups.

Same algorithm as v4 but with float16:
- 2× less memory bandwidth (half → 2 bytes vs float → 4 bytes)
- Apple M1+ has native half-precision ALUs
- simdgroup_half8x8 hardware matmul
- Shared memory uses half (threadgroup half[])

Expected: faster than float32 v4 due to 2× bandwidth savings.
"""

import time
import numpy as np
import locomp

BM = 64
BN = 64
BK = 8


@locomp.kernel
def matmul_f16(A: locomp.Float16, B: locomp.Float16, C: locomp.Float16,
               M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    sgid = locomp.simd_group_id()
    lane = locomp.simd_lane_id()

    tile_n = locomp.program_id(0)
    tile_m = locomp.program_id(1)

    sg_row = sgid // 4
    sg_col = sgid % 4

    As = locomp.shared_memory(BM * BK, locomp.Float16)
    Bs = locomp.shared_memory(BK * BN, locomp.Float16)

    acc00 = locomp.simdgroup_matrix(0.0, locomp.Float16)
    acc01 = locomp.simdgroup_matrix(0.0, locomp.Float16)
    acc10 = locomp.simdgroup_matrix(0.0, locomp.Float16)
    acc11 = locomp.simdgroup_matrix(0.0, locomp.Float16)

    for k_tile in range(K // BK):
        tid = sgid * 32 + lane

        a_row = tile_m * BM + tid // BK
        a_col = k_tile * BK + tid % BK
        locomp.shared_store(As, tid, locomp.load(A + (a_row * K + a_col)))

        b_row = k_tile * BK + tid // BN
        b_col = tile_n * BN + tid % BN
        locomp.shared_store(Bs, tid, locomp.load(B + (b_row * N + b_col)))

        locomp.barrier()

        a0 = locomp.simdgroup_matrix_load(As, (sg_row * 16) * BK, BK)
        a1 = locomp.simdgroup_matrix_load(As, (sg_row * 16 + 8) * BK, BK)

        b0 = locomp.simdgroup_matrix_load(Bs, sg_col * 16, BN)
        b1 = locomp.simdgroup_matrix_load(Bs, sg_col * 16 + 8, BN)

        acc00 = locomp.simdgroup_mac(acc00, a0, b0)
        acc01 = locomp.simdgroup_mac(acc01, a0, b1)
        acc10 = locomp.simdgroup_mac(acc10, a1, b0)
        acc11 = locomp.simdgroup_mac(acc11, a1, b1)

        locomp.barrier()

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

    WARMUP = 5; RUNS = 15

    print("Float16 Simdgroup Matmul: 64×64 tile, 16 SIMD groups, simdgroup_half8x8")
    print(f"{'Size':>12} | {'F16 Locomp':>10} | {'F32 MLX':>8} | {'F16 MLX':>8} | {'F16/F32MLX':>10} | {'F16/F16MLX':>10} | {'Error':>8}")
    print("-" * 90)

    for M_sz, N_sz, K_sz in sizes:
        np.random.seed(42)
        A_np = np.random.randn(M_sz, K_sz).astype(np.float16)
        B_np = np.random.randn(K_sz, N_sz).astype(np.float16)
        expected = (A_np.astype(np.float32) @ B_np.astype(np.float32))

        A_t = locomp.tensor(A_np.flatten())
        B_t = locomp.tensor(B_np.flatten())
        C_t = locomp.empty(M_sz * N_sz, dtype=np.float16)

        grid_x = N_sz // BN
        grid_y = M_sz // BM

        # Correctness check
        matmul_f16[(grid_x, grid_y), (32, 16)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
        result = C_t.numpy().reshape(M_sz, N_sz).astype(np.float32)
        err = np.max(np.abs(result - expected))

        # Benchmark locomp f16
        for _ in range(WARMUP):
            matmul_f16[(grid_x, grid_y), (32, 16)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            matmul_f16[(grid_x, grid_y), (32, 16)](A_t, B_t, C_t, M_sz, N_sz, K_sz)
            times.append((time.perf_counter() - t0) * 1000)
        t_f16 = sorted(times)[RUNS // 2]

        # MLX f32 baseline
        mA32 = mx.array(A_np.astype(np.float32)); mB32 = mx.array(B_np.astype(np.float32))
        mx.eval(mA32, mB32)
        for _ in range(WARMUP):
            out = mA32 @ mB32; mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mA32 @ mB32; mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx32 = sorted(mt)[RUNS // 2]

        # MLX f16
        mA16 = mx.array(A_np); mB16 = mx.array(B_np)
        mx.eval(mA16, mB16)
        for _ in range(WARMUP):
            out = mA16 @ mB16; mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mA16 @ mB16; mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx16 = sorted(mt)[RUNS // 2]

        r32 = t_f16 / t_mlx32
        r16 = t_f16 / t_mlx16
        print(f"{M_sz:>4}×{K_sz:>4}×{N_sz:>2} | {t_f16:>8.3f}ms | {t_mlx32:>6.3f}ms | {t_mlx16:>6.3f}ms | {r32:>9.2f}x | {r16:>9.2f}x | {err:.2e}")
