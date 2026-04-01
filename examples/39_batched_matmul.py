"""
Example 39: Batched Matmul — [B, M, K] @ [B, K, N] → [B, M, N]

Model-agnostic batched matrix multiply using simdgroup_matrix.
Core primitive for batched linear projections in any transformer.
Uses the same 64×64 tiled simdgroup approach as v4 with batch dim on program_id(2).
"""

import time
import numpy as np
import locomp

BM = 64
BN = 64
BK = 8


@locomp.kernel
def batched_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                   M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    sgid = locomp.simd_group_id()
    lane = locomp.simd_lane_id()

    tile_n = locomp.program_id(0)
    tile_m = locomp.program_id(1)
    batch = locomp.program_id(2)

    sg_row = sgid // 4
    sg_col = sgid % 4

    a_batch_off = batch * M * K
    b_batch_off = batch * K * N
    c_batch_off = batch * M * N

    As = locomp.shared_memory(BM * BK)
    Bs = locomp.shared_memory(BK * BN)

    acc00 = locomp.simdgroup_matrix(0.0)
    acc01 = locomp.simdgroup_matrix(0.0)
    acc10 = locomp.simdgroup_matrix(0.0)
    acc11 = locomp.simdgroup_matrix(0.0)

    for k_tile in range(K // BK):
        tid = sgid * 32 + lane

        a_row = tile_m * BM + tid // BK
        a_col = k_tile * BK + tid % BK
        locomp.shared_store(As, tid, locomp.load(A + (a_batch_off + a_row * K + a_col)))

        b_row = k_tile * BK + tid // BN
        b_col = tile_n * BN + tid % BN
        locomp.shared_store(Bs, tid, locomp.load(B + (b_batch_off + b_row * N + b_col)))

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
    locomp.simdgroup_matrix_store_device(acc00, C + (c_batch_off + c_base_r * N + c_base_c), N)
    locomp.simdgroup_matrix_store_device(acc01, C + (c_batch_off + c_base_r * N + (c_base_c + 8)), N)
    locomp.simdgroup_matrix_store_device(acc10, C + (c_batch_off + (c_base_r + 8) * N + c_base_c), N)
    locomp.simdgroup_matrix_store_device(acc11, C + (c_batch_off + (c_base_r + 8) * N + (c_base_c + 8)), N)


# =============================================================================
# Dispatch
# =============================================================================

def gpu_batched_matmul(A, B):
    """Batched matmul. A:[B,M,K] @ B_mat:[B,K,N] → C:[B,M,N]."""
    batch, M, K = A.shape
    _, _, N = B.shape
    assert M % BM == 0 and N % BN == 0 and K % BK == 0

    A_g = locomp.tensor(A.reshape(-1))
    B_g = locomp.tensor(B.reshape(-1))
    C_g = locomp.empty(batch * M * N)

    grid = (N // BN, M // BM, batch)
    batched_matmul[grid, (32, 16)](A_g, B_g, C_g, M, N, K)
    return C_g.numpy().reshape(batch, M, N)


if __name__ == "__main__":
    WARMUP = 5
    RUNS = 15

    print(f"\n{'='*70}")
    print("Batched Matmul: [B,M,K] @ [B,K,N] — simdgroup 64×64 tiles")
    print(f"{'='*70}")
    print(f"{'Config':>30} | {'GPU':>8} | {'NumPy':>8} | GPU/NP | {'Error':>8}")
    print("-" * 70)

    configs = [
        (1, 64, 64, 64),
        (4, 128, 128, 128),
        (8, 256, 256, 256),
        (12, 128, 768, 768),    # typical transformer: 12 heads, seq=128, d=768
        (32, 64, 128, 128),     # Llama-style: 32 heads, seq=64, d=128
        (4, 256, 256, 1024),
    ]

    for B, M, K, N in configs:
        np.random.seed(42)
        A_np = np.random.randn(B, M, K).astype(np.float32) * 0.1
        B_np = np.random.randn(B, K, N).astype(np.float32) * 0.1
        ref = np.matmul(A_np, B_np)

        gpu_out = gpu_batched_matmul(A_np, B_np)
        err = np.max(np.abs(gpu_out - ref))

        for _ in range(WARMUP):
            gpu_batched_matmul(A_np, B_np)
        times_gpu = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_batched_matmul(A_np, B_np)
            times_gpu.append((time.perf_counter() - t0) * 1000)
        t_gpu = sorted(times_gpu)[RUNS // 2]

        for _ in range(WARMUP):
            np.matmul(A_np, B_np)
        times_np = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            np.matmul(A_np, B_np)
            times_np.append((time.perf_counter() - t0) * 1000)
        t_np = sorted(times_np)[RUNS // 2]

        r = t_gpu / t_np
        label = f"B={B} M={M} K={K} N={N}"
        print(f"{label:>30} | {t_gpu:>6.3f}ms | {t_np:>6.3f}ms | {r:>5.2f}x | {err:.2e}")
