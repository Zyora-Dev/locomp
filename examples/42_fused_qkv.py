"""
Example 42: Fused QKV Projection — single kernel for Q, K, V linear projections.

Instead of 3 separate matmuls (X@Wq, X@Wk, X@Wv), this does one kernel
reading X once and writing Q, K, V outputs.

Architecture:
  Weight is fused: W_qkv [D, 3*D] = [Wq | Wk | Wv] concatenated.
  Output is [N, 3*D], later split into Q, K, V each [N, D].
  Uses simdgroup_matrix 64×64 tiled approach.
  Saves 2× global memory reads of X compared to 3 separate matmuls.
"""

import time
import numpy as np
import locomp

BM = 64
BN = 64
BK = 8


@locomp.kernel
def fused_qkv_proj(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
                   N_seq: locomp.constexpr, D_in: locomp.constexpr,
                   D_out3: locomp.constexpr):
    """X:[N,D_in] @ W:[D_in, 3*D_out] → O:[N, 3*D_out].
    D_out3 = 3*D_out. Standard tiled matmul."""
    sgid = locomp.simd_group_id()
    lane = locomp.simd_lane_id()

    tile_n = locomp.program_id(0)   # along D_out3 dimension
    tile_m = locomp.program_id(1)   # along N_seq dimension

    sg_row = sgid // 4
    sg_col = sgid % 4

    As = locomp.shared_memory(BM * BK)
    Bs = locomp.shared_memory(BK * BN)

    acc00 = locomp.simdgroup_matrix(0.0)
    acc01 = locomp.simdgroup_matrix(0.0)
    acc10 = locomp.simdgroup_matrix(0.0)
    acc11 = locomp.simdgroup_matrix(0.0)

    for k_tile in range(D_in // BK):
        tid = sgid * 32 + lane

        a_row = tile_m * BM + tid // BK
        a_col = k_tile * BK + tid % BK
        locomp.shared_store(As, tid, locomp.load(X + (a_row * D_in + a_col)))

        b_row = k_tile * BK + tid // BN
        b_col = tile_n * BN + tid % BN
        locomp.shared_store(Bs, tid, locomp.load(W + (b_row * D_out3 + b_col)))

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
    locomp.simdgroup_matrix_store_device(acc00, O + (c_base_r * D_out3 + c_base_c), D_out3)
    locomp.simdgroup_matrix_store_device(acc01, O + (c_base_r * D_out3 + (c_base_c + 8)), D_out3)
    locomp.simdgroup_matrix_store_device(acc10, O + ((c_base_r + 8) * D_out3 + c_base_c), D_out3)
    locomp.simdgroup_matrix_store_device(acc11, O + ((c_base_r + 8) * D_out3 + (c_base_c + 8)), D_out3)


# =============================================================================
# Dispatch
# =============================================================================

def gpu_fused_qkv(X, Wq, Wk, Wv):
    """Fused QKV projection. X:[N,D], Wq/Wk/Wv:[D,D_out] → Q,K,V each [N,D_out]."""
    N, D_in = X.shape
    D_out = Wq.shape[1]
    D_out3 = 3 * D_out

    # Fuse weights: [D_in, 3*D_out]
    W_fused = np.concatenate([Wq, Wk, Wv], axis=1).astype(np.float32)

    assert N % BM == 0 and D_out3 % BN == 0 and D_in % BK == 0

    X_g = locomp.tensor(X.reshape(-1))
    W_g = locomp.tensor(W_fused.reshape(-1))
    O_g = locomp.empty(N * D_out3)

    grid = (D_out3 // BN, N // BM)
    fused_qkv_proj[grid, (32, 16)](X_g, W_g, O_g, N, D_in, D_out3)

    out = O_g.numpy().reshape(N, D_out3)
    Q = out[:, :D_out]
    K = out[:, D_out:2*D_out]
    V = out[:, 2*D_out:]
    return Q, K, V


if __name__ == "__main__":
    WARMUP = 5
    RUNS = 15

    print(f"\n{'='*70}")
    print("Fused QKV Projection: X @ [Wq|Wk|Wv] → Q, K, V")
    print(f"{'='*70}")
    print(f"{'Config':>30} | {'Fused':>8} | {'3×Sep':>8} | Fused/Sep | {'Error':>8}")
    print("-" * 72)

    configs = [
        (64, 768, 768, "Small D=768"),
        (128, 768, 768, "Med D=768"),
        (64, 4096, 4096, "Llama D=4096"),
        (128, 4096, 4096, "Llama long"),
    ]

    for N, D_in, D_out, label in configs:
        np.random.seed(42)
        X = np.random.randn(N, D_in).astype(np.float32) * 0.02
        Wq = np.random.randn(D_in, D_out).astype(np.float32) * 0.02
        Wk = np.random.randn(D_in, D_out).astype(np.float32) * 0.02
        Wv = np.random.randn(D_in, D_out).astype(np.float32) * 0.02

        # Reference: 3 separate matmuls
        ref_Q = X @ Wq
        ref_K = X @ Wk
        ref_V = X @ Wv

        Q, K, V = gpu_fused_qkv(X, Wq, Wk, Wv)
        err = max(np.max(np.abs(Q - ref_Q)), np.max(np.abs(K - ref_K)), np.max(np.abs(V - ref_V)))

        # Benchmark fused
        for _ in range(WARMUP):
            gpu_fused_qkv(X, Wq, Wk, Wv)
        times_fused = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_fused_qkv(X, Wq, Wk, Wv)
            times_fused.append((time.perf_counter() - t0) * 1000)
        t_fused = sorted(times_fused)[RUNS // 2]

        # Benchmark 3 separate numpy matmuls
        for _ in range(WARMUP):
            X @ Wq; X @ Wk; X @ Wv
        times_sep = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            X @ Wq; X @ Wk; X @ Wv
            times_sep.append((time.perf_counter() - t0) * 1000)
        t_sep = sorted(times_sep)[RUNS // 2]

        r = t_fused / t_sep
        print(f"{label+' N='+str(N):>30} | {t_fused:>6.3f}ms | {t_sep:>6.3f}ms | {r:>8.2f}x | {err:.2e}")
