"""
Example 37: GPU Causal Flash Attention — D=64 and D=128.

Model-agnostic causal flash attention kernels using Apple simdgroup_matrix.
Supports D=64 (GPT-2, Phi) and D=128 (Llama, Mistral, Gemma).

Architecture:
  TG = 128 threads = 4 SIMD groups (2×2 grid)
  Br=16, Bc=16

  SIMD group layout:
    sg0: rows 0:8,  cols 0:(D/2-1)     sg1: rows 0:8,  cols (D/2):(D-1)
    sg2: rows 8:16, cols 0:(D/2-1)     sg3: rows 8:16, cols (D/2):(D-1)

  D=64:  4 acc blocks per SG half, Qs=1024,  KTs=1024,  Vs=1024
  D=128: 8 acc blocks per SG half, Qs=2048, KTs=2048, Vs=2048

  Q must be pre-scaled by 1/sqrt(D) before passing to kernel.
  Now also supports float constexpr (compiler bug fixed).
"""

import time
import numpy as np
import locomp

Br = 16
Bc = 16


# =============================================================================
# D=64 kernel — 4 acc blocks per SG half
# =============================================================================

@locomp.kernel
def causal_flash_d64(Q: locomp.Tensor, K: locomp.Tensor, V: locomp.Tensor,
                     O: locomp.Tensor,
                     N: locomp.constexpr, NUM_KV_BLOCKS: locomp.constexpr):
    sgid = locomp.simd_group_id()
    lane = locomp.simd_lane_id()
    bq = locomp.program_id(0)
    bh = locomp.program_id(1)
    tid = sgid * 32 + lane
    sg_row = sgid // 2
    sg_col = sgid % 2
    head_off = bh * N * 64

    Qs = locomp.shared_memory(1024)
    KTs = locomp.shared_memory(1024)
    Vs = locomp.shared_memory(1024)
    Ss = locomp.shared_memory(256)
    row_m = locomp.shared_memory(16)
    row_l = locomp.shared_memory(16)
    old_m_snap = locomp.shared_memory(16)
    Ds = locomp.shared_memory(128)

    for iq in range(8):
        q_idx = tid + iq * 128
        locomp.shared_store(Qs, q_idx, locomp.load(Q + (head_off + bq * 16 * 64 + q_idx)))

    if tid < 16:
        locomp.shared_store(row_m, tid, -1000000.0)
        locomp.shared_store(row_l, tid, 0.0)

    acc_o0 = locomp.simdgroup_matrix(0.0)
    acc_o1 = locomp.simdgroup_matrix(0.0)
    acc_o2 = locomp.simdgroup_matrix(0.0)
    acc_o3 = locomp.simdgroup_matrix(0.0)
    locomp.barrier()

    for bk in range(NUM_KV_BLOCKS):
        kv_base = bk * 16
        for il in range(8):
            kv_idx = tid + il * 128
            k_bc = kv_idx // 64
            k_d = kv_idx % 64
            locomp.shared_store(KTs, k_d * 16 + k_bc,
                                locomp.load(K + (head_off + (kv_base + k_bc) * 64 + k_d)))
            locomp.shared_store(Vs, kv_idx,
                                locomp.load(V + (head_off + kv_base * 64 + kv_idx)))
        locomp.barrier()

        s_acc = locomp.simdgroup_matrix(0.0)
        for dk in range(8):
            q_blk = locomp.simdgroup_matrix_load(Qs, sg_row * 8 * 64 + dk * 8, 64)
            kt_blk = locomp.simdgroup_matrix_load(KTs, dk * 8 * 16 + sg_col * 8, 16)
            s_acc = locomp.simdgroup_mac(s_acc, q_blk, kt_blk)
        locomp.simdgroup_matrix_store(s_acc, Ss, sg_row * 8 * 16 + sg_col * 8, 16)
        locomp.barrier()

        if tid < 16:
            locomp.shared_store(old_m_snap, tid, locomp.shared_load(row_m, tid))
        locomp.barrier()

        if tid < 16:
            row = tid
            old_m = locomp.shared_load(old_m_snap, row)
            q_pos = bq * 16 + row
            block_max = -1000000.0
            for j in range(16):
                kv_pos = kv_base + j
                s_val = locomp.shared_load(Ss, row * 16 + j)
                s_val = locomp.where(kv_pos <= q_pos, s_val, -1000000.0)
                locomp.shared_store(Ss, row * 16 + j, s_val)
                block_max = locomp.where(s_val > block_max, s_val, block_max)
            new_max = locomp.where(block_max > old_m, block_max, old_m)
            locomp.shared_store(row_m, row, new_max)
            old_sum = locomp.shared_load(row_l, row)
            rescaled_sum = old_sum * locomp.exp(old_m - new_max)
            block_sum = 0.0
            for j in range(16):
                s_val = locomp.shared_load(Ss, row * 16 + j)
                p_val = locomp.exp(s_val - new_max)
                block_sum = block_sum + p_val
                locomp.shared_store(Ss, row * 16 + j, p_val)
            locomp.shared_store(row_l, row, rescaled_sum + block_sum)
        locomp.barrier()

        locomp.shared_store(Ds, tid, 0.0)
        locomp.barrier()
        if tid < 8:
            locomp.shared_store(Ds, tid * 8 + tid,
                locomp.exp(locomp.shared_load(old_m_snap, tid) - locomp.shared_load(row_m, tid)))
            locomp.shared_store(Ds, 64 + tid * 8 + tid,
                locomp.exp(locomp.shared_load(old_m_snap, tid + 8) - locomp.shared_load(row_m, tid + 8)))
        locomp.barrier()

        d_mat = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
        zero = locomp.simdgroup_matrix(0.0)
        acc_o0 = locomp.simdgroup_mac(zero, d_mat, acc_o0)
        acc_o1 = locomp.simdgroup_mac(zero, d_mat, acc_o1)
        acc_o2 = locomp.simdgroup_mac(zero, d_mat, acc_o2)
        acc_o3 = locomp.simdgroup_mac(zero, d_mat, acc_o3)

        for jj in range(2):
            p_blk = locomp.simdgroup_matrix_load(Ss, sg_row * 8 * 16 + jj * 8, 16)
            v0 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 64 + sg_col * 32, 64)
            v1 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 64 + sg_col * 32 + 8, 64)
            v2 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 64 + sg_col * 32 + 16, 64)
            v3 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 64 + sg_col * 32 + 24, 64)
            acc_o0 = locomp.simdgroup_mac(acc_o0, p_blk, v0)
            acc_o1 = locomp.simdgroup_mac(acc_o1, p_blk, v1)
            acc_o2 = locomp.simdgroup_mac(acc_o2, p_blk, v2)
            acc_o3 = locomp.simdgroup_mac(acc_o3, p_blk, v3)
        locomp.barrier()

    locomp.shared_store(Ds, tid, 0.0)
    locomp.barrier()
    if tid < 8:
        locomp.shared_store(Ds, tid * 8 + tid, 1.0 / locomp.shared_load(row_l, tid))
        locomp.shared_store(Ds, 64 + tid * 8 + tid, 1.0 / locomp.shared_load(row_l, tid + 8))
    locomp.barrier()
    d_final = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
    z2 = locomp.simdgroup_matrix(0.0)
    acc_o0 = locomp.simdgroup_mac(z2, d_final, acc_o0)
    acc_o1 = locomp.simdgroup_mac(z2, d_final, acc_o1)
    acc_o2 = locomp.simdgroup_mac(z2, d_final, acc_o2)
    acc_o3 = locomp.simdgroup_mac(z2, d_final, acc_o3)

    c_row = bq * 16 + sg_row * 8
    c_col = sg_col * 32
    locomp.simdgroup_matrix_store_device(acc_o0, O + (head_off + c_row * 64 + c_col), 64)
    locomp.simdgroup_matrix_store_device(acc_o1, O + (head_off + c_row * 64 + c_col + 8), 64)
    locomp.simdgroup_matrix_store_device(acc_o2, O + (head_off + c_row * 64 + c_col + 16), 64)
    locomp.simdgroup_matrix_store_device(acc_o3, O + (head_off + c_row * 64 + c_col + 24), 64)


# =============================================================================
# D=128 kernel — 8 acc blocks per SG half
# =============================================================================

@locomp.kernel
def causal_flash_d128(Q: locomp.Tensor, K: locomp.Tensor, V: locomp.Tensor,
                      O: locomp.Tensor,
                      N: locomp.constexpr, NUM_KV_BLOCKS: locomp.constexpr):
    sgid = locomp.simd_group_id()
    lane = locomp.simd_lane_id()
    bq = locomp.program_id(0)
    bh = locomp.program_id(1)
    tid = sgid * 32 + lane
    sg_row = sgid // 2
    sg_col = sgid % 2
    head_off = bh * N * 128

    Qs = locomp.shared_memory(2048)    # 16*128
    KTs = locomp.shared_memory(2048)   # 128*16
    Vs = locomp.shared_memory(2048)    # 16*128
    Ss = locomp.shared_memory(256)     # 16*16
    row_m = locomp.shared_memory(16)
    row_l = locomp.shared_memory(16)
    old_m_snap = locomp.shared_memory(16)
    Ds = locomp.shared_memory(128)

    # Load Q: 2048 values / 128 threads = 16 per thread
    for iq in range(16):
        q_idx = tid + iq * 128
        locomp.shared_store(Qs, q_idx, locomp.load(Q + (head_off + bq * 16 * 128 + q_idx)))

    if tid < 16:
        locomp.shared_store(row_m, tid, -1000000.0)
        locomp.shared_store(row_l, tid, 0.0)

    # 8 output accumulators per SG half (64 cols / 8 = 8 blocks)
    acc_o0 = locomp.simdgroup_matrix(0.0)
    acc_o1 = locomp.simdgroup_matrix(0.0)
    acc_o2 = locomp.simdgroup_matrix(0.0)
    acc_o3 = locomp.simdgroup_matrix(0.0)
    acc_o4 = locomp.simdgroup_matrix(0.0)
    acc_o5 = locomp.simdgroup_matrix(0.0)
    acc_o6 = locomp.simdgroup_matrix(0.0)
    acc_o7 = locomp.simdgroup_matrix(0.0)
    locomp.barrier()

    for bk in range(NUM_KV_BLOCKS):
        kv_base = bk * 16

        # Load K^T and V: 2048 values / 128 threads = 16 per thread
        for il in range(16):
            kv_idx = tid + il * 128
            k_bc = kv_idx // 128
            k_d = kv_idx % 128
            locomp.shared_store(KTs, k_d * 16 + k_bc,
                                locomp.load(K + (head_off + (kv_base + k_bc) * 128 + k_d)))
            locomp.shared_store(Vs, kv_idx,
                                locomp.load(V + (head_off + kv_base * 128 + kv_idx)))
        locomp.barrier()

        # QK: [16,128] × [128,16] = [16,16]
        s_acc = locomp.simdgroup_matrix(0.0)
        for dk in range(16):   # D/8 = 128/8 = 16
            q_blk = locomp.simdgroup_matrix_load(Qs, sg_row * 8 * 128 + dk * 8, 128)
            kt_blk = locomp.simdgroup_matrix_load(KTs, dk * 8 * 16 + sg_col * 8, 16)
            s_acc = locomp.simdgroup_mac(s_acc, q_blk, kt_blk)
        locomp.simdgroup_matrix_store(s_acc, Ss, sg_row * 8 * 16 + sg_col * 8, 16)
        locomp.barrier()

        if tid < 16:
            locomp.shared_store(old_m_snap, tid, locomp.shared_load(row_m, tid))
        locomp.barrier()

        # Causal masked softmax
        if tid < 16:
            row = tid
            old_m = locomp.shared_load(old_m_snap, row)
            q_pos = bq * 16 + row
            block_max = -1000000.0
            for j in range(16):
                kv_pos = kv_base + j
                s_val = locomp.shared_load(Ss, row * 16 + j)
                s_val = locomp.where(kv_pos <= q_pos, s_val, -1000000.0)
                locomp.shared_store(Ss, row * 16 + j, s_val)
                block_max = locomp.where(s_val > block_max, s_val, block_max)
            new_max = locomp.where(block_max > old_m, block_max, old_m)
            locomp.shared_store(row_m, row, new_max)
            old_sum = locomp.shared_load(row_l, row)
            rescaled_sum = old_sum * locomp.exp(old_m - new_max)
            block_sum = 0.0
            for j in range(16):
                s_val = locomp.shared_load(Ss, row * 16 + j)
                p_val = locomp.exp(s_val - new_max)
                block_sum = block_sum + p_val
                locomp.shared_store(Ss, row * 16 + j, p_val)
            locomp.shared_store(row_l, row, rescaled_sum + block_sum)
        locomp.barrier()

        # Diagonal rescale
        locomp.shared_store(Ds, tid, 0.0)
        locomp.barrier()
        if tid < 8:
            locomp.shared_store(Ds, tid * 8 + tid,
                locomp.exp(locomp.shared_load(old_m_snap, tid) - locomp.shared_load(row_m, tid)))
            locomp.shared_store(Ds, 64 + tid * 8 + tid,
                locomp.exp(locomp.shared_load(old_m_snap, tid + 8) - locomp.shared_load(row_m, tid + 8)))
        locomp.barrier()

        d_mat = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
        zero = locomp.simdgroup_matrix(0.0)
        acc_o0 = locomp.simdgroup_mac(zero, d_mat, acc_o0)
        acc_o1 = locomp.simdgroup_mac(zero, d_mat, acc_o1)
        acc_o2 = locomp.simdgroup_mac(zero, d_mat, acc_o2)
        acc_o3 = locomp.simdgroup_mac(zero, d_mat, acc_o3)
        acc_o4 = locomp.simdgroup_mac(zero, d_mat, acc_o4)
        acc_o5 = locomp.simdgroup_mac(zero, d_mat, acc_o5)
        acc_o6 = locomp.simdgroup_mac(zero, d_mat, acc_o6)
        acc_o7 = locomp.simdgroup_mac(zero, d_mat, acc_o7)

        # PV: [16,16] × [16,128] → [16,128]
        # Each SG half covers 64 cols (sg_col * 64), 8 blocks of 8
        for jj in range(2):
            p_blk = locomp.simdgroup_matrix_load(Ss, sg_row * 8 * 16 + jj * 8, 16)
            v0 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 128 + sg_col * 64, 128)
            v1 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 128 + sg_col * 64 + 8, 128)
            v2 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 128 + sg_col * 64 + 16, 128)
            v3 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 128 + sg_col * 64 + 24, 128)
            v4 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 128 + sg_col * 64 + 32, 128)
            v5 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 128 + sg_col * 64 + 40, 128)
            v6 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 128 + sg_col * 64 + 48, 128)
            v7 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 128 + sg_col * 64 + 56, 128)
            acc_o0 = locomp.simdgroup_mac(acc_o0, p_blk, v0)
            acc_o1 = locomp.simdgroup_mac(acc_o1, p_blk, v1)
            acc_o2 = locomp.simdgroup_mac(acc_o2, p_blk, v2)
            acc_o3 = locomp.simdgroup_mac(acc_o3, p_blk, v3)
            acc_o4 = locomp.simdgroup_mac(acc_o4, p_blk, v4)
            acc_o5 = locomp.simdgroup_mac(acc_o5, p_blk, v5)
            acc_o6 = locomp.simdgroup_mac(acc_o6, p_blk, v6)
            acc_o7 = locomp.simdgroup_mac(acc_o7, p_blk, v7)
        locomp.barrier()

    # Final divide
    locomp.shared_store(Ds, tid, 0.0)
    locomp.barrier()
    if tid < 8:
        locomp.shared_store(Ds, tid * 8 + tid, 1.0 / locomp.shared_load(row_l, tid))
        locomp.shared_store(Ds, 64 + tid * 8 + tid, 1.0 / locomp.shared_load(row_l, tid + 8))
    locomp.barrier()
    d_final = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
    z2 = locomp.simdgroup_matrix(0.0)
    acc_o0 = locomp.simdgroup_mac(z2, d_final, acc_o0)
    acc_o1 = locomp.simdgroup_mac(z2, d_final, acc_o1)
    acc_o2 = locomp.simdgroup_mac(z2, d_final, acc_o2)
    acc_o3 = locomp.simdgroup_mac(z2, d_final, acc_o3)
    acc_o4 = locomp.simdgroup_mac(z2, d_final, acc_o4)
    acc_o5 = locomp.simdgroup_mac(z2, d_final, acc_o5)
    acc_o6 = locomp.simdgroup_mac(z2, d_final, acc_o6)
    acc_o7 = locomp.simdgroup_mac(z2, d_final, acc_o7)

    c_row = bq * 16 + sg_row * 8
    c_col = sg_col * 64
    locomp.simdgroup_matrix_store_device(acc_o0, O + (head_off + c_row * 128 + c_col), 128)
    locomp.simdgroup_matrix_store_device(acc_o1, O + (head_off + c_row * 128 + c_col + 8), 128)
    locomp.simdgroup_matrix_store_device(acc_o2, O + (head_off + c_row * 128 + c_col + 16), 128)
    locomp.simdgroup_matrix_store_device(acc_o3, O + (head_off + c_row * 128 + c_col + 24), 128)
    locomp.simdgroup_matrix_store_device(acc_o4, O + (head_off + c_row * 128 + c_col + 32), 128)
    locomp.simdgroup_matrix_store_device(acc_o5, O + (head_off + c_row * 128 + c_col + 40), 128)
    locomp.simdgroup_matrix_store_device(acc_o6, O + (head_off + c_row * 128 + c_col + 48), 128)
    locomp.simdgroup_matrix_store_device(acc_o7, O + (head_off + c_row * 128 + c_col + 56), 128)


# =============================================================================
# Unified dispatch
# =============================================================================

_kernels = {64: causal_flash_d64, 128: causal_flash_d128}


def gpu_causal_attention(Q, K, V):
    """GPU causal flash attention. Q[H,N,D], K[H,N,D], V[H,N,D] → O[H,N,D].
    Supports D=64 and D=128. Q is pre-scaled internally."""
    H, N, D = Q.shape
    assert D in _kernels, f"Unsupported head dim D={D}. Supported: {list(_kernels.keys())}"
    assert N % 16 == 0, f"Sequence length must be multiple of 16, got {N}"

    scale = 1.0 / np.sqrt(D)
    Q_scaled = (Q * scale).astype(np.float32)

    Q_g = locomp.tensor(Q_scaled.reshape(-1))
    K_g = locomp.tensor(K.reshape(-1))
    V_g = locomp.tensor(V.reshape(-1))
    O_g = locomp.empty(H * N * D)

    nkv = N // 16
    _kernels[D][(nkv, H), (32, 4)](Q_g, K_g, V_g, O_g, N, nkv)
    return O_g.numpy().reshape(H, N, D)


def numpy_causal_attention(Q, K, V):
    H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    mask = np.triu(np.ones((N, N), dtype=np.float32), k=1) * -1e9
    out = np.zeros_like(Q)
    for h in range(H):
        scores = (Q[h] @ K[h].T) * scale + mask
        sm = scores.max(axis=-1, keepdims=True)
        es = np.exp(scores - sm)
        aw = es / es.sum(axis=-1, keepdims=True)
        out[h] = aw @ V[h]
    return out


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    WARMUP = 5
    RUNS = 15

    for D in [64, 128]:
        num_heads = 12 if D == 64 else 8
        sizes = [16, 32, 64, 128, 256]

        print(f"\n{'='*60}")
        print(f"Causal Flash Attention D={D}, H={num_heads}")
        print(f"{'='*60}")
        print(f"{'N':>6} | {'GPU':>8} | {'NumPy':>8} | GPU/NP | {'Error':>8}")
        print("-" * 52)

        for N in sizes:
            np.random.seed(42)
            Q_np = np.random.randn(num_heads, N, D).astype(np.float32) * 0.1
            K_np = np.random.randn(num_heads, N, D).astype(np.float32) * 0.1
            V_np = np.random.randn(num_heads, N, D).astype(np.float32) * 0.1

            ref = numpy_causal_attention(Q_np, K_np, V_np)
            gpu_out = gpu_causal_attention(Q_np, K_np, V_np)
            err = np.max(np.abs(gpu_out - ref))

            for _ in range(WARMUP):
                gpu_causal_attention(Q_np, K_np, V_np)
            times_gpu = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                gpu_causal_attention(Q_np, K_np, V_np)
                times_gpu.append((time.perf_counter() - t0) * 1000)
            t_gpu = sorted(times_gpu)[RUNS // 2]

            for _ in range(WARMUP):
                numpy_causal_attention(Q_np, K_np, V_np)
            times_np = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                numpy_causal_attention(Q_np, K_np, V_np)
                times_np.append((time.perf_counter() - t0) * 1000)
            t_np = sorted(times_np)[RUNS // 2]

            r_np = t_gpu / t_np
            print(f"{N:>6} | {t_gpu:>6.3f}ms | {t_np:>6.3f}ms | {r_np:>5.2f}x | {err:.2e}")
