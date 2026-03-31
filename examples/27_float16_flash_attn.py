"""
Float16 Simdgroup Flash Attention — half precision QKV with float32 softmax.

Mixed precision: data in half (2× bandwidth), softmax in float32 (numerical stability).
Uses simdgroup_half8x8 for QK and PV matmuls.
Shares the same algorithmic structure as example 22.
"""

import time
import numpy as np
import locomp

Br = 16
Bc = 16
D_HEAD = 32
NUM_LOADS = 4


@locomp.kernel
def flash_attn_f16(Q: locomp.Float16, K: locomp.Float16, V: locomp.Float16,
                   O: locomp.Float16,
                   N: locomp.constexpr, D: locomp.constexpr,
                   NUM_KV_BLOCKS: locomp.constexpr,
                   BLOCK_R: locomp.constexpr, BLOCK_C: locomp.constexpr):
    sgid = locomp.simd_group_id()
    lane = locomp.simd_lane_id()
    bq = locomp.program_id(0)
    tid = sgid * 32 + lane

    sg_row = sgid // 2
    sg_col = sgid % 2

    # Data in half, softmax stats in float32
    Qs = locomp.shared_memory(Br * D_HEAD, locomp.Float16)
    KTs = locomp.shared_memory(D_HEAD * Bc, locomp.Float16)
    Vs = locomp.shared_memory(Bc * D_HEAD, locomp.Float16)
    Ss = locomp.shared_memory(Br * Bc, locomp.Float16) # half (from QK matmul)
    row_m = locomp.shared_memory(Br)             # float32
    row_l = locomp.shared_memory(Br)             # float32
    old_m_snap = locomp.shared_memory(Br)        # float32
    Ds = locomp.shared_memory(128, locomp.Float16)  # diagonal scale matrices in half

    # Load Q pre-scaled
    for iq in range(NUM_LOADS):
        q_idx = tid + iq * 128
        q_row = bq * BLOCK_R + q_idx // D
        q_col = q_idx % D
        q_val = locomp.load(Q + (q_row * D + q_col))
        locomp.shared_store(Qs, q_idx, q_val * 0.176776695)

    if tid < 16:
        locomp.shared_store(row_m, tid, -1000000.0)
        locomp.shared_store(row_l, tid, 0.0)

    acc_o0 = locomp.simdgroup_matrix(0.0, locomp.Float16)
    acc_o1 = locomp.simdgroup_matrix(0.0, locomp.Float16)

    locomp.barrier()

    for bk in range(NUM_KV_BLOCKS):
        kv_base = bk * BLOCK_C

        for il in range(NUM_LOADS):
            kv_idx = tid + il * 128
            k_bc = kv_idx // D
            k_d = kv_idx % D
            k_val = locomp.load(K + ((kv_base + k_bc) * D + k_d))
            locomp.shared_store(KTs, k_d * BLOCK_C + k_bc, k_val)
            v_row = kv_idx // D
            v_col = kv_idx % D
            v_val = locomp.load(V + ((kv_base + v_row) * D + v_col))
            locomp.shared_store(Vs, v_row * D + v_col, v_val)

        locomp.barrier()

        # QK matmul (half precision)
        s_acc = locomp.simdgroup_matrix(0.0, locomp.Float16)
        for dk in range(4):
            q_blk = locomp.simdgroup_matrix_load(Qs, sg_row * 8 * D + dk * 8, D)
            kt_blk = locomp.simdgroup_matrix_load(KTs, dk * 8 * BLOCK_C + sg_col * 8, BLOCK_C)
            s_acc = locomp.simdgroup_mac(s_acc, q_blk, kt_blk)
        # Store S to float32 shared memory for softmax
        locomp.simdgroup_matrix_store(s_acc, Ss, sg_row * 8 * BLOCK_C + sg_col * 8, BLOCK_C)

        locomp.barrier()

        if tid < 16:
            locomp.shared_store(old_m_snap, tid, locomp.shared_load(row_m, tid))

        locomp.barrier()

        # Softmax in float32
        if tid < 16:
            row = tid
            old_m = locomp.shared_load(old_m_snap, row)
            block_max = locomp.shared_load(Ss, row * BLOCK_C)
            for j in range(1, BLOCK_C):
                s_val = locomp.shared_load(Ss, row * BLOCK_C + j)
                block_max = locomp.where(s_val > block_max, s_val, block_max)
            new_max = locomp.where(block_max > old_m, block_max, old_m)
            locomp.shared_store(row_m, row, new_max)
            old_sum = locomp.shared_load(row_l, row)
            rescaled_sum = old_sum * locomp.exp(old_m - new_max)
            block_sum = 0.0
            for j in range(BLOCK_C):
                s_val = locomp.shared_load(Ss, row * BLOCK_C + j)
                p_val = locomp.exp(s_val - new_max)
                block_sum = block_sum + p_val
                locomp.shared_store(Ss, row * BLOCK_C + j, p_val)
            locomp.shared_store(row_l, row, rescaled_sum + block_sum)

        locomp.barrier()

        # Build diagonal scale matrices (in half for simdgroup ops)
        locomp.shared_store(Ds, tid, 0.0)
        locomp.barrier()
        if tid < 8:
            scale_top = locomp.exp(locomp.shared_load(old_m_snap, tid) - locomp.shared_load(row_m, tid))
            locomp.shared_store(Ds, tid * 8 + tid, scale_top)
            scale_bot = locomp.exp(locomp.shared_load(old_m_snap, tid + 8) - locomp.shared_load(row_m, tid + 8))
            locomp.shared_store(Ds, 64 + tid * 8 + tid, scale_bot)
        locomp.barrier()

        # Rescale O: O = D × O
        d_mat = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
        zero = locomp.simdgroup_matrix(0.0, locomp.Float16)
        acc_o0 = locomp.simdgroup_mac(zero, d_mat, acc_o0)
        acc_o1 = locomp.simdgroup_mac(zero, d_mat, acc_o1)

        # PV matmul (half precision)
        for jj in range(2):
            p_blk = locomp.simdgroup_matrix_load(Ss, sg_row * 8 * BLOCK_C + jj * 8, BLOCK_C)
            v_blk0 = locomp.simdgroup_matrix_load(Vs, jj * 8 * D + sg_col * 16, D)
            v_blk1 = locomp.simdgroup_matrix_load(Vs, jj * 8 * D + sg_col * 16 + 8, D)
            acc_o0 = locomp.simdgroup_mac(acc_o0, p_blk, v_blk0)
            acc_o1 = locomp.simdgroup_mac(acc_o1, p_blk, v_blk1)

        locomp.barrier()

    # Final: divide by row_l
    locomp.shared_store(Ds, tid, 0.0)
    locomp.barrier()
    if tid < 8:
        inv_l_top = 1.0 / locomp.shared_load(row_l, tid)
        locomp.shared_store(Ds, tid * 8 + tid, inv_l_top)
        inv_l_bot = 1.0 / locomp.shared_load(row_l, tid + 8)
        locomp.shared_store(Ds, 64 + tid * 8 + tid, inv_l_bot)
    locomp.barrier()
    d_final = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
    zero2 = locomp.simdgroup_matrix(0.0, locomp.Float16)
    acc_o0 = locomp.simdgroup_mac(zero2, d_final, acc_o0)
    acc_o1 = locomp.simdgroup_mac(zero2, d_final, acc_o1)

    c_row = bq * BLOCK_R + sg_row * 8
    c_col = sg_col * 16
    locomp.simdgroup_matrix_store_device(acc_o0, O + (c_row * D + c_col), D)
    locomp.simdgroup_matrix_store_device(acc_o1, O + (c_row * D + c_col + 8), D)


def naive_attention_np(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return attn @ V


if __name__ == "__main__":
    import mlx.core as mx

    sizes = [64, 128, 256, 512]
    d = D_HEAD
    WARMUP = 5; RUNS = 15

    print(f"Float16 Simdgroup Flash Attention: Br={Br} Bc={Bc} D={d}, 4 SG, async dispatch")
    print(f"{'N':>6} | {'F16 Locomp':>10} | {'F32 FA':>8} | {'MLX sdpa':>8} | {'F16/MLX':>8} | {'Error':>10}")
    print("-" * 70)

    for N in sizes:
        np.random.seed(42)
        Q_np = np.random.randn(N, d).astype(np.float16)
        K_np = np.random.randn(N, d).astype(np.float16)
        V_np = np.random.randn(N, d).astype(np.float16)
        expected = naive_attention_np(Q_np.astype(np.float32), K_np.astype(np.float32),
                                      V_np.astype(np.float32))

        Q_t = locomp.tensor(Q_np.flatten())
        K_t = locomp.tensor(K_np.flatten())
        V_t = locomp.tensor(V_np.flatten())
        O_t = locomp.empty(N * d, dtype=np.float16)

        n_kv = N // Bc
        grid = (N // Br,)

        flash_attn_f16[grid, (32, 4)](Q_t, K_t, V_t, O_t, N, d, n_kv, Br, Bc)
        result = O_t.numpy().reshape(N, d).astype(np.float32)
        err = np.max(np.abs(result - expected))

        # Benchmark F16
        for _ in range(WARMUP):
            flash_attn_f16[grid, (32, 4)](Q_t, K_t, V_t, O_t, N, d, n_kv, Br, Bc)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            flash_attn_f16[grid, (32, 4)](Q_t, K_t, V_t, O_t, N, d, n_kv, Br, Bc)
            times.append((time.perf_counter() - t0) * 1000)
        t_f16 = sorted(times)[RUNS // 2]

        # MLX sdpa
        mQ = mx.array(Q_np.astype(np.float32)).reshape(1, 1, N, d)
        mK = mx.array(K_np.astype(np.float32)).reshape(1, 1, N, d)
        mV = mx.array(V_np.astype(np.float32)).reshape(1, 1, N, d)
        mx.eval(mQ, mK, mV)
        for _ in range(WARMUP):
            out = mx.fast.scaled_dot_product_attention(mQ, mK, mV, scale=1.0/np.sqrt(d))
            mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mx.fast.scaled_dot_product_attention(mQ, mK, mV, scale=1.0/np.sqrt(d))
            mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[RUNS // 2]

        ratio = t_f16 / t_mlx
        print(f"{N:>6} | {t_f16:>8.3f}ms | {'—':>8} | {t_mlx:>6.3f}ms | {ratio:>7.2f}x | {err:.2e}")
