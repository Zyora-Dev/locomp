"""
Multi-Head Attention — simdgroup flash attention over [B, H, N, D].

Extends example 22's single-head kernel with a batch-head dimension.
Each threadgroup handles one (batch, head, query_block) triple.
Grid: (N/Br, B*H) — x=query block, y=batch*head.

Input layout: Q, K, V = [B, H, N, D] contiguous (B*H*N*D floats)
Output: O = [B, H, N, D]

Same architecture as flash_attn_simd:
  TG = 128 threads = 4 SIMD groups
  Br=16, Bc=16, D=32
  simdgroup_mac for QK and PV, diagonal rescaling for online softmax
"""

import time
import numpy as np
import locomp

Br = 16
Bc = 16
D_HEAD = 32
NUM_LOADS = 4


@locomp.kernel
def multi_head_attn(Q: locomp.Tensor, K: locomp.Tensor, V: locomp.Tensor,
                    O: locomp.Tensor,
                    N: locomp.constexpr, D: locomp.constexpr,
                    NUM_KV_BLOCKS: locomp.constexpr,
                    BLOCK_R: locomp.constexpr, BLOCK_C: locomp.constexpr):
    sgid = locomp.simd_group_id()
    lane = locomp.simd_lane_id()
    bq = locomp.program_id(0)      # query block index within head
    bh = locomp.program_id(1)      # batch * num_heads + head

    tid = sgid * 32 + lane
    sg_row = sgid // 2
    sg_col = sgid % 2

    # Per-head offset into Q/K/V/O
    head_off = bh * N * D

    Qs = locomp.shared_memory(Br * D_HEAD)
    KTs = locomp.shared_memory(D_HEAD * Bc)
    Vs = locomp.shared_memory(Bc * D_HEAD)
    Ss = locomp.shared_memory(Br * Bc)
    row_m = locomp.shared_memory(Br)
    row_l = locomp.shared_memory(Br)
    old_m_snap = locomp.shared_memory(Br)
    Ds = locomp.shared_memory(128)

    # Load Q (pre-scaled)
    for iq in range(NUM_LOADS):
        q_idx = tid + iq * 128
        q_row = bq * BLOCK_R + q_idx // D
        q_col = q_idx % D
        q_val = locomp.load(Q + (head_off + q_row * D + q_col))
        locomp.shared_store(Qs, q_idx, q_val * 0.176776695)

    if tid < 16:
        locomp.shared_store(row_m, tid, -1000000.0)
        locomp.shared_store(row_l, tid, 0.0)

    acc_o0 = locomp.simdgroup_matrix(0.0)
    acc_o1 = locomp.simdgroup_matrix(0.0)

    locomp.barrier()

    for bk in range(NUM_KV_BLOCKS):
        kv_base = bk * BLOCK_C

        for il in range(NUM_LOADS):
            kv_idx = tid + il * 128
            k_bc = kv_idx // D
            k_d = kv_idx % D
            k_val = locomp.load(K + (head_off + (kv_base + k_bc) * D + k_d))
            locomp.shared_store(KTs, k_d * BLOCK_C + k_bc, k_val)
            v_row = kv_idx // D
            v_col = kv_idx % D
            v_val = locomp.load(V + (head_off + (kv_base + v_row) * D + v_col))
            locomp.shared_store(Vs, v_row * D + v_col, v_val)

        locomp.barrier()

        # QK matmul
        s_acc = locomp.simdgroup_matrix(0.0)
        for dk in range(4):
            q_blk = locomp.simdgroup_matrix_load(Qs, sg_row * 8 * D + dk * 8, D)
            kt_blk = locomp.simdgroup_matrix_load(KTs, dk * 8 * BLOCK_C + sg_col * 8, BLOCK_C)
            s_acc = locomp.simdgroup_mac(s_acc, q_blk, kt_blk)
        locomp.simdgroup_matrix_store(s_acc, Ss, sg_row * 8 * BLOCK_C + sg_col * 8, BLOCK_C)

        locomp.barrier()

        if tid < 16:
            locomp.shared_store(old_m_snap, tid, locomp.shared_load(row_m, tid))
        locomp.barrier()

        # Softmax
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

        # Rescale accumulators
        locomp.shared_store(Ds, tid, 0.0)
        locomp.barrier()
        if tid < 8:
            scale_top = locomp.exp(locomp.shared_load(old_m_snap, tid) - locomp.shared_load(row_m, tid))
            locomp.shared_store(Ds, tid * 8 + tid, scale_top)
            scale_bot = locomp.exp(locomp.shared_load(old_m_snap, tid + 8) - locomp.shared_load(row_m, tid + 8))
            locomp.shared_store(Ds, 64 + tid * 8 + tid, scale_bot)
        locomp.barrier()

        d_mat = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
        zero = locomp.simdgroup_matrix(0.0)
        acc_o0 = locomp.simdgroup_mac(zero, d_mat, acc_o0)
        acc_o1 = locomp.simdgroup_mac(zero, d_mat, acc_o1)

        # PV matmul
        for jj in range(2):
            p_blk = locomp.simdgroup_matrix_load(Ss, sg_row * 8 * BLOCK_C + jj * 8, BLOCK_C)
            v_blk0 = locomp.simdgroup_matrix_load(Vs, jj * 8 * D + sg_col * 16, D)
            v_blk1 = locomp.simdgroup_matrix_load(Vs, jj * 8 * D + sg_col * 16 + 8, D)
            acc_o0 = locomp.simdgroup_mac(acc_o0, p_blk, v_blk0)
            acc_o1 = locomp.simdgroup_mac(acc_o1, p_blk, v_blk1)

        locomp.barrier()

    # Final divide by row_l
    locomp.shared_store(Ds, tid, 0.0)
    locomp.barrier()
    if tid < 8:
        inv_l_top = 1.0 / locomp.shared_load(row_l, tid)
        locomp.shared_store(Ds, tid * 8 + tid, inv_l_top)
        inv_l_bot = 1.0 / locomp.shared_load(row_l, tid + 8)
        locomp.shared_store(Ds, 64 + tid * 8 + tid, inv_l_bot)
    locomp.barrier()
    d_final = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
    zero2 = locomp.simdgroup_matrix(0.0)
    acc_o0 = locomp.simdgroup_mac(zero2, d_final, acc_o0)
    acc_o1 = locomp.simdgroup_mac(zero2, d_final, acc_o1)

    # Store output
    c_row = bq * BLOCK_R + sg_row * 8
    c_col = sg_col * 16
    locomp.simdgroup_matrix_store_device(acc_o0, O + (head_off + c_row * D + c_col), D)
    locomp.simdgroup_matrix_store_device(acc_o1, O + (head_off + c_row * D + c_col + 8), D)


def ref_mha_np(Q, K, V):
    """Reference: [B, H, N, D] multi-head attention via NumPy."""
    B, H, N, D = Q.shape
    O = np.zeros_like(Q)
    from scipy.special import softmax
    for b in range(B):
        for h in range(H):
            scores = Q[b, h] @ K[b, h].T / np.sqrt(D)
            attn = softmax(scores, axis=-1)
            O[b, h] = attn @ V[b, h]
    return O


if __name__ == "__main__":
    import mlx.core as mx

    d = D_HEAD

    configs = [
        (1, 1, 64),    # single-head baseline
        (1, 4, 64),    # 4 heads
        (1, 8, 128),   # 8 heads, longer seq
        (2, 8, 128),   # batched
        (4, 8, 256),   # larger batch
    ]

    print(f"Multi-Head Attention: d={d}, Br={Br}, Bc={Bc}, TG=128 (4 SIMD groups)")
    print(f"{'B×H×N':>12} | {'Locust':>8} | {'MLX':>8} | {'Ratio':>7} | {'Error':>8}")
    print("-" * 58)

    for B, H, N in configs:
        np.random.seed(42)
        Q_np = np.random.randn(B, H, N, d).astype(np.float32) * 0.1
        K_np = np.random.randn(B, H, N, d).astype(np.float32) * 0.1
        V_np = np.random.randn(B, H, N, d).astype(np.float32) * 0.1
        expected = ref_mha_np(Q_np, K_np, V_np)

        Q_t = locomp.tensor(Q_np.flatten())
        K_t = locomp.tensor(K_np.flatten())
        V_t = locomp.tensor(V_np.flatten())
        O_t = locomp.empty(B * H * N * d)
        nkv = N // Bc

        grid_x = N // Br
        grid_y = B * H

        # Correctness
        multi_head_attn[(grid_x, grid_y), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        result = O_t.numpy().reshape(B, H, N, d)
        err = np.max(np.abs(result - expected))

        # Benchmark Locust
        WARMUP = 5; RUNS = 15
        for _ in range(WARMUP):
            multi_head_attn[(grid_x, grid_y), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            multi_head_attn[(grid_x, grid_y), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[RUNS // 2]

        # Benchmark MLX
        mQ = mx.array(Q_np); mK = mx.array(K_np); mV = mx.array(V_np)
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

        ratio = t_loc / t_mlx
        label = f"{B}×{H}×{N}"
        print(f"{label:>12} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
