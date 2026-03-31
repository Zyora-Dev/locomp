"""
Simdgroup Flash Attention — hardware 8×8 matrix multiply for QK and PV.

Architecture:
  TG = 128 threads = 4 SIMD groups
  Br=16, Bc=16, D=32 (same tile sizes as flash_attn_v3)

  SIMD group layout (2×2):
    sg0: S[0:8, 0:8]    sg1: S[0:8, 8:16]
    sg2: S[8:16, 0:8]   sg3: S[8:16, 8:16]

  Replaces:
    - Scalar QK dot product (32 FMAs/element) → 4 simdgroup_macs
    - Scalar PV accumulation (16 FMAs/element) → 2 simdgroup_macs per output block

  Shared memory (7.9 KB):
    Qs[16×32]=512  KTs[32×16]=512  Vs[16×32]=512  Ss[16×16]=256
    row_m[16]  row_l[16]  old_m_snap[16]  Ds[128]  (2× diagonal 8×8)

  Rescaling via diagonal matrix multiply: D×O where D=diag(exp(old_m - new_m))
"""

import time
import numpy as np
import locomp

Br = 16
Bc = 16
D_HEAD = 32
NUM_LOADS = 4   # 512 elements / 128 threads


@locomp.kernel
def flash_attn_simd(Q: locomp.Tensor, K: locomp.Tensor, V: locomp.Tensor,
                    O: locomp.Tensor,
                    N: locomp.constexpr, D: locomp.constexpr,
                    NUM_KV_BLOCKS: locomp.constexpr,
                    BLOCK_R: locomp.constexpr, BLOCK_C: locomp.constexpr):
    sgid = locomp.simd_group_id()   # 0..3
    lane = locomp.simd_lane_id()    # 0..31
    bq = locomp.program_id(0)

    tid = sgid * 32 + lane  # 0..127

    # SIMD group's position in 2×2 grid
    sg_row = sgid // 2   # 0 or 1 (row half: 0=top 8 rows, 1=bottom 8 rows)
    sg_col = sgid % 2    # 0 or 1 (col half: 0=left 8 cols, 1=right 8 cols)

    # Shared memory
    Qs = locomp.shared_memory(Br * D_HEAD)        # [16×32] = 512
    KTs = locomp.shared_memory(D_HEAD * Bc)       # [32×16] = 512 (K transposed)
    Vs = locomp.shared_memory(Bc * D_HEAD)        # [16×32] = 512
    Ss = locomp.shared_memory(Br * Bc)            # [16×16] = 256
    row_m = locomp.shared_memory(Br)              # 16
    row_l = locomp.shared_memory(Br)              # 16
    old_m_snap = locomp.shared_memory(Br)         # 16
    Ds = locomp.shared_memory(128)                # 2× diagonal 8×8 matrices

    # ── Load Q once (pre-scaled by 1/sqrt(d)) ──
    # 512 values, 128 threads = 4 per thread
    for iq in range(NUM_LOADS):
        q_idx = tid + iq * 128
        q_row = bq * BLOCK_R + q_idx // D
        q_col = q_idx % D
        q_val = locomp.load(Q + (q_row * D + q_col))
        locomp.shared_store(Qs, q_idx, q_val * 0.176776695)

    # Init row stats
    if tid < 16:
        locomp.shared_store(row_m, tid, -1000000.0)
        locomp.shared_store(row_l, tid, 0.0)

    # Output accumulators: each group owns 2 blocks of O[sg_row*8:+8, ...]
    # sg0: O[0:8, 0:8] O[0:8, 8:16]
    # sg1: O[0:8, 16:24] O[0:8, 24:32]
    # sg2: O[8:16, 0:8] O[8:16, 8:16]
    # sg3: O[8:16, 16:24] O[8:16, 24:32]
    acc_o0 = locomp.simdgroup_matrix(0.0)
    acc_o1 = locomp.simdgroup_matrix(0.0)

    locomp.barrier()

    for bk in range(NUM_KV_BLOCKS):
        kv_base = bk * BLOCK_C

        # ── Load K transposed and V ──
        for il in range(NUM_LOADS):
            kv_idx = tid + il * 128
            # K: read K[kv_base + bc, d], store transposed as KTs[d, bc]
            k_bc = kv_idx // D
            k_d = kv_idx % D
            k_val = locomp.load(K + ((kv_base + k_bc) * D + k_d))
            locomp.shared_store(KTs, k_d * BLOCK_C + k_bc, k_val)
            # V: read V[kv_base + row, col], store as Vs[row, col]
            v_row = kv_idx // D
            v_col = kv_idx % D
            v_val = locomp.load(V + ((kv_base + v_row) * D + v_col))
            locomp.shared_store(Vs, v_row * D + v_col, v_val)

        locomp.barrier()

        # ── QK matmul: S = Q × K^T using simdgroup_matrix ──
        # Each group computes S[sg_row*8:+8, sg_col*8:+8]
        s_acc = locomp.simdgroup_matrix(0.0)
        for dk in range(4):   # D/8 = 32/8 = 4
            q_blk = locomp.simdgroup_matrix_load(Qs, sg_row * 8 * D + dk * 8, D)
            kt_blk = locomp.simdgroup_matrix_load(KTs, dk * 8 * BLOCK_C + sg_col * 8, BLOCK_C)
            s_acc = locomp.simdgroup_mac(s_acc, q_blk, kt_blk)
        # Store S block to shared
        locomp.simdgroup_matrix_store(s_acc, Ss, sg_row * 8 * BLOCK_C + sg_col * 8, BLOCK_C)

        locomp.barrier()

        # ── Snapshot old max ──
        if tid < 16:
            locomp.shared_store(old_m_snap, tid, locomp.shared_load(row_m, tid))

        locomp.barrier()

        # ── Softmax: 16 threads (tid < 16) each handle one row ──
        if tid < 16:
            row = tid
            old_m = locomp.shared_load(old_m_snap, row)

            # Row max
            block_max = locomp.shared_load(Ss, row * BLOCK_C)
            for j in range(1, BLOCK_C):
                s_val = locomp.shared_load(Ss, row * BLOCK_C + j)
                block_max = locomp.where(s_val > block_max, s_val, block_max)

            new_max = locomp.where(block_max > old_m, block_max, old_m)
            locomp.shared_store(row_m, row, new_max)

            # Row sum + store P = exp(S - max) into Ss
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

        # ── Build diagonal scale matrices for rescaling O accumulators ──
        # Zero Ds[0:128] (128 threads, 1 each)
        locomp.shared_store(Ds, tid, 0.0)
        locomp.barrier()
        # Fill diagonals: Ds_top[i,i] = scale for row i, Ds_bot[i,i] = scale for row 8+i
        if tid < 8:
            scale_top = locomp.exp(locomp.shared_load(old_m_snap, tid) - locomp.shared_load(row_m, tid))
            locomp.shared_store(Ds, tid * 8 + tid, scale_top)
            scale_bot = locomp.exp(locomp.shared_load(old_m_snap, tid + 8) - locomp.shared_load(row_m, tid + 8))
            locomp.shared_store(Ds, 64 + tid * 8 + tid, scale_bot)
        locomp.barrier()

        # ── Rescale accumulators via diagonal matrix multiply: O = D × O ──
        d_mat = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
        zero = locomp.simdgroup_matrix(0.0)
        acc_o0 = locomp.simdgroup_mac(zero, d_mat, acc_o0)
        acc_o1 = locomp.simdgroup_mac(zero, d_mat, acc_o1)

        # ── PV matmul: O += P × V ──
        # sg0: O[0:8, 0:8]+=[0:8,8:16],  sg1: O[0:8, 16:24]+=[0:8,24:32]
        # sg2: O[8:16, 0:8]+=[8:16,8:16], sg3: O[8:16, 16:24]+=[8:16,24:32]
        for jj in range(2):   # Bc/8 = 16/8 = 2
            p_blk = locomp.simdgroup_matrix_load(Ss, sg_row * 8 * BLOCK_C + jj * 8, BLOCK_C)
            v_blk0 = locomp.simdgroup_matrix_load(Vs, jj * 8 * D + sg_col * 16, D)
            v_blk1 = locomp.simdgroup_matrix_load(Vs, jj * 8 * D + sg_col * 16 + 8, D)
            acc_o0 = locomp.simdgroup_mac(acc_o0, p_blk, v_blk0)
            acc_o1 = locomp.simdgroup_mac(acc_o1, p_blk, v_blk1)

        locomp.barrier()

    # ── Final: divide by row_l using diagonal matrix ──
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

    # ── Store output ──
    c_row = bq * BLOCK_R + sg_row * 8
    c_col = sg_col * 16
    locomp.simdgroup_matrix_store_device(acc_o0, O + (c_row * D + c_col), D)
    locomp.simdgroup_matrix_store_device(acc_o1, O + (c_row * D + c_col + 8), D)


def naive_attention_np(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    from scipy.special import softmax
    attn = softmax(scores, axis=-1)
    return attn @ V


if __name__ == "__main__":
    import mlx.core as mx

    d = D_HEAD
    sizes = [64, 128, 256, 512]

    print(f"Simdgroup Flash Attention: d={d}, Br={Br}, Bc={Bc}, TG=128 (4 SIMD groups)")
    print(f"{'N':>6} | {'simd':>8} | {'v3':>8} | {'MLX':>8} | simd/MLX | simd/v3 | {'Error':>8}")
    print("-" * 75)

    import importlib
    v3_mod = importlib.import_module("examples.17_flash_attention_v3")
    flash_attn_v3 = v3_mod.flash_attn_v3

    for N in sizes:
        np.random.seed(42)
        Q_np = np.random.randn(N, d).astype(np.float32) * 0.1
        K_np = np.random.randn(N, d).astype(np.float32) * 0.1
        V_np = np.random.randn(N, d).astype(np.float32) * 0.1
        expected = naive_attention_np(Q_np, K_np, V_np)

        Q_t = locomp.tensor(Q_np.flatten())
        K_t = locomp.tensor(K_np.flatten())
        V_t = locomp.tensor(V_np.flatten())
        nkv = N // Bc

        # ── simdgroup flash attention ──
        O_t = locomp.empty(N * d)
        flash_attn_simd[(N // Br,), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        result = O_t.numpy().reshape(N, d)
        err = np.max(np.abs(result - expected))

        WARMUP = 5; RUNS = 15
        for _ in range(WARMUP):
            flash_attn_simd[(N // Br,), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        times_simd = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            flash_attn_simd[(N // Br,), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
            times_simd.append((time.perf_counter() - t0) * 1000)
        t_simd = sorted(times_simd)[RUNS // 2]

        # ── v3 flash attention ──
        O2 = locomp.empty(N * d)
        for _ in range(WARMUP):
            flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O2, N, d, nkv, Br, Bc)
        times_v3 = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O2, N, d, nkv, Br, Bc)
            times_v3.append((time.perf_counter() - t0) * 1000)
        t_v3 = sorted(times_v3)[RUNS // 2]

        # ── MLX ──
        mQ = mx.array(Q_np); mK = mx.array(K_np); mV = mx.array(V_np)
        mQ4 = mQ.reshape(1, N, 1, d); mK4 = mK.reshape(1, N, 1, d); mV4 = mV.reshape(1, N, 1, d)
        mx.eval(mQ4, mK4, mV4)
        for _ in range(WARMUP):
            out = mx.fast.scaled_dot_product_attention(mQ4, mK4, mV4, scale=1.0/np.sqrt(d))
            mx.eval(out)
        times_mlx = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mx.fast.scaled_dot_product_attention(mQ4, mK4, mV4, scale=1.0/np.sqrt(d))
            mx.eval(out)
            times_mlx.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(times_mlx)[RUNS // 2]

        r_mlx = t_simd / t_mlx
        r_v3 = t_simd / t_v3
        print(f"{N:>6} | {t_simd:>6.3f}ms | {t_v3:>6.3f}ms | {t_mlx:>6.3f}ms | {r_mlx:>7.2f}x | {r_v3:>6.2f}x | {err:.2e}")
