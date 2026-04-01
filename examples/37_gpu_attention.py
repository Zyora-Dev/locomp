"""
Example 37: GPU Causal Flash Attention — D=64, multi-head.

Simdgroup-accelerated causal flash attention for GPT-2 (D=64, 12 heads).
Pre-scales Q on CPU (locomp constexpr truncates floats to int).

Architecture:
  TG = 128 threads = 4 SIMD groups (2×2 grid)
  Br=16, Bc=16, D=64
  Each SG covers 8 rows × 32 output columns (4 acc blocks of 8×8)

  SIMD group layout:
    sg0: rows 0:8,  cols 0:31    sg1: rows 0:8,  cols 32:63
    sg2: rows 8:16, cols 0:31    sg3: rows 8:16, cols 32:63

  Shared memory (13.7 KB):
    Qs[16×64]=1024  KTs[64×16]=1024  Vs[16×64]=1024  Ss[16×16]=256
    row_m[16]  row_l[16]  old_m_snap[16]  Ds[128]  (2× diagonal 8×8)

  Q must be pre-scaled by 1/sqrt(D) before passing to kernel.
"""

import time
import numpy as np
import locomp

Br = 16
Bc = 16
D_HEAD = 64


@locomp.kernel
def causal_flash_attn(Q: locomp.Tensor, K: locomp.Tensor, V: locomp.Tensor,
                      O: locomp.Tensor,
                      N: locomp.constexpr, D: locomp.constexpr,
                      NUM_KV_BLOCKS: locomp.constexpr,
                      BLOCK_R: locomp.constexpr, BLOCK_C: locomp.constexpr):
    sgid = locomp.simd_group_id()
    lane = locomp.simd_lane_id()
    bq = locomp.program_id(0)
    bh = locomp.program_id(1)
    tid = sgid * 32 + lane
    sg_row = sgid // 2
    sg_col = sgid % 2
    head_off = bh * N * D

    Qs = locomp.shared_memory(1024)   # Br*D = 16*64
    KTs = locomp.shared_memory(1024)  # D*Bc = 64*16
    Vs = locomp.shared_memory(1024)   # Bc*D = 16*64
    Ss = locomp.shared_memory(256)    # Br*Bc = 16*16
    row_m = locomp.shared_memory(16)
    row_l = locomp.shared_memory(16)
    old_m_snap = locomp.shared_memory(16)
    Ds = locomp.shared_memory(128)

    # Load Q (already pre-scaled by 1/sqrt(d))
    for iq in range(8):
        q_idx = tid + iq * 128
        q_row = bq * 16 + q_idx // 64
        q_col = q_idx % 64
        locomp.shared_store(Qs, q_idx, locomp.load(Q + (head_off + q_row * 64 + q_col)))

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

        # Load K^T and V
        for il in range(8):
            kv_idx = tid + il * 128
            k_bc = kv_idx // 64
            k_d = kv_idx % 64
            locomp.shared_store(KTs, k_d * 16 + k_bc,
                                locomp.load(K + (head_off + (kv_base + k_bc) * 64 + k_d)))
            v_row = kv_idx // 64
            v_col = kv_idx % 64
            locomp.shared_store(Vs, v_row * 64 + v_col,
                                locomp.load(V + (head_off + (kv_base + v_row) * 64 + v_col)))
        locomp.barrier()

        # QK matmul: S = Q × K^T  [16,64]×[64,16] = [16,16]
        s_acc = locomp.simdgroup_matrix(0.0)
        for dk in range(8):   # D/8 = 64/8 = 8
            q_blk = locomp.simdgroup_matrix_load(Qs, sg_row * 8 * 64 + dk * 8, 64)
            kt_blk = locomp.simdgroup_matrix_load(KTs, dk * 8 * 16 + sg_col * 8, 16)
            s_acc = locomp.simdgroup_mac(s_acc, q_blk, kt_blk)
        locomp.simdgroup_matrix_store(s_acc, Ss, sg_row * 8 * 16 + sg_col * 8, 16)
        locomp.barrier()

        # Snapshot old max
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

        # Diagonal rescale: O = diag(exp(old_m - new_m)) × O
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
        acc_o2 = locomp.simdgroup_mac(zero, d_mat, acc_o2)
        acc_o3 = locomp.simdgroup_mac(zero, d_mat, acc_o3)

        # PV matmul: O += P × V  [16,16]×[16,64] → [16,64]
        for jj in range(2):   # Bc/8 = 2
            p_blk = locomp.simdgroup_matrix_load(Ss, sg_row * 8 * 16 + jj * 8, 16)
            v_blk0 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 64 + sg_col * 32, 64)
            v_blk1 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 64 + sg_col * 32 + 8, 64)
            v_blk2 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 64 + sg_col * 32 + 16, 64)
            v_blk3 = locomp.simdgroup_matrix_load(Vs, jj * 8 * 64 + sg_col * 32 + 24, 64)
            acc_o0 = locomp.simdgroup_mac(acc_o0, p_blk, v_blk0)
            acc_o1 = locomp.simdgroup_mac(acc_o1, p_blk, v_blk1)
            acc_o2 = locomp.simdgroup_mac(acc_o2, p_blk, v_blk2)
            acc_o3 = locomp.simdgroup_mac(acc_o3, p_blk, v_blk3)
        locomp.barrier()

    # Final: divide by row_l
    locomp.shared_store(Ds, tid, 0.0)
    locomp.barrier()
    if tid < 8:
        inv_top = 1.0 / locomp.shared_load(row_l, tid)
        locomp.shared_store(Ds, tid * 8 + tid, inv_top)
        inv_bot = 1.0 / locomp.shared_load(row_l, tid + 8)
        locomp.shared_store(Ds, 64 + tid * 8 + tid, inv_bot)
    locomp.barrier()
    d_final = locomp.simdgroup_matrix_load(Ds, sg_row * 64, 8)
    zero2 = locomp.simdgroup_matrix(0.0)
    acc_o0 = locomp.simdgroup_mac(zero2, d_final, acc_o0)
    acc_o1 = locomp.simdgroup_mac(zero2, d_final, acc_o1)
    acc_o2 = locomp.simdgroup_mac(zero2, d_final, acc_o2)
    acc_o3 = locomp.simdgroup_mac(zero2, d_final, acc_o3)

    # Store output
    c_row = bq * 16 + sg_row * 8
    c_col = sg_col * 32
    locomp.simdgroup_matrix_store_device(acc_o0, O + (head_off + c_row * 64 + c_col), 64)
    locomp.simdgroup_matrix_store_device(acc_o1, O + (head_off + c_row * 64 + c_col + 8), 64)
    locomp.simdgroup_matrix_store_device(acc_o2, O + (head_off + c_row * 64 + c_col + 16), 64)
    locomp.simdgroup_matrix_store_device(acc_o3, O + (head_off + c_row * 64 + c_col + 24), 64)


def gpu_causal_attention(Q_np, K_np, V_np, num_heads):
    """Run GPU causal attention. Q must NOT be pre-scaled (we do it here)."""
    H, N, D = Q_np.shape[0], Q_np.shape[1], Q_np.shape[2]
    scale = 1.0 / np.sqrt(D)
    Q_scaled = (Q_np * scale).astype(np.float32)

    Q_g = locomp.tensor(Q_scaled.reshape(-1))
    K_g = locomp.tensor(K_np.reshape(-1))
    V_g = locomp.tensor(V_np.reshape(-1))
    O_g = locomp.empty(H * N * D)

    nkv = N // Bc
    causal_flash_attn[(nkv, H), (32, 4)](Q_g, K_g, V_g, O_g, N, D, nkv, Br, Bc)
    return O_g.numpy().reshape(H, N, D)


def numpy_causal_attention(Q, K, V):
    """Reference causal attention."""
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


if __name__ == "__main__":
    import mlx.core as mx

    D = D_HEAD
    num_heads = 12
    sizes = [16, 32, 64, 128, 256]

    print(f"GPU Causal Flash Attention D={D}: Br={Br}, Bc={Bc}, TG=128 (4 SIMD groups)")
    print(f"{'N':>6} {'H':>4} | {'GPU':>8} | {'NumPy':>8} | {'MLX':>8} | GPU/NP | GPU/MLX | {'Error':>8}")
    print("-" * 78)

    for N in sizes:
        np.random.seed(42)
        Q_np = np.random.randn(num_heads, N, D).astype(np.float32) * 0.1
        K_np = np.random.randn(num_heads, N, D).astype(np.float32) * 0.1
        V_np = np.random.randn(num_heads, N, D).astype(np.float32) * 0.1

        # Correctness
        ref = numpy_causal_attention(Q_np, K_np, V_np)
        gpu_out = gpu_causal_attention(Q_np, K_np, V_np, num_heads)
        err = np.max(np.abs(gpu_out - ref))

        # GPU benchmark
        WARMUP = 5; RUNS = 15
        for _ in range(WARMUP):
            gpu_causal_attention(Q_np, K_np, V_np, num_heads)
        times_gpu = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_causal_attention(Q_np, K_np, V_np, num_heads)
            times_gpu.append((time.perf_counter() - t0) * 1000)
        t_gpu = sorted(times_gpu)[RUNS // 2]

        # NumPy benchmark
        for _ in range(WARMUP):
            numpy_causal_attention(Q_np, K_np, V_np)
        times_np = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            numpy_causal_attention(Q_np, K_np, V_np)
            times_np.append((time.perf_counter() - t0) * 1000)
        t_np = sorted(times_np)[RUNS // 2]

        # MLX benchmark (causal SDPA)
        mQ = mx.array(Q_np).reshape(1, num_heads, N, D)
        mK = mx.array(K_np).reshape(1, num_heads, N, D)
        mV = mx.array(V_np).reshape(1, num_heads, N, D)
        mx.eval(mQ, mK, mV)
        # Build causal mask: upper triangle = -inf
        causal = np.triu(np.full((N, N), -np.inf, dtype=np.float32), k=1)
        mask_mlx = mx.array(causal)
        for _ in range(WARMUP):
            out = mx.fast.scaled_dot_product_attention(mQ, mK, mV, scale=1.0/np.sqrt(D), mask=mask_mlx)
            mx.eval(out)
        times_mlx = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mx.fast.scaled_dot_product_attention(mQ, mK, mV, scale=1.0/np.sqrt(D), mask=mask_mlx)
            mx.eval(out)
            times_mlx.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(times_mlx)[RUNS // 2]

        r_np = t_gpu / t_np
        r_mlx = t_gpu / t_mlx
        print(f"{N:>6} {num_heads:>4} | {t_gpu:>6.3f}ms | {t_np:>6.3f}ms | {t_mlx:>6.3f}ms | {r_np:>5.2f}x | {r_mlx:>6.2f}x | {err:.2e}")
