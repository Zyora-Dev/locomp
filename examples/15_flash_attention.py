"""
Flash Attention — Tiled, fused QKV attention with online softmax.

Algorithm (Dao et al., 2022):
  For each Q block (Br rows):
    For each K/V block (Bc cols):
      1. S[Br, Bc] = Q_block @ K_block^T / sqrt(d)
      2. Online softmax: m_new, l_new, rescale O
      3. O += softmax_weights @ V_block
    Final: O = O / l

Layout:
  - Threadgroup: (d, Br) = (32, 16) = 512 threads
  - Each thread (tr, tc) where tr = row in Q block, tc = column in output
  - tr iterates over Bc K-columns for dot products
  - Reductions over Bc use shared memory

Parameters: N=seq_len, d=head_dim=32, Br=Bc=16
Shared memory: Ks[Bc*d]=512, Vs[Bc*d]=512, Ss[Br*Bc]=256, smem_reduce[Br*Bc]=256
Total: 1536 floats * 4 = 6144 bytes
"""

import time
import numpy as np
import locomp

Br = 16
Bc = 16
D_HEAD = 32


@locomp.kernel
def flash_attention(Q: locomp.Tensor, K: locomp.Tensor, V: locomp.Tensor,
                    O: locomp.Tensor,
                    N: locomp.constexpr, D: locomp.constexpr,
                    NUM_KV_BLOCKS: locomp.constexpr,
                    BLOCK_R: locomp.constexpr, BLOCK_C: locomp.constexpr):
    tc = locomp.local_id(0)   # output column (0..D-1 = 0..31)
    tr = locomp.local_id(1)   # row within Q block (0..Br-1 = 0..15)
    bq = locomp.program_id(0) # which Q block

    Ks = locomp.shared_memory(Bc * D_HEAD)    # [16, 32]
    Vs = locomp.shared_memory(Bc * D_HEAD)    # [16, 32]
    Ss = locomp.shared_memory(Br * Bc)        # [16, 16] attention scores
    row_m = locomp.shared_memory(Br)          # row max for reduction
    row_l = locomp.shared_memory(Br)          # row sum for reduction
    old_m = locomp.shared_memory(Br)          # previous max for rescaling

    q_row = bq * BLOCK_R + tr

    # Load Q[q_row, tc] into register (reused across all KV blocks)
    q_val = locomp.load(Q + (q_row * D + tc))

    # Initialize per-row running max and sum
    if tc == 0:
        locomp.shared_store(row_m, tr, -1000000.0)
        locomp.shared_store(row_l, tr, 0.0)
    locomp.barrier()

    # Output accumulator — one per thread (for output column tc)
    acc = 0.0

    for bk in range(NUM_KV_BLOCKS):
        # ── Load K tile [Bc, D] ──
        # 512 threads, 512 elements → 1 element per thread
        kv_row = bk * BLOCK_C + tr
        k_val = locomp.load(K + (kv_row * D + tc))
        locomp.shared_store(Ks, tr * D + tc, k_val)

        # ── Load V tile [Bc, D] ──
        v_val = locomp.load(V + (kv_row * D + tc))
        locomp.shared_store(Vs, tr * D + tc, v_val)
        locomp.barrier()

        # ── Compute S[tr, j] for j = tc (only first Bc=16 threads per row) ──
        # S[tr, j] = sum_k Q[q_row, k] * K[bk*Bc+j, k]
        if tc < BLOCK_C:
            dot = 0.0
            for k in range(D):
                q_k = locomp.load(Q + (q_row * D + k))
                k_k = locomp.shared_load(Ks, tc * D + k)
                dot = dot + q_k * k_k
            dot = dot * 0.176776695
            locomp.shared_store(Ss, tr * BLOCK_C + tc, dot)
        locomp.barrier()

        # ── Online softmax reduction (thread tc==0 per row) ──
        if tc == 0:
            # Save old max
            prev_max = locomp.shared_load(row_m, tr)
            locomp.shared_store(old_m, tr, prev_max)

            # Row max of S[tr, :]
            block_max = locomp.shared_load(Ss, tr * BLOCK_C + 0)
            for j in range(1, BLOCK_C):
                s_val = locomp.shared_load(Ss, tr * BLOCK_C + j)
                block_max = locomp.where(s_val > block_max, s_val, block_max)

            new_max = locomp.where(block_max > prev_max, block_max, prev_max)
            locomp.shared_store(row_m, tr, new_max)

            # Rescale old sum and add new block sum
            prev_sum = locomp.shared_load(row_l, tr)
            rescaled_sum = prev_sum * locomp.exp(prev_max - new_max)
            block_sum = 0.0
            for j in range(BLOCK_C):
                s_val = locomp.shared_load(Ss, tr * BLOCK_C + j)
                block_sum = block_sum + locomp.exp(s_val - new_max)
            locomp.shared_store(row_l, tr, rescaled_sum + block_sum)
        locomp.barrier()

        # ── Rescale accumulator and add weighted V ──
        # All D threads participate here
        prev_m = locomp.shared_load(old_m, tr)
        curr_m = locomp.shared_load(row_m, tr)
        scale = locomp.exp(prev_m - curr_m)
        acc = acc * scale

        # acc += sum_j exp(S[tr,j] - curr_m) * V[j, tc]
        for j in range(BLOCK_C):
            s_val = locomp.shared_load(Ss, tr * BLOCK_C + j)
            weight = locomp.exp(s_val - curr_m)
            v_shared = locomp.shared_load(Vs, j * D + tc)
            acc = acc + weight * v_shared

        locomp.barrier()

    # ── Final normalization: O = acc / l ──
    l_val = locomp.shared_load(row_l, tr)
    out_val = acc / l_val
    locomp.store(O + (q_row * D + tc), out_val)


def naive_attention_np(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    from scipy.special import softmax
    attn = softmax(scores, axis=-1)
    return attn @ V


if __name__ == "__main__":
    import mlx.core as mx
    import torch

    d = D_HEAD
    sizes = [64, 128, 256, 512]

    print(f"Flash Attention Benchmark: d={d}, Br={Br}, Bc={Bc}")
    print(f"{'N':>6} | {'Locust':>8} | {'MLX':>8} | {'Torch':>8} | {'Ratio':>6} | {'Error':>8}")
    print("-" * 60)

    for N in sizes:
        np.random.seed(42)
        Q_np = np.random.randn(N, d).astype(np.float32) * 0.1
        K_np = np.random.randn(N, d).astype(np.float32) * 0.1
        V_np = np.random.randn(N, d).astype(np.float32) * 0.1
        expected = naive_attention_np(Q_np, K_np, V_np)

        # Locust Flash Attention
        Q_t = locomp.tensor(Q_np.flatten())
        K_t = locomp.tensor(K_np.flatten())
        V_t = locomp.tensor(V_np.flatten())
        O_t = locomp.empty(N * d)
        nkv = N // Bc
        grid = (N // Br,)
        tg = (D_HEAD, Br)

        flash_attention[grid, tg](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        result = O_t.numpy().reshape(N, d)
        err = np.max(np.abs(result - expected))

        WARMUP = 3; RUNS = 10
        for _ in range(WARMUP):
            flash_attention[grid, tg](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        loc_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            flash_attention[grid, tg](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
            loc_times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(loc_times)[5]

        # MLX naive attention (no flash attn in MLX python API)
        mQ = mx.array(Q_np); mK = mx.array(K_np); mV = mx.array(V_np)
        mx.eval(mQ, mK, mV)
        for _ in range(WARMUP):
            scores = (mQ @ mK.T) * (1.0 / np.sqrt(d))
            attn = mx.softmax(scores, axis=-1)
            out = attn @ mV
            mx.eval(out)
        mlx_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            scores = (mQ @ mK.T) * (1.0 / np.sqrt(d))
            attn = mx.softmax(scores, axis=-1)
            out = attn @ mV
            mx.eval(out)
            mlx_times.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mlx_times)[5]

        # PyTorch MPS scaled_dot_product_attention
        dev = torch.device("mps")
        tQ = torch.tensor(Q_np, device=dev).unsqueeze(0).unsqueeze(0)  # [1,1,N,d]
        tK = torch.tensor(K_np, device=dev).unsqueeze(0).unsqueeze(0)
        tV = torch.tensor(V_np, device=dev).unsqueeze(0).unsqueeze(0)
        torch.mps.synchronize()
        for _ in range(WARMUP):
            torch.nn.functional.scaled_dot_product_attention(tQ, tK, tV)
            torch.mps.synchronize()
        torch_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            torch.nn.functional.scaled_dot_product_attention(tQ, tK, tV)
            torch.mps.synchronize()
            torch_times.append((time.perf_counter() - t0) * 1000)
        t_torch = sorted(torch_times)[5]

        ratio = t_loc / t_mlx
        print(f"{N:>6} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {t_torch:>6.3f}ms | {ratio:>5.2f}x | {err:.2e}")
