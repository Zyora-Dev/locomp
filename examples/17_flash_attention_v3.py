"""
Flash Attention v3 — Optimized v2 with P-in-shared and eliminated prev_m.

Optimizations over v2:
  1. tc==0 stores P = exp(S-max) into Ss during sum loop (fused, no extra exp)
  2. V accumulation reads P directly — eliminates 15 exp() per thread per KV block
  3. prev_m eliminated: all threads read old_max from row_m BEFORE softmax writes
  4. Same barrier count as v2 (4 per KV block)

Net effect: V accum loop goes from (shared_load S + exp + 2 shared_load V + 2 FMA)
to (shared_load P + 2 shared_load V + 2 FMA) — removing exp saves ~300 cycles/block.
tc==0 adds only 16 shared_stores to the already-computed P values.
"""

import time
import numpy as np
import locomp

Br = 16
Bc = 16
D_HEAD = 32


@locomp.kernel
def flash_attn_v3(Q: locomp.Tensor, K: locomp.Tensor, V: locomp.Tensor,
                  O: locomp.Tensor,
                  N: locomp.constexpr, D: locomp.constexpr,
                  NUM_KV_BLOCKS: locomp.constexpr,
                  BLOCK_R: locomp.constexpr, BLOCK_C: locomp.constexpr):
    tc = locomp.local_id(0)   # 0..15
    tr = locomp.local_id(1)   # 0..15
    bq = locomp.program_id(0)

    Qs = locomp.shared_memory(Br * D_HEAD)    # [16, 32] = 512
    Ks = locomp.shared_memory(Bc * D_HEAD)    # [16, 32] = 512
    Vs = locomp.shared_memory(Bc * D_HEAD)    # [16, 32] = 512
    Ss = locomp.shared_memory(Br * Bc)        # [16, 16] = 256
    row_m = locomp.shared_memory(Br)          # 16
    row_l = locomp.shared_memory(Br)          # 16

    q_row = bq * BLOCK_R + tr

    # ── Load Q once ──
    locomp.shared_store(Qs, tr * D + tc, locomp.load(Q + (q_row * D + tc)))
    locomp.shared_store(Qs, tr * D + tc + 16, locomp.load(Q + (q_row * D + tc + 16)))

    if tc == 0:
        locomp.shared_store(row_m, tr, -1000000.0)
        locomp.shared_store(row_l, tr, 0.0)

    acc0 = 0.0
    acc1 = 0.0

    locomp.barrier()

    for bk in range(NUM_KV_BLOCKS):
        kv_row = bk * BLOCK_C + tr

        # ── Load K, V ──
        locomp.shared_store(Ks, tr * D + tc, locomp.load(K + (kv_row * D + tc)))
        locomp.shared_store(Ks, tr * D + tc + 16, locomp.load(K + (kv_row * D + tc + 16)))
        locomp.shared_store(Vs, tr * D + tc, locomp.load(V + (kv_row * D + tc)))
        locomp.shared_store(Vs, tr * D + tc + 16, locomp.load(V + (kv_row * D + tc + 16)))
        locomp.barrier()

        # ── S[tr, tc] = dot(Q[tr,:], K[tc,:]) / sqrt(d) ──
        dot = 0.0
        for k in range(D):
            dot = dot + locomp.shared_load(Qs, tr * D + k) * locomp.shared_load(Ks, tc * D + k)
        dot = dot * 0.176776695
        locomp.shared_store(Ss, tr * BLOCK_C + tc, dot)
        locomp.barrier()

        # ── ALL threads: snapshot old_max BEFORE tc==0 overwrites ──
        old_m = locomp.shared_load(row_m, tr)

        # ── tc==0: online softmax + store P into Ss ──
        if tc == 0:
            # Row max
            block_max = locomp.shared_load(Ss, tr * BLOCK_C + 0)
            for j in range(1, BLOCK_C):
                s_val = locomp.shared_load(Ss, tr * BLOCK_C + j)
                block_max = locomp.where(s_val > block_max, s_val, block_max)

            new_max = locomp.where(block_max > old_m, block_max, old_m)
            locomp.shared_store(row_m, tr, new_max)

            # Row sum AND store P = exp(S - max) into Ss
            old_sum = locomp.shared_load(row_l, tr)
            rescaled_sum = old_sum * locomp.exp(old_m - new_max)
            block_sum = 0.0
            for j in range(BLOCK_C):
                s_val = locomp.shared_load(Ss, tr * BLOCK_C + j)
                p_val = locomp.exp(s_val - new_max)
                block_sum = block_sum + p_val
                locomp.shared_store(Ss, tr * BLOCK_C + j, p_val)  # overwrite S with P
            locomp.shared_store(row_l, tr, rescaled_sum + block_sum)
        locomp.barrier()

        # ── ALL threads: rescale + V accumulation (P already in Ss!) ──
        new_m = locomp.shared_load(row_m, tr)
        scale = locomp.exp(old_m - new_m)
        acc0 = acc0 * scale
        acc1 = acc1 * scale

        for j in range(BLOCK_C):
            weight = locomp.shared_load(Ss, tr * BLOCK_C + j)  # P value directly!
            acc0 = acc0 + weight * locomp.shared_load(Vs, j * D + tc)
            acc1 = acc1 + weight * locomp.shared_load(Vs, j * D + tc + 16)

        locomp.barrier()

    # ── Final ──
    l_val = locomp.shared_load(row_l, tr)
    locomp.store(O + (q_row * D + tc), acc0 / l_val)
    locomp.store(O + (q_row * D + tc + 16), acc1 / l_val)


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

    print(f"Flash Attention v3 (P-in-Ss): d={d}, Br={Br}, Bc={Bc}, TG=(16,16)=256")
    print(f"{'N':>6} | {'v3':>8} | {'v2':>8} | {'v1':>8} | {'MLX':>8} | {'Torch':>8} | v3/MLX | v3/v2 | {'Error':>8}")
    print("-" * 88)

    from examples.flash_attention_v1 import flash_attention as flash_v1
    import importlib
    v2_mod = importlib.import_module("examples.16_flash_attention_v2")
    flash_attn_v2 = v2_mod.flash_attn_v2

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

        # ── v3 (P-in-Ss) ──
        O_t = locomp.empty(N * d)
        flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        result = O_t.numpy().reshape(N, d)
        err = np.max(np.abs(result - expected))

        WARMUP = 3; RUNS = 10
        for _ in range(WARMUP):
            flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        v3_t = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
            v3_t.append((time.perf_counter() - t0) * 1000)
        t_v3 = sorted(v3_t)[5]

        # ── v2 ──
        O_t2 = locomp.empty(N * d)
        for _ in range(WARMUP):
            flash_attn_v2[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t2, N, d, nkv, Br, Bc)
        v2_t = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            flash_attn_v2[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t2, N, d, nkv, Br, Bc)
            v2_t.append((time.perf_counter() - t0) * 1000)
        t_v2 = sorted(v2_t)[5]

        # ── v1 ──
        O_t3 = locomp.empty(N * d)
        for _ in range(WARMUP):
            flash_v1[(N // 16,), (D_HEAD, 16)](Q_t, K_t, V_t, O_t3, N, d, nkv, 16, Bc)
        v1_t = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            flash_v1[(N // 16,), (D_HEAD, 16)](Q_t, K_t, V_t, O_t3, N, d, nkv, 16, Bc)
            v1_t.append((time.perf_counter() - t0) * 1000)
        t_v1 = sorted(v1_t)[5]

        # ── MLX ──
        mQ = mx.array(Q_np); mK = mx.array(K_np); mV = mx.array(V_np)
        mx.eval(mQ, mK, mV)
        for _ in range(WARMUP):
            out = mx.softmax((mQ @ mK.T) * (1.0 / np.sqrt(d)), axis=-1) @ mV; mx.eval(out)
        mlx_t = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = mx.softmax((mQ @ mK.T) * (1.0 / np.sqrt(d)), axis=-1) @ mV; mx.eval(out)
            mlx_t.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mlx_t)[5]

        # ── Torch ──
        dev = torch.device("mps")
        tQ = torch.tensor(Q_np, device=dev).unsqueeze(0).unsqueeze(0)
        tK = torch.tensor(K_np, device=dev).unsqueeze(0).unsqueeze(0)
        tV = torch.tensor(V_np, device=dev).unsqueeze(0).unsqueeze(0)
        torch.mps.synchronize()
        for _ in range(WARMUP):
            torch.nn.functional.scaled_dot_product_attention(tQ, tK, tV); torch.mps.synchronize()
        torch_t = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            torch.nn.functional.scaled_dot_product_attention(tQ, tK, tV); torch.mps.synchronize()
            torch_t.append((time.perf_counter() - t0) * 1000)
        t_torch = sorted(torch_t)[5]

        ratio = t_v3 / t_mlx
        ratio_v2 = t_v3 / t_v2
        print(f"{N:>6} | {t_v3:>6.3f}ms | {t_v2:>6.3f}ms | {t_v1:>6.3f}ms | {t_mlx:>6.3f}ms | {t_torch:>6.3f}ms | {ratio:>5.2f}x | {ratio_v2:>5.2f}x | {err:.2e}")
