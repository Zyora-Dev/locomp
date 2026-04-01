"""
Example 36: GPT-2 with KV-Cache — fast autoregressive generation.

Two-phase inference:
  1. Prefill: process full prompt, cache K/V for all layers
  2. Decode: process 1 token at a time, reuse cached K/V

Includes a dedicated vector-matrix kernel for single-token decode, avoiding
the wasteful 64×64 tiled matmul when M=1.

Architecture: 12 layers, 12 heads, D=768, FFN=3072, vocab=50257
"""

import time
import numpy as np
import locomp

# =============================================================================
# Kernel definitions
# =============================================================================

BM, BN, BK = 64, 64, 8  # Tiled matmul sizes


@locomp.kernel
def matmul_tiled(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    sgid = locomp.simd_group_id()
    lane = locomp.simd_lane_id()
    tile_n = locomp.program_id(0)
    tile_m = locomp.program_id(1)
    sg_row = sgid // 4
    sg_col = sgid % 4
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
    c_r = tile_m * BM + sg_row * 16
    c_c = tile_n * BN + sg_col * 16
    locomp.simdgroup_matrix_store_device(acc00, C + (c_r * N + c_c), N)
    locomp.simdgroup_matrix_store_device(acc01, C + (c_r * N + (c_c + 8)), N)
    locomp.simdgroup_matrix_store_device(acc10, C + ((c_r + 8) * N + c_c), N)
    locomp.simdgroup_matrix_store_device(acc11, C + ((c_r + 8) * N + (c_c + 8)), N)


@locomp.kernel
def matvec(X: locomp.Tensor, W: locomp.Tensor, OUT: locomp.Tensor,
           N: locomp.constexpr, K: locomp.constexpr,
           K_TILE: locomp.constexpr, N_TILES: locomp.constexpr,
           X_LOADS: locomp.constexpr):
    group = locomp.program_id(0)
    tid = locomp.local_id(0)
    col = group * 256 + tid
    x_smem = locomp.shared_memory(K_TILE)
    acc = 0.0
    for t in range(N_TILES):
        k_base = t * K_TILE
        for i in range(X_LOADS):
            locomp.shared_store(x_smem, tid + i * 256, locomp.load(X + (k_base + tid + i * 256)))
        locomp.barrier()
        for k in range(K_TILE):
            acc = acc + locomp.shared_load(x_smem, k) * locomp.load(W + ((k_base + k) * N + col))
        locomp.barrier()
    locomp.store(OUT + col, acc)


@locomp.kernel
def add_bias(X: locomp.Tensor, Bias: locomp.Tensor,
             ROWS: locomp.constexpr, COLS: locomp.constexpr):
    row = locomp.program_id(0)
    col = locomp.local_id(0)
    for j in range(COLS // 256):
        idx = col + j * 256
        val = locomp.load(X + (row * COLS + idx)) + locomp.load(Bias + idx)
        locomp.store(X + (row * COLS + idx), val)


@locomp.kernel
def add_bias_vec(X: locomp.Tensor, Bias: locomp.Tensor,
                 COLS: locomp.constexpr):
    tid = locomp.local_id(0)
    for j in range(COLS // 256):
        idx = tid + j * 256
        locomp.store(X + idx, locomp.load(X + idx) + locomp.load(Bias + idx))


@locomp.kernel
def layer_norm(X: locomp.Tensor, W: locomp.Tensor, B: locomp.Tensor,
               OUT: locomp.Tensor,
               ROWS: locomp.constexpr, D: locomp.constexpr,
               THREADS: locomp.constexpr, NUM_SIMD: locomp.constexpr,
               ELEMS: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    simd_gid = locomp.simd_group_id()
    smem = locomp.shared_memory(NUM_SIMD)
    base = row * D
    local_sum = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        local_sum = local_sum + locomp.load(X + (base + idx))
    group_sum = locomp.simd_sum(local_sum)
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_sum)
    locomp.barrier()
    total = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        total = total + locomp.shared_load(smem, g)
    mean = total / D
    locomp.barrier()
    local_var = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        diff = locomp.load(X + (base + idx)) - mean
        local_var = local_var + diff * diff
    group_var = locomp.simd_sum(local_var)
    if lane == 0:
        locomp.shared_store(smem, simd_gid, group_var)
    locomp.barrier()
    var_total = locomp.shared_load(smem, 0)
    for g in range(1, NUM_SIMD):
        var_total = var_total + locomp.shared_load(smem, g)
    inv_std = locomp.rsqrt(var_total / D + 1e-5)
    locomp.barrier()
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (base + idx))
        w = locomp.load(W + idx)
        b = locomp.load(B + idx)
        locomp.store(OUT + (base + idx), (val - mean) * inv_std * w + b)


@locomp.kernel
def gelu_inplace(X: locomp.Tensor, N: locomp.constexpr):
    pid = locomp.program_id(0)
    tid = locomp.local_id(0)
    idx = pid * 256 + tid
    x = locomp.load(X + idx)
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    inner = locomp.clamp(inner, -10.0, 10.0)
    locomp.store(X + idx, 0.5 * x * (1.0 + locomp.tanh(inner)))


@locomp.kernel
def add_inplace(A: locomp.Tensor, B: locomp.Tensor, N: locomp.constexpr):
    pid = locomp.program_id(0)
    tid = locomp.local_id(0)
    idx = pid * 256 + tid
    locomp.store(A + idx, locomp.load(A + idx) + locomp.load(B + idx))


@locomp.kernel
def copy_kernel(Src: locomp.Tensor, Dst: locomp.Tensor, N: locomp.constexpr):
    pid = locomp.program_id(0)
    tid = locomp.local_id(0)
    idx = pid * 256 + tid
    locomp.store(Dst + idx, locomp.load(Src + idx))


# =============================================================================
# GPT-2 Model with KV-Cache
# =============================================================================

def pad64(n):
    return ((n + 63) // 64) * 64

def pad256(n):
    return ((n + 255) // 256) * 256


class GPT2KV:
    N_LAYERS = 12
    N_HEADS = 12
    D = 768
    D_FFN = 3072
    VOCAB = 50257
    MAX_SEQ = 1024
    HEAD_DIM = 64  # D // N_HEADS

    def __init__(self, weights: dict):
        self.w = {}
        for k, v in weights.items():
            self.w[k] = locomp.tensor(v)

        self.LN_THREADS = 256
        self.LN_NUM_SIMD = 8
        self.LN_ELEMS = 3  # 768 // 256

        # KV cache: stored as numpy, uploaded per-layer for attention
        # Shape per layer: [MAX_SEQ, D] for both K and V
        self.k_cache = [np.zeros((self.MAX_SEQ, self.D), dtype=np.float32)
                        for _ in range(self.N_LAYERS)]
        self.v_cache = [np.zeros((self.MAX_SEQ, self.D), dtype=np.float32)
                        for _ in range(self.N_LAYERS)]
        self.cache_len = 0  # Number of tokens currently in cache

        # Embedding tables (keep as numpy for embedding lookup)
        self.wte = weights["wte.weight"].reshape(self.VOCAB, self.D)
        self.wpe = weights["wpe.weight"].reshape(self.MAX_SEQ, self.D)

        # Pre-allocate GPU buffers for prefill (batch) and decode (single)
        self._alloc_prefill_buffers(64)
        self._alloc_decode_buffers()

    def _alloc_prefill_buffers(self, max_seq):
        M = pad64(max_seq)
        self.M_PAD = M
        D, FFN = self.D, self.D_FFN
        V_PAD = pad256(self.VOCAB)
        self.buf_ln = locomp.empty(M * D)
        self.buf_qkv = locomp.empty(M * 2304)
        self.buf_attn_out = locomp.empty(M * D)
        self.buf_proj = locomp.empty(M * D)
        self.buf_ffn_up = locomp.empty(M * 3072)
        self.buf_ffn_down = locomp.empty(M * D)
        self.buf_logits = locomp.empty(M * V_PAD)
        self.buf_hidden = locomp.empty(M * D)

    def _alloc_decode_buffers(self):
        D = self.D
        # Single-row buffers for decode
        self.dec_hidden = locomp.empty(D)
        self.dec_ln = locomp.empty(D)
        self.dec_qkv = locomp.empty(2304)
        self.dec_proj = locomp.empty(D)
        self.dec_ffn_up = locomp.empty(pad256(self.D_FFN))
        self.dec_ffn_down = locomp.empty(D)
        self.dec_logits = locomp.empty(pad256(self.VOCAB))

    # --- Batch ops (prefill) ---
    def _matmul_batch(self, A, B, C, M, N, K):
        matmul_tiled[(N // 64, M // 64), (32, 16)](A, B, C, M, N, K)

    def _add_bias_batch(self, X, bias, rows, cols):
        add_bias[(rows,), (256,)](X, bias, rows, cols)

    def _layer_norm_batch(self, X, W, B, OUT, rows):
        layer_norm[(rows,), (self.LN_THREADS,)](
            X, W, B, OUT, rows, self.D, self.LN_THREADS,
            self.LN_NUM_SIMD, self.LN_ELEMS)

    # --- Single-row ops (decode) ---
    def _matvec(self, X, W, OUT, N, K):
        K_TILE = 256
        N_TILES = K // K_TILE
        X_LOADS = K_TILE // 256  # 1
        matvec[(N // 256,), (256,)](X, W, OUT, N, K, K_TILE, N_TILES, X_LOADS)

    def _add_bias_vec(self, X, bias, cols):
        add_bias_vec[(1,), (256,)](X, bias, cols)

    def _layer_norm_one(self, X, W, B, OUT):
        layer_norm[(1,), (self.LN_THREADS,)](
            X, W, B, OUT, 1, self.D, self.LN_THREADS,
            self.LN_NUM_SIMD, self.LN_ELEMS)

    def _gelu_vec(self, X, n):
        n_pad = pad256(n)
        gelu_inplace[(n_pad // 256,), (256,)](X, n_pad)

    def _add_vec(self, A, B, n):
        n_pad = pad256(n)
        add_inplace[(n_pad // 256,), (256,)](A, B, n_pad)

    # --- Attention helpers ---
    def _attention_prefill(self, qkv_buf, layer, seq_len, M):
        D, H, HD = self.D, self.N_HEADS, self.HEAD_DIM

        qkv = qkv_buf.numpy()[:M * 2304].reshape(M, 2304)[:seq_len, :D * 3]
        q, k, v = np.split(qkv, 3, axis=-1)

        # Store in KV cache
        self.k_cache[layer][:seq_len] = k
        self.v_cache[layer][:seq_len] = v

        # Multi-head attention
        q = q.reshape(seq_len, H, HD).transpose(1, 0, 2)
        k = k.reshape(seq_len, H, HD).transpose(1, 0, 2)
        v = v.reshape(seq_len, H, HD).transpose(1, 0, 2)

        scale = 1.0 / np.sqrt(HD)
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * -1e9
        scores += mask[np.newaxis, :, :]
        sm = scores.max(axis=-1, keepdims=True)
        es = np.exp(scores - sm)
        aw = es / es.sum(axis=-1, keepdims=True)
        attn_out = np.matmul(aw, v).transpose(1, 0, 2).reshape(seq_len, D)
        return attn_out

    def _attention_decode(self, qkv_vec, layer, pos):
        D, H, HD = self.D, self.N_HEADS, self.HEAD_DIM
        cache_len = pos + 1  # includes current token

        qkv = qkv_vec[:D * 3]
        q, k, v = qkv[:D], qkv[D:D*2], qkv[D*2:D*3]

        # Store new K/V in cache
        self.k_cache[layer][pos] = k
        self.v_cache[layer][pos] = v

        # Query against all cached keys
        q_heads = q.reshape(H, HD)
        k_cached = self.k_cache[layer][:cache_len].reshape(cache_len, H, HD).transpose(1, 0, 2)
        v_cached = self.v_cache[layer][:cache_len].reshape(cache_len, H, HD).transpose(1, 0, 2)

        scale = 1.0 / np.sqrt(HD)
        # scores: [H, 1, cache_len]
        scores = np.einsum('hd,hsd->hs', q_heads, k_cached) * scale
        # No causal mask needed — we only have past + current tokens
        sm = scores.max(axis=-1, keepdims=True)
        es = np.exp(scores - sm)
        aw = es / es.sum(axis=-1, keepdims=True)
        # aw: [H, cache_len], v_cached: [H, cache_len, HD]
        attn_out = np.einsum('hs,hsd->hd', aw, v_cached)
        return attn_out.reshape(D)

    # === Prefill: process full prompt ===
    def prefill(self, token_ids: np.ndarray) -> np.ndarray:
        seq_len = len(token_ids)
        M = pad64(seq_len)
        if M > self.M_PAD:
            self._alloc_prefill_buffers(M)
        D = self.D

        # Embedding
        hidden = np.zeros((M, D), dtype=np.float32)
        for i, tid in enumerate(token_ids):
            hidden[i] = self.wte[tid] + self.wpe[i]
        h_gpu = locomp.tensor(hidden.flatten())
        n_pad = pad256(M * D)
        copy_kernel[(n_pad // 256,), (256,)](h_gpu, self.buf_hidden, n_pad)

        for layer in range(self.N_LAYERS):
            pfx = f"h.{layer}."

            # LN1 → QKV
            self._layer_norm_batch(self.buf_hidden, self.w[pfx + "ln_1.weight"],
                                   self.w[pfx + "ln_1.bias"], self.buf_ln, M)
            self._matmul_batch(self.buf_ln, self.w[pfx + "attn.c_attn.weight"],
                               self.buf_qkv, M, 2304, D)
            self._add_bias_batch(self.buf_qkv, self.w[pfx + "attn.c_attn.bias"], M, 2304)

            # Attention + cache
            attn_out = self._attention_prefill(self.buf_qkv, layer, seq_len, M)

            # Upload + out proj + residual
            ap = np.zeros((M, D), dtype=np.float32)
            ap[:seq_len] = attn_out
            at = locomp.tensor(ap.flatten())
            copy_kernel[(n_pad // 256,), (256,)](at, self.buf_attn_out, n_pad)

            self._matmul_batch(self.buf_attn_out, self.w[pfx + "attn.c_proj.weight"],
                               self.buf_proj, M, D, D)
            self._add_bias_batch(self.buf_proj, self.w[pfx + "attn.c_proj.bias"], M, D)
            add_inplace[(n_pad // 256,), (256,)](self.buf_hidden, self.buf_proj, n_pad)

            # LN2 → FFN
            self._layer_norm_batch(self.buf_hidden, self.w[pfx + "ln_2.weight"],
                                   self.w[pfx + "ln_2.bias"], self.buf_ln, M)
            self._matmul_batch(self.buf_ln, self.w[pfx + "mlp.c_fc.weight"],
                               self.buf_ffn_up, M, 3072, D)
            self._add_bias_batch(self.buf_ffn_up, self.w[pfx + "mlp.c_fc.bias"], M, 3072)
            gp = pad256(M * 3072)
            gelu_inplace[(gp // 256,), (256,)](self.buf_ffn_up, gp)
            self._matmul_batch(self.buf_ffn_up, self.w[pfx + "mlp.c_proj.weight"],
                               self.buf_ffn_down, M, D, 3072)
            self._add_bias_batch(self.buf_ffn_down, self.w[pfx + "mlp.c_proj.bias"], M, D)
            add_inplace[(n_pad // 256,), (256,)](self.buf_hidden, self.buf_ffn_down, n_pad)

        # Final LN + logits
        self._layer_norm_batch(self.buf_hidden, self.w["ln_f.weight"],
                               self.w["ln_f.bias"], self.buf_ln, M)
        V_PAD = pad256(self.VOCAB)
        self._matmul_batch(self.buf_ln, self.w["wte_t"], self.buf_logits, M, V_PAD, D)

        self.cache_len = seq_len
        logits = self.buf_logits.numpy()
        return logits[(seq_len - 1) * V_PAD: seq_len * V_PAD][:self.VOCAB]

    # === Decode: process single new token ===
    def decode(self, token_id: int) -> np.ndarray:
        pos = self.cache_len
        D = self.D

        # Embedding: single token
        hidden_np = (self.wte[token_id] + self.wpe[pos]).astype(np.float32)
        self.dec_hidden = locomp.tensor(hidden_np)

        for layer in range(self.N_LAYERS):
            pfx = f"h.{layer}."

            # LN1
            self._layer_norm_one(self.dec_hidden, self.w[pfx + "ln_1.weight"],
                                 self.w[pfx + "ln_1.bias"], self.dec_ln)

            # QKV: [1, 768] × [768, 2304] → [1, 2304]
            self._matvec(self.dec_ln, self.w[pfx + "attn.c_attn.weight"],
                         self.dec_qkv, 2304, D)
            self._add_bias_vec(self.dec_qkv, self.w[pfx + "attn.c_attn.bias"], 2304)

            # Attention with KV cache
            qkv_np = self.dec_qkv.numpy()[:D * 3]
            attn_out = self._attention_decode(qkv_np, layer, pos)
            self.dec_proj = locomp.tensor(attn_out.astype(np.float32))

            # Out projection: [1, 768] × [768, 768]
            dec_attn_proj = locomp.empty(D)
            self._matvec(self.dec_proj, self.w[pfx + "attn.c_proj.weight"],
                         dec_attn_proj, D, D)
            self._add_bias_vec(dec_attn_proj, self.w[pfx + "attn.c_proj.bias"], D)

            # Residual
            self._add_vec(self.dec_hidden, dec_attn_proj, D)

            # LN2
            self._layer_norm_one(self.dec_hidden, self.w[pfx + "ln_2.weight"],
                                 self.w[pfx + "ln_2.bias"], self.dec_ln)

            # FFN up: [1, 768] × [768, 3072]
            self._matvec(self.dec_ln, self.w[pfx + "mlp.c_fc.weight"],
                         self.dec_ffn_up, 3072, D)
            self._add_bias_vec(self.dec_ffn_up, self.w[pfx + "mlp.c_fc.bias"], 3072)

            # GELU
            self._gelu_vec(self.dec_ffn_up, 3072)

            # FFN down: [1, 3072] × [3072, 768]
            self._matvec(self.dec_ffn_up, self.w[pfx + "mlp.c_proj.weight"],
                         self.dec_ffn_down, D, 3072)
            self._add_bias_vec(self.dec_ffn_down, self.w[pfx + "mlp.c_proj.bias"], D)

            # Residual
            self._add_vec(self.dec_hidden, self.dec_ffn_down, D)

        # Final LN
        self._layer_norm_one(self.dec_hidden, self.w["ln_f.weight"],
                             self.w["ln_f.bias"], self.dec_ln)

        # Logits: [1, 768] × [768, V_PAD256] — pad256 for matvec grid
        V_PAD = pad256(self.VOCAB)
        self._matvec(self.dec_ln, self.w["wte_t"], self.dec_logits, V_PAD, D)

        self.cache_len = pos + 1
        return self.dec_logits.numpy()[:self.VOCAB]

    def reset_cache(self):
        for layer in range(self.N_LAYERS):
            self.k_cache[layer][:] = 0
            self.v_cache[layer][:] = 0
        self.cache_len = 0


# =============================================================================
# Weight loading & sampling
# =============================================================================

def load_gpt2_weights() -> dict:
    print("Loading GPT-2 weights from HuggingFace...")
    import torch
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = model.state_dict()

    weights = {}
    for k, v in sd.items():
        name = k.replace("transformer.", "")
        weights[name] = v.detach().float().numpy().flatten().astype(np.float32)

    # Transposed embedding for logits — pad to multiple of 256 for matvec
    wte = sd["transformer.wte.weight"].detach().float().numpy()
    V_PAD = pad256(50257)  # 50432
    wte_t = np.zeros((768, V_PAD), dtype=np.float32)
    wte_t[:, :50257] = wte.T
    weights["wte_t"] = wte_t.flatten().astype(np.float32)

    del model, sd
    print(f"  Loaded {len(weights)} weight tensors")
    return weights


def sample_token(logits, temperature=0.8, top_k=40):
    logits = logits / temperature
    top_k_indices = np.argsort(logits)[-top_k:]
    top_k_logits = logits[top_k_indices]
    top_k_logits -= top_k_logits.max()
    probs = np.exp(top_k_logits)
    probs /= probs.sum()
    idx = np.random.choice(len(probs), p=probs)
    return top_k_indices[idx]


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    weights = load_gpt2_weights()
    model = GPT2KV(weights)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # --- Correctness: prefill logits vs HuggingFace ---
    print("\n=== Correctness Check: Prefill ===")
    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'  tokens: {input_ids}")

    t0 = time.perf_counter()
    logits = model.prefill(np.array(input_ids, dtype=np.int32))
    t_prefill = (time.perf_counter() - t0) * 1000

    top5 = np.argsort(logits)[-5:][::-1]
    print(f"Prefill: {t_prefill:.1f}ms")
    print(f"Top-5: {[tokenizer.decode([t]) for t in top5]}")

    import torch
    from transformers import GPT2LMHeadModel
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(torch.tensor([input_ids])).logits[0, -1].numpy()
    hf_top5 = np.argsort(hf_logits)[-5:][::-1]
    print(f"HF-5:  {[tokenizer.decode([t]) for t in hf_top5]}")
    corr = np.corrcoef(logits, hf_logits)[0, 1]
    print(f"Correlation: {corr:.6f}")

    # --- Correctness: decode single token vs HuggingFace ---
    print("\n=== Correctness Check: Decode (1 token) ===")
    next_id = int(np.argmax(logits))  # greedy
    print(f"Decoding token: {next_id} ('{tokenizer.decode([next_id])}')")

    t0 = time.perf_counter()
    dec_logits = model.decode(next_id)
    t_decode = (time.perf_counter() - t0) * 1000

    dec_top5 = np.argsort(dec_logits)[-5:][::-1]
    print(f"Decode: {t_decode:.1f}ms")
    print(f"Top-5: {[tokenizer.decode([t]) for t in dec_top5]}")

    with torch.no_grad():
        hf_dec = hf_model(torch.tensor([input_ids + [next_id]])).logits[0, -1].numpy()
    hf_dec_top5 = np.argsort(hf_dec)[-5:][::-1]
    print(f"HF-5:  {[tokenizer.decode([t]) for t in hf_dec_top5]}")
    dec_corr = np.corrcoef(dec_logits, hf_dec)[0, 1]
    print(f"Correlation: {dec_corr:.6f}")

    # --- Benchmark: generation with KV cache ---
    print("\n=== Generation Benchmark (KV-Cache) ===")
    model.reset_cache()
    prompt = "In a shocking finding, scientists discovered a"
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'")

    np.random.seed(42)

    t_start = time.perf_counter()
    logits = model.prefill(np.array(input_ids, dtype=np.int32))
    t_prefill = time.perf_counter() - t_start
    gen_tokens = list(input_ids)

    n_generate = 50
    decode_times = []
    for i in range(n_generate):
        next_token = sample_token(logits, temperature=0.8, top_k=40)
        gen_tokens.append(int(next_token))
        t0 = time.perf_counter()
        logits = model.decode(int(next_token))
        decode_times.append(time.perf_counter() - t0)

    t_total = time.perf_counter() - t_start
    avg_decode = np.mean(decode_times) * 1000
    tok_per_sec = n_generate / sum(decode_times)

    text = tokenizer.decode(gen_tokens)
    print(f"\nGenerated ({n_generate} tokens):")
    print(f"  {text}")
    print(f"\nPrefill ({len(input_ids)} tokens): {t_prefill*1000:.1f}ms")
    print(f"Avg decode: {avg_decode:.1f}ms/token")
    print(f"Decode throughput: {tok_per_sec:.1f} tokens/sec")
    print(f"Total: {t_total:.2f}s")

    # --- Compare: without KV cache (full recompute each step) ---
    print("\n=== Comparison: Without KV-Cache ===")
    model_nc = GPT2KV(weights)
    np.random.seed(42)
    gen_tokens_nc = list(input_ids)

    t_start_nc = time.perf_counter()
    for i in range(n_generate):
        model_nc.reset_cache()
        logits_nc = model_nc.prefill(np.array(gen_tokens_nc, dtype=np.int32))
        next_token = sample_token(logits_nc, temperature=0.8, top_k=40)
        gen_tokens_nc.append(int(next_token))
    t_total_nc = time.perf_counter() - t_start_nc
    tok_per_sec_nc = n_generate / t_total_nc

    print(f"No-cache: {tok_per_sec_nc:.1f} tok/s ({t_total_nc:.2f}s)")
    print(f"KV-cache: {tok_per_sec:.1f} tok/s ({sum(decode_times):.2f}s decode)")
    speedup = tok_per_sec / tok_per_sec_nc
    print(f"Speedup:  {speedup:.1f}x")
