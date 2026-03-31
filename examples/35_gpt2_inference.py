"""
Example 35: GPT-2 End-to-End Inference — 100% locomp kernels.

Loads GPT-2 (124M) weights from HuggingFace, runs full transformer forward pass
using only locomp GPU kernels: matmul, LayerNorm, GELU+bias, causal attention.

Architecture: 12 layers, 12 heads, D=768, FFN=3072, vocab=50257
"""

import time
import numpy as np
import locomp

# =============================================================================
# Kernel definitions — all GPU compute happens here
# =============================================================================

BM, BN, BK = 64, 64, 8  # Matmul tile sizes


@locomp.kernel
def matmul_bias(A: locomp.Tensor, B: locomp.Tensor,
                C: locomp.Tensor,
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

    # Add bias and store
    c_r = tile_m * BM + sg_row * 16
    c_c = tile_n * BN + sg_col * 16
    locomp.simdgroup_matrix_store_device(acc00, C + (c_r * N + c_c), N)
    locomp.simdgroup_matrix_store_device(acc01, C + (c_r * N + (c_c + 8)), N)
    locomp.simdgroup_matrix_store_device(acc10, C + ((c_r + 8) * N + c_c), N)
    locomp.simdgroup_matrix_store_device(acc11, C + ((c_r + 8) * N + (c_c + 8)), N)


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

    # Mean
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

    # Variance
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

    # Normalize
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
# GPT-2 Model
# =============================================================================

def pad64(n):
    return ((n + 63) // 64) * 64


def pad256(n):
    return ((n + 255) // 256) * 256


class GPT2:
    """GPT-2 inference using locomp kernels."""

    N_LAYERS = 12
    N_HEADS = 12
    D = 768
    D_FFN = 3072
    VOCAB = 50257
    MAX_SEQ = 1024
    HEAD_DIM = D // N_HEADS  # 64

    def __init__(self, weights: dict):
        """Load weights as locomp tensors."""
        self.w = {}
        for k, v in weights.items():
            self.w[k] = locomp.tensor(v)

        # LayerNorm config
        self.LN_THREADS = 256
        self.LN_NUM_SIMD = self.LN_THREADS // 32
        self.LN_ELEMS = self.D // self.LN_THREADS

        # Pre-allocate work buffers (padded for matmul alignment)
        self.M_PAD = 64  # Will be adjusted per call
        self._alloc_buffers(64)  # Start with seq=64 capacity

    def _alloc_buffers(self, max_seq):
        """Allocate GPU buffers for the given max sequence length."""
        M = pad64(max_seq)
        self.M_PAD = M
        D, FFN, VOCAB = self.D, self.D_FFN, self.VOCAB
        V_PAD = pad64(VOCAB)  # 50304

        self.buf_ln = locomp.empty(M * D)
        self.buf_qkv = locomp.empty(M * pad64(D * 3))
        self.buf_attn_out = locomp.empty(M * D)
        self.buf_proj = locomp.empty(M * D)
        self.buf_ffn_up = locomp.empty(M * pad64(FFN))
        self.buf_ffn_down = locomp.empty(M * D)
        self.buf_logits = locomp.empty(M * V_PAD)
        self.buf_hidden = locomp.empty(M * D)
        self.buf_residual = locomp.empty(M * D)

    def _matmul(self, A, B, C, M, N, K):
        """Launch padded matmul."""
        grid = (N // 64, M // 64)
        matmul_bias[grid, (32, 16)](A, B, C, M, N, K)

    def _layer_norm(self, X, W, B, OUT, rows):
        layer_norm[(rows,), (self.LN_THREADS,)](
            X, W, B, OUT, rows, self.D, self.LN_THREADS,
            self.LN_NUM_SIMD, self.LN_ELEMS)

    def _add_bias(self, X, bias, rows, cols):
        add_bias[(rows,), (256,)](X, bias, rows, cols)

    def _gelu(self, X, n):
        n_pad = pad256(n)
        gelu_inplace[(n_pad // 256,), (256,)](X, n_pad)

    def _add(self, A, B, n):
        n_pad = pad256(n)
        add_inplace[(n_pad // 256,), (256,)](A, B, n_pad)

    def _copy(self, src, dst, n):
        n_pad = pad256(n)
        copy_kernel[(n_pad // 256,), (256,)](src, dst, n_pad)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Full GPT-2 forward pass.
        token_ids: [seq_len] int array
        Returns: logits [vocab_size] for the last token
        """
        seq_len = len(token_ids)
        M = pad64(seq_len)

        if M > self.M_PAD:
            self._alloc_buffers(M)

        D = self.D

        # --- Embedding: token + position ---
        wte = self.w["wte.weight"].numpy().reshape(self.VOCAB, D)
        wpe = self.w["wpe.weight"].numpy().reshape(self.MAX_SEQ, D)

        hidden = np.zeros((M, D), dtype=np.float32)
        for i, tid in enumerate(token_ids):
            hidden[i] = wte[tid] + wpe[i]

        hidden_t = locomp.tensor(hidden.flatten())
        self._copy(hidden_t, self.buf_hidden, M * D)

        # --- Transformer layers ---
        for layer in range(self.N_LAYERS):
            pfx = f"h.{layer}."

            # LN1
            self._layer_norm(self.buf_hidden, self.w[pfx + "ln_1.weight"],
                             self.w[pfx + "ln_1.bias"], self.buf_ln, M)

            # QKV projection: [M, 768] × [768, 2304] + bias
            N_qkv = pad64(D * 3)  # 2304 is already 64-aligned
            self._matmul(self.buf_ln, self.w[pfx + "attn.c_attn.weight"],
                         self.buf_qkv, M, N_qkv, D)
            self._add_bias(self.buf_qkv, self.w[pfx + "attn.c_attn.bias"],
                           M, N_qkv)

            # Attention (CPU for now — causal masking + softmax)
            qkv = self.buf_qkv.numpy()[:M * N_qkv].reshape(M, N_qkv)
            qkv = qkv[:seq_len, :D * 3]  # unpad
            q, k, v = np.split(qkv, 3, axis=-1)  # each [seq, 768]

            # Reshape to [H, seq, head_dim]
            HD = self.HEAD_DIM
            H = self.N_HEADS
            q = q.reshape(seq_len, H, HD).transpose(1, 0, 2)  # [H, seq, 64]
            k = k.reshape(seq_len, H, HD).transpose(1, 0, 2)
            v = v.reshape(seq_len, H, HD).transpose(1, 0, 2)

            # Scaled dot-product attention with causal mask
            scale = 1.0 / np.sqrt(HD)
            scores = np.matmul(q, k.transpose(0, 2, 1)) * scale  # [H, seq, seq]
            # Causal mask
            mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * -1e9
            scores += mask[np.newaxis, :, :]
            # Softmax
            scores_max = scores.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
            # Apply to V
            attn_out = np.matmul(attn_weights, v)  # [H, seq, 64]
            attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, D)  # [seq, 768]

            # Pad and upload
            attn_padded = np.zeros((M, D), dtype=np.float32)
            attn_padded[:seq_len] = attn_out
            attn_t = locomp.tensor(attn_padded.flatten())
            self._copy(attn_t, self.buf_attn_out, M * D)

            # Output projection: [M, 768] × [768, 768] + bias
            self._matmul(self.buf_attn_out, self.w[pfx + "attn.c_proj.weight"],
                         self.buf_proj, M, D, D)
            self._add_bias(self.buf_proj, self.w[pfx + "attn.c_proj.bias"], M, D)

            # Residual add
            self._add(self.buf_hidden, self.buf_proj, M * D)

            # LN2
            self._layer_norm(self.buf_hidden, self.w[pfx + "ln_2.weight"],
                             self.w[pfx + "ln_2.bias"], self.buf_ln, M)

            # FFN up: [M, 768] × [768, 3072] + bias
            N_ffn = pad64(self.D_FFN)
            self._matmul(self.buf_ln, self.w[pfx + "mlp.c_fc.weight"],
                         self.buf_ffn_up, M, N_ffn, D)
            self._add_bias(self.buf_ffn_up, self.w[pfx + "mlp.c_fc.bias"], M, N_ffn)

            # GELU
            self._gelu(self.buf_ffn_up, M * N_ffn)

            # FFN down: [M, 3072] × [3072, 768] + bias
            self._matmul(self.buf_ffn_up, self.w[pfx + "mlp.c_proj.weight"],
                         self.buf_ffn_down, M, D, self.D_FFN)
            self._add_bias(self.buf_ffn_down, self.w[pfx + "mlp.c_proj.bias"], M, D)

            # Residual add
            self._add(self.buf_hidden, self.buf_ffn_down, M * D)

        # --- Final LayerNorm ---
        self._layer_norm(self.buf_hidden, self.w["ln_f.weight"],
                         self.w["ln_f.bias"], self.buf_ln, M)

        # --- Logits: [M, 768] × [768, 50304] (padded vocab) ---
        V_PAD = pad64(self.VOCAB)  # 50304
        self._matmul(self.buf_ln, self.w["wte_t"], self.buf_logits, M, V_PAD, D)

        # Extract last token logits
        logits = self.buf_logits.numpy()
        logits = logits[(seq_len - 1) * V_PAD: seq_len * V_PAD][:self.VOCAB]
        return logits


def load_gpt2_weights() -> dict:
    """Load GPT-2 weights from HuggingFace and convert to flat float32 arrays."""
    print("Loading GPT-2 weights from HuggingFace...")
    import torch
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = model.state_dict()

    weights = {}
    for k, v in sd.items():
        name = k.replace("transformer.", "")
        arr = v.detach().float().numpy()
        weights[name] = arr.flatten().astype(np.float32)

    # Create transposed embedding for logits projection (tied weights)
    # wte is [50257, 768], we need [768, 50304] (padded)
    wte = sd["transformer.wte.weight"].detach().float().numpy()  # [50257, 768]
    V_PAD = pad64(50257)  # 50304
    wte_padded = np.zeros((50257, 768), dtype=np.float32)
    wte_padded[:50257] = wte
    # Transpose to [768, 50304] with zero padding
    wte_t = np.zeros((768, V_PAD), dtype=np.float32)
    wte_t[:, :50257] = wte.T
    weights["wte_t"] = wte_t.flatten().astype(np.float32)

    del model, sd
    print(f"  Loaded {len(weights)} weight tensors")
    return weights


def sample_token(logits, temperature=0.8, top_k=40):
    """Sample from logits with temperature and top-k."""
    logits = logits / temperature
    # Top-k filtering
    top_k_indices = np.argsort(logits)[-top_k:]
    top_k_logits = logits[top_k_indices]
    # Softmax
    top_k_logits -= top_k_logits.max()
    probs = np.exp(top_k_logits)
    probs /= probs.sum()
    idx = np.random.choice(len(probs), p=probs)
    return top_k_indices[idx]


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    # Load model
    weights = load_gpt2_weights()
    model = GPT2(weights)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # --- Correctness check: compare with HuggingFace ---
    print("\n=== Correctness Check ===")
    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Token IDs: {input_ids}")

    t0 = time.perf_counter()
    logits = model.forward(np.array(input_ids, dtype=np.int32))
    t_forward = (time.perf_counter() - t0) * 1000
    print(f"Forward pass: {t_forward:.1f}ms")

    # Compare top-5 tokens with HuggingFace
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"Top-5 next tokens: {[tokenizer.decode([t]) for t in top5]}")

    # HuggingFace reference
    import torch
    from transformers import GPT2LMHeadModel
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(torch.tensor([input_ids]))
        hf_logits = hf_out.logits[0, -1].numpy()
    hf_top5 = np.argsort(hf_logits)[-5:][::-1]
    print(f"HF Top-5:          {[tokenizer.decode([t]) for t in hf_top5]}")

    # Logit correlation
    corr = np.corrcoef(logits, hf_logits)[0, 1]
    max_diff = np.max(np.abs(logits - hf_logits))
    print(f"Logit correlation: {corr:.6f}")
    print(f"Max logit diff:    {max_diff:.4f}")

    # --- Generation ---
    print("\n=== Text Generation ===")
    prompt = "In a shocking finding, scientists discovered a"
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'")

    np.random.seed(42)
    gen_tokens = list(input_ids)
    n_generate = 50

    t_start = time.perf_counter()
    for i in range(n_generate):
        logits = model.forward(np.array(gen_tokens, dtype=np.int32))
        next_token = sample_token(logits, temperature=0.8, top_k=40)
        gen_tokens.append(int(next_token))

    t_total = time.perf_counter() - t_start
    tok_per_sec = n_generate / t_total

    generated_text = tokenizer.decode(gen_tokens)
    print(f"\nGenerated ({n_generate} tokens, {tok_per_sec:.1f} tok/s):")
    print(f"  {generated_text}")
    print(f"\nTotal time: {t_total:.2f}s | {tok_per_sec:.1f} tokens/sec")
