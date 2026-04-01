"""
SmolLM2-135M inference using only locomp GPU kernels.

Architecture: Llama-style decoder-only transformer
- 30 layers, hidden=576, heads=9, kv_heads=3 (GQA group=3)
- SiLU activation, RMSNorm, RoPE (theta=100000)
- Vocab: 49152, tied embeddings

All matrix operations and normalization run on Apple Metal GPU via locomp.
"""

import time
import json
import numpy as np
import locomp
from safetensors import safe_open

# =========================================================================
# Model config
# =========================================================================
MODEL_PATH = "/Users/redfoxhotels/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors"
TOKENIZER_PATH = "/Users/redfoxhotels/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/tokenizer.json"

HIDDEN = 576
N_HEADS = 9
N_KV_HEADS = 3
HEAD_DIM = HIDDEN // N_HEADS  # 64
INTER = 1536
N_LAYERS = 30
VOCAB = 49152
RMS_EPS = 1e-5
ROPE_THETA = 100000.0
KV_GROUP = N_HEADS // N_KV_HEADS  # 3

# =========================================================================
# GPU Kernels
# =========================================================================

@locomp.kernel
def rms_norm_kernel(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
                    N: locomp.constexpr, EPS_X1000: locomp.constexpr):
    row = locomp.program_id(0)
    tid = locomp.local_id(0)
    eps = EPS_X1000 * 0.000001

    # Compute sum of squares
    smem = locomp.shared_memory(32)
    local_sum = 0.0
    for i in range(tid, N, 128):
        val = locomp.load(X + (row * N + i))
        local_sum = local_sum + val * val

    local_sum = locomp.simd_sum(local_sum)
    sg = locomp.simd_group_id()
    lane = locomp.simd_lane_id()
    if lane == 0:
        locomp.shared_store(smem, sg, local_sum)
    locomp.barrier()

    if tid == 0:
        total = 0.0
        for g in range(4):
            total = total + locomp.shared_load(smem, g)
        rms = locomp.rsqrt(total / N + eps)
        locomp.shared_store(smem, 0, rms)
    locomp.barrier()

    rms = locomp.shared_load(smem, 0)
    for i in range(tid, N, 128):
        val = locomp.load(X + (row * N + i))
        w = locomp.load(W + i)
        locomp.store(O + (row * N + i), val * rms * w)


@locomp.kernel
def matmul_kernel(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                  M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    idx = locomp.program_id(0)
    m = idx // N
    n = idx - m * N
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(A + (m * K + k)) * locomp.load(B + (k * N + n))
    locomp.store(C + (m * N + n), acc)


@locomp.kernel
def matvec_kernel(W: locomp.Tensor, X: locomp.Tensor, O: locomp.Tensor,
                  N: locomp.constexpr, K: locomp.constexpr):
    n = locomp.program_id(0)
    tid = locomp.local_id(0)

    smem = locomp.shared_memory(32)
    local_sum = 0.0
    for k in range(tid, K, 128):
        local_sum = local_sum + locomp.load(W + (n * K + k)) * locomp.load(X + k)

    local_sum = locomp.simd_sum(local_sum)
    sg = locomp.simd_group_id()
    lane = locomp.simd_lane_id()
    if lane == 0:
        locomp.shared_store(smem, sg, local_sum)
    locomp.barrier()

    if tid == 0:
        total = 0.0
        for g in range(4):
            total = total + locomp.shared_load(smem, g)
        locomp.store(O + n, total)


@locomp.kernel
def silu_mul_kernel(gate: locomp.Tensor, up: locomp.Tensor, O: locomp.Tensor,
                    N: locomp.constexpr):
    i = locomp.program_id(0)
    g = locomp.load(gate + i)
    u = locomp.load(up + i)
    s = g * locomp.sigmoid(g)
    locomp.store(O + i, s * u)


@locomp.kernel
def rope_kernel(Q: locomp.Tensor, K: locomp.Tensor, FREQ: locomp.Tensor,
                POS: locomp.constexpr, N_Q: locomp.constexpr,
                N_K: locomp.constexpr, HD: locomp.constexpr):
    idx = locomp.program_id(0)
    half = HD // 2
    head = idx // half
    d = idx - head * half

    freq = locomp.load(FREQ + d)
    angle = POS * freq

    cos_a = locomp.cos(angle)
    sin_a = locomp.sin(angle)

    # Q heads — half-half rotation (Llama/SmolLM2 rotate_half format)
    if head < N_Q:
        i0 = head * HD + d
        i1 = head * HD + d + half
        q0 = locomp.load(Q + i0)
        q1 = locomp.load(Q + i1)
        locomp.store(Q + i0, q0 * cos_a - q1 * sin_a)
        locomp.store(Q + i1, q1 * cos_a + q0 * sin_a)

    # K heads — half-half rotation
    if head < N_K:
        i0 = head * HD + d
        i1 = head * HD + d + half
        k0 = locomp.load(K + i0)
        k1 = locomp.load(K + i1)
        locomp.store(K + i0, k0 * cos_a - k1 * sin_a)
        locomp.store(K + i1, k1 * cos_a + k0 * sin_a)


@locomp.kernel
def add_kernel(A: locomp.Tensor, B: locomp.Tensor, O: locomp.Tensor,
               N: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(O + i, locomp.load(A + i) + locomp.load(B + i))


@locomp.kernel
def copy_kernel(src: locomp.Tensor, dst: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(dst + i, locomp.load(src + i))


@locomp.kernel
def embed_kernel(tokens: locomp.Tensor, table: locomp.Tensor, out: locomp.Tensor,
                 DIM: locomp.constexpr):
    seq = locomp.program_id(0)
    d = locomp.program_id(1)
    tok = locomp.load(tokens + seq)
    # Convert float token to int index
    tok_int = tok * 1  # stays float, used as index
    val = locomp.load(table + (tok_int * DIM + d))
    locomp.store(out + (seq * DIM + d), val)


@locomp.kernel
def gqa_attn_kernel(Q: locomp.Tensor, K_cache: locomp.Tensor, V_cache: locomp.Tensor,
                    O: locomp.Tensor, SCORES: locomp.Tensor,
                    SEQ_LEN: locomp.constexpr, HD: locomp.constexpr,
                    KV_GROUP: locomp.constexpr, MAX_SEQ: locomp.constexpr):
    q_head = locomp.program_id(0)
    kv_head = q_head // KV_GROUP
    tid = locomp.local_id(0)

    smem = locomp.shared_memory(256)  # max seq_len scores

    # Phase 1: compute all attention scores (one thread per KV position)
    scale = locomp.rsqrt(HD * 1.0)

    for s_base in range(0, SEQ_LEN, 64):
        s = s_base + tid
        if s < SEQ_LEN:
            dot = 0.0
            for d in range(HD):
                q_val = locomp.load(Q + (q_head * HD + d))
                k_val = locomp.load(K_cache + (kv_head * MAX_SEQ * HD + s * HD + d))
                dot = dot + q_val * k_val
            locomp.shared_store(smem, s, dot * scale)
    locomp.barrier()

    # Phase 2: softmax (thread 0)
    if tid == 0:
        m = locomp.shared_load(smem, 0)
        for s in range(1, SEQ_LEN):
            m = locomp.max(m, locomp.shared_load(smem, s))
        total = 0.0
        for s in range(SEQ_LEN):
            e = locomp.exp(locomp.shared_load(smem, s) - m)
            locomp.shared_store(smem, s, e)
            total = total + e
        inv = 1.0 / total
        for s in range(SEQ_LEN):
            locomp.shared_store(smem, s, locomp.shared_load(smem, s) * inv)
    locomp.barrier()

    # Phase 3: weighted sum of V (thread tid handles output dim tid)
    if tid < HD:
        out_val = 0.0
        for s in range(SEQ_LEN):
            w = locomp.shared_load(smem, s)
            v_val = locomp.load(V_cache + (kv_head * MAX_SEQ * HD + s * HD + tid))
            out_val = out_val + w * v_val
        locomp.store(O + (q_head * HD + tid), out_val)


# =========================================================================
# Weight loading
# =========================================================================

def load_weights(path):
    """Load safetensors weights as numpy float32 arrays."""
    import torch
    weights = {}
    with safe_open(path, framework='pt') as f:
        for key in f.keys():
            t = f.get_tensor(key).to(torch.float32).numpy()
            weights[key] = t
    return weights


# =========================================================================
# Tokenizer (minimal)
# =========================================================================

class SimpleTokenizer:
    """Minimal tokenizer using HuggingFace tokenizer.json."""

    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        # Build vocab: token string → id
        self.vocab = data['model']['vocab']
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        # BOS/EOS
        self.bos_id = 0
        self.eos_id = 0
        # Merges for BPE
        self.merges = data['model'].get('merges', [])
        self._build_bpe()

    def _build_bpe(self):
        """Build BPE merge rules."""
        self.bpe_ranks = {}
        for i, merge in enumerate(self.merges):
            pair = tuple(merge.split())
            self.bpe_ranks[pair] = i

    def _bpe_encode(self, text):
        """Byte-level BPE encoding."""
        # Convert to bytes, then to token strings  
        tokens = [bytes([b]).decode('latin-1') for b in text.encode('utf-8')]
        # Map byte chars to the vocab's byte representation
        # HuggingFace byte-level BPE uses special unicode chars for bytes
        byte_encoder = self._bytes_to_unicode()
        tokens = [byte_encoder.get(b, b) for b in text.encode('utf-8')]

        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            best_pair = None
            best_rank = float('inf')
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None or best_rank == float('inf'):
                break
            # Merge best pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert merged tokens to ids
        ids = []
        for t in tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            else:
                ids.append(0)  # UNK
        return ids

    @staticmethod
    def _bytes_to_unicode():
        """Map bytes to unicode chars (GPT-2 byte-level BPE)."""
        bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def encode(self, text):
        """Encode text to token ids."""
        return self._bpe_encode(text)

    def decode(self, ids):
        """Decode token ids to text."""
        byte_decoder = {v: k for k, v in self._bytes_to_unicode().items()}
        tokens = [self.id_to_token.get(i, '') for i in ids]
        text = ''.join(tokens)
        # Decode byte-level chars back to bytes
        byte_arr = bytearray([byte_decoder.get(c, ord(c)) for c in text])
        return byte_arr.decode('utf-8', errors='replace')


# =========================================================================
# Model inference
# =========================================================================

class SmolLM2:
    """SmolLM2-135M inference engine using locomp GPU kernels."""

    def __init__(self, weights_path, tokenizer_path):
        print("Loading weights...")
        t0 = time.time()
        raw = load_weights(weights_path)
        print(f"  Loaded {len(raw)} tensors in {time.time()-t0:.1f}s")

        print("Uploading to GPU...")
        t0 = time.time()
        self.w = {}
        total_mb = 0
        for k, v in raw.items():
            self.w[k] = locomp.tensor(v)
            total_mb += v.nbytes / 1e6
        print(f"  {total_mb:.0f}MB uploaded in {time.time()-t0:.1f}s")

        # Pre-compute RoPE frequencies
        freqs = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
        self.rope_freqs = locomp.tensor(freqs)

        self.tokenizer = SimpleTokenizer(tokenizer_path)

        # Allocate KV cache
        self.max_seq = 256
        self.k_cache = [locomp.tensor(np.zeros((N_KV_HEADS, self.max_seq, HEAD_DIM), dtype=np.float32))
                        for _ in range(N_LAYERS)]
        self.v_cache = [locomp.tensor(np.zeros((N_KV_HEADS, self.max_seq, HEAD_DIM), dtype=np.float32))
                        for _ in range(N_LAYERS)]

        # Scratch buffers (reused across layers)
        self.hidden_buf = locomp.empty(HIDDEN)
        self.norm_buf = locomp.empty(HIDDEN)
        self.q_buf = locomp.empty(N_HEADS * HEAD_DIM)
        self.k_buf = locomp.empty(N_KV_HEADS * HEAD_DIM)
        self.v_buf = locomp.empty(N_KV_HEADS * HEAD_DIM)
        self.attn_out_buf = locomp.empty(N_HEADS * HEAD_DIM)
        self.o_proj_buf = locomp.empty(HIDDEN)
        self.gate_buf = locomp.empty(INTER)
        self.up_buf = locomp.empty(INTER)
        self.mlp_buf = locomp.empty(INTER)
        self.down_buf = locomp.empty(HIDDEN)
        self.residual_buf = locomp.empty(HIDDEN)
        self.scores_buf = locomp.empty(self.max_seq)
        self.logits_buf = locomp.empty(VOCAB)

    def _rms_norm(self, x, w, out):
        rms_norm_kernel[(1,), (128,)](x, w, out, HIDDEN, int(RMS_EPS * 1000000))

    def _matvec(self, W, x, out, N, K):
        matvec_kernel[(N,), (128,)](W, x, out, N, K)

    def _layer(self, layer_idx, pos):
        """Run one transformer layer."""
        prefix = f"model.layers.{layer_idx}"

        # Input RMSNorm
        self._rms_norm(self.hidden_buf, self.w[f"{prefix}.input_layernorm.weight"],
                       self.norm_buf)

        # Q, K, V projections
        self._matvec(self.w[f"{prefix}.self_attn.q_proj.weight"], self.norm_buf,
                     self.q_buf, N_HEADS * HEAD_DIM, HIDDEN)
        self._matvec(self.w[f"{prefix}.self_attn.k_proj.weight"], self.norm_buf,
                     self.k_buf, N_KV_HEADS * HEAD_DIM, HIDDEN)
        self._matvec(self.w[f"{prefix}.self_attn.v_proj.weight"], self.norm_buf,
                     self.v_buf, N_KV_HEADS * HEAD_DIM, HIDDEN)

        # RoPE
        n_rope_programs = max(N_HEADS, N_KV_HEADS) * (HEAD_DIM // 2)
        rope_kernel[(n_rope_programs,)](
            self.q_buf, self.k_buf, self.rope_freqs,
            pos, N_HEADS, N_KV_HEADS, HEAD_DIM
        )

        # Store K, V into cache at position `pos`
        # Copy k_buf → k_cache[layer][kv_head, pos, :]
        # We need a kernel for this — or use numpy for now
        from locomp.backends.metal_runtime import get_runtime
        rt = get_runtime()
        rt.sync()
        k_np = self.k_buf.numpy().reshape(N_KV_HEADS, HEAD_DIM)
        v_np = self.v_buf.numpy().reshape(N_KV_HEADS, HEAD_DIM)

        # Update cache numpy and re-upload (naive but correct)
        k_cache_np = self.k_cache[layer_idx].numpy()
        v_cache_np = self.v_cache[layer_idx].numpy()
        k_cache_np[:, pos, :] = k_np
        v_cache_np[:, pos, :] = v_np
        self.k_cache[layer_idx] = locomp.tensor(k_cache_np)
        self.v_cache[layer_idx] = locomp.tensor(v_cache_np)

        # GQA Attention
        seq_len = pos + 1
        gqa_attn_kernel[(N_HEADS,), (64,)](
            self.q_buf, self.k_cache[layer_idx], self.v_cache[layer_idx],
            self.attn_out_buf, self.scores_buf,
            seq_len, HEAD_DIM, KV_GROUP, self.max_seq
        )

        # O projection
        self._matvec(self.w[f"{prefix}.self_attn.o_proj.weight"], self.attn_out_buf,
                     self.o_proj_buf, HIDDEN, HIDDEN)

        # Residual add: hidden = hidden + o_proj
        add_kernel[(HIDDEN,)](self.hidden_buf, self.o_proj_buf, self.residual_buf, HIDDEN)
        copy_kernel[(HIDDEN,)](self.residual_buf, self.hidden_buf, HIDDEN)

        # Post-attention RMSNorm
        self._rms_norm(self.hidden_buf, self.w[f"{prefix}.post_attention_layernorm.weight"],
                       self.norm_buf)

        # MLP: gate_proj, up_proj → SiLU(gate) * up → down_proj
        self._matvec(self.w[f"{prefix}.mlp.gate_proj.weight"], self.norm_buf,
                     self.gate_buf, INTER, HIDDEN)
        self._matvec(self.w[f"{prefix}.mlp.up_proj.weight"], self.norm_buf,
                     self.up_buf, INTER, HIDDEN)
        silu_mul_kernel[(INTER,)](self.gate_buf, self.up_buf, self.mlp_buf, INTER)
        self._matvec(self.w[f"{prefix}.mlp.down_proj.weight"], self.mlp_buf,
                     self.down_buf, HIDDEN, INTER)

        # Residual add: hidden = hidden + down
        add_kernel[(HIDDEN,)](self.hidden_buf, self.down_buf, self.residual_buf, HIDDEN)
        copy_kernel[(HIDDEN,)](self.residual_buf, self.hidden_buf, HIDDEN)

    def forward(self, token_id, pos):
        """Forward pass for a single token at position `pos`. Returns logits numpy array."""
        # Embedding lookup (CPU for now — single row copy)
        embed_np = self.w["model.embed_tokens.weight"].numpy()[token_id]
        # Upload to hidden_buf
        self.hidden_buf = locomp.tensor(embed_np.astype(np.float32))

        # Run all layers
        for layer_idx in range(N_LAYERS):
            self._layer(layer_idx, pos)

        # Final RMSNorm
        self._rms_norm(self.hidden_buf, self.w["model.norm.weight"], self.norm_buf)

        # LM head (tied embeddings): logits = norm_buf @ embed_tokens^T
        # = matvec with embed_tokens as weight matrix
        self._matvec(self.w["model.embed_tokens.weight"], self.norm_buf,
                     self.logits_buf, VOCAB, HIDDEN)

        from locomp.backends.metal_runtime import get_runtime
        get_runtime().sync()
        return self.logits_buf.numpy()

    def generate(self, prompt, max_tokens=50, temperature=0.0):
        """Generate text from a prompt."""
        tokens = self.tokenizer.encode(prompt)
        print(f"  Prompt tokens: {tokens[:10]}... ({len(tokens)} tokens)")

        generated = []
        all_tokens = list(tokens)

        # Prefill
        print("  Prefilling...", end="", flush=True)
        t0 = time.time()
        for i, tok in enumerate(tokens):
            logits = self.forward(tok, i)
        prefill_time = time.time() - t0
        print(f" {prefill_time:.2f}s ({len(tokens)/prefill_time:.1f} tok/s)")

        # Decode
        print("  Generating: ", end="", flush=True)
        t0 = time.time()
        for step in range(max_tokens):
            if temperature <= 0:
                next_tok = int(np.argmax(logits))
            else:
                logits_t = logits / temperature
                logits_t -= np.max(logits_t)
                probs = np.exp(logits_t) / np.sum(np.exp(logits_t))
                next_tok = int(np.random.choice(len(probs), p=probs))

            if next_tok == self.tokenizer.eos_id and step > 0:
                break

            generated.append(next_tok)
            all_tokens.append(next_tok)

            # Print token as we decode
            text = self.tokenizer.decode([next_tok])
            print(text, end="", flush=True)

            # Forward next token
            pos = len(tokens) + step
            logits = self.forward(next_tok, pos)

        gen_time = time.time() - t0
        tokens_generated = len(generated)
        print()
        print(f"  Generated {tokens_generated} tokens in {gen_time:.2f}s "
              f"({tokens_generated/gen_time:.1f} tok/s)")

        return self.tokenizer.decode(generated)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SmolLM2-135M — locomp GPU inference")
    print("=" * 60)

    model = SmolLM2(MODEL_PATH, TOKENIZER_PATH)

    print("\n--- Generation ---")
    prompt = "The meaning of life is"
    print(f"Prompt: \"{prompt}\"")
    output = model.generate(prompt, max_tokens=30, temperature=0.0)
    print(f"\nFull output: {prompt}{output}")
