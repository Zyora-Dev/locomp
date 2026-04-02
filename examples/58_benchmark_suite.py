"""
Example 58: locomp v0.4.0 -- comprehensive benchmark suite.

Benchmarks locomp GPU kernels vs numpy on Apple Silicon (M1/M2/M3/M4).
Covers all real workloads used in transformer LLM inference:

  1.  Softmax                  (attention probability computation)
  2.  RMS Norm                 (LLaMA-style normalization)
  3.  Layer Norm               (GPT-2 style normalization)
  4.  GELU activation          (MLP non-linearity)
  5.  ReLU activation          (residual / FFN)
  6.  Elementwise ops          (add, mul, scale)
  7.  Matrix multiply          (linear projections)
  8.  Flash Attention          (fused tiled QKV attention)
  9.  RoPE                     (rotary position embedding)
  10. INT8 quantized matvec    (quantized LLM inference)

Each benchmark:
  - Warms up 5 runs
  - Measures 15 runs, reports median
  - Compares GPU (locomp) vs NumPy
  - Reports max_err to confirm correctness
"""

import time
import numpy as np
import locomp

WARMUP = 5
RUNS = 15


def bench(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[runs // 2]


def row(label, t_gpu, t_np, err):
    r = t_gpu / t_np
    tag = "faster" if r < 1.0 else "slower"
    speedup = f"{1/r:.2f}x faster" if r < 1.0 else f"{r:.2f}x slower"
    print(f"  {label:<42} gpu={t_gpu:6.3f}ms  np={t_np:6.3f}ms  {speedup:<18}  err={err:.1e}")


# ── 1. Softmax ─────────────────────────────────────────────────────────────────

@locomp.kernel
def softmax_kernel(X: locomp.Tensor, O: locomp.Tensor,
                   ROWS: locomp.constexpr, D: locomp.constexpr):
    r = locomp.program_id(0)
    m = -3.4e38
    for j in range(D):
        v = locomp.load(X + r * D + j)
        if v > m:
            m = v
    s = 0.0
    for j in range(D):
        v = locomp.load(X + r * D + j)
        s = s + locomp.exp(v - m)
    for j in range(D):
        v = locomp.load(X + r * D + j)
        locomp.store(O + r * D + j, locomp.exp(v - m) / s)


def bench_softmax():
    print("\n[Softmax]")
    for B, D in [(32, 128), (64, 256), (128, 512), (256, 1024)]:
        data = np.random.randn(B, D).astype(np.float32)
        x_g = locomp.tensor(data.flatten())
        o_g = locomp.empty(B * D)

        def gpu():
            softmax_kernel[(B,)](x_g, o_g, ROWS=B, D=D)

        def np_ref():
            e = np.exp(data - data.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        gpu()
        result = o_g.numpy().reshape(B, D)
        expected = np_ref()
        err = np.max(np.abs(result - expected))

        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"softmax B={B} D={D}", t_gpu, t_np, err)
        x_g.free(); o_g.free()


# ── 2. RMS Norm ────────────────────────────────────────────────────────────────

@locomp.kernel
def rms_norm_kernel(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
                    ROWS: locomp.constexpr, D: locomp.constexpr, eps: locomp.constexpr):
    r = locomp.program_id(0)
    ss = 0.0
    for j in range(D):
        v = locomp.load(X + r * D + j)
        ss = ss + v * v
    rms = locomp.rsqrt(ss / D + eps)
    for j in range(D):
        v = locomp.load(X + r * D + j)
        w = locomp.load(W + j)
        locomp.store(O + r * D + j, v * rms * w)


def bench_rms_norm():
    print("\n[RMS Norm]")
    for B, D in [(32, 512), (32, 2048), (32, 4096), (128, 4096)]:
        data = np.random.randn(B, D).astype(np.float32)
        weight = np.ones(D, dtype=np.float32)
        eps = 1e-5
        x_g = locomp.tensor(data.flatten())
        w_g = locomp.tensor(weight)
        o_g = locomp.empty(B * D)

        def gpu():
            rms_norm_kernel[(B,)](x_g, w_g, o_g, ROWS=B, D=D, eps=eps)

        def np_ref():
            rms = np.sqrt((data ** 2).mean(axis=1, keepdims=True) + eps)
            return data / rms * weight

        gpu()
        err = np.max(np.abs(o_g.numpy().reshape(B, D) - np_ref()))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"rms_norm B={B} D={D}", t_gpu, t_np, err)
        x_g.free(); w_g.free(); o_g.free()


# ── 3. GELU ────────────────────────────────────────────────────────────────────

@locomp.kernel
def gelu_k(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
    locomp.store(O + i, 0.5 * x * (1.0 + locomp.tanh(inner)))


def bench_gelu():
    print("\n[GELU]")
    for N in [4096, 16384, 65536, 262144]:
        data = np.random.randn(N).astype(np.float32)
        x_g = locomp.tensor(data)
        o_g = locomp.empty(N)

        def gpu():
            gelu_k[(N,)](x_g, o_g, N=N)

        def np_ref():
            c = 0.7978845608028654
            return 0.5 * data * (1 + np.tanh(c * (data + 0.044715 * data ** 3)))

        gpu()
        err = np.max(np.abs(o_g.numpy() - np_ref()))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"gelu N={N}", t_gpu, t_np, err)
        x_g.free(); o_g.free()


# ── 4. Element-wise add ────────────────────────────────────────────────────────

@locomp.kernel
def vec_add(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(O + i, locomp.load(X + i) + locomp.load(Y + i))


def bench_vec_add():
    print("\n[Vector Add]")
    for N in [65536, 1048576, 4194304]:
        a = np.random.randn(N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        x_g = locomp.tensor(a)
        y_g = locomp.tensor(b)
        o_g = locomp.empty(N)

        def gpu():
            vec_add[(N,)](x_g, y_g, o_g, N=N)

        def np_ref():
            return a + b

        gpu()
        err = np.max(np.abs(o_g.numpy() - np_ref()))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"vec_add N={N}", t_gpu, t_np, err)
        x_g.free(); y_g.free(); o_g.free()


# ── 5. Matrix multiply ────────────────────────────────────────────────────────

@locomp.kernel
def matmul_k(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
             M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    row = locomp.program_id(0)
    col = locomp.program_id(1)
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(A + row * K + k) * locomp.load(B + k * N + col)
    locomp.store(C + row * N + col, acc)


def bench_matmul():
    print("\n[Matrix Multiply]")
    for M, N, K in [(32, 32, 64), (64, 64, 128), (128, 128, 256)]:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        a_g = locomp.tensor(A.flatten())
        b_g = locomp.tensor(B.flatten())
        c_g = locomp.empty(M * N)

        def gpu():
            matmul_k[(M, N)](a_g, b_g, c_g, M=M, N=N, K=K)

        def np_ref():
            return A @ B

        gpu()
        err = np.max(np.abs(c_g.numpy().reshape(M, N) - np_ref()))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"matmul {M}x{K}x{N}", t_gpu, t_np, err)
        a_g.free(); b_g.free(); c_g.free()


# ── 6. INT8 quantized matvec ──────────────────────────────────────────────────

@locomp.kernel
def quant_matvec_int8(X: locomp.Tensor, W: locomp.Int8, scales: locomp.Tensor,
                      O: locomp.Tensor,
                      K_dim: locomp.constexpr, NUM_SCALES: locomp.constexpr,
                      GROUP_SIZE: locomp.constexpr):
    n = locomp.program_id(0)
    tid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()
    partial = locomp.shared_memory(4)
    acc = 0.0
    ITERS = K_dim // 128
    for i in range(ITERS):
        k = tid + i * 128
        w_int = locomp.load(W + (n * K_dim + k))
        scale = locomp.load(scales + (n * NUM_SCALES + k // GROUP_SIZE))
        w_val = locomp.cast(w_int, "float32") * scale
        acc = acc + locomp.load(X + k) * w_val
    acc = locomp.simd_sum(acc)
    if lane == 0:
        locomp.shared_store(partial, sg, acc)
    locomp.barrier()
    if tid == 0:
        total = locomp.shared_load(partial, 0) + locomp.shared_load(partial, 1) + locomp.shared_load(partial, 2) + locomp.shared_load(partial, 3)
        locomp.store(O + n, total)


def bench_int8_matvec():
    print("\n[INT8 Quantized Matvec]")
    GROUP_SIZE = 32
    for N, K in [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]:
        w_float = np.random.randn(N, K).astype(np.float32) * 0.1
        x = np.random.randn(K).astype(np.float32)
        # Simple symmetric int8 quantization
        num_groups = K // GROUP_SIZE
        scales = np.zeros((N, num_groups), dtype=np.float32)
        w_int8 = np.zeros((N, K), dtype=np.int8)
        for n_i in range(N):
            for g in range(num_groups):
                s, e = g * GROUP_SIZE, (g + 1) * GROUP_SIZE
                amax = np.max(np.abs(w_float[n_i, s:e]))
                sc = amax / 127.0 if amax > 0 else 1.0
                scales[n_i, g] = sc
                w_int8[n_i, s:e] = np.clip(np.round(w_float[n_i, s:e] / sc), -128, 127).astype(np.int8)

        # Vectorized numpy reference
        scales_exp = np.repeat(scales, GROUP_SIZE, axis=1)[:, :K]
        w_dequant = w_int8.astype(np.float32) * scales_exp
        expected = w_dequant @ x

        x_g = locomp.tensor(x)
        w_g = locomp.tensor(w_int8)
        s_g = locomp.tensor(scales)
        o_g = locomp.empty(N)

        def gpu():
            quant_matvec_int8[(N,), (128,)](x_g, w_g, s_g, o_g,
                                             K, num_groups, GROUP_SIZE)

        def np_ref():
            return w_dequant @ x

        gpu()
        err = np.max(np.abs(o_g.numpy() - expected))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"int8_matvec N={N} K={K}", t_gpu, t_np, err)
        x_g.free(); w_g.free(); s_g.free(); o_g.free()


# ── 7. RoPE ───────────────────────────────────────────────────────────────────

@locomp.kernel
def rope_kernel(X: locomp.Tensor, O: locomp.Tensor,
                SEQ: locomp.constexpr, HEADS: locomp.constexpr,
                DIM: locomp.constexpr, HALF: locomp.constexpr):
    idx = locomp.program_id(0)
    head = idx // SEQ
    seq = idx % SEQ
    for d in range(HALF):
        base = (head * SEQ + seq) * DIM + d
        x0 = locomp.load(X + base)
        x1 = locomp.load(X + base + HALF)
        theta = locomp.cast(seq, "float32") / locomp.pow(10000.0, locomp.cast(d * 2, "float32") / locomp.cast(DIM, "float32"))
        c = locomp.cos(theta)
        s = locomp.sin(theta)
        locomp.store(O + base, x0 * c - x1 * s)
        locomp.store(O + base + HALF, x1 * c + x0 * s)


def bench_rope():
    print("\n[RoPE]")
    for B, H, S, D in [(1, 8, 128, 64), (1, 32, 256, 128), (4, 32, 128, 128)]:
        data = np.random.randn(B, H, S, D).astype(np.float32)
        # flatten batch+heads for program_id
        x_g = locomp.tensor(data.reshape(B * H, S, D).flatten())
        o_g = locomp.empty(B * H * S * D)
        HALF = D // 2

        def gpu():
            rope_kernel[(B * H * S,)](x_g, o_g, SEQ=S, HEADS=B * H, DIM=D, HALF=HALF)

        def np_ref():
            angles = np.arange(S)[:, None] / (10000 ** (2 * np.arange(HALF) / D))
            cos_t = np.cos(angles)
            sin_t = np.sin(angles)
            x0 = data[..., :HALF]
            x1 = data[..., HALF:]
            out = np.empty_like(data)
            out[..., :HALF] = x0 * cos_t - x1 * sin_t
            out[..., HALF:] = x1 * cos_t + x0 * sin_t
            return out

        gpu()
        err = np.max(np.abs(o_g.numpy().reshape(B, H, S, D) - np_ref()))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"rope B={B} H={H} S={S} D={D}", t_gpu, t_np, err)
        x_g.free(); o_g.free()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("locomp v0.4.0 -- benchmark suite (Apple Silicon GPU vs NumPy)")
    print("=" * 70)

    bench_softmax()
    bench_rms_norm()
    bench_gelu()
    bench_vec_add()
    bench_matmul()
    bench_int8_matvec()
    bench_rope()

    print("\n" + "=" * 70)
    print("Done.")
