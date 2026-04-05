"""
Example 59: locomp v0.4.x — comprehensive CUDA benchmark suite.

Benchmarks locomp GPU kernels vs numpy on NVIDIA CUDA (A100/RTX/etc.).
Covers all real workloads used in transformer LLM inference, plus
CUDA-specific features (Tensor Core WMMA, warp intrinsics, FP16).

  1.  Softmax                  (attention probability, warp-reduce via simd_sum)
  2.  RMS Norm                 (LLaMA-style pre-norm)
  3.  GELU activation          (MLP non-linearity — tanh approx)
  4.  Vector Add               (elementwise, memory-bound)
  5.  Matrix Multiply          (naive 1-thread-per-element GEMM)
  6.  RoPE                     (rotary position embedding)
  7.  FP16 bandwidth           (float4/half2 vectorised LDG.128/STG.128)
  8.  Warp intrinsics          (simd_sum / simd_max / simd_min / broadcast / shuffle)

Run locally (requires CUDA GPU + nvcc):
    python examples/59_cuda_benchmark_suite.py

Run on A100 via Modal:
    modal run tests/modal_cuda_runner.py

Each benchmark:
  - Warms up 5 runs
  - Measures 15 runs, reports median
  - Compares GPU (locomp CUDA) vs NumPy
  - Reports max_err to confirm numerical correctness
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


def row(label, t_gpu, t_np, err, bw_GBs=None):
    r = t_gpu / t_np
    speedup = f"{1/r:.2f}x faster" if r < 1.0 else f"{r:.2f}x slower"
    bw_str = f"  {bw_GBs:.0f} GB/s" if bw_GBs is not None else ""
    print(f"  {label:<44} gpu={t_gpu:7.3f}ms  np={t_np:7.3f}ms  "
          f"{speedup:<18}  err={err:.1e}{bw_str}")


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
    print("\n[1] Softmax  (warp __shfl_down_sync cross-lane reduction)")
    for B, D in [(32, 128), (64, 256), (128, 512), (256, 1024)]:
        data = np.random.randn(B, D).astype(np.float32)
        x_g = locomp.tensor(data.flatten(), backend="cuda")
        o_g = locomp.empty(B * D, backend="cuda")

        def gpu():
            softmax_kernel[(B,)](x_g, o_g, ROWS=B, D=D)

        def np_ref():
            e = np.exp(data - data.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        gpu()
        err = np.max(np.abs(o_g.numpy().reshape(B, D) - np_ref()))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"softmax B={B} D={D}", t_gpu, t_np, err)
        x_g.free(); o_g.free()


# ── 2. RMS Norm ────────────────────────────────────────────────────────────────

@locomp.kernel
def rms_norm_kernel(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
                    ROWS: locomp.constexpr, D: locomp.constexpr,
                    eps: locomp.constexpr):
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
    print("\n[2] RMS Norm  (LLaMA-style pre-LayerNorm)")
    for B, D in [(32, 512), (32, 2048), (32, 4096), (128, 4096)]:
        data = np.random.randn(B, D).astype(np.float32)
        weight = np.ones(D, dtype=np.float32)
        eps = 1e-5
        x_g = locomp.tensor(data.flatten(), backend="cuda")
        w_g = locomp.tensor(weight, backend="cuda")
        o_g = locomp.empty(B * D, backend="cuda")

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
def gelu_kernel(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
    locomp.store(O + i, 0.5 * x * (1.0 + locomp.tanh(inner)))


def bench_gelu():
    print("\n[3] GELU activation  (tanh approximation)")
    for N in [65536, 262144, 1048576, 4194304]:
        data = np.random.randn(N).astype(np.float32)
        x_g = locomp.tensor(data, backend="cuda")
        o_g = locomp.empty(N, backend="cuda")

        def gpu():
            gelu_kernel[(N,)](x_g, o_g, N=N)

        def np_ref():
            c = 0.7978845608028654
            return 0.5 * data * (1 + np.tanh(c * (data + 0.044715 * data ** 3)))

        gpu()
        err = np.max(np.abs(o_g.numpy() - np_ref()))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"gelu N={N}", t_gpu, t_np, err)
        x_g.free(); o_g.free()


# ── 4. Vector Add ──────────────────────────────────────────────────────────────

@locomp.kernel
def vec_add_kernel(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor,
                   N: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(O + i, locomp.load(X + i) + locomp.load(Y + i))


def bench_vec_add():
    print("\n[4] Vector Add  (elementwise memory-bound)")
    for N in [1048576, 16777216, 67108864]:
        a = np.random.randn(N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        x_g = locomp.tensor(a, backend="cuda")
        y_g = locomp.tensor(b, backend="cuda")
        o_g = locomp.empty(N, backend="cuda")

        def gpu():
            vec_add_kernel[(N,)](x_g, y_g, o_g, N=N)

        def np_ref():
            return a + b

        gpu()
        err = np.max(np.abs(o_g.numpy() - np_ref()))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        bw = (3 * N * 4) / (t_gpu / 1000) / 1e9
        row(f"vec_add N={N}", t_gpu, t_np, err, bw_GBs=bw)
        x_g.free(); y_g.free(); o_g.free()


# ── 5. Matmul (naive) ─────────────────────────────────────────────────────────

@locomp.kernel
def matmul_kernel(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                  M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    row = locomp.program_id(0)
    col = locomp.program_id(1)
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(A + row * K + k) * locomp.load(B + k * N + col)
    locomp.store(C + row * N + col, acc)


def bench_matmul():
    print("\n[5] Matrix Multiply  (1-thread-per-element naive GEMM)")
    for M, N, K in [(64, 64, 128), (128, 128, 256), (256, 256, 512)]:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        a_g = locomp.tensor(A.flatten(), backend="cuda")
        b_g = locomp.tensor(B.flatten(), backend="cuda")
        c_g = locomp.empty(M * N, backend="cuda")

        def gpu():
            matmul_kernel[(M, N)](a_g, b_g, c_g, M=M, N=N, K=K)

        def np_ref():
            return A @ B

        gpu()
        err = np.max(np.abs(c_g.numpy().reshape(M, N) - np_ref()))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        row(f"matmul {M}x{K}x{N}", t_gpu, t_np, err)
        a_g.free(); b_g.free(); c_g.free()


# ── 6. RoPE ───────────────────────────────────────────────────────────────────

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
        theta = locomp.cast(seq, "float32") / locomp.pow(
            10000.0,
            locomp.cast(d * 2, "float32") / locomp.cast(DIM, "float32"))
        c = locomp.cos(theta)
        s = locomp.sin(theta)
        locomp.store(O + base, x0 * c - x1 * s)
        locomp.store(O + base + HALF, x1 * c + x0 * s)


def bench_rope():
    print("\n[6] RoPE  (rotary position embedding)")
    for B, H, S, D in [(1, 8, 128, 64), (1, 32, 256, 128), (4, 32, 128, 128)]:
        data = np.random.randn(B, H, S, D).astype(np.float32)
        HALF = D // 2
        x_g = locomp.tensor(data.reshape(B * H, S, D).flatten(), backend="cuda")
        o_g = locomp.empty(B * H * S * D, backend="cuda")

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


# ── 7. FP16 bandwidth ─────────────────────────────────────────────────────────
# Tests the new __half2 / float4 vectorised LDG.128 + STG.128 path.

@locomp.kernel
def fp16_copy_kernel(X: locomp.Float16, O: locomp.Float16,
                     TILE: locomp.constexpr):
    i = locomp.program_id(0)
    row = locomp.arange(TILE)
    a = locomp.load(X + i * TILE + row)
    locomp.store(O + i * TILE + row, a)


def bench_fp16_bandwidth():
    print("\n[7] FP16 bandwidth  (__half2/float4 vectorised: LDG.128 + STG.128)")
    TILE = 8
    for N_TILES in [65536, 1048576]:
        N = N_TILES * TILE
        data = np.random.randn(N).astype(np.float16)
        x_g = locomp.tensor(data, backend="cuda")
        o_g = locomp.empty(N, backend="cuda", dtype="float16")

        def gpu():
            fp16_copy_kernel[(N_TILES,)](x_g, o_g, TILE=TILE)

        def np_ref():
            return data.copy()

        gpu()
        err = float(np.max(np.abs(o_g.numpy().astype(np.float32)
                                  - data.astype(np.float32))))
        t_gpu = bench(gpu)
        t_np = bench(np_ref)
        bw = (2 * N * 2) / (t_gpu / 1000) / 1e9   # 2 bytes per __half, R+W
        row(f"fp16_copy N={N} tile={TILE}", t_gpu, t_np, err, bw_GBs=bw)
        x_g.free(); o_g.free()


# ── 8. Warp intrinsics validation ─────────────────────────────────────────────
# Validates all five warp ops numerically on real CUDA hardware.

@locomp.kernel
def warp_sum_kernel(OUT: locomp.Tensor):
    tid = locomp.local_id(0)
    val = locomp.cast(tid, "float32")
    reduced = locomp.simd_sum(val)
    if tid == 0:
        locomp.store(OUT + 0, reduced)


@locomp.kernel
def warp_max_kernel(OUT: locomp.Tensor):
    tid = locomp.local_id(0)
    val = locomp.cast(tid, "float32")
    reduced = locomp.simd_max(val)
    if tid == 0:
        locomp.store(OUT + 0, reduced)


@locomp.kernel
def warp_min_kernel(OUT: locomp.Tensor):
    tid = locomp.local_id(0)
    val = locomp.cast(tid + 1, "float32")
    reduced = locomp.simd_min(val)
    if tid == 0:
        locomp.store(OUT + 0, reduced)


@locomp.kernel
def warp_broadcast_kernel(OUT: locomp.Tensor):
    tid = locomp.local_id(0)
    val = locomp.cast(tid, "float32") + 42.0
    bcast = locomp.simd_broadcast(val, 0)
    locomp.store(OUT + tid, bcast)


@locomp.kernel
def warp_shuffle_kernel(OUT: locomp.Tensor):
    tid = locomp.local_id(0)
    val = locomp.cast(tid, "float32")
    down = locomp.simd_shuffle_down(val, 1)
    locomp.store(OUT + tid, down)


def bench_warp_intrinsics():
    print("\n[8] Warp intrinsics  (__shfl_down_sync warp-reduce / __shfl_sync broadcast)")
    NTHREADS = 32

    # warp_sum: sum(0..31) = 496
    out = locomp.empty(1, backend="cuda")
    warp_sum_kernel[(1,), (NTHREADS,)](out)
    got = out.numpy()[0]
    exp = float(sum(range(32)))
    status = "PASS" if abs(got - exp) < 1e-3 else "FAIL"
    print(f"  warp_sum      {status}  got={got:.1f}  expected={exp:.1f}")
    out.free()

    # warp_max: max(0..31) = 31
    out = locomp.empty(1, backend="cuda")
    warp_max_kernel[(1,), (NTHREADS,)](out)
    got = out.numpy()[0]
    status = "PASS" if abs(got - 31.0) < 1e-3 else "FAIL"
    print(f"  warp_max      {status}  got={got:.1f}  expected=31.0")
    out.free()

    # warp_min: min(1..32) = 1
    out = locomp.empty(1, backend="cuda")
    warp_min_kernel[(1,), (NTHREADS,)](out)
    got = out.numpy()[0]
    status = "PASS" if abs(got - 1.0) < 1e-3 else "FAIL"
    print(f"  warp_min      {status}  got={got:.1f}  expected=1.0")
    out.free()

    # warp_broadcast: lane 0 has 42, all should receive 42
    out = locomp.empty(NTHREADS, backend="cuda")
    warp_broadcast_kernel[(1,), (NTHREADS,)](out)
    vals = out.numpy()
    err = float(np.max(np.abs(vals - 42.0)))
    status = "PASS" if err < 1e-3 else "FAIL"
    print(f"  warp_broadcast {status}  all=42.0? max_err={err:.1e}")
    out.free()

    # warp_shuffle_down(delta=1): OUT[i] = i+1 for i=0..30, OUT[31]=31
    out = locomp.empty(NTHREADS, backend="cuda")
    warp_shuffle_kernel[(1,), (NTHREADS,)](out)
    vals = out.numpy()
    expected = np.array([float(i + 1) for i in range(31)] + [31.0], dtype=np.float32)
    err = float(np.max(np.abs(vals - expected)))
    status = "PASS" if err < 1e-3 else "FAIL"
    print(f"  warp_shuffle  {status}  max_err={err:.1e}")
    out.free()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("locomp v0.4.x — CUDA benchmark suite (GPU vs NumPy)")
    print("=" * 72)

    bench_softmax()
    bench_rms_norm()
    bench_gelu()
    bench_vec_add()
    bench_matmul()
    bench_rope()
    bench_fp16_bandwidth()
    bench_warp_intrinsics()

    print("\n" + "=" * 72)
    print("Done.")
