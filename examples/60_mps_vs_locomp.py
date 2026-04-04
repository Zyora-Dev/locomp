"""
Example 60 — MPS vs locomp benchmark on Apple Silicon.

Compares Apple MPS (PyTorch) vs locomp GPU vs NumPy CPU for:
  - GELU
  - RMS Norm
  - RoPE
  - Softmax
  - Vector Add
  - MatMul (fp32)
  - INT8 quantized matvec

Run: python examples/60_mps_vs_locomp.py
Requires: pip install torch  (PyTorch with MPS support)
"""

import time
import numpy as np
import locomp

try:
    import torch
    assert torch.backends.mps.is_available(), "MPS not available"
    MPS = True
except (ImportError, AssertionError):
    print("PyTorch MPS not available — install torch to enable MPS comparison.")
    MPS = False

# ── Config ────────────────────────────────────────────────────────────────────

WARMUP = 10
REPS   = 50

# ── locomp kernels ────────────────────────────────────────────────────────────

@locomp.kernel
def lc_gelu(X: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    t = locomp.tanh(0.7978845608 * (x + 0.044715 * x * x * x))
    locomp.store(O + i, x * 0.5 * (1.0 + t))


@locomp.kernel
def lc_rms_norm(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
                N: locomp.constexpr, eps: locomp.constexpr):
    i = locomp.program_id(0)
    acc = 0.0
    for j in range(N):
        v = locomp.load(X + j)
        acc = acc + v * v
    scale = locomp.rsqrt(acc / N + eps)
    v = locomp.load(X + i)
    locomp.store(O + i, v * scale * locomp.load(W + i))


@locomp.kernel
def lc_rope(X: locomp.Tensor, Cos: locomp.Tensor, Sin: locomp.Tensor,
            O: locomp.Tensor, D: locomp.constexpr, HALF_D: locomp.constexpr):
    # program_id(0)=position, program_id(1)=dim_pair (0..HALF_D-1)
    pos = locomp.program_id(0)
    i   = locomp.program_id(1)
    base = pos * D
    x0 = locomp.load(X + base + i)
    x1 = locomp.load(X + base + i + HALF_D)
    c  = locomp.load(Cos + pos * HALF_D + i)
    s  = locomp.load(Sin + pos * HALF_D + i)
    locomp.store(O + base + i,         x0 * c - x1 * s)
    locomp.store(O + base + i + HALF_D, x0 * s + x1 * c)


@locomp.kernel
def lc_softmax(X: locomp.Tensor, O: locomp.Tensor,
               N: locomp.constexpr, stride: locomp.constexpr):
    row = locomp.program_id(0)
    base = row * stride
    m = locomp.load(X + base)
    for j in range(1, N):
        v = locomp.load(X + base + j)
        if v > m:
            m = v
    s = 0.0
    for j in range(N):
        v = locomp.exp(locomp.load(X + base + j) - m)
        locomp.store(O + base + j, v)
        s = s + v
    for j in range(N):
        locomp.store(O + base + j, locomp.load(O + base + j) / s)


@locomp.kernel
def lc_vec_add(X: locomp.Tensor, Y: locomp.Tensor, O: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(O + i, locomp.load(X + i) + locomp.load(Y + i))


@locomp.kernel
def lc_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
              M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    row = locomp.program_id(0)
    col = locomp.program_id(1)
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(A + row * K + k) * locomp.load(B + k * N + col)
    locomp.store(C + row * N + col, acc)


@locomp.kernel
def lc_int8_matvec(W: locomp.Int8, X: locomp.Tensor, O: locomp.Tensor,
                   N: locomp.constexpr, K: locomp.constexpr):
    row = locomp.program_id(0)
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(W + row * K + k) * locomp.load(X + k)
    locomp.store(O + row, acc)


# ── Timing helpers ────────────────────────────────────────────────────────────

def bench_locomp(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000  # ms


def bench_mps(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / reps * 1000  # ms


def bench_numpy(fn, warmup=3, reps=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


# ── Print helpers ─────────────────────────────────────────────────────────────

HDR = f"  {'Benchmark':<22} {'N':>8}  {'locomp':>9}  {'MPS':>9}  {'NumPy':>9}  {'lc/MPS':>8}  {'lc/NP':>8}"
SEP = "  " + "─" * 84

def row(name, N, lc, mps, np_, lo_vs_mps, lo_vs_np):
    mps_s  = f"{mps:.3f}ms"  if mps  else "   n/a   "
    lvm    = f"{lo_vs_mps:.2f}x" if lo_vs_mps else "  n/a  "
    lvn    = f"{lo_vs_np:.2f}x"
    print(f"  {name:<22} {N:>8}  {lc:>7.3f}ms  {mps_s:>9}  {np_:>7.3f}ms  {lvm:>8}  {lvn:>8}")


# ── Benchmarks ────────────────────────────────────────────────────────────────

print(f"\nlocomp v{locomp.__version__}  |  PyTorch {torch.__version__ if MPS else 'n/a'}  |  MPS={'yes' if MPS else 'no'}")
print(SEP)
print(HDR)
print(SEP)

# 1. GELU ──────────────────────────────────────────────────────────────────────
for N in [65536, 262144, 1048576]:
    x_np = np.random.randn(N).astype(np.float32)
    x_lc = locomp.tensor(x_np); o_lc = locomp.empty(N)

    lc_time = bench_locomp(lambda: lc_gelu[(N,)](x_lc, o_lc, N=N))

    if MPS:
        x_t = torch.tensor(x_np, device="mps")
        mps_time = bench_mps(lambda: torch.nn.functional.gelu(x_t))
    else:
        mps_time = None

    np_time = bench_numpy(lambda: x_np * 0.5 * (1.0 + np.tanh(0.7978845608 * (x_np + 0.044715 * x_np**3))))

    row("GELU", N, lc_time, mps_time, np_time,
        mps_time / lc_time if mps_time else None,
        np_time / lc_time)

# 2. RMS Norm ──────────────────────────────────────────────────────────────────
for N in [1024, 4096]:
    x_np = np.random.randn(N).astype(np.float32)
    w_np = np.ones(N, dtype=np.float32)
    x_lc = locomp.tensor(x_np); w_lc = locomp.tensor(w_np); o_lc = locomp.empty(N)

    lc_time = bench_locomp(lambda: lc_rms_norm[(N,)](x_lc, w_lc, o_lc, N=N, eps=1e-5))

    if MPS:
        x_t = torch.tensor(x_np, device="mps")
        rms = torch.nn.RMSNorm(N, device="mps")
        mps_time = bench_mps(lambda: rms(x_t))
    else:
        mps_time = None

    def np_rms():
        rms_v = np.sqrt(np.mean(x_np**2) + 1e-5)
        return x_np / rms_v * w_np
    np_time = bench_numpy(np_rms)

    row("RMS Norm", N, lc_time, mps_time, np_time,
        mps_time / lc_time if mps_time else None,
        np_time / lc_time)

# 3. RoPE ──────────────────────────────────────────────────────────────────────
for SEQ, D in [(512, 64), (2048, 128)]:
    N_rope = SEQ * D
    x_np  = np.random.randn(N_rope).astype(np.float32)
    half  = D // 2
    pos   = np.arange(SEQ, dtype=np.float32)
    freqs = 1.0 / (10000 ** (np.arange(half, dtype=np.float32) / half))
    theta = np.outer(pos, freqs).astype(np.float32)
    cos_np = np.cos(theta).astype(np.float32)
    sin_np = np.sin(theta).astype(np.float32)

    x_lc   = locomp.tensor(x_np)
    cos_lc = locomp.tensor(cos_np.ravel())
    sin_lc = locomp.tensor(sin_np.ravel())
    o_lc   = locomp.empty(N_rope)

    lc_time = bench_locomp(lambda: lc_rope[(SEQ, half)](x_lc, cos_lc, sin_lc, o_lc, D=D, HALF_D=half))

    if MPS:
        xr = torch.tensor(x_np.reshape(SEQ, D), device="mps")
        cos_t = torch.tensor(cos_np, device="mps")
        sin_t = torch.tensor(sin_np, device="mps")
        def mps_rope():
            x1, x2 = xr[..., :half], xr[..., half:]
            return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)
        mps_time = bench_mps(mps_rope)
    else:
        mps_time = None

    xr_np = x_np.reshape(SEQ, D)
    def np_rope():
        x1, x2 = xr_np[..., :half], xr_np[..., half:]
        return np.concatenate([x1 * cos_np - x2 * sin_np, x2 * cos_np + x1 * sin_np], axis=-1)
    np_time = bench_numpy(np_rope)

    row(f"RoPE seq={SEQ} d={D}", N_rope, lc_time, mps_time, np_time,
        mps_time / lc_time if mps_time else None,
        np_time / lc_time)

# 4. Softmax ───────────────────────────────────────────────────────────────────
for ROWS, COLS in [(256, 512), (1024, 1024)]:
    N_sm = ROWS * COLS
    x_np = np.random.randn(N_sm).astype(np.float32)
    x_lc = locomp.tensor(x_np); o_lc = locomp.empty(N_sm)

    lc_time = bench_locomp(lambda: lc_softmax[(ROWS,)](x_lc, o_lc, N=COLS, stride=COLS))

    if MPS:
        x_t = torch.tensor(x_np.reshape(ROWS, COLS), device="mps")
        mps_time = bench_mps(lambda: torch.nn.functional.softmax(x_t, dim=-1))
    else:
        mps_time = None

    def np_sm():
        xr = x_np.reshape(ROWS, COLS)
        xm = xr - xr.max(axis=1, keepdims=True)
        e  = np.exp(xm)
        return e / e.sum(axis=1, keepdims=True)
    np_time = bench_numpy(np_sm)

    row(f"Softmax {ROWS}x{COLS}", N_sm, lc_time, mps_time, np_time,
        mps_time / lc_time if mps_time else None,
        np_time / lc_time)

# 5. Vec Add ───────────────────────────────────────────────────────────────────
for N in [262144, 1048576]:
    x_np = np.random.randn(N).astype(np.float32)
    y_np = np.random.randn(N).astype(np.float32)
    x_lc = locomp.tensor(x_np); y_lc = locomp.tensor(y_np); o_lc = locomp.empty(N)

    lc_time = bench_locomp(lambda: lc_vec_add[(N,)](x_lc, y_lc, o_lc, N=N))

    if MPS:
        x_t = torch.tensor(x_np, device="mps")
        y_t = torch.tensor(y_np, device="mps")
        mps_time = bench_mps(lambda: x_t + y_t)
    else:
        mps_time = None

    np_time = bench_numpy(lambda: x_np + y_np)

    row("Vec Add", N, lc_time, mps_time, np_time,
        mps_time / lc_time if mps_time else None,
        np_time / lc_time)

# 6. MatMul ────────────────────────────────────────────────────────────────────
for SZ in [128, 256]:
    M = N_mm = K = SZ
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N_mm).astype(np.float32)
    a_lc = locomp.tensor(a_np.ravel()); b_lc = locomp.tensor(b_np.ravel()); c_lc = locomp.empty(M * N_mm)

    lc_time = bench_locomp(lambda: lc_matmul[(M, N_mm)](a_lc, b_lc, c_lc, M=M, N=N_mm, K=K))

    if MPS:
        a_t = torch.tensor(a_np, device="mps")
        b_t = torch.tensor(b_np, device="mps")
        mps_time = bench_mps(lambda: torch.mm(a_t, b_t))
    else:
        mps_time = None

    np_time = bench_numpy(lambda: np.dot(a_np, b_np))

    row(f"MatMul {SZ}x{SZ}", M * N_mm, lc_time, mps_time, np_time,
        mps_time / lc_time if mps_time else None,
        np_time / lc_time)

# 7. INT8 Matvec ───────────────────────────────────────────────────────────────
for N_mv, K_mv in [(2048, 2048), (4096, 4096)]:
    w_np = np.random.randint(-127, 127, (N_mv, K_mv), dtype=np.int8)
    x_np = np.random.randn(K_mv).astype(np.float32)
    w_lc = locomp.tensor(w_np.ravel()); x_lc = locomp.tensor(x_np); o_lc = locomp.empty(N_mv)

    lc_time = bench_locomp(lambda: lc_int8_matvec[(N_mv,)](w_lc, x_lc, o_lc, N=N_mv, K=K_mv))

    if MPS:
        # MPS doesn't support int8 matmul directly — use float32 cast
        w_t = torch.tensor(w_np.astype(np.float32), device="mps")
        x_t = torch.tensor(x_np, device="mps")
        mps_time = bench_mps(lambda: torch.mv(w_t, x_t))
    else:
        mps_time = None

    np_time = bench_numpy(lambda: w_np.astype(np.float32) @ x_np)

    row(f"INT8 Matvec N={N_mv}", N_mv * K_mv, lc_time, mps_time, np_time,
        mps_time / lc_time if mps_time else None,
        np_time / lc_time)

print(SEP)
print(f"\n  lc/MPS > 1.0x = locomp faster than MPS")
print(f"  lc/MPS < 1.0x = MPS faster than locomp")
print(f"  lc/NP  > 1.0x = locomp faster than NumPy CPU\n")
