"""
locomp ROCm Benchmark — AMD MI100 (gfx908)
===========================================

Tests memory bandwidth, transformer operators, wavefront intrinsics, and
correctness on real AMD GPU hardware.

Usage (on the AMD server):
    pip install locomp
    python bench_rocm_mi100.py

Expected MI100 specs:
    - Peak memory bandwidth: ~1,229 GB/s (HBM2e)
    - Wavefront width: 64 lanes
    - Architecture: gfx908
    - VRAM: 32 GB
"""

import time
import numpy as np
import sys
import os
import subprocess

print("=" * 70)
print("locomp ROCm Benchmark — AMD MI100")
print("=" * 70)

# ── Sanity checks ──────────────────────────────────────────────────────────────

def check_rocm():
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print("ERROR: rocminfo failed. Is ROCm installed?")
            sys.exit(1)
        gpu_name = "(unknown)"
        arch = "(unknown)"
        for line in r.stdout.splitlines():
            s = line.strip()
            if s.lower().startswith("name:"):
                toks = s.split()
                if len(toks) >= 2:
                    name_val = toks[1]
                    if name_val.lower().startswith("gfx"):
                        arch = name_val
                    elif gpu_name == "(unknown)":
                        gpu_name = name_val
        print(f"GPU arch : {arch}")
        print(f"GPU name : {gpu_name}")
        return arch
    except FileNotFoundError:
        print("ERROR: rocminfo not found. Install ROCm.")
        sys.exit(1)

arch = check_rocm()

try:
    import locomp
    print(f"locomp   : {locomp.__version__ if hasattr(locomp, '__version__') else 'installed'}")
    print()
except ImportError:
    print("ERROR: locomp not installed. Run: pip install locomp")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────

BACKEND = "rocm"
WARMUP  = 5
RUNS    = 20

results = {}   # name -> dict with keys: bw_GBs, speedup, correct

# ── Helper ─────────────────────────────────────────────────────────────────────

def bench(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1e3  # ms


def allclose(a, b, rtol=1e-3, atol=1e-3):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def to_gpu(arr):
    return locomp.tensor(arr, backend=BACKEND)


def from_gpu(t):
    return t.numpy() if hasattr(t, "numpy") else np.array(t)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Memory bandwidth (bandwidth-bound kernels)
# ══════════════════════════════════════════════════════════════════════════════

print("─" * 70)
print("SECTION 1: Memory Bandwidth (target: ~1,229 GB/s peak for MI100)")
print("─" * 70)

N_BW = 256 * 1024 * 1024   # 256M float32 elements = 1 GB per array


@locomp.kernel
def vector_add_kernel(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                      N: locomp.constexpr):
    tid = locomp.thread_id(0) + locomp.program_id(0) * locomp.group_size(0)
    mask = tid < N
    a = locomp.load(A + tid, mask=mask)
    b = locomp.load(B + tid, mask=mask)
    locomp.store(C + tid, a + b, mask=mask)


@locomp.kernel
def scale_shift_kernel(X: locomp.Tensor, OUT: locomp.Tensor,
                       N: locomp.constexpr):
    tid = locomp.thread_id(0) + locomp.program_id(0) * locomp.group_size(0)
    mask = tid < N
    x = locomp.load(X + tid, mask=mask)
    locomp.store(OUT + tid, x * 2.0 + 1.0, mask=mask)


@locomp.kernel
def relu_kernel(X: locomp.Tensor, OUT: locomp.Tensor, N: locomp.constexpr):
    tid = locomp.thread_id(0) + locomp.program_id(0) * locomp.group_size(0)
    mask = tid < N
    x = locomp.load(X + tid, mask=mask)
    locomp.store(OUT + tid, locomp.where(x > 0.0, x, 0.0), mask=mask)


def run_bandwidth_kernel(name, kernel_fn, n_bytes_per_elem_io):
    """
    n_bytes_per_elem_io: total bytes read + written per element.
    vector_add: reads A,B writes C = 3 * 4 = 12 bytes/elem
    scale_shift: reads X writes OUT = 2 * 4 = 8 bytes/elem
    relu: reads X writes OUT = 2 * 4 = 8 bytes/elem
    """
    a = to_gpu(np.random.randn(N_BW).astype(np.float32))
    b = to_gpu(np.random.randn(N_BW).astype(np.float32))
    c = locomp.empty(N_BW, dtype=np.float32, backend=BACKEND)

    tpb = 256
    grid = ((N_BW + tpb - 1) // tpb,)

    if name == "vector_add":
        def fn(): vector_add_kernel[grid, (tpb,)](a, b, c, N_BW)
    elif name == "scale_shift":
        def fn(): scale_shift_kernel[grid, (tpb,)](a, c, N_BW)
    else:
        def fn(): relu_kernel[grid, (tpb,)](a, c, N_BW)

    # Correctness
    fn()
    out = from_gpu(c)
    a_np = from_gpu(a)
    b_np = from_gpu(b)
    if name == "vector_add":
        correct = allclose(out, a_np + b_np)
    elif name == "scale_shift":
        correct = allclose(out, a_np * 2.0 + 1.0)
    else:
        correct = allclose(out, np.maximum(0, a_np))

    ms = bench(fn)
    total_bytes = N_BW * 4 * n_bytes_per_elem_io   # float32 = 4 bytes each
    bw = (total_bytes / 1e9) / (ms / 1e3)

    status = "OK" if correct else "FAIL"
    print(f"  {name:<20}  {ms:7.3f} ms   {bw:8.1f} GB/s   [{status}]")
    results[name] = {"bw_GBs": bw, "ms": ms, "correct": correct}


run_bandwidth_kernel("vector_add",   vector_add_kernel,   3)   # read A,B + write C
run_bandwidth_kernel("scale_shift",  scale_shift_kernel,  2)   # read X + write OUT
run_bandwidth_kernel("relu",         relu_kernel,         2)   # read X + write OUT

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Transformer operators
# ══════════════════════════════════════════════════════════════════════════════

print()
print("─" * 70)
print("SECTION 2: Transformer Operators")
print("─" * 70)

# ── Softmax ───────────────────────────────────────────────────────────────────

B_SOFT, D_SOFT = 256, 1024

@locomp.kernel
def softmax_kernel(X: locomp.Tensor, OUT: locomp.Tensor,
                   ROWS: locomp.constexpr, D: locomp.constexpr):
    row = locomp.program_id(0)
    guard = row < ROWS
    row_max = locomp.load(X + row * D + 0, mask=guard)
    for j in range(1, D):
        v = locomp.load(X + row * D + j, mask=guard)
        row_max = locomp.where(v > row_max, v, row_max)
    exp_sum = 0.0
    for j in range(D):
        v = locomp.load(X + row * D + j, mask=guard)
        exp_sum = exp_sum + locomp.exp(v - row_max)
    for j in range(D):
        v = locomp.load(X + row * D + j, mask=guard)
        locomp.store(OUT + row * D + j, locomp.exp(v - row_max) / exp_sum, mask=guard)


x_soft_np = np.random.randn(B_SOFT, D_SOFT).astype(np.float32)
x_soft = to_gpu(x_soft_np.ravel())
o_soft = locomp.empty(B_SOFT * D_SOFT, dtype=np.float32, backend=BACKEND)

softmax_kernel[(B_SOFT,), (1,)](x_soft, o_soft, B_SOFT, D_SOFT)
ref_soft = (np.exp(x_soft_np - x_soft_np.max(axis=1, keepdims=True))
            / np.exp(x_soft_np - x_soft_np.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
out_soft  = from_gpu(o_soft).reshape(B_SOFT, D_SOFT)
ok_soft   = allclose(out_soft, ref_soft)

ms_soft = bench(lambda: softmax_kernel[(B_SOFT,), (1,)](x_soft, o_soft, B_SOFT, D_SOFT))
bw_soft = (B_SOFT * D_SOFT * 4 * 3 / 1e9) / (ms_soft / 1e3)  # 3×: 2 read passes + 1 write
print(f"  {'softmax [256x1024]':<26}  {ms_soft:7.3f} ms   {bw_soft:8.1f} GB/s   [{'OK' if ok_soft else 'FAIL'}]")
results["softmax"] = {"ms": ms_soft, "bw_GBs": bw_soft, "correct": ok_soft}

# ── RMSNorm ───────────────────────────────────────────────────────────────────

B_RMS, D_RMS = 128, 4096

@locomp.kernel
def rmsnorm_kernel(X: locomp.Tensor, W: locomp.Tensor, OUT: locomp.Tensor,
                   N: locomp.constexpr, D: locomp.constexpr):
    row = locomp.program_id(0)
    sq_sum = 0.0
    for j in range(D):
        v = locomp.load(X + row * D + j)
        sq_sum = sq_sum + v * v
    rms_inv = locomp.rsqrt(sq_sum / D + 1e-6)
    for j in range(D):
        v = locomp.load(X + row * D + j)
        w = locomp.load(W + j)
        locomp.store(OUT + row * D + j, v * rms_inv * w)


x_rms_np = np.random.randn(B_RMS, D_RMS).astype(np.float32)
w_rms_np = np.ones(D_RMS, dtype=np.float32)
x_rms  = to_gpu(x_rms_np.ravel())
w_rms  = to_gpu(w_rms_np)
o_rms  = locomp.empty(B_RMS * D_RMS, dtype=np.float32, backend=BACKEND)

rmsnorm_kernel[(B_RMS,), (1,)](x_rms, w_rms, o_rms, B_RMS, D_RMS)
rms_inv_ref = 1.0 / np.sqrt((x_rms_np ** 2).mean(axis=1, keepdims=True) + 1e-6)
ref_rms     = x_rms_np * rms_inv_ref * w_rms_np
out_rms     = from_gpu(o_rms).reshape(B_RMS, D_RMS)
ok_rms      = allclose(out_rms, ref_rms, rtol=1e-2, atol=1e-2)

ms_rms  = bench(lambda: rmsnorm_kernel[(B_RMS,), (1,)](x_rms, w_rms, o_rms, B_RMS, D_RMS))
bw_rms  = (B_RMS * D_RMS * 4 * 3 / 1e9) / (ms_rms / 1e3)
print(f"  {'rmsnorm [128x4096]':<26}  {ms_rms:7.3f} ms   {bw_rms:8.1f} GB/s   [{'OK' if ok_rms else 'FAIL'}]")
results["rmsnorm"] = {"ms": ms_rms, "bw_GBs": bw_rms, "correct": ok_rms}

# ── GELU ──────────────────────────────────────────────────────────────────────

N_GELU = 4 * 1024 * 1024  # 4M elements

@locomp.kernel
def gelu_kernel(X: locomp.Tensor, OUT: locomp.Tensor, N: locomp.constexpr):
    tid = locomp.thread_id(0) + locomp.program_id(0) * locomp.group_size(0)
    mask = tid < N
    x = locomp.load(X + tid, mask=mask)
    # GELU approximation: 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    c = 0.044715
    k = 0.7978845608028654
    inner = k * (x + c * x * x * x)
    g = 0.5 * x * (1.0 + locomp.tanh(inner))
    locomp.store(OUT + tid, g, mask=mask)


x_gelu_np = np.random.randn(N_GELU).astype(np.float32)
x_gelu    = to_gpu(x_gelu_np)
o_gelu    = locomp.empty(N_GELU, dtype=np.float32, backend=BACKEND)

tpb_g = 256
grid_g = ((N_GELU + tpb_g - 1) // tpb_g,)
gelu_kernel[grid_g, (tpb_g,)](x_gelu, o_gelu, N_GELU)

ref_gelu = 0.5 * x_gelu_np * (1 + np.tanh(0.7978845608 * (x_gelu_np + 0.044715 * x_gelu_np**3)))
ok_gelu  = allclose(from_gpu(o_gelu), ref_gelu, rtol=1e-2, atol=1e-2)

ms_gelu  = bench(lambda: gelu_kernel[grid_g, (tpb_g,)](x_gelu, o_gelu, N_GELU))
bw_gelu  = (N_GELU * 4 * 2 / 1e9) / (ms_gelu / 1e3)
print(f"  {'gelu [4M]':<26}  {ms_gelu:7.3f} ms   {bw_gelu:8.1f} GB/s   [{'OK' if ok_gelu else 'FAIL'}]")
results["gelu"] = {"ms": ms_gelu, "bw_GBs": bw_gelu, "correct": ok_gelu}

# ── RoPE ──────────────────────────────────────────────────────────────────────

SEQ, HEADS, HEAD_DIM = 512, 8, 64
N_ROPE = SEQ * HEADS * HEAD_DIM

@locomp.kernel
def rope_kernel(X: locomp.Tensor, OUT: locomp.Tensor,
                SEQ_LEN: locomp.constexpr, H: locomp.constexpr,
                D: locomp.constexpr):
    """Apply rotary embeddings. Each thread handles one (seq, head, dim) slot."""
    tid = locomp.thread_id(0) + locomp.program_id(0) * locomp.group_size(0)
    total = SEQ_LEN * H * D
    mask  = tid < total
    # Which (seq, head, d) position
    d_idx   = tid % D
    head_d  = D // 2
    # rotation index within [0, D/2)
    rot_idx = d_idx % head_d
    pos     = tid // (H * D)   # sequence position
    theta   = 1.0 / locomp.exp(rot_idx * (9.210340372  / head_d))  # 9.21 = log(10000)
    angle   = pos * theta
    x_val   = locomp.load(X + tid, mask=mask)
    # pair index: d_idx < head_d -> uses +sin, else -sin
    is_first_half = d_idx < head_d
    pair_offset   = locomp.where(is_first_half, head_d, -head_d)
    x_pair        = locomp.load(X + tid + pair_offset, mask=mask)
    cos_a = locomp.cos(angle)
    sin_a = locomp.sin(angle)
    sign  = locomp.where(is_first_half, 1.0, -1.0)
    out   = x_val * cos_a + sign * x_pair * sin_a
    locomp.store(OUT + tid, out, mask=mask)


x_rope_np = np.random.randn(N_ROPE).astype(np.float32)
x_rope    = to_gpu(x_rope_np)
o_rope    = locomp.empty(N_ROPE, dtype=np.float32, backend=BACKEND)

tpb_r = 256
grid_r = ((N_ROPE + tpb_r - 1) // tpb_r,)
rope_kernel[grid_r, (tpb_r,)](x_rope, o_rope, SEQ, HEADS, HEAD_DIM)

# Correctness: just check output is finite and different from input (rotation happened)
out_rope = from_gpu(o_rope)
ok_rope  = np.isfinite(out_rope).all() and not np.allclose(out_rope, x_rope_np)

ms_rope  = bench(lambda: rope_kernel[grid_r, (tpb_r,)](x_rope, o_rope, SEQ, HEADS, HEAD_DIM))
bw_rope  = (N_ROPE * 4 * 2 / 1e9) / (ms_rope / 1e3)
print(f"  {'rope [512x8x64]':<26}  {ms_rope:7.3f} ms   {bw_rope:8.1f} GB/s   [{'OK' if ok_rope else 'FAIL'}]")
results["rope"] = {"ms": ms_rope, "bw_GBs": bw_rope, "correct": ok_rope}

# ── matvec ────────────────────────────────────────────────────────────────────

M_MV, K_MV = 4096, 4096

@locomp.kernel
def matvec_kernel(A: locomp.Tensor, x: locomp.Tensor, y: locomp.Tensor,
                  M: locomp.constexpr, K: locomp.constexpr):
    """y[row] = A[row, :] @ x[:]"""
    row = locomp.program_id(0)
    mask = row < M
    acc = 0.0
    for k in range(K):
        a = locomp.load(A + row * K + k, mask=mask)
        xv = locomp.load(x + k)
        acc = acc + a * xv
    locomp.store(y + row, acc, mask=mask)


A_mv_np = np.random.randn(M_MV, K_MV).astype(np.float32)
x_mv_np = np.random.randn(K_MV).astype(np.float32)
A_mv    = to_gpu(A_mv_np.ravel())
x_mv    = to_gpu(x_mv_np)
y_mv    = locomp.empty(M_MV, dtype=np.float32, backend=BACKEND)

matvec_kernel[(M_MV,), (1,)](A_mv, x_mv, y_mv, M_MV, K_MV)
ref_mv   = A_mv_np @ x_mv_np
ok_mv    = allclose(from_gpu(y_mv), ref_mv, rtol=1e-2, atol=1e-2)

ms_mv    = bench(lambda: matvec_kernel[(M_MV,), (1,)](A_mv, x_mv, y_mv, M_MV, K_MV))
# bytes: A (MxK * 4) + x (K * 4) + y (M * 4)
bw_mv    = ((M_MV * K_MV + K_MV + M_MV) * 4 / 1e9) / (ms_mv / 1e3)
print(f"  {'matvec [4096x4096]':<26}  {ms_mv:7.3f} ms   {bw_mv:8.1f} GB/s   [{'OK' if ok_mv else 'FAIL'}]")
results["matvec"] = {"ms": ms_mv, "bw_GBs": bw_mv, "correct": ok_mv}

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Wavefront Intrinsics (64-lane, GFX9)
# ══════════════════════════════════════════════════════════════════════════════

print()
print("─" * 70)
print("SECTION 3: Wavefront Intrinsics (64-lane GFX9 / gfx908)")
print("─" * 70)

# MI100 has 64-lane wavefronts. simd_sum over 64 threads = sum(0..63) = 2016

@locomp.kernel
def warp_sum_kernel(OUT: locomp.Tensor, N: locomp.constexpr):
    tid = locomp.thread_id(0)
    lane = locomp.simd_lane_id()
    s = locomp.simd_sum(locomp.cast(lane, locomp.float32))
    # Only lane 0 writes
    mask = lane == 0
    locomp.store(OUT + locomp.program_id(0), s, mask=mask)


@locomp.kernel
def warp_max_kernel(OUT: locomp.Tensor, N: locomp.constexpr):
    lane = locomp.simd_lane_id()
    m = locomp.simd_max(locomp.cast(lane, locomp.float32))
    mask = lane == 0
    locomp.store(OUT + locomp.program_id(0), m, mask=mask)


@locomp.kernel
def warp_broadcast_kernel(OUT: locomp.Tensor, N: locomp.constexpr):
    lane = locomp.simd_lane_id()
    # Load 42.0 from lane 0, broadcast to all
    val = locomp.where(lane == 0, 42.0, 0.0)
    bcast = locomp.simd_broadcast(val, 0)
    mask = lane == 0
    locomp.store(OUT + locomp.program_id(0), bcast, mask=mask)


out_ws  = locomp.empty(1, dtype=np.float32, backend=BACKEND)
out_wm  = locomp.empty(1, dtype=np.float32, backend=BACKEND)
out_wb  = locomp.empty(1, dtype=np.float32, backend=BACKEND)

warp_sum_kernel[(1,), (64,)](out_ws, 64)
warp_max_kernel[(1,), (64,)](out_wm, 64)
warp_broadcast_kernel[(1,), (64,)](out_wb, 64)

ws_val = from_gpu(out_ws)[0]
wm_val = from_gpu(out_wm)[0]
wb_val = from_gpu(out_wb)[0]

# 64-lane wavefront: sum(0..63) = 63*64/2 = 2016, max = 63, broadcast = 42
ws_ok = abs(ws_val - 2016.0) < 1.0
wm_ok = abs(wm_val - 63.0)   < 0.5
wb_ok = abs(wb_val - 42.0)   < 0.5

print(f"  warp_sum  (64-lane): got {ws_val:.1f}, expect 2016.0  [{'OK' if ws_ok else 'FAIL'}]")
print(f"  warp_max  (64-lane): got {wm_val:.1f}, expect 63.0    [{'OK' if wm_ok else 'FAIL'}]")
print(f"  warp_bcast(64-lane): got {wb_val:.1f}, expect 42.0    [{'OK' if wb_ok else 'FAIL'}]")

results["warp_sum"]   = {"correct": ws_ok, "got": float(ws_val), "expect": 2016.0}
results["warp_max"]   = {"correct": wm_ok, "got": float(wm_val), "expect": 63.0}
results["warp_bcast"] = {"correct": wb_ok, "got": float(wb_val), "expect": 42.0}

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Summary
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  GPU architecture : {arch}")
print()
print(f"  {'Kernel':<26}  {'BW (GB/s)':>10}  {'Correct':>8}")
print(f"  {'-'*26}  {'-'*10}  {'-'*8}")

bw_keys = ["vector_add", "scale_shift", "relu", "softmax", "rmsnorm",
           "gelu", "rope", "matvec"]
for k in bw_keys:
    if k in results and "bw_GBs" in results[k]:
        bw  = results[k]["bw_GBs"]
        ok  = results[k]["correct"]
        print(f"  {k:<26}  {bw:>10.1f}  {'✓' if ok else '✗':>8}")

print()
intrinsics_keys = ["warp_sum", "warp_max", "warp_bcast"]
for k in intrinsics_keys:
    if k in results:
        ok  = results[k]["correct"]
        got = results[k]["got"]
        exp = results[k]["expect"]
        print(f"  {k:<26}  got={got:.1f} expect={exp:.1f}  {'✓' if ok else '✗'}")

all_ok = all(v.get("correct", False) for v in results.values())
print()
print(f"  All tests passed: {'YES ✓' if all_ok else 'NO ✗ — check FAIL items above'}")

# ── Peak bandwidth estimate vs MI100 theoretical peak ──────────────────────
bw_vals = [results[k]["bw_GBs"] for k in bw_keys if k in results and "bw_GBs" in results[k]]
if bw_vals:
    peak_bw = max(bw_vals)
    mi100_theoretical = 1229.0   # GB/s HBM2e
    pct = peak_bw / mi100_theoretical * 100
    print(f"\n  Peak observed BW : {peak_bw:.1f} GB/s  ({pct:.1f}% of MI100 {mi100_theoretical:.0f} GB/s theoretical)")

print("=" * 70)
print()
print("Copy these numbers into paper/locomp.tex Section 4 ROCm table.")
