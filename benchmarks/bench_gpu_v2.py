"""
Locust Benchmark v2 — Naive vs Optimized vs MLX vs PyTorch MPS.

Compares:
  - Locust naive matmul (1 thread per element, sequential K-loop)
  - Locust tiled matmul (shared memory, cooperative loading, 16x16 tiles)
  - Locust naive softmax (1 thread per row, sequential passes)
  - Locust parallel softmax (256 threads per row, tree reduction)
  - MLX (Apple's ML framework)
  - PyTorch MPS (Metal backend)
  - NumPy (CPU baseline)
"""

import time
import numpy as np

MATMUL_SIZES = [16, 32, 64, 128, 256, 512]
SOFTMAX_SHAPES = [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
WARMUP = 3
RUNS = 10
TILE = 16  # matmul tile size

# ─── Locust Kernels ──────────────────────────────────────────────────────────

import locomp

@locomp.kernel
def naive_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    tid = locomp.thread_id(0)
    row = tid / N
    col = tid % N
    acc = 0.0
    for k in range(K):
        a_val = locomp.load(A + (row * K + k))
        b_val = locomp.load(B + (k * N + col))
        acc = acc + a_val * b_val
    total = M * N
    mask = tid < total
    locomp.store(C + tid, acc, mask=mask)


@locomp.kernel
def tiled_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
                 NUM_TILES: locomp.constexpr, BLOCK: locomp.constexpr):
    row = locomp.local_id(1)
    col = locomp.local_id(0)
    brow = locomp.program_id(1)
    bcol = locomp.program_id(0)
    As = locomp.shared_memory(TILE * TILE)
    Bs = locomp.shared_memory(TILE * TILE)
    acc = 0.0
    for t in range(NUM_TILES):
        a_row = brow * BLOCK + row
        a_col = t * BLOCK + col
        a_val = locomp.load(A + (a_row * K + a_col))
        locomp.shared_store(As, row * BLOCK + col, a_val)
        b_row = t * BLOCK + row
        b_col = bcol * BLOCK + col
        b_val = locomp.load(B + (b_row * N + b_col))
        locomp.shared_store(Bs, row * BLOCK + col, b_val)
        locomp.barrier()
        for k in range(BLOCK):
            a_shared = locomp.shared_load(As, row * BLOCK + k)
            b_shared = locomp.shared_load(Bs, k * BLOCK + col)
            acc = acc + a_shared * b_shared
        locomp.barrier()
    out_idx = (brow * BLOCK + row) * N + (bcol * BLOCK + col)
    locomp.store(C + out_idx, acc)


@locomp.kernel
def naive_softmax(X: locomp.Tensor, OUT: locomp.Tensor,
                  ROWS: locomp.constexpr, D: locomp.constexpr):
    row = locomp.thread_id(0)
    guard = row < ROWS
    row_max = locomp.load(X + (row * D + 0))
    for j in range(1, D):
        val = locomp.load(X + (row * D + j))
        row_max = locomp.where(val > row_max, val, row_max)
    exp_sum = 0.0
    for j in range(D):
        val = locomp.load(X + (row * D + j))
        e = locomp.exp(val - row_max)
        exp_sum = exp_sum + e
    for j in range(D):
        val = locomp.load(X + (row * D + j))
        e = locomp.exp(val - row_max)
        result = e / exp_sum
        locomp.store(OUT + (row * D + j), result, mask=guard)


@locomp.kernel
def parallel_softmax(X: locomp.Tensor, OUT: locomp.Tensor,
                     ROWS: locomp.constexpr, D: locomp.constexpr,
                     THREADS: locomp.constexpr, LOG_T: locomp.constexpr,
                     ELEMS: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS
    smem = locomp.shared_memory(256)
    local_max = locomp.load(X + (row * D + lid))
    for j in range(1, ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        local_max = locomp.where(val > local_max, val, local_max)
    locomp.shared_store(smem, lid, local_max)
    locomp.barrier()
    stride = THREADS / 2
    for s in range(LOG_T):
        if lid < stride:
            a = locomp.shared_load(smem, lid)
            b = locomp.shared_load(smem, lid + stride)
            mx = locomp.where(b > a, b, a)
            locomp.shared_store(smem, lid, mx)
        locomp.barrier()
        stride = stride / 2
    row_max = locomp.shared_load(smem, 0)
    locomp.barrier()
    local_sum = 0.0
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        local_sum = local_sum + e
    locomp.shared_store(smem, lid, local_sum)
    locomp.barrier()
    stride2 = THREADS / 2
    for s in range(LOG_T):
        if lid < stride2:
            a = locomp.shared_load(smem, lid)
            b = locomp.shared_load(smem, lid + stride2)
            locomp.shared_store(smem, lid, a + b)
        locomp.barrier()
        stride2 = stride2 / 2
    total_sum = locomp.shared_load(smem, 0)
    locomp.barrier()
    for j in range(ELEMS):
        idx = lid + j * THREADS
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        result = e / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)


# ─── Locust Benchmarks ──────────────────────────────────────────────────────

def bench_locust_naive_matmul(M, N, K):
    a = locomp.tensor(np.random.randn(M * K).astype(np.float32))
    b = locomp.tensor(np.random.randn(K * N).astype(np.float32))
    c = locomp.empty(M * N)
    total = M * N
    tpg = min(256, total)
    ng = (total + tpg - 1) // tpg
    for _ in range(WARMUP):
        naive_matmul[(ng,), (tpg,)](a, b, c, M, N, K)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        naive_matmul[(ng,), (tpg,)](a, b, c, M, N, K)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


def bench_locust_tiled_matmul(M, N, K):
    a = locomp.tensor(np.random.randn(M * K).astype(np.float32))
    b = locomp.tensor(np.random.randn(K * N).astype(np.float32))
    c = locomp.empty(M * N)
    nt = K // TILE
    grid = (N // TILE, M // TILE)
    tg = (TILE, TILE)
    for _ in range(WARMUP):
        tiled_matmul[grid, tg](a, b, c, M, N, K, nt, TILE)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        tiled_matmul[grid, tg](a, b, c, M, N, K, nt, TILE)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


def bench_locust_naive_softmax(ROWS, D):
    x = locomp.tensor(np.random.randn(ROWS * D).astype(np.float32))
    out = locomp.empty(ROWS * D)
    tpg = min(256, ROWS)
    ng = (ROWS + tpg - 1) // tpg
    for _ in range(WARMUP):
        naive_softmax[(ng,), (tpg,)](x, out, ROWS, D)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        naive_softmax[(ng,), (tpg,)](x, out, ROWS, D)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


def bench_locust_parallel_softmax(ROWS, D):
    THREADS = min(256, D)
    LOG_T = int(np.log2(THREADS))
    ELEMS = D // THREADS
    if ELEMS < 1:
        return None  # D < THREADS, skip
    x = locomp.tensor(np.random.randn(ROWS * D).astype(np.float32))
    out = locomp.empty(ROWS * D)
    for _ in range(WARMUP):
        parallel_softmax[(ROWS,), (THREADS,)](x, out, ROWS, D, THREADS, LOG_T, ELEMS)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        parallel_softmax[(ROWS,), (THREADS,)](x, out, ROWS, D, THREADS, LOG_T, ELEMS)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


# ─── MLX ─────────────────────────────────────────────────────────────────────

import mlx.core as mx

def bench_mlx_matmul(M, N, K):
    a = mx.random.normal((M, K))
    b = mx.random.normal((K, N))
    mx.eval(a, b)
    for _ in range(WARMUP):
        c = a @ b; mx.eval(c)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        c = a @ b; mx.eval(c)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000

def bench_mlx_softmax(ROWS, D):
    x = mx.random.normal((ROWS, D))
    mx.eval(x)
    for _ in range(WARMUP):
        out = mx.softmax(x, axis=1); mx.eval(out)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = mx.softmax(x, axis=1); mx.eval(out)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


# ─── PyTorch MPS ─────────────────────────────────────────────────────────────

import torch

def bench_torch_matmul(M, N, K):
    device = torch.device("mps")
    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)
    torch.mps.synchronize()
    for _ in range(WARMUP):
        c = a @ b; torch.mps.synchronize()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        c = a @ b; torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000

def bench_torch_softmax(ROWS, D):
    device = torch.device("mps")
    x = torch.randn(ROWS, D, device=device)
    torch.mps.synchronize()
    for _ in range(WARMUP):
        out = torch.nn.functional.softmax(x, dim=1); torch.mps.synchronize()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = torch.nn.functional.softmax(x, dim=1); torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


# ─── NumPy ───────────────────────────────────────────────────────────────────

def bench_numpy_matmul(M, N, K):
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    for _ in range(WARMUP):
        c = a @ b
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        c = a @ b
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000

def bench_numpy_softmax(ROWS, D):
    x = np.random.randn(ROWS, D).astype(np.float32)
    for _ in range(WARMUP):
        xm = x - x.max(axis=1, keepdims=True)
        e = np.exp(xm)
        out = e / e.sum(axis=1, keepdims=True)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        xm = x - x.max(axis=1, keepdims=True)
        e = np.exp(xm)
        out = e / e.sum(axis=1, keepdims=True)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


# ─── Runner ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 95)
    print("LOCUST BENCHMARK v2 — Naive vs Optimized vs MLX vs PyTorch MPS")
    print(f"Apple M1 | Warmup: {WARMUP} | Runs: {RUNS} (median) | All times in ms")
    print("=" * 95)

    # ── Matmul ──
    print("\nMATMUL (M×K @ K×N, float32)")
    print(f"{'Size':>8} | {'Naive':>8} | {'Tiled':>8} | {'MLX':>8} | {'Torch':>8} | {'NumPy':>8} | {'Tiled vs MLX':>12}")
    print("-" * 80)

    for sz in MATMUL_SIZES:
        M = N = K = sz
        t_naive = bench_locust_naive_matmul(M, N, K)
        t_tiled = bench_locust_tiled_matmul(M, N, K)
        t_mlx = bench_mlx_matmul(M, N, K)
        t_torch = bench_torch_matmul(M, N, K)
        t_numpy = bench_numpy_matmul(M, N, K)
        ratio = t_tiled / t_mlx
        print(f"{sz:>4}x{sz:<3} | {t_naive:>7.3f} | {t_tiled:>7.3f} | {t_mlx:>7.3f} | {t_torch:>7.3f} | {t_numpy:>7.3f} | {ratio:>10.2f}x")

    # ── Softmax ──
    print(f"\nSOFTMAX (ROWS × D, float32)")
    print(f"{'Shape':>8} | {'Naive':>8} | {'Parallel':>8} | {'MLX':>8} | {'Torch':>8} | {'NumPy':>8} | {'Par vs MLX':>12}")
    print("-" * 80)

    for ROWS, D in SOFTMAX_SHAPES:
        t_naive = bench_locust_naive_softmax(ROWS, D)
        t_par = bench_locust_parallel_softmax(ROWS, D)
        t_mlx = bench_mlx_softmax(ROWS, D)
        t_torch = bench_torch_softmax(ROWS, D)
        t_numpy = bench_numpy_softmax(ROWS, D)
        par_str = f"{t_par:>7.3f}" if t_par else "    N/A"
        ratio = t_par / t_mlx if t_par else float('inf')
        ratio_str = f"{ratio:>10.2f}x" if t_par else "        N/A"
        print(f"{ROWS:>3}x{D:<4} | {t_naive:>7.3f} | {par_str} | {t_mlx:>7.3f} | {t_torch:>7.3f} | {t_numpy:>7.3f} | {ratio_str}")


if __name__ == "__main__":
    main()
