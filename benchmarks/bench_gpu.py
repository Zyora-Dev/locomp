"""
Locust Benchmark — Matmul & Softmax vs MLX & PyTorch MPS on Apple Silicon.

Measures GPU kernel execution time (excluding compilation/setup).
All frameworks warm up first, then average over multiple runs.
"""

import time
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────

MATMUL_SIZES = [16, 32, 64, 128, 256, 512]
SOFTMAX_SHAPES = [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
WARMUP = 3
RUNS = 10


# ─── Locust ──────────────────────────────────────────────────────────────────

import locomp

@locomp.kernel
def locust_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
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
def locust_softmax(X: locomp.Tensor, OUT: locomp.Tensor,
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


def bench_locust_matmul(M, N, K):
    a = locomp.tensor(np.random.randn(M * K).astype(np.float32))
    b = locomp.tensor(np.random.randn(K * N).astype(np.float32))
    c = locomp.empty(M * N)
    total = M * N
    tpg = min(256, total)
    ng = (total + tpg - 1) // tpg

    # Warmup (includes first compile)
    for _ in range(WARMUP):
        locust_matmul[(ng,), (tpg,)](a, b, c, M, N, K)

    # Timed runs
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        locust_matmul[(ng,), (tpg,)](a, b, c, M, N, K)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000  # ms


def bench_locust_softmax(ROWS, D):
    x = locomp.tensor(np.random.randn(ROWS * D).astype(np.float32))
    out = locomp.empty(ROWS * D)
    tpg = min(256, ROWS)
    ng = (ROWS + tpg - 1) // tpg

    for _ in range(WARMUP):
        locust_softmax[(ng,), (tpg,)](x, out, ROWS, D)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        locust_softmax[(ng,), (tpg,)](x, out, ROWS, D)
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
        c = a @ b
        mx.eval(c)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        c = a @ b
        mx.eval(c)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


def bench_mlx_softmax(ROWS, D):
    x = mx.random.normal((ROWS, D))
    mx.eval(x)

    for _ in range(WARMUP):
        out = mx.softmax(x, axis=1)
        mx.eval(out)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = mx.softmax(x, axis=1)
        mx.eval(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


# ─── PyTorch MPS ─────────────────────────────────────────────────────────────

import torch

def bench_torch_matmul(M, N, K):
    device = torch.device("mps")
    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)
    torch.mps.synchronize()

    for _ in range(WARMUP):
        c = a @ b
        torch.mps.synchronize()

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        c = a @ b
        torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


def bench_torch_softmax(ROWS, D):
    device = torch.device("mps")
    x = torch.randn(ROWS, D, device=device)
    torch.mps.synchronize()

    for _ in range(WARMUP):
        out = torch.nn.functional.softmax(x, dim=1)
        torch.mps.synchronize()

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = torch.nn.functional.softmax(x, dim=1)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


# ─── NumPy (CPU baseline) ───────────────────────────────────────────────────

def bench_numpy_matmul(M, N, K):
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    for _ in range(WARMUP):
        c = a @ b

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        c = a @ b
        t1 = time.perf_counter()
        times.append(t1 - t0)
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
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


# ─── Runner ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("LOCUST BENCHMARK — Apple M1 GPU")
    print(f"Warmup: {WARMUP} | Runs: {RUNS} (median) | All times in ms")
    print("=" * 80)

    # ── Matmul ──
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MATMUL (M×K @ K×N)                                           │")
    print("├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤")
    print("│   Size   │  Locust  │   MLX    │ PyTorch  │  NumPy   │  Notes  │")
    print("├──────────┼──────────┼──────────┼──────────┼──────────┼─────────┤")

    for sz in MATMUL_SIZES:
        M = N = K = sz
        t_locust = bench_locust_matmul(M, N, K)
        t_mlx = bench_mlx_matmul(M, N, K)
        t_torch = bench_torch_matmul(M, N, K)
        t_numpy = bench_numpy_matmul(M, N, K)

        # Find fastest GPU
        gpu_times = {"Locust": t_locust, "MLX": t_mlx, "Torch": t_torch}
        fastest = min(gpu_times, key=gpu_times.get)

        print(f"│ {sz:>4}x{sz:<4}│ {t_locust:>7.3f}  │ {t_mlx:>7.3f}  │ {t_torch:>7.3f}  │ {t_numpy:>7.3f}  │ {fastest:<7} │")

    print("└──────────┴──────────┴──────────┴──────────┴──────────┴─────────┘")

    # ── Softmax ──
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  SOFTMAX (ROWS × D)                                           │")
    print("├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤")
    print("│   Shape  │  Locust  │   MLX    │ PyTorch  │  NumPy   │  Notes  │")
    print("├──────────┼──────────┼──────────┼──────────┼──────────┼─────────┤")

    for ROWS, D in SOFTMAX_SHAPES:
        t_locust = bench_locust_softmax(ROWS, D)
        t_mlx = bench_mlx_softmax(ROWS, D)
        t_torch = bench_torch_softmax(ROWS, D)
        t_numpy = bench_numpy_softmax(ROWS, D)

        gpu_times = {"Locust": t_locust, "MLX": t_mlx, "Torch": t_torch}
        fastest = min(gpu_times, key=gpu_times.get)

        label = f"{ROWS}x{D}"
        print(f"│ {label:>8} │ {t_locust:>7.3f}  │ {t_mlx:>7.3f}  │ {t_torch:>7.3f}  │ {t_numpy:>7.3f}  │ {fastest:<7} │")

    print("└──────────┴──────────┴──────────┴──────────┴──────────┴─────────┘")

    print("\nAll times in milliseconds (lower is better)")
    print("Locomp: custom compiler, pure Python → MSL")
    print("MLX: Apple's ML framework (v0.31.1)")
    print("PyTorch: MPS backend (v2.9.1)")
    print("NumPy: CPU baseline (Accelerate BLAS)")


if __name__ == "__main__":
    main()
