"""Quick benchmark: tiled matmul vs naive matmul at various sizes."""
import locomp
import numpy as np
import time

TILE = 16

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


def bench(name, fn, args, grid, tg, M, N):
    # warmup
    for _ in range(3):
        c = locomp.empty(M * N)
        args_with_c = list(args[:2]) + [c] + list(args[3:])
        fn[grid, tg](*args_with_c)

    times = []
    for _ in range(10):
        c = locomp.empty(M * N)
        args_with_c = list(args[:2]) + [c] + list(args[3:])
        t0 = time.perf_counter()
        fn[grid, tg](*args_with_c)
        times.append((time.perf_counter() - t0) * 1000)

    result = c.numpy().reshape(M, N)
    return sorted(times)[5], result


print(f"{'Size':>8} | {'Naive':>10} | {'Tiled':>10} | {'Speedup':>8} | {'Error':>10}")
print("-" * 60)

for size in [16, 32, 64, 128, 256, 512]:
    M = N = K = size
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    a = locomp.tensor(a_np.flatten())
    b = locomp.tensor(b_np.flatten())
    expected = a_np @ b_np

    # Naive
    total = M * N
    tpg = min(256, total)
    ng = (total + tpg - 1) // tpg
    c_naive = locomp.empty(M * N)
    naive_matmul[(ng,), (tpg,)](a, b, c_naive, M, N, K)
    # warmup + time
    for _ in range(3):
        c = locomp.empty(M * N)
        naive_matmul[(ng,), (tpg,)](a, b, c, M, N, K)
    naive_times = []
    for _ in range(10):
        c = locomp.empty(M * N)
        t0 = time.perf_counter()
        naive_matmul[(ng,), (tpg,)](a, b, c, M, N, K)
        naive_times.append((time.perf_counter() - t0) * 1000)
    naive_med = sorted(naive_times)[5]

    # Tiled
    nt = K // TILE
    grid = (N // TILE, M // TILE)
    tg = (TILE, TILE)
    for _ in range(3):
        c = locomp.empty(M * N)
        tiled_matmul[grid, tg](a, b, c, M, N, K, nt, TILE)
    tiled_times = []
    for _ in range(10):
        c = locomp.empty(M * N)
        t0 = time.perf_counter()
        tiled_matmul[grid, tg](a, b, c, M, N, K, nt, TILE)
        tiled_times.append((time.perf_counter() - t0) * 1000)
    tiled_med = sorted(tiled_times)[5]

    result = c.numpy().reshape(M, N)
    err = np.max(np.abs(result - expected))
    speedup = naive_med / tiled_med

    print(f"{size:>4}x{size} | {naive_med:>8.3f}ms | {tiled_med:>8.3f}ms | {speedup:>7.2f}x | {err:.6f}")
