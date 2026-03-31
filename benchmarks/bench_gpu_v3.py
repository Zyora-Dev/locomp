"""
Locust Benchmark v3 — All optimizations vs MLX vs PyTorch MPS.

Matmul variants: naive, tiled (16×16 smem), micro-tiled (2×2 per thread)
Softmax variants: naive, parallel tree, SIMD (32 threads)
"""

import time
import numpy as np

MATMUL_SIZES = [32, 64, 128, 256, 512]
SOFTMAX_SHAPES = [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
WARMUP = 3
RUNS = 10
TILE = 16  # matmul tile
BM = 32; BN = 32; BK = 16  # micro-tile block dims

import locomp, mlx.core as mx, torch

# ─── Locust Kernels ──────────────────────────────────────────────────────────

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
def micro_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
                 NUM_K_TILES: locomp.constexpr, BLOCK_K: locomp.constexpr,
                 BLOCK_M: locomp.constexpr, BLOCK_N: locomp.constexpr):
    trow = locomp.local_id(1)
    tcol = locomp.local_id(0)
    brow = locomp.program_id(1)
    bcol = locomp.program_id(0)
    As = locomp.shared_memory(BM * BK)
    Bs = locomp.shared_memory(BK * BN)
    acc00 = 0.0
    acc01 = 0.0
    acc10 = 0.0
    acc11 = 0.0
    for t in range(NUM_K_TILES):
        a_base_row = brow * BLOCK_M + trow * 2
        a_base_col = t * BLOCK_K + tcol
        a_val0 = locomp.load(A + ((a_base_row + 0) * K + a_base_col))
        a_val1 = locomp.load(A + ((a_base_row + 1) * K + a_base_col))
        locomp.shared_store(As, (trow * 2 + 0) * BLOCK_K + tcol, a_val0)
        locomp.shared_store(As, (trow * 2 + 1) * BLOCK_K + tcol, a_val1)
        b_base_row = t * BLOCK_K + trow
        b_base_col = bcol * BLOCK_N + tcol * 2
        b_val0 = locomp.load(B + (b_base_row * N + b_base_col + 0))
        b_val1 = locomp.load(B + (b_base_row * N + b_base_col + 1))
        locomp.shared_store(Bs, trow * BLOCK_N + tcol * 2 + 0, b_val0)
        locomp.shared_store(Bs, trow * BLOCK_N + tcol * 2 + 1, b_val1)
        locomp.barrier()
        for k in range(BLOCK_K):
            a0 = locomp.shared_load(As, (trow * 2 + 0) * BLOCK_K + k)
            a1 = locomp.shared_load(As, (trow * 2 + 1) * BLOCK_K + k)
            b0 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 2 + 0)
            b1 = locomp.shared_load(Bs, k * BLOCK_N + tcol * 2 + 1)
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1
        locomp.barrier()
    out_row = brow * BLOCK_M + trow * 2
    out_col = bcol * BLOCK_N + tcol * 2
    locomp.store(C + ((out_row + 0) * N + out_col + 0), acc00)
    locomp.store(C + ((out_row + 0) * N + out_col + 1), acc01)
    locomp.store(C + ((out_row + 1) * N + out_col + 0), acc10)
    locomp.store(C + ((out_row + 1) * N + out_col + 1), acc11)


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


@locomp.kernel
def simd_softmax(X: locomp.Tensor, OUT: locomp.Tensor,
                 ROWS: locomp.constexpr, D: locomp.constexpr,
                 ELEMS: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS
    local_max = locomp.load(X + (row * D + lid))
    for j in range(1, ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        local_max = locomp.where(val > local_max, val, local_max)
    row_max = locomp.simd_max(local_max)
    local_sum = 0.0
    for j in range(ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        local_sum = local_sum + e
    total_sum = locomp.simd_sum(local_sum)
    for j in range(ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        e = locomp.exp(val - row_max)
        result = e / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)


@locomp.kernel
def online_softmax(X: locomp.Tensor, OUT: locomp.Tensor,
                   ROWS: locomp.constexpr, D: locomp.constexpr,
                   ELEMS: locomp.constexpr):
    row = locomp.program_id(0)
    lid = locomp.local_id(0)
    guard = row < ROWS
    local_max = locomp.load(X + (row * D + lid))
    local_sum = 1.0
    for j in range(1, ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        new_max = locomp.where(val > local_max, val, local_max)
        local_sum = local_sum * locomp.exp(local_max - new_max) + locomp.exp(val - new_max)
        local_max = new_max
    row_max = locomp.simd_max(local_max)
    local_sum = local_sum * locomp.exp(local_max - row_max)
    total_sum = locomp.simd_sum(local_sum)
    for j in range(ELEMS):
        idx = lid + j * 32
        val = locomp.load(X + (row * D + idx))
        result = locomp.exp(val - row_max) / total_sum
        locomp.store(OUT + (row * D + idx), result, mask=guard)


# ─── Bench helpers ───────────────────────────────────────────────────────────

def bench(fn, args, grid, tg):
    for _ in range(WARMUP):
        fn[grid, tg](*args)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn[grid, tg](*args)
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[5]


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("LOCUST BENCHMARK v3 — All Optimizations vs MLX vs PyTorch MPS — Apple M1")
    print(f"Warmup: {WARMUP} | Runs: {RUNS} (median) | All times in ms")
    print("=" * 100)

    # ── MATMUL ──
    print(f"\nMATMUL (float32)")
    print(f"{'Size':>8} | {'Naive':>7} | {'Tiled':>7} | {'Micro':>7} | {'MLX':>7} | {'Torch':>7} | {'Best/MLX':>9}")
    print("-" * 75)

    for sz in MATMUL_SIZES:
        M = N = K = sz
        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)
        a = locomp.tensor(a_np.flatten())
        b = locomp.tensor(b_np.flatten())

        # Naive
        total = M * N
        tpg = min(256, total)
        ng = (total + tpg - 1) // tpg
        c = locomp.empty(M * N)
        t_naive = bench(naive_matmul, [a, b, c, M, N, K], (ng,), (tpg,))

        # Tiled
        nt = K // TILE
        c = locomp.empty(M * N)
        t_tiled = bench(tiled_matmul, [a, b, c, M, N, K, nt, TILE], (N//TILE, M//TILE), (TILE, TILE))

        # Micro-tiled (needs size >= 32)
        if sz >= BM:
            nkt = K // BK
            c = locomp.empty(M * N)
            t_micro = bench(micro_matmul, [a, b, c, M, N, K, nkt, BK, BM, BN],
                           (N//BN, M//BM), (TILE, TILE))
        else:
            t_micro = None

        # MLX
        ma = mx.random.normal((M, K)); mb = mx.random.normal((K, N)); mx.eval(ma, mb)
        for _ in range(WARMUP):
            mc = ma @ mb; mx.eval(mc)
        mlx_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            mc = ma @ mb; mx.eval(mc)
            mlx_times.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mlx_times)[5]

        # Torch MPS
        dev = torch.device("mps")
        ta = torch.randn(M, K, device=dev); tb = torch.randn(K, N, device=dev)
        torch.mps.synchronize()
        for _ in range(WARMUP):
            tc = ta @ tb; torch.mps.synchronize()
        torch_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            tc = ta @ tb; torch.mps.synchronize()
            torch_times.append((time.perf_counter() - t0) * 1000)
        t_torch = sorted(torch_times)[5]

        best = min(t_naive, t_tiled, t_micro or 999)
        ratio = best / t_mlx
        micro_str = f"{t_micro:>6.3f}" if t_micro else "   N/A"
        print(f"{sz:>4}x{sz:<3} | {t_naive:>6.3f} | {t_tiled:>6.3f} | {micro_str} | {t_mlx:>6.3f} | {t_torch:>6.3f} | {ratio:>8.2f}x")

    # ── SOFTMAX ──
    print(f"\nSOFTMAX (float32)")
    print(f"{'Shape':>8} | {'Naive':>7} | {'Tree':>7} | {'SIMD32':>7} | {'Online':>7} | {'MLX':>7} | {'Torch':>7} | {'Best/MLX':>9}")
    print("-" * 90)

    for ROWS, D in SOFTMAX_SHAPES:
        x = locomp.tensor(np.random.randn(ROWS * D).astype(np.float32))

        # Naive
        tpg = min(256, ROWS); ng = (ROWS + tpg - 1) // tpg
        o = locomp.empty(ROWS * D)
        t_naive = bench(naive_softmax, [x, o, ROWS, D], (ng,), (tpg,))

        # Parallel tree
        THREADS = min(256, D)
        LOG_T = int(np.log2(THREADS))
        ELEMS = D // THREADS
        if ELEMS >= 1 and D % THREADS == 0:
            o = locomp.empty(ROWS * D)
            t_tree = bench(parallel_softmax, [x, o, ROWS, D, THREADS, LOG_T, ELEMS],
                          (ROWS,), (THREADS,))
        else:
            t_tree = None

        # SIMD 32
        ELEMS32 = D // 32
        if D % 32 == 0 and ELEMS32 >= 1:
            o = locomp.empty(ROWS * D)
            t_simd = bench(simd_softmax, [x, o, ROWS, D, ELEMS32], (ROWS,), (32,))
        else:
            t_simd = None

        # Online softmax (2-pass, SIMD 32)
        if D % 32 == 0 and ELEMS32 >= 1:
            o = locomp.empty(ROWS * D)
            t_online = bench(online_softmax, [x, o, ROWS, D, ELEMS32], (ROWS,), (32,))
        else:
            t_online = None

        # MLX
        mx_x = mx.random.normal((ROWS, D)); mx.eval(mx_x)
        for _ in range(WARMUP):
            mx_o = mx.softmax(mx_x, axis=1); mx.eval(mx_o)
        mlx_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            mx_o = mx.softmax(mx_x, axis=1); mx.eval(mx_o)
            mlx_times.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mlx_times)[5]

        # Torch
        dev = torch.device("mps")
        tx = torch.randn(ROWS, D, device=dev); torch.mps.synchronize()
        for _ in range(WARMUP):
            to = torch.nn.functional.softmax(tx, dim=1); torch.mps.synchronize()
        torch_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            to = torch.nn.functional.softmax(tx, dim=1); torch.mps.synchronize()
            torch_times.append((time.perf_counter() - t0) * 1000)
        t_torch = sorted(torch_times)[5]

        best = min(t_naive, t_tree or 999, t_simd or 999, t_online or 999)
        ratio = best / t_mlx
        tree_str = f"{t_tree:>6.3f}" if t_tree else "   N/A"
        simd_str = f"{t_simd:>6.3f}" if t_simd else "   N/A"
        online_str = f"{t_online:>6.3f}" if t_online else "   N/A"
        print(f"{ROWS:>3}x{D:<4} | {t_naive:>6.3f} | {tree_str} | {simd_str} | {online_str} | {t_mlx:>6.3f} | {t_torch:>6.3f} | {ratio:>8.2f}x")


if __name__ == "__main__":
    main()
