"""
Benchmark tiled matmul vs naive, with reduced dispatch overhead.
Pre-allocates buffers, reuses them, measures GPU-only time.
"""
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


def raw_dispatch(kernel_launcher, args, grid, tg):
    """Dispatch without readback — time only the GPU compute."""
    from locomp.backends.metal_runtime import get_runtime
    from locomp.api import LocustTensor
    runtime = get_runtime()
    kernel_launcher._compile()
    if kernel_launcher._pipeline is None:
        kernel_launcher._pipeline = runtime.compile_msl(
            kernel_launcher._msl_source, kernel_launcher.func_name)

    buffers = []
    for arg in args:
        if isinstance(arg, LocustTensor):
            buffers.append(arg.to_metal_buffer(runtime))
        elif isinstance(arg, (int, float)):
            buffers.append(runtime.allocate_int_buffer(int(arg)))

    g = grid + (1,) * (3 - len(grid))
    t = tg + (1,) * (3 - len(tg))

    t0 = time.perf_counter()
    runtime.dispatch(kernel_launcher._pipeline, buffers, grid=g, threadgroup_size=t)
    return (time.perf_counter() - t0) * 1000


print(f"{'Size':>8} | {'Naive':>10} | {'Tiled':>10} | {'Speedup':>8} | {'Error':>12}")
print("-" * 65)

for size in [16, 32, 64, 128, 256, 512, 1024]:
    M = N = K = size
    if size < TILE:
        continue

    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    a = locomp.tensor(a_np.flatten())
    b = locomp.tensor(b_np.flatten())
    c = locomp.empty(M * N)

    # Naive params
    total = M * N
    tpg = min(256, total)
    ng = (total + tpg - 1) // tpg

    # Tiled params
    nt = K // TILE
    tgrid = (N // TILE, M // TILE)
    tg_size = (TILE, TILE)

    # Warmup (3 runs each)
    for _ in range(3):
        raw_dispatch(naive_matmul, [a, b, c, M, N, K], (ng,), (tpg,))
        raw_dispatch(tiled_matmul, [a, b, c, M, N, K, nt, TILE], tgrid, tg_size)

    # Benchmark naive (10 runs, median)
    naive_times = []
    for _ in range(10):
        t = raw_dispatch(naive_matmul, [a, b, c, M, N, K], (ng,), (tpg,))
        naive_times.append(t)
    naive_med = sorted(naive_times)[5]

    # Benchmark tiled (10 runs, median)
    tiled_times = []
    for _ in range(10):
        t = raw_dispatch(tiled_matmul, [a, b, c, M, N, K, nt, TILE], tgrid, tg_size)
        tiled_times.append(t)
    tiled_med = sorted(tiled_times)[5]

    # Verify correctness (one run with readback)
    c_verify = locomp.empty(M * N)
    tiled_matmul[tgrid, tg_size](a, b, c_verify, M, N, K, nt, TILE)
    result = c_verify.numpy().reshape(M, N)
    expected = a_np @ b_np
    err = np.max(np.abs(result - expected))
    speedup = naive_med / tiled_med

    print(f"{size:>4}x{size} | {naive_med:>8.3f}ms | {tiled_med:>8.3f}ms | {speedup:>7.2f}x | {err:.8f}")
