"""
Profile simdgroup flash attention: GPU kernel time vs wall-clock to identify the real bottleneck.
"""
import time
import numpy as np
import locomp
from locomp.backends.metal_runtime import get_runtime

Br = 16
Bc = 16
D_HEAD = 32
NUM_LOADS = 4

# Import the kernel
import importlib
mod = importlib.import_module("examples.22_simdgroup_flash_attn")
flash_attn_simd = mod.flash_attn_simd

d = D_HEAD
runtime = get_runtime()

print(f"Profiling simdgroup flash attention: GPU kernel time vs wall-clock")
print(f"{'N':>6} | {'Wall':>8} | {'GPU':>8} | {'Dispatch':>8} | {'MLX':>8} | GPU/MLX | Wall/MLX")
print("-" * 78)

import mlx.core as mx

for N in [64, 128, 256, 512]:
    np.random.seed(42)
    Q_np = np.random.randn(N, d).astype(np.float32) * 0.1
    K_np = np.random.randn(N, d).astype(np.float32) * 0.1
    V_np = np.random.randn(N, d).astype(np.float32) * 0.1

    Q_t = locomp.tensor(Q_np.flatten())
    K_t = locomp.tensor(K_np.flatten())
    V_t = locomp.tensor(V_np.flatten())
    O_t = locomp.empty(N * d)
    nkv = N // Bc

    grid_x = N // Br

    # Warmup
    for _ in range(5):
        flash_attn_simd[(grid_x,), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)

    # Wall-clock timing
    wall_times = []
    for _ in range(15):
        t0 = time.perf_counter()
        flash_attn_simd[(grid_x,), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        wall_times.append((time.perf_counter() - t0) * 1000)
    t_wall = sorted(wall_times)[7]

    # GPU kernel timing via dispatch_repeat
    constexpr_key = tuple(sorted({
        "N_0": N, "D_1": d, "NUM_KV_BLOCKS_2": nkv, "BLOCK_R_3": Br, "BLOCK_C_4": Bc
    }.items()))

    # Access internals to get pipeline/buffers
    # First ensure compiled
    flash_attn_simd[(grid_x,), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
    gpu_start = runtime._last_gpu_start
    gpu_end = runtime._last_gpu_end
    t_single_gpu = (gpu_end - gpu_start) * 1000

    # Multiple single-dispatch GPU measurements
    gpu_times = []
    for _ in range(15):
        flash_attn_simd[(grid_x,), (32, 4)](Q_t, K_t, V_t, O_t, N, d, nkv, Br, Bc)
        gpu_times.append((runtime._last_gpu_end - runtime._last_gpu_start) * 1000)
    t_gpu = sorted(gpu_times)[7]

    dispatch_overhead = t_wall - t_gpu

    # MLX
    mQ = mx.array(Q_np); mK = mx.array(K_np); mV = mx.array(V_np)
    mQ4 = mQ.reshape(1, N, 1, d); mK4 = mK.reshape(1, N, 1, d); mV4 = mV.reshape(1, N, 1, d)
    mx.eval(mQ4, mK4, mV4)
    for _ in range(5):
        out = mx.fast.scaled_dot_product_attention(mQ4, mK4, mV4, scale=1.0/np.sqrt(d))
        mx.eval(out)
    mt = []
    for _ in range(15):
        t0 = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(mQ4, mK4, mV4, scale=1.0/np.sqrt(d))
        mx.eval(out)
        mt.append((time.perf_counter() - t0) * 1000)
    t_mlx = sorted(mt)[7]

    print(f"{N:>6} | {t_wall:>6.3f}ms | {t_gpu:>6.3f}ms | {dispatch_overhead:>6.3f}ms | {t_mlx:>6.3f}ms | {t_gpu/t_mlx:>6.2f}x | {t_wall/t_mlx:>6.2f}x")
