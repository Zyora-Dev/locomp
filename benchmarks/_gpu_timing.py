import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np, locust
from locomp.backends.metal_runtime import get_runtime
import importlib

v3_mod = importlib.import_module("examples.17_flash_attention_v3")
flash_attn_v3 = v3_mod.flash_attn_v3
Br, Bc, D = 16, 16, 32
runtime = get_runtime()

print("GPU Kernel Time (batch dispatch, no Python overhead)")
print("N     | GPU/iter  | WallClock | MLX wall  | GPU/MLX")
print("-" * 55)

import mlx.core as mx

for N in [64, 128, 256, 512, 1024, 2048]:
    np.random.seed(42)
    Q_np = np.random.randn(N, D).astype(np.float32) * 0.1
    K_np = np.random.randn(N, D).astype(np.float32) * 0.1
    V_np = np.random.randn(N, D).astype(np.float32) * 0.1
    Q_t = locomp.tensor(Q_np.flatten())
    K_t = locomp.tensor(K_np.flatten())
    V_t = locomp.tensor(V_np.flatten())
    O_t = locomp.empty(N * D)
    nkv = N // Bc

    # Warmup (triggers specialization + pipeline compile)
    flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t, N, D, nkv, Br, Bc)

    # Get the specialized pipeline — find the matching key
    # The constexpr key is stored during dispatch
    assert len(flash_attn_v3._specialized) > 0, "No specialized pipeline found"
    constexpr_key = list(flash_attn_v3._specialized.keys())[-1]
    pipeline, buffer_map, msl = flash_attn_v3._specialized[constexpr_key]
    buffers = [
        Q_t.to_metal_buffer(runtime),
        K_t.to_metal_buffer(runtime),
        V_t.to_metal_buffer(runtime),
        O_t.to_metal_buffer(runtime),
    ]

    # Batch GPU timing
    gpu_ms = runtime.dispatch_repeat(pipeline, buffers, (N // Br,), (Bc, Br), repeat=50)

    # Wall-clock for comparison
    for _ in range(3):
        flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t, N, D, nkv, Br, Bc)
    wall_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t, N, D, nkv, Br, Bc)
        wall_times.append((time.perf_counter() - t0) * 1000)
    wall = sorted(wall_times)[5]

    # MLX wall-clock
    mQ = mx.array(Q_np); mK = mx.array(K_np); mV = mx.array(V_np)
    mx.eval(mQ, mK, mV)
    for _ in range(3):
        out = mx.softmax((mQ @ mK.T) * (1.0 / np.sqrt(D)), axis=-1) @ mV; mx.eval(out)
    mt = []
    for _ in range(10):
        t0 = time.perf_counter()
        out = mx.softmax((mQ @ mK.T) * (1.0 / np.sqrt(D)), axis=-1) @ mV; mx.eval(out)
        mt.append((time.perf_counter() - t0) * 1000)
    t_mlx = sorted(mt)[5]

    ratio = gpu_ms / t_mlx
    print(f"{N:>5} | {gpu_ms:>7.3f}ms | {wall:>7.3f}ms | {t_mlx:>7.3f}ms | {ratio:>6.2f}x")
