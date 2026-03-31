"""Quick benchmark: Locust flash attention v3 vs MLX at various N sizes."""
import time, numpy as np, locust
from examples import flash_attention_v1
import importlib, mlx.core as mx

v3_mod = importlib.import_module("examples.17_flash_attention_v3")
flash_attn_v3 = v3_mod.flash_attn_v3
Br, Bc, D = 16, 16, 32

print(f"Flash Attention v3 (constexpr inlined) vs MLX, d={D}")
print(f"{'N':>6} | {'Locust':>8} | {'MLX':>8} | {'Ratio':>7}")
print("-" * 40)

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

    # Locust v3
    flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t, N, D, nkv, Br, Bc)
    for _ in range(3):
        flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t, N, D, nkv, Br, Bc)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        flash_attn_v3[(N // Br,), (Bc, Br)](Q_t, K_t, V_t, O_t, N, D, nkv, Br, Bc)
        times.append((time.perf_counter() - t0) * 1000)
    t_v3 = sorted(times)[5]

    # MLX
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

    ratio = t_v3 / t_mlx
    print(f"{N:>6} | {t_v3:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x")
