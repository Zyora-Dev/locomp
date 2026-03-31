"""
Example 32: Fused GELU + Bias — FFN activation for GPT-2/BERT style models.

Fuses bias add + GELU activation into one kernel:
  out = GELU(x + bias)
  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""

import time
import numpy as np
import locomp


@locomp.kernel
def fused_gelu_bias(X: locomp.Tensor, Bias: locomp.Tensor, OUT: locomp.Tensor,
                    N: locomp.constexpr):
    pid = locomp.program_id(0)
    tid = locomp.local_id(0)
    idx = pid * 256 + tid

    x = locomp.load(X + idx) + locomp.load(Bias + (idx % N))

    # GELU approximation (tanh version)
    # 0.7978845608 = sqrt(2/pi)
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    out = 0.5 * x * (1.0 + locomp.tanh(inner))
    locomp.store(OUT + idx, out)


def gelu_bias_np(x, bias):
    """Reference GELU + bias in numpy."""
    x = x + bias
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


if __name__ == "__main__":
    import mlx.core as mx
    import mlx.nn as nn

    configs = [
        (32, 768),     # GPT-2 small
        (32, 1024),    # GPT-2 medium
        (32, 3072),    # GPT-2 small FFN (4× hidden)
        (32, 4096),    # GPT-2 medium FFN / LLaMA hidden
        (128, 4096),   # Large batch
    ]

    WARMUP = 5; RUNS = 15
    THREADS = 256

    print("Fused GELU+Bias: tanh approximation, 256 threads/group")
    print(f"{'Config':>15} | {'Locomp':>8} | {'MLX':>8} | {'Ratio':>7} | {'Error':>8}")
    print("-" * 60)

    for rows, D in configs:
        total = rows * D
        total_padded = ((total + 255) // 256) * 256
        label = f"{rows}×{D}"

        np.random.seed(42)
        x_np = np.random.randn(rows, D).astype(np.float32)
        b_np = np.random.randn(D).astype(np.float32)
        expected = gelu_bias_np(x_np, b_np)

        x_t = locomp.tensor(np.pad(x_np.flatten(), (0, total_padded - total)))
        b_t = locomp.tensor(b_np)
        o_t = locomp.empty(total_padded)

        grid = (total_padded // THREADS,)
        tg = (THREADS,)

        # Correctness
        fused_gelu_bias[grid, tg](x_t, b_t, o_t, D)
        result = o_t.numpy()[:total].reshape(rows, D)
        err = np.max(np.abs(result - expected))

        # Benchmark locomp
        for _ in range(WARMUP):
            fused_gelu_bias[grid, tg](x_t, b_t, o_t, D)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            fused_gelu_bias[grid, tg](x_t, b_t, o_t, D)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[RUNS // 2]

        # MLX: gelu(x + bias)
        x_mlx = mx.array(x_np)
        b_mlx = mx.array(b_np)
        mx.eval(x_mlx, b_mlx)
        for _ in range(WARMUP):
            out = nn.gelu_approx(x_mlx + b_mlx)
            mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = nn.gelu_approx(x_mlx + b_mlx)
            mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[RUNS // 2]

        ratio = t_loc / t_mlx
        print(f"{label:>15} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
