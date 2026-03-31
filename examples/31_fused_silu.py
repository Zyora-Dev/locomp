"""
Example 31: Fused SiLU (SwiGLU) — the FFN activation in LLaMA/Mistral.

SwiGLU: out = SiLU(gate) * up = (gate * sigmoid(gate)) * up
Fuses two loads, sigmoid, multiply, and store into one kernel.
Used in LLaMA, Mistral, Gemma, Qwen — all modern LLMs.
"""

import time
import numpy as np
import locomp


@locomp.kernel
def fused_silu_mul(Gate: locomp.Tensor, Up: locomp.Tensor, OUT: locomp.Tensor,
                   N: locomp.constexpr):
    pid = locomp.program_id(0)
    tid = locomp.local_id(0)
    idx = pid * 256 + tid

    gate = locomp.load(Gate + idx)
    up = locomp.load(Up + idx)

    # SiLU(gate) * up = gate * sigmoid(gate) * up
    silu_gate = gate * locomp.sigmoid(gate)
    locomp.store(OUT + idx, silu_gate * up)


def silu_mul_np(gate, up):
    """Reference SwiGLU in numpy."""
    return gate * (1.0 / (1.0 + np.exp(-gate))) * up


if __name__ == "__main__":
    import mlx.core as mx
    import mlx.nn as nn

    sizes = [4096, 11008, 14336, 32768, 65536, 131072]
    # 11008 = LLaMA-7B FFN, 14336 = LLaMA-13B/Mistral, rest = stress tests

    WARMUP = 5; RUNS = 15
    THREADS = 256

    print("Fused SiLU×Up (SwiGLU): element-wise, 256 threads/group")
    print(f"{'N':>10} | {'Locomp':>8} | {'MLX':>8} | {'Ratio':>7} | {'Error':>8}")
    print("-" * 55)

    for N in sizes:
        # Round up to multiple of 256
        N_padded = ((N + 255) // 256) * 256
        label = str(N)

        np.random.seed(42)
        gate_np = np.random.randn(N_padded).astype(np.float32)
        up_np = np.random.randn(N_padded).astype(np.float32)
        expected = silu_mul_np(gate_np[:N], up_np[:N])

        gate_t = locomp.tensor(gate_np)
        up_t = locomp.tensor(up_np)
        o_t = locomp.empty(N_padded)

        grid = (N_padded // THREADS,)
        tg = (THREADS,)

        # Correctness
        fused_silu_mul[grid, tg](gate_t, up_t, o_t, N_padded)
        result = o_t.numpy()[:N]
        err = np.max(np.abs(result - expected))

        # Benchmark locomp
        for _ in range(WARMUP):
            fused_silu_mul[grid, tg](gate_t, up_t, o_t, N_padded)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            fused_silu_mul[grid, tg](gate_t, up_t, o_t, N_padded)
            times.append((time.perf_counter() - t0) * 1000)
        t_loc = sorted(times)[RUNS // 2]

        # MLX: silu(gate) * up — MLX doesn't have fused SwiGLU, compose it
        gate_mlx = mx.array(gate_np[:N])
        up_mlx = mx.array(up_np[:N])
        mx.eval(gate_mlx, up_mlx)
        for _ in range(WARMUP):
            out = nn.silu(gate_mlx) * up_mlx
            mx.eval(out)
        mt = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = nn.silu(gate_mlx) * up_mlx
            mx.eval(out)
            mt.append((time.perf_counter() - t0) * 1000)
        t_mlx = sorted(mt)[RUNS // 2]

        ratio = t_loc / t_mlx
        print(f"{label:>10} | {t_loc:>6.3f}ms | {t_mlx:>6.3f}ms | {ratio:>6.2f}x | {err:.2e}")
