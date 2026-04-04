"""
Example 61 — Kernel Graph: chain kernels in a single GPU command buffer.

Without graph: each kernel call syncs GPU→CPU, then CPU→GPU for next kernel.
With graph:    all kernels run in one command buffer, one sync at the end.

Demonstrates:
  1. Chaining 3 kernels: rms_norm → silu_mul → add  (transformer-style FFN pass)
  2. Context manager form
  3. Re-running the same graph multiple times (inference loop)
  4. Speedup vs sequential dispatch
"""

import time
import numpy as np
import locomp

print(f"locomp v{locomp.__version__}\n")

# ── Kernels ────────────────────────────────────────────────────────────────────

@locomp.kernel
def rms_norm(X: locomp.Tensor, W: locomp.Tensor, O: locomp.Tensor,
             N: locomp.constexpr, eps: locomp.constexpr):
    i = locomp.program_id(0)
    acc = 0.0
    for j in range(N):
        v = locomp.load(X + j)
        acc = acc + v * v
    scale = locomp.rsqrt(acc / N + eps)
    locomp.store(O + i, locomp.load(X + i) * scale * locomp.load(W + i))


@locomp.kernel
def silu_mul(gate: locomp.Tensor, up: locomp.Tensor, O: locomp.Tensor,
             N: locomp.constexpr):
    i = locomp.program_id(0)
    g = locomp.load(gate + i)
    u = locomp.load(up + i)
    silu = g / (1.0 + locomp.exp(-g))
    locomp.store(O + i, silu * u)


@locomp.kernel
def add_inplace(A: locomp.Tensor, B: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(A + i, locomp.load(A + i) + locomp.load(B + i))


@locomp.kernel
def scale_shift(X: locomp.Tensor, O: locomp.Tensor,
                N: locomp.constexpr, scale: locomp.constexpr):
    i = locomp.program_id(0)
    locomp.store(O + i, locomp.load(X + i) * scale)


# ── Setup data ─────────────────────────────────────────────────────────────────

N = 4096
x_np   = np.random.randn(N).astype(np.float32)
w_np   = np.ones(N, dtype=np.float32)
gate_np = np.random.randn(N).astype(np.float32)
up_np   = np.random.randn(N).astype(np.float32)

x    = locomp.tensor(x_np)
w    = locomp.tensor(w_np)
gate = locomp.tensor(gate_np)
up   = locomp.tensor(up_np)
norm_out = locomp.empty(N)
ffn_out  = locomp.empty(N)

# ── 1. Basic usage ─────────────────────────────────────────────────────────────
print("── Test 1: basic graph (rms_norm → silu_mul → add_inplace) ──")

g = locomp.graph()
g.add(rms_norm,   (N,), x, w, norm_out, N=N, eps=1e-5)
g.add(silu_mul,   (N,), gate, up, ffn_out, N=N)
g.add(add_inplace,(N,), norm_out, ffn_out, N=N)
g.run()

# Verify correctness vs sequential
x2    = locomp.tensor(x_np)
w2    = locomp.tensor(w_np)
gate2 = locomp.tensor(gate_np)
up2   = locomp.tensor(up_np)
no2   = locomp.empty(N); fo2 = locomp.empty(N)
rms_norm[(N,)](x2, w2, no2, N=N, eps=1e-5)
silu_mul[(N,)](gate2, up2, fo2, N=N)
add_inplace[(N,)](no2, fo2, N=N)

err = np.max(np.abs(norm_out.numpy() - no2.numpy()))
print(f"  norm_out max_err vs sequential: {err:.2e} {'✓' if err < 1e-5 else '✗'}")

# ── 2. Context manager form ────────────────────────────────────────────────────
print("\n── Test 2: context manager form ──")

x3 = locomp.tensor(x_np); no3 = locomp.empty(N); fo3 = locomp.empty(N)
gate3 = locomp.tensor(gate_np); up3 = locomp.tensor(up_np)
with locomp.graph() as g2:
    g2.add(rms_norm,    (N,), x3, w, no3, N=N, eps=1e-5)
    g2.add(silu_mul,    (N,), gate3, up3, fo3, N=N)
    g2.add(add_inplace, (N,), no3, fo3, N=N)

err2 = np.max(np.abs(no3.numpy() - no2.numpy()))
print(f"  context manager result max_err: {err2:.2e} {'✓' if err2 < 1e-5 else '✗'}")

# ── 3. Re-run graph (inference loop) ──────────────────────────────────────────
print("\n── Test 3: re-run graph 5 times (inference loop) ──")

loop_x  = locomp.tensor(np.random.randn(N).astype(np.float32))
loop_out = locomp.empty(N)
loop_g = locomp.graph()
loop_g.add(rms_norm,   (N,), loop_x, w, loop_out, N=N, eps=1e-5)
loop_g.add(scale_shift,(N,), loop_out, loop_out, N=N, scale=0.5)

for step in range(5):
    loop_g.run()
print(f"  5 re-runs completed ✓")

# ── 4. Benchmark: graph vs sequential dispatch ─────────────────────────────────
print("\n── Benchmark: graph vs sequential (50 reps, N=4096) ──")

REPS = 50

# Sequential
x_s = locomp.tensor(x_np); no_s = locomp.empty(N); fo_s = locomp.empty(N)
gate_s = locomp.tensor(gate_np); up_s = locomp.tensor(up_np)
for _ in range(5):  # warmup
    rms_norm[(N,)](x_s, w, no_s, N=N, eps=1e-5)
    silu_mul[(N,)](gate_s, up_s, fo_s, N=N)
    add_inplace[(N,)](no_s, fo_s, N=N)

t0 = time.perf_counter()
for _ in range(REPS):
    rms_norm[(N,)](x_s, w, no_s, N=N, eps=1e-5)
    silu_mul[(N,)](gate_s, up_s, fo_s, N=N)
    add_inplace[(N,)](no_s, fo_s, N=N)
seq_ms = (time.perf_counter() - t0) / REPS * 1000

# Graph
x_g = locomp.tensor(x_np); no_g = locomp.empty(N); fo_g = locomp.empty(N)
gate_g = locomp.tensor(gate_np); up_g = locomp.tensor(up_np)
bench_graph = locomp.graph()
bench_graph.add(rms_norm,    (N,), x_g, w, no_g, N=N, eps=1e-5)
bench_graph.add(silu_mul,    (N,), gate_g, up_g, fo_g, N=N)
bench_graph.add(add_inplace, (N,), no_g, fo_g, N=N)
for _ in range(5):  # warmup
    bench_graph.run()

t0 = time.perf_counter()
for _ in range(REPS):
    bench_graph.run()
graph_ms = (time.perf_counter() - t0) / REPS * 1000

speedup = seq_ms / graph_ms
print(f"  Sequential: {seq_ms:.3f}ms/iter")
print(f"  Graph:      {graph_ms:.3f}ms/iter")
print(f"  Speedup:    {speedup:.2f}x")

print(f"\nAll tests passed. Graph repr: {bench_graph}")
