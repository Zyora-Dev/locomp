"""
Benchmark new kernels (44-53) — locomp vs MLX 0.31
Apple M1, float32, median of 10 runs after 3 warmup.
"""

import time
import numpy as np
import locomp
import mlx.core as mx
import importlib.util
import sys
import os

# Add examples to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

WARMUP = 3
RUNS = 10

def median_time(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return sorted(times)[len(times) // 2]

def load_example(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# 1. TRANSPOSE
# ============================================================
print("=" * 60)
print("TRANSPOSE 2D")
print("=" * 60)

ex44 = load_example("ex44", "examples/44_transpose.py")

for shape in [(64, 128), (256, 512), (512, 1024)]:
    R, C = shape
    data = np.random.randn(R, C).astype(np.float32)

    def run_locomp(d=data):
        return ex44.gpu_transpose_2d(d)
    t_locomp = median_time(run_locomp)

    mx_data = mx.array(data)
    def run_mlx(m=mx_data):
        out = mx.transpose(m)
        mx.eval(out)
        return out
    t_mlx = median_time(run_mlx)

    ratio = t_locomp / t_mlx
    print(f"  [{R}×{C}]  locomp={t_locomp:.3f}ms  MLX={t_mlx:.3f}ms  ratio={ratio:.2f}×")


# ============================================================
# 2. REDUCE SUM
# ============================================================
print("\n" + "=" * 60)
print("REDUCE SUM (last axis)")
print("=" * 60)

ex45 = load_example("ex45", "examples/45_reduce.py")

for ROWS, D in [(32, 128), (64, 256), (128, 512), (32, 1024)]:
    data = np.random.randn(ROWS, D).astype(np.float32)

    def run_lc(d=data):
        return ex45.gpu_reduce_sum(d)
    t_lc = median_time(run_lc)

    mx_data = mx.array(data)
    def run_mx(m=mx_data):
        out = mx.sum(m, axis=-1)
        mx.eval(out)
        return out
    t_mx = median_time(run_mx)

    ratio = t_lc / t_mx
    print(f"  [{ROWS}×{D}]  locomp={t_lc:.3f}ms  MLX={t_mx:.3f}ms  ratio={ratio:.2f}×")


# ============================================================
# 3. GATHER (row indexing)
# ============================================================
print("\n" + "=" * 60)
print("GATHER (row indexing)")
print("=" * 60)

ex47 = load_example("ex47", "examples/47_scatter_gather.py")

for rows, cols, num_idx in [(256, 128, 64), (1024, 256, 128), (4096, 128, 256)]:
    table = np.random.randn(rows, cols).astype(np.float32)
    indices = np.random.randint(0, rows, size=num_idx).astype(np.int32)

    def run_lc(t=table, i=indices):
        return ex47.gpu_gather_rows(t, i)
    t_lc = median_time(run_lc)

    mx_table = mx.array(table)
    mx_idx = mx.array(indices)
    def run_mx(t=mx_table, i=mx_idx):
        out = t[i]
        mx.eval(out)
        return out
    t_mx = median_time(run_mx)

    ratio = t_lc / t_mx
    print(f"  [{rows}×{cols}] idx={num_idx}  locomp={t_lc:.3f}ms  MLX={t_mx:.3f}ms  ratio={ratio:.2f}×")


# ============================================================
# 4. CONCAT (axis=0)
# ============================================================
print("\n" + "=" * 60)
print("CONCAT (axis=0)")
print("=" * 60)

ex48 = load_example("ex48", "examples/48_concat_split.py")

for R, C in [(128, 256), (256, 512), (512, 256)]:
    a = np.random.randn(R, C).astype(np.float32)
    b = np.random.randn(R, C).astype(np.float32)

    def run_lc(a_=a, b_=b):
        return ex48.gpu_concat_first(a_, b_)
    t_lc = median_time(run_lc)

    mx_a, mx_b = mx.array(a), mx.array(b)
    def run_mx(a_=mx_a, b_=mx_b):
        out = mx.concatenate([a_, b_], axis=0)
        mx.eval(out)
        return out
    t_mx = median_time(run_mx)

    ratio = t_lc / t_mx
    print(f"  [{R}×{C}]+[{R}×{C}]  locomp={t_lc:.3f}ms  MLX={t_mx:.3f}ms  ratio={ratio:.2f}×")


# ============================================================
# 5. BATCH NORM (inference)
# ============================================================
print("\n" + "=" * 60)
print("BATCH NORM (inference)")
print("=" * 60)

ex49 = load_example("ex49", "examples/49_batch_norm.py")

for N, C, H, W in [(1, 64, 8, 8), (4, 128, 4, 4), (8, 256, 4, 4)]:
    x = np.random.randn(N, C, H, W).astype(np.float32)
    gamma = np.ones(C, dtype=np.float32)
    beta = np.zeros(C, dtype=np.float32)
    rmean = np.random.randn(C).astype(np.float32) * 0.1
    rvar = np.abs(np.random.randn(C).astype(np.float32)) + 0.5
    eps = 1e-5

    def run_lc(x_=x, g=gamma, b=beta, m=rmean, v=rvar):
        return ex49.gpu_batch_norm_infer(x_, g, b, m, v)
    t_lc = median_time(run_lc)

    mx_x = mx.array(x.reshape(N, H, W, C))  # MLX uses NHWC
    mx_g = mx.array(gamma)
    mx_b = mx.array(beta)
    mx_m = mx.array(rmean)
    mx_v = mx.array(rvar)
    def run_mx(x_=mx_x, g=mx_g, b=mx_b, m=mx_m, v=mx_v):
        out = g * (x_ - m) / mx.sqrt(v + eps) + b
        mx.eval(out)
        return out
    t_mx = median_time(run_mx)

    ratio = t_lc / t_mx
    print(f"  N={N} C={C} H={H} W={W}  locomp={t_lc:.3f}ms  MLX={t_mx:.3f}ms  ratio={ratio:.2f}×")


# ============================================================
# 6. AVG POOL 2D
# ============================================================
print("\n" + "=" * 60)
print("AVG POOL 2D")
print("=" * 60)

ex50 = load_example("ex50", "examples/50_pooling.py")

for N, C, H, W, K in [(1, 1, 32, 32, 2), (1, 1, 64, 64, 2), (1, 1, 128, 128, 4)]:
    data = np.random.randn(N, C, H, W).astype(np.float32)

    def run_lc(d=data, k=K):
        return ex50.gpu_avg_pool2d(d, (k, k), (k, k))
    t_lc = median_time(run_lc)

    mx_data = mx.array(data.reshape(N, H, W, C))  # NHWC for MLX
    def run_mx(m=mx_data, h=H, w=W, k=K, c=C, n=N):
        r = mx.reshape(m, (n, h//k, k, w//k, k, c))
        out = mx.mean(r, axis=(2, 4))
        mx.eval(out)
        return out
    t_mx = median_time(run_mx)

    ratio = t_lc / t_mx
    print(f"  [{H}×{W}] K={K}  locomp={t_lc:.3f}ms  MLX={t_mx:.3f}ms  ratio={ratio:.2f}×")


# ============================================================
# 7. CROSS ATTENTION
# ============================================================
print("\n" + "=" * 60)
print("CROSS ATTENTION")
print("=" * 60)

ex51 = load_example("ex51", "examples/51_cross_attention.py")

for B, H, Nq, Nkv, D in [(1, 2, 1, 16, 32), (1, 4, 4, 32, 32), (1, 8, 1, 64, 32)]:
    q = np.random.randn(B, H, Nq, D).astype(np.float32) * 0.1
    k = np.random.randn(B, H, Nkv, D).astype(np.float32) * 0.1
    v = np.random.randn(B, H, Nkv, D).astype(np.float32) * 0.1

    def run_lc(q_=q, k_=k, v_=v):
        return ex51.gpu_cross_attention(q_, k_, v_)
    t_lc = median_time(run_lc)

    mx_q, mx_k, mx_v = mx.array(q), mx.array(k), mx.array(v)
    scale = 1.0 / np.sqrt(D)
    def run_mx(q_=mx_q, k_=mx_k, v_=mx_v, s=scale):
        scores = (q_ @ mx.transpose(k_, axes=(0,1,3,2))) * s
        scores = mx.softmax(scores, axis=-1)
        out = scores @ v_
        mx.eval(out)
        return out
    t_mx = median_time(run_mx)

    ratio = t_lc / t_mx
    print(f"  B={B} H={H} Nq={Nq} Nkv={Nkv} D={D}  locomp={t_lc:.3f}ms  MLX={t_mx:.3f}ms  ratio={ratio:.2f}×")


# ============================================================
# 8. DEQUANTIZE INT4
# ============================================================
print("\n" + "=" * 60)
print("DEQUANTIZE INT4 (vs NumPy CPU)")
print("=" * 60)

ex52 = load_example("ex52", "examples/52_dequantize.py")

for N, K in [(64, 256), (256, 1024), (512, 2048)]:
    GS = 32
    w = np.random.randn(N, K).astype(np.float32) * 0.5
    packed, scales = ex52.quantize_int4_ref(w, GS)

    def run_lc(p=packed, s=scales, n=N, k=K):
        return ex52.gpu_dequant_int4(p, s, n, k, GS)
    t_lc = median_time(run_lc)

    def run_np(p=packed, s=scales, n=N, k=K):
        return ex52.dequant_int4_ref(p, s, n, k, GS)
    t_np = median_time(run_np)

    ratio = t_lc / t_np
    print(f"  [{N}×{K}] group=32  locomp={t_lc:.3f}ms  NumPy={t_np:.3f}ms  ratio={ratio:.2f}×")


# ============================================================
# 9. BROADCAST (element-wise add)
# ============================================================
print("\n" + "=" * 60)
print("ELEMENT-WISE ADD")
print("=" * 60)

ex53 = load_example("ex53", "examples/53_broadcast.py")

for N in [1024, 4096, 16384, 65536]:
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)

    def run_lc(a_=a, b_=b):
        return ex53.gpu_ewise_add(a_, b_)
    t_lc = median_time(run_lc)

    mx_a, mx_b = mx.array(a), mx.array(b)
    def run_mx(a_=mx_a, b_=mx_b):
        out = a_ + b_
        mx.eval(out)
        return out
    t_mx = median_time(run_mx)

    ratio = t_lc / t_mx
    print(f"  N={N}  locomp={t_lc:.3f}ms  MLX={t_mx:.3f}ms  ratio={ratio:.2f}×")


# ============================================================
# 10. ROW BROADCAST ADD (bias add)
# ============================================================
print("\n" + "=" * 60)
print("ROW BROADCAST ADD (bias add)")
print("=" * 60)

for R, C in [(64, 128), (256, 512), (512, 1024)]:
    x = np.random.randn(R, C).astype(np.float32)
    bias = np.random.randn(C).astype(np.float32)

    def run_lc(x_=x, b=bias):
        return ex53.gpu_broadcast_row_add(x_, b)
    t_lc = median_time(run_lc)

    mx_x = mx.array(x)
    mx_bias = mx.array(bias)
    def run_mx(x_=mx_x, b=mx_bias):
        out = x_ + b
        mx.eval(out)
        return out
    t_mx = median_time(run_mx)

    ratio = t_lc / t_mx
    print(f"  [{R}×{C}]+[{C}]  locomp={t_lc:.3f}ms  MLX={t_mx:.3f}ms  ratio={ratio:.2f}×")


print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
