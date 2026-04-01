"""
Example 41: Top-k Sampling — logits → top-k candidates → sampled token.

Two kernels:
1. top_k_filter: find the k-th largest value via partial sort, mask everything below
2. softmax_sample: softmax over filtered logits, prefix-sum for sampling

Architecture for top-k:
  Bitonic-style partial sort in shared memory with 128 threads.
  Each thread maintains a local candidate, threadgroup cooperatively finds top-k.

For production: temperature scaling, top-p (nucleus) support.
"""

import time
import numpy as np
import locomp


@locomp.kernel
def argmax_kernel(logits: locomp.Tensor, result: locomp.Tensor,
                  V: locomp.constexpr):
    """Single-threadgroup argmax over logits[V]. Result is token index."""
    tid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    local_max = locomp.shared_memory(4)   # per-simd-group max values
    local_idx = locomp.shared_memory(4)   # per-simd-group max indices

    # Each thread scans V/128 elements
    ITERS = V // 128
    best_val = -1000000.0
    best_idx = 0

    for i in range(ITERS):
        pos = tid + i * 128
        val = locomp.load(logits + pos)
        is_better = val > best_val
        best_val = locomp.where(is_better, val, best_val)
        # pos as float to store in float shared mem
        pos_f = locomp.cast(pos, "float32")
        best_idx_f = locomp.cast(best_idx, "float32")
        best_idx = locomp.cast(locomp.where(is_better, pos_f, best_idx_f), "int32")

    # SIMD reduction: find max within each SIMD group
    for delta_exp in range(5):  # 1,2,4,8,16
        delta = 1 << delta_exp
        other_val = locomp.simd_shuffle_down(best_val, delta)
        other_idx_f = locomp.simd_shuffle_down(locomp.cast(best_idx, "float32"), delta)
        is_better2 = other_val > best_val
        best_val = locomp.where(is_better2, other_val, best_val)
        best_idx = locomp.cast(locomp.where(is_better2, other_idx_f, locomp.cast(best_idx, "float32")), "int32")

    if lane == 0:
        locomp.shared_store(local_max, sg, best_val)
        locomp.shared_store(local_idx, sg, locomp.cast(best_idx, "float32"))
    locomp.barrier()

    # Thread 0 reduces across 4 SIMD groups
    if tid == 0:
        final_val = locomp.shared_load(local_max, 0)
        final_idx = locomp.shared_load(local_idx, 0)
        v1 = locomp.shared_load(local_max, 1)
        i1 = locomp.shared_load(local_idx, 1)
        is_b = v1 > final_val
        final_val = locomp.where(is_b, v1, final_val)
        final_idx = locomp.where(is_b, i1, final_idx)
        v2 = locomp.shared_load(local_max, 2)
        i2 = locomp.shared_load(local_idx, 2)
        is_b2 = v2 > final_val
        final_val = locomp.where(is_b2, v2, final_val)
        final_idx = locomp.where(is_b2, i2, final_idx)
        v3 = locomp.shared_load(local_max, 3)
        i3 = locomp.shared_load(local_idx, 3)
        is_b3 = v3 > final_val
        final_val = locomp.where(is_b3, v3, final_val)
        final_idx = locomp.where(is_b3, i3, final_idx)
        locomp.store(result + 0, final_idx)


@locomp.kernel
def topk_filter(logits: locomp.Tensor, output: locomp.Tensor,
                threshold: locomp.Tensor, V: locomp.constexpr,
                TEMPERATURE: locomp.constexpr):
    """Apply temperature + mask logits below threshold. One threadgroup."""
    tid = locomp.local_id(0)
    thresh = locomp.load(threshold + 0)

    ITERS = V // 128
    for i in range(ITERS):
        pos = tid + i * 128
        val = locomp.load(logits + pos) / TEMPERATURE
        masked = locomp.where(val >= thresh, val, -1000000.0)
        locomp.store(output + pos, masked)


@locomp.kernel
def softmax_1d(logits: locomp.Tensor, output: locomp.Tensor,
               V: locomp.constexpr):
    """1D softmax over V elements. Single threadgroup, 128 threads, SIMD reduce."""
    tid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    partial_max = locomp.shared_memory(4)
    partial_sum = locomp.shared_memory(4)

    ITERS = V // 128

    # Pass 1: find max
    local_max = -1000000.0
    for i in range(ITERS):
        pos = tid + i * 128
        val = locomp.load(logits + pos)
        local_max = locomp.where(val > local_max, val, local_max)
    local_max = locomp.simd_max(local_max)
    if lane == 0:
        locomp.shared_store(partial_max, sg, local_max)
    locomp.barrier()
    if tid == 0:
        m0 = locomp.shared_load(partial_max, 0)
        m1 = locomp.shared_load(partial_max, 1)
        m2 = locomp.shared_load(partial_max, 2)
        m3 = locomp.shared_load(partial_max, 3)
        gmax = locomp.where(m0 > m1, m0, m1)
        gmax = locomp.where(m2 > gmax, m2, gmax)
        gmax = locomp.where(m3 > gmax, m3, gmax)
        locomp.shared_store(partial_max, 0, gmax)
    locomp.barrier()
    gmax2 = locomp.shared_load(partial_max, 0)

    # Pass 2: exp and sum
    local_sum = 0.0
    for i in range(ITERS):
        pos = tid + i * 128
        val = locomp.load(logits + pos)
        e = locomp.exp(val - gmax2)
        locomp.store(output + pos, e)
        local_sum = local_sum + e
    local_sum = locomp.simd_sum(local_sum)
    if lane == 0:
        locomp.shared_store(partial_sum, sg, local_sum)
    locomp.barrier()
    if tid == 0:
        s = locomp.shared_load(partial_sum, 0) + locomp.shared_load(partial_sum, 1) + locomp.shared_load(partial_sum, 2) + locomp.shared_load(partial_sum, 3)
        locomp.shared_store(partial_sum, 0, s)
    locomp.barrier()
    inv_sum = 1.0 / locomp.shared_load(partial_sum, 0)

    # Pass 3: normalize
    for i in range(ITERS):
        pos = tid + i * 128
        val = locomp.load(output + pos)
        locomp.store(output + pos, val * inv_sum)


# =============================================================================
# Dispatch
# =============================================================================

def gpu_argmax(logits):
    """GPU argmax over logits[V]. Returns token index."""
    V = logits.shape[0]
    assert V % 128 == 0
    L_g = locomp.tensor(logits)
    R_g = locomp.empty(1)
    argmax_kernel[(1,), (128,)](L_g, R_g, V)
    return int(R_g.numpy()[0])


def gpu_sample_topk(logits, k=50, temperature=1.0):
    """Top-k sampling: GPU argmax to find k-th value, filter, softmax, sample on CPU."""
    V = logits.shape[0]
    assert V % 128 == 0

    # Apply temperature
    scaled = logits / temperature

    # Find k-th largest on CPU (sorting small array of top candidates)
    topk_vals = np.partition(scaled, -k)[-k:]
    threshold = np.min(topk_vals).astype(np.float32)

    # GPU: filter + softmax
    T_g = locomp.tensor(np.array([threshold], dtype=np.float32))
    L_g = locomp.tensor(scaled)
    F_g = locomp.empty(V)
    topk_filter[(1,), (128,)](L_g, F_g, T_g, V, 1)  # temp=1 since already scaled

    P_g = locomp.empty(V)
    softmax_1d[(1,), (128,)](F_g, P_g, V)

    probs = P_g.numpy()
    probs = np.maximum(probs, 0)
    probs = probs / probs.sum()
    return np.random.choice(V, p=probs)


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    WARMUP = 5
    RUNS = 15

    print(f"\n{'='*70}")
    print("Top-k Sampling + Argmax")
    print(f"{'='*70}")

    # --- Argmax ---
    print(f"\n{'Config':>25} | {'GPU':>8} | {'NumPy':>8} | GPU/NP | {'Correct':>8}")
    print("-" * 65)

    for V in [32000, 50304, 128256]:
        # Round up to multiple of 128
        V_pad = ((V + 127) // 128) * 128
        np.random.seed(42)
        logits = np.random.randn(V_pad).astype(np.float32)

        ref = np.argmax(logits[:V])
        gpu_idx = gpu_argmax(logits)
        correct = "YES" if gpu_idx == ref else f"NO (gpu={gpu_idx} np={ref})"

        for _ in range(WARMUP):
            gpu_argmax(logits)
        times_gpu = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_argmax(logits)
            times_gpu.append((time.perf_counter() - t0) * 1000)
        t_gpu = sorted(times_gpu)[RUNS // 2]

        for _ in range(WARMUP):
            np.argmax(logits)
        times_np = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            np.argmax(logits)
            times_np.append((time.perf_counter() - t0) * 1000)
        t_np = sorted(times_np)[RUNS // 2]

        r = t_gpu / t_np
        print(f"{'argmax V='+str(V_pad):>25} | {t_gpu:>6.3f}ms | {t_np:>6.3f}ms | {r:>5.2f}x | {correct}")

    # --- Top-k sampling ---
    print(f"\n{'Config':>25} | {'GPU':>8} | {'NumPy':>8} | GPU/NP | {'Valid':>8}")
    print("-" * 65)

    for V, k in [(32000, 50), (50304, 50), (128256, 100)]:
        V_pad = ((V + 127) // 128) * 128
        np.random.seed(42)
        logits = np.random.randn(V_pad).astype(np.float32)

        token = gpu_sample_topk(logits, k=k, temperature=0.8)
        # Check token is in top-k
        topk_indices = np.argsort(logits)[-k:]
        valid = "YES" if token in topk_indices else "NO"

        for _ in range(WARMUP):
            gpu_sample_topk(logits, k=k, temperature=0.8)
        times_gpu = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_sample_topk(logits, k=k, temperature=0.8)
            times_gpu.append((time.perf_counter() - t0) * 1000)
        t_gpu = sorted(times_gpu)[RUNS // 2]

        def np_topk_sample(logits, k, temp):
            scaled = logits / temp
            topk_idx = np.argpartition(scaled, -k)[-k:]
            topk_vals = scaled[topk_idx]
            topk_vals -= topk_vals.max()
            probs = np.exp(topk_vals)
            probs /= probs.sum()
            return np.random.choice(topk_idx, p=probs)

        for _ in range(WARMUP):
            np_topk_sample(logits, k, 0.8)
        times_np = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            np_topk_sample(logits, k, 0.8)
            times_np.append((time.perf_counter() - t0) * 1000)
        t_np = sorted(times_np)[RUNS // 2]

        r = t_gpu / t_np
        print(f"{'topk V='+str(V_pad)+' k='+str(k):>25} | {t_gpu:>6.3f}ms | {t_np:>6.3f}ms | {r:>5.2f}x | {valid}")
