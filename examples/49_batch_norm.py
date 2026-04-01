"""
Example 49: Batch Normalization — for vision models.

  out = gamma * (x - mean) / sqrt(var + eps) + beta

Used in ResNet, EfficientNet, YOLO, MobileNet, etc.
Training computes per-channel statistics from the batch;
inference uses running mean/var precomputed during training.

Architecture:
  Inference mode: one threadgroup per channel (C).
  Each thread handles multiple spatial elements (B × H × W).
  Input: [N, C, HW] (channels-first, spatial flattened).
"""

import time
import numpy as np
import locomp


# =============================================================================
# BatchNorm inference: uses precomputed running_mean, running_var
# =============================================================================

@locomp.kernel
def batch_norm_infer(X: locomp.Tensor, GAMMA: locomp.Tensor, BETA: locomp.Tensor,
                     MEAN: locomp.Tensor, VAR: locomp.Tensor,
                     OUT: locomp.Tensor,
                     N: locomp.constexpr, C: locomp.constexpr,
                     HW: locomp.constexpr, TOTAL: locomp.constexpr,
                     THREADS: locomp.constexpr, ELEMS: locomp.constexpr):
    c = locomp.program_id(0)      # channel index
    tid = locomp.local_id(0)

    # Load per-channel params
    gamma = locomp.load(GAMMA + c)
    beta = locomp.load(BETA + c)
    mean = locomp.load(MEAN + c)
    var = locomp.load(VAR + c)
    inv_std = locomp.rsqrt(var + 1e-5)

    scale = gamma * inv_std
    shift = beta - mean * scale

    # Process all spatial positions across batch for this channel
    for e in range(ELEMS):
        idx = tid + e * THREADS
        if idx < TOTAL:
            # Linear index within channel c: idx = n * HW + hw
            n = idx // HW
            hw = idx % HW
            src = n * C * HW + c * HW + hw
            dst = src  # same layout in/out
            val = locomp.load(X + src)
            locomp.store(OUT + dst, val * scale + shift)


# =============================================================================
# BatchNorm forward (training): compute mean/var from batch, then normalize
# =============================================================================

@locomp.kernel
def batch_norm_train_stats(X: locomp.Tensor, OUT_MEAN: locomp.Tensor,
                           OUT_VAR: locomp.Tensor,
                           N: locomp.constexpr, C: locomp.constexpr,
                           HW: locomp.constexpr, TOTAL: locomp.constexpr,
                           THREADS: locomp.constexpr, ELEMS: locomp.constexpr,
                           NUM_SIMD: locomp.constexpr):
    c = locomp.program_id(0)
    tid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    smem_sum = locomp.shared_memory(NUM_SIMD)
    smem_sq = locomp.shared_memory(NUM_SIMD)

    local_sum = 0.0
    local_sq = 0.0

    for e in range(ELEMS):
        idx = tid + e * THREADS
        if idx < TOTAL:
            n = idx // HW
            hw = idx % HW
            src = n * C * HW + c * HW + hw
            val = locomp.load(X + src)
            local_sum = local_sum + val
            local_sq = local_sq + val * val

    # SIMD reduce
    local_sum = locomp.simd_sum(local_sum)
    local_sq = locomp.simd_sum(local_sq)

    if lane == 0:
        locomp.shared_store(smem_sum, sg, local_sum)
        locomp.shared_store(smem_sq, sg, local_sq)
    locomp.barrier()

    if tid == 0:
        total_sum = 0.0
        total_sq = 0.0
        for g in range(NUM_SIMD):
            total_sum = total_sum + locomp.shared_load(smem_sum, g)
            total_sq = total_sq + locomp.shared_load(smem_sq, g)
        mean = total_sum / TOTAL
        var = total_sq / TOTAL - mean * mean
        locomp.store(OUT_MEAN + c, mean)
        locomp.store(OUT_VAR + c, var)


# =============================================================================
# Dispatch helpers
# =============================================================================

def gpu_batch_norm_infer(x, gamma, beta, running_mean, running_var):
    """BN inference. x: [N, C, H, W], gamma/beta/mean/var: [C]."""
    N, C, H, W = x.shape
    HW = H * W
    TOTAL = N * HW   # elements per channel
    THREADS = min(128, TOTAL)
    ELEMS = (TOTAL + THREADS - 1) // THREADS

    X_g = locomp.tensor(x.flatten())
    G_g = locomp.tensor(gamma)
    B_g = locomp.tensor(beta)
    M_g = locomp.tensor(running_mean)
    V_g = locomp.tensor(running_var)
    O_g = locomp.empty(x.size)

    batch_norm_infer[(C,), (THREADS,)](X_g, G_g, B_g, M_g, V_g, O_g,
                                        N, C, HW, TOTAL, THREADS, ELEMS)
    result = O_g.numpy().reshape(N, C, H, W)
    X_g.free(); G_g.free(); B_g.free(); M_g.free(); V_g.free(); O_g.free()
    return result


def batch_norm_np(x, gamma, beta, mean, var, eps=1e-5):
    """Reference BN in numpy. x: [N, C, H, W]."""
    # Reshape [C] → [1, C, 1, 1] for broadcasting
    g = gamma.reshape(1, -1, 1, 1)
    b = beta.reshape(1, -1, 1, 1)
    m = mean.reshape(1, -1, 1, 1)
    v = var.reshape(1, -1, 1, 1)
    return g * (x - m) / np.sqrt(v + eps) + b


if __name__ == "__main__":
    np.random.seed(42)

    print("=== BatchNorm Inference ===")
    for N, C, H, W in [(1, 16, 8, 8), (2, 32, 16, 16), (4, 64, 8, 8), (1, 128, 4, 4)]:
        x = np.random.randn(N, C, H, W).astype(np.float32)
        gamma = np.random.randn(C).astype(np.float32) * 0.5 + 1.0
        beta = np.random.randn(C).astype(np.float32) * 0.1
        running_mean = np.random.randn(C).astype(np.float32) * 0.5
        running_var = np.abs(np.random.randn(C).astype(np.float32)) + 0.1

        out = gpu_batch_norm_infer(x, gamma, beta, running_mean, running_var)
        expected = batch_norm_np(x, gamma, beta, running_mean, running_var)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
        print(f"  [{N}×{C}×{H}×{W}] ✓")

    print("\nAll batch norm tests passed.")
