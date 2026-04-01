"""
Example 50: Average Pooling and Max Pooling — for vision models.

2D pooling over spatial dimensions (H, W) with stride.
Used in: ResNet (avg pool before FC), YOLO (max pool downscale),
  MobileNet, EfficientNet, sentence embeddings (pool over tokens).

Architecture:
  One thread per output element.
  Grid: (OW, OH) per (N, C) — each thread reads a KxK window.
  Input: [N, C, H, W] channels-first.
"""

import time
import numpy as np
import locomp


# =============================================================================
# Average Pool 2D: [N, C, H, W] → [N, C, OH, OW]
# =============================================================================

@locomp.kernel
def avg_pool2d(X: locomp.Tensor, OUT: locomp.Tensor,
               C: locomp.constexpr, H: locomp.constexpr, W: locomp.constexpr,
               KH: locomp.constexpr, KW: locomp.constexpr,
               SH: locomp.constexpr, SW: locomp.constexpr,
               OH: locomp.constexpr, OW: locomp.constexpr):
    # program_id(0) = output spatial (oh * OW + ow)
    # program_id(1) = n * C + c
    spatial = locomp.program_id(0)
    nc = locomp.program_id(1)

    oh = spatial // OW
    ow = spatial % OW
    n = nc // C
    c = nc % C

    # Input base for this (n, c) feature map
    base = n * C * H * W + c * H * W

    acc = 0.0
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh
            iw = ow * SW + kw
            acc = acc + locomp.load(X + (base + ih * W + iw))

    avg = acc / (KH * KW)
    out_base = n * C * OH * OW + c * OH * OW
    locomp.store(OUT + (out_base + oh * OW + ow), avg)


# =============================================================================
# Max Pool 2D: [N, C, H, W] → [N, C, OH, OW]
# =============================================================================

@locomp.kernel
def max_pool2d(X: locomp.Tensor, OUT: locomp.Tensor,
               C: locomp.constexpr, H: locomp.constexpr, W: locomp.constexpr,
               KH: locomp.constexpr, KW: locomp.constexpr,
               SH: locomp.constexpr, SW: locomp.constexpr,
               OH: locomp.constexpr, OW: locomp.constexpr):
    spatial = locomp.program_id(0)
    nc = locomp.program_id(1)

    oh = spatial // OW
    ow = spatial % OW
    n = nc // C
    c = nc % C

    base = n * C * H * W + c * H * W

    m = -1e30
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh
            iw = ow * SW + kw
            val = locomp.load(X + (base + ih * W + iw))
            m = locomp.max(m, val)

    out_base = n * C * OH * OW + c * OH * OW
    locomp.store(OUT + (out_base + oh * OW + ow), m)


# =============================================================================
# Global Average Pool: [N, C, H, W] → [N, C]  (pool entire spatial)
# =============================================================================

@locomp.kernel
def global_avg_pool(X: locomp.Tensor, OUT: locomp.Tensor,
                    C: locomp.constexpr, HW: locomp.constexpr,
                    THREADS: locomp.constexpr, ELEMS: locomp.constexpr,
                    NUM_SIMD: locomp.constexpr):
    nc = locomp.program_id(0)  # n * C + c
    tid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    smem = locomp.shared_memory(NUM_SIMD)
    base = nc * HW

    acc = 0.0
    for e in range(ELEMS):
        idx = tid + e * THREADS
        if idx < HW:
            acc = acc + locomp.load(X + (base + idx))

    acc = locomp.simd_sum(acc)
    if lane == 0:
        locomp.shared_store(smem, sg, acc)
    locomp.barrier()

    if tid == 0:
        total = 0.0
        for g in range(NUM_SIMD):
            total = total + locomp.shared_load(smem, g)
        locomp.store(OUT + nc, total / HW)


# =============================================================================
# Dispatch helpers
# =============================================================================

def gpu_avg_pool2d(x, kernel_size, stride):
    N, C, H, W = x.shape
    KH, KW = kernel_size
    SH, SW = stride
    OH = (H - KH) // SH + 1
    OW = (W - KW) // SW + 1

    X_g = locomp.tensor(x.flatten())
    O_g = locomp.empty(N * C * OH * OW)
    avg_pool2d[(OH * OW, N * C)](X_g, O_g, C, H, W, KH, KW, SH, SW, OH, OW)
    result = O_g.numpy().reshape(N, C, OH, OW)
    X_g.free(); O_g.free()
    return result


def gpu_max_pool2d(x, kernel_size, stride):
    N, C, H, W = x.shape
    KH, KW = kernel_size
    SH, SW = stride
    OH = (H - KH) // SH + 1
    OW = (W - KW) // SW + 1

    X_g = locomp.tensor(x.flatten())
    O_g = locomp.empty(N * C * OH * OW)
    max_pool2d[(OH * OW, N * C)](X_g, O_g, C, H, W, KH, KW, SH, SW, OH, OW)
    result = O_g.numpy().reshape(N, C, OH, OW)
    X_g.free(); O_g.free()
    return result


def gpu_global_avg_pool(x):
    N, C, H, W = x.shape
    HW = H * W
    THREADS = min(128, HW)
    ELEMS = (HW + THREADS - 1) // THREADS
    NUM_SIMD = max(1, THREADS // 32)

    X_g = locomp.tensor(x.flatten())
    O_g = locomp.empty(N * C)
    global_avg_pool[(N * C,), (THREADS,)](X_g, O_g, C, HW, THREADS, ELEMS, NUM_SIMD)
    result = O_g.numpy().reshape(N, C)
    X_g.free(); O_g.free()
    return result


def avg_pool2d_np(x, kernel_size, stride):
    N, C, H, W = x.shape
    KH, KW = kernel_size
    SH, SW = stride
    OH = (H - KH) // SH + 1
    OW = (W - KW) // SW + 1
    out = np.zeros((N, C, OH, OW), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    patch = x[n, c, oh*SH:oh*SH+KH, ow*SW:ow*SW+KW]
                    out[n, c, oh, ow] = patch.mean()
    return out


def max_pool2d_np(x, kernel_size, stride):
    N, C, H, W = x.shape
    KH, KW = kernel_size
    SH, SW = stride
    OH = (H - KH) // SH + 1
    OW = (W - KW) // SW + 1
    out = np.zeros((N, C, OH, OW), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    patch = x[n, c, oh*SH:oh*SH+KH, ow*SW:ow*SW+KW]
                    out[n, c, oh, ow] = patch.max()
    return out


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Avg Pool 2D ===")
    for N, C, H, W, K, S in [(1, 4, 8, 8, 2, 2), (2, 16, 16, 16, 3, 2), (1, 32, 32, 32, 2, 2)]:
        x = np.random.randn(N, C, H, W).astype(np.float32)
        out = gpu_avg_pool2d(x, (K, K), (S, S))
        expected = avg_pool2d_np(x, (K, K), (S, S))
        np.testing.assert_allclose(out, expected, rtol=1e-4)
        print(f"  [{N}×{C}×{H}×{W}] k={K} s={S} ✓")

    print("\n=== Max Pool 2D ===")
    for N, C, H, W, K, S in [(1, 4, 8, 8, 2, 2), (2, 16, 16, 16, 3, 2), (1, 32, 32, 32, 2, 2)]:
        x = np.random.randn(N, C, H, W).astype(np.float32)
        out = gpu_max_pool2d(x, (K, K), (S, S))
        expected = max_pool2d_np(x, (K, K), (S, S))
        np.testing.assert_allclose(out, expected, rtol=1e-5)
        print(f"  [{N}×{C}×{H}×{W}] k={K} s={S} ✓")

    print("\n=== Global Average Pool ===")
    for N, C, H, W in [(1, 16, 8, 8), (2, 64, 4, 4), (4, 128, 2, 2)]:
        x = np.random.randn(N, C, H, W).astype(np.float32)
        out = gpu_global_avg_pool(x)
        expected = x.mean(axis=(2, 3))
        np.testing.assert_allclose(out, expected, rtol=1e-4)
        print(f"  [{N}×{C}×{H}×{W}] → [{N}×{C}] ✓")

    print("\nAll pooling tests passed.")
