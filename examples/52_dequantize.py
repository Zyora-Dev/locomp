"""
Example 52: Standalone Dequantize — INT4/INT8 weights → float32.

Serving primitive: load quantized weight tensors from disk/memory
and dequantize them on GPU before compute. Used when you want to
decouple dequantization from the matmul (e.g., prefetch + dequant
while previous layer's matmul is running).

INT4: packed as uint8 (2 values/byte), symmetric quantization.
  float_val = (nibble - 8) * scale

INT8: stored as int8, symmetric quantization.
  float_val = int8_val * scale

Both use per-group scales (GROUP_SIZE typically 32 or 128).
"""

import time
import numpy as np
import locomp


# =============================================================================
# Dequant INT4: uint8[N, K//2] → float32[N, K]
# =============================================================================

@locomp.kernel
def dequant_int4(W: locomp.UInt8, SCALES: locomp.Tensor, OUT: locomp.Tensor,
                 K: locomp.constexpr, HALF_K: locomp.constexpr,
                 NUM_SCALES: locomp.constexpr, GROUP_SIZE: locomp.constexpr):
    row = locomp.program_id(0)       # output row
    tid = locomp.local_id(0)         # thread ID

    ITERS = HALF_K // 128   # each thread unpacks 1 byte per iter

    for i in range(ITERS):
        byte_idx = tid + i * 128
        packed = locomp.load(W + (row * HALF_K + byte_idx))

        low = packed & 15
        high = packed >> 4

        k_low = byte_idx * 2
        k_high = byte_idx * 2 + 1

        scale_low = locomp.load(SCALES + (row * NUM_SCALES + k_low // GROUP_SIZE))
        scale_high = locomp.load(SCALES + (row * NUM_SCALES + k_high // GROUP_SIZE))

        f_low = (locomp.cast(low, "float32") - 8.0) * scale_low
        f_high = (locomp.cast(high, "float32") - 8.0) * scale_high

        locomp.store(OUT + (row * K + k_low), f_low)
        locomp.store(OUT + (row * K + k_high), f_high)


# =============================================================================
# Dequant INT8: int8[N, K] → float32[N, K]
# =============================================================================

@locomp.kernel
def dequant_int8(W: locomp.Int8, SCALES: locomp.Tensor, OUT: locomp.Tensor,
                 K: locomp.constexpr, NUM_SCALES: locomp.constexpr,
                 GROUP_SIZE: locomp.constexpr):
    row = locomp.program_id(0)
    tid = locomp.local_id(0)

    ITERS = K // 128

    for i in range(ITERS):
        k = tid + i * 128
        w_int = locomp.load(W + (row * K + k))
        scale = locomp.load(SCALES + (row * NUM_SCALES + k // GROUP_SIZE))
        f_val = locomp.cast(w_int, "float32") * scale
        locomp.store(OUT + (row * K + k), f_val)


# =============================================================================
# Dispatch helpers
# =============================================================================

def gpu_dequant_int4(w_packed, scales, N, K, group_size=32):
    """Dequantize INT4 packed weights. w_packed:[N, K//2] uint8, scales:[N, K//GS]."""
    HALF_K = K // 2
    NUM_SCALES = K // group_size

    W_g = locomp.tensor(w_packed.flatten())
    S_g = locomp.tensor(scales.flatten())
    O_g = locomp.empty(N * K)

    dequant_int4[(N,), (128,)](W_g, S_g, O_g, K, HALF_K, NUM_SCALES, group_size)
    result = O_g.numpy().reshape(N, K)
    W_g.free(); S_g.free(); O_g.free()
    return result


def gpu_dequant_int8(w_int8, scales, N, K, group_size=32):
    """Dequantize INT8 weights. w_int8:[N, K] int8, scales:[N, K//GS]."""
    NUM_SCALES = K // group_size

    W_g = locomp.tensor(w_int8.flatten())
    S_g = locomp.tensor(scales.flatten())
    O_g = locomp.empty(N * K)

    dequant_int8[(N,), (128,)](W_g, S_g, O_g, K, NUM_SCALES, group_size)
    result = O_g.numpy().reshape(N, K)
    W_g.free(); S_g.free(); O_g.free()
    return result


def quantize_int4_ref(w, group_size=32):
    """Quantize float32 → INT4 (symmetric, per-group scales). Returns packed uint8 + scales."""
    N, K = w.shape
    assert K % group_size == 0
    num_groups = K // group_size
    w_grouped = w.reshape(N, num_groups, group_size)
    absmax = np.abs(w_grouped).max(axis=2, keepdims=True)
    absmax = np.maximum(absmax, 1e-10)
    scales = absmax.reshape(N, num_groups) / 7.0  # range [-7, 7] mapped to [-8, 7] symmetric

    # Quantize: q = round(w / scale) + 8, clamp to [0, 15]
    q = np.round(w_grouped / (scales.reshape(N, num_groups, 1) + 1e-10)) + 8
    q = np.clip(q, 0, 15).astype(np.uint8).reshape(N, K)

    # Pack: two nibbles per byte
    packed = (q[:, 1::2] << 4) | q[:, 0::2]
    return packed.astype(np.uint8), scales.astype(np.float32)


def dequant_int4_ref(packed, scales, N, K, group_size=32):
    """Reference INT4 dequantize in numpy."""
    low = (packed & 0xF).astype(np.float32) - 8.0
    high = (packed >> 4).astype(np.float32) - 8.0
    # Interleave: low is even indices, high is odd
    result = np.zeros((N, K), dtype=np.float32)
    result[:, 0::2] = low
    result[:, 1::2] = high
    # Scale per group
    num_groups = K // group_size
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        result[:, start:end] *= scales[:, g:g+1]
    return result


if __name__ == "__main__":
    np.random.seed(42)
    GS = 32

    print("=== Dequantize INT4 ===")
    for N, K in [(64, 256), (128, 512), (256, 1024)]:
        w = np.random.randn(N, K).astype(np.float32) * 0.5
        packed, scales = quantize_int4_ref(w, GS)
        expected = dequant_int4_ref(packed, scales, N, K, GS)
        out = gpu_dequant_int4(packed, scales, N, K, GS)
        np.testing.assert_allclose(out, expected, atol=1e-5)
        print(f"  [{N}×{K}] group_size={GS} ✓")

    print("\n=== Dequantize INT8 ===")
    for N, K in [(64, 256), (128, 512), (256, 1024)]:
        w = np.random.randn(N, K).astype(np.float32) * 0.5
        # Quantize to INT8
        absmax = np.abs(w.reshape(N, K // GS, GS)).max(axis=2)
        scales = (absmax / 127.0).astype(np.float32)
        w_int8 = np.clip(np.round(w / (np.repeat(scales, GS, axis=1) + 1e-10)), -128, 127).astype(np.int8)
        expected = w_int8.astype(np.float32) * np.repeat(scales, GS, axis=1)

        out = gpu_dequant_int8(w_int8, scales, N, K, GS)
        np.testing.assert_allclose(out, expected, atol=1e-5)
        print(f"  [{N}×{K}] group_size={GS} ✓")

    print("\nAll dequantize tests passed.")
