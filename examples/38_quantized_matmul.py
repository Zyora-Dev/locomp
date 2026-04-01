"""
Example 38: Quantized Matrix-Vector Multiply — INT4 and INT8.

Model-agnostic dequantizing matvec kernels for Apple Metal.
These are the core primitives for quantized LLM inference (token generation).

INT4: weights packed as uint8 (2 values/byte), symmetric quantization.
INT8: weights stored as int8, symmetric quantization.
Both use per-group scales (GROUP_SIZE typically 32 or 128).

Architecture:
  One threadgroup per output element (row of W).
  128 threads/group → parallelize the dot product over K.
  SIMD reduction + shared memory for final summation.

  Dequantization:
    INT4: nibble = packed >> shift & 0xF; w = (float(nibble) - 8) * scale
    INT8: w = float(w_int8) * scale
"""

import time
import numpy as np
import locomp


# =============================================================================
# INT4 dequantizing matvec — y[n] = sum_k(x[k] * dequant_int4(W[n,k]))
# =============================================================================

@locomp.kernel
def dequant_matvec_int4(
    X: locomp.Tensor,       # [K] float32 input vector
    W: locomp.UInt8,         # [N, K//2] uint8 packed INT4 weights
    scales: locomp.Tensor,   # [N, K//GROUP_SIZE] float32 per-group scales
    O: locomp.Tensor,        # [N] float32 output
    K_dim: locomp.constexpr,
    HALF_K: locomp.constexpr,       # K // 2
    NUM_SCALES: locomp.constexpr,   # K // GROUP_SIZE
    GROUP_SIZE: locomp.constexpr,
):
    n = locomp.program_id(0)        # output index
    tid = locomp.local_id(0)        # thread in group [0..127]
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    partial = locomp.shared_memory(4)  # one float per SIMD group

    acc = 0.0

    # Each thread processes 2 elements per iteration (one packed byte)
    # 128 threads × 2 values = 256 values per iteration
    # Total iterations = K / 256
    ITERS = K_dim // 256

    for i in range(ITERS):
        byte_offset = tid + i * 128
        packed = locomp.load(W + (n * HALF_K + byte_offset))

        low = packed & 15
        high = packed >> 4

        k_low = byte_offset * 2
        k_high = byte_offset * 2 + 1

        scale_low = locomp.load(scales + (n * NUM_SCALES + k_low // GROUP_SIZE))
        scale_high = locomp.load(scales + (n * NUM_SCALES + k_high // GROUP_SIZE))

        w_low = (locomp.cast(low, "float32") - 8.0) * scale_low
        w_high = (locomp.cast(high, "float32") - 8.0) * scale_high

        x_low = locomp.load(X + k_low)
        x_high = locomp.load(X + k_high)

        acc = acc + x_low * w_low + x_high * w_high

    # SIMD reduction within each SIMD group
    acc = locomp.simd_sum(acc)

    # Write per-simd-group partial sum to shared memory
    if lane == 0:
        locomp.shared_store(partial, sg, acc)
    locomp.barrier()

    # Thread 0 reduces 4 partial sums and writes output
    if tid == 0:
        total = locomp.shared_load(partial, 0) + locomp.shared_load(partial, 1) + locomp.shared_load(partial, 2) + locomp.shared_load(partial, 3)
        locomp.store(O + n, total)


# =============================================================================
# INT8 dequantizing matvec — y[n] = sum_k(x[k] * dequant_int8(W[n,k]))
# =============================================================================

@locomp.kernel
def dequant_matvec_int8(
    X: locomp.Tensor,       # [K] float32 input vector
    W: locomp.Int8,          # [N, K] int8 weights
    scales: locomp.Tensor,   # [N, K//GROUP_SIZE] float32 per-group scales
    O: locomp.Tensor,        # [N] float32 output
    K_dim: locomp.constexpr,
    NUM_SCALES: locomp.constexpr,
    GROUP_SIZE: locomp.constexpr,
):
    n = locomp.program_id(0)
    tid = locomp.local_id(0)
    lane = locomp.simd_lane_id()
    sg = locomp.simd_group_id()

    partial = locomp.shared_memory(4)

    acc = 0.0

    # Each thread processes K/128 elements
    ITERS = K_dim // 128

    for i in range(ITERS):
        k = tid + i * 128

        w_int = locomp.load(W + (n * K_dim + k))
        scale = locomp.load(scales + (n * NUM_SCALES + k // GROUP_SIZE))
        w_val = locomp.cast(w_int, "float32") * scale
        x_val = locomp.load(X + k)

        acc = acc + x_val * w_val

    acc = locomp.simd_sum(acc)

    if lane == 0:
        locomp.shared_store(partial, sg, acc)
    locomp.barrier()

    if tid == 0:
        total = locomp.shared_load(partial, 0) + locomp.shared_load(partial, 1) + locomp.shared_load(partial, 2) + locomp.shared_load(partial, 3)
        locomp.store(O + n, total)


# =============================================================================
# Unified dispatch
# =============================================================================

def gpu_matvec_int4(x, w_packed, scales, group_size=32):
    """INT4 quantized matvec. x:[K], w_packed:[N,K//2] uint8, scales:[N,K//G]."""
    K = x.shape[0]
    N = w_packed.shape[0]
    assert K % 256 == 0, f"K must be multiple of 256, got {K}"
    half_k = K // 2
    num_scales = K // group_size

    X_g = locomp.tensor(x)
    W_g = locomp.tensor(w_packed)
    S_g = locomp.tensor(scales)
    O_g = locomp.empty(N)

    dequant_matvec_int4[(N,), (128,)](X_g, W_g, S_g, O_g, K, half_k, num_scales, group_size)
    return O_g.numpy()


def gpu_matvec_int8(x, w_int8, scales, group_size=32):
    """INT8 quantized matvec. x:[K], w_int8:[N,K] int8, scales:[N,K//G]."""
    K = x.shape[0]
    N = w_int8.shape[0]
    assert K % 128 == 0, f"K must be multiple of 128, got {K}"
    num_scales = K // group_size

    X_g = locomp.tensor(x)
    W_g = locomp.tensor(w_int8)
    S_g = locomp.tensor(scales)
    O_g = locomp.empty(N)

    dequant_matvec_int8[(N,), (128,)](X_g, W_g, S_g, O_g, K, num_scales, group_size)
    return O_g.numpy()


# =============================================================================
# NumPy reference implementations
# =============================================================================

def numpy_matvec_int4(x, w_packed, scales, group_size=32):
    """Reference INT4 dequant matvec."""
    N, half_k = w_packed.shape
    K = half_k * 2
    num_scales = K // group_size
    output = np.zeros(N, dtype=np.float32)

    for n_idx in range(N):
        acc = 0.0
        for byte_i in range(half_k):
            packed = w_packed[n_idx, byte_i]
            low = int(packed) & 0xF
            high = int(packed) >> 4
            k_low = byte_i * 2
            k_high = byte_i * 2 + 1
            s_low = scales[n_idx, k_low // group_size]
            s_high = scales[n_idx, k_high // group_size]
            w_low = (float(low) - 8.0) * s_low
            w_high = (float(high) - 8.0) * s_high
            acc += x[k_low] * w_low + x[k_high] * w_high
        output[n_idx] = acc
    return output


def numpy_matvec_int8(x, w_int8, scales, group_size=32):
    """Reference INT8 dequant matvec."""
    N, K = w_int8.shape
    output = np.zeros(N, dtype=np.float32)
    for n_idx in range(N):
        acc = 0.0
        for k in range(K):
            w_val = float(w_int8[n_idx, k]) * scales[n_idx, k // group_size]
            acc += x[k] * w_val
        output[n_idx] = acc
    return output


def fast_numpy_matvec_int4(x, w_packed, scales, group_size=32):
    """Vectorized NumPy INT4 dequant matvec (for benchmarking)."""
    N, half_k = w_packed.shape
    K = half_k * 2

    low = (w_packed.astype(np.int32) & 0xF).astype(np.float32) - 8.0
    high = (w_packed.astype(np.int32) >> 4).astype(np.float32) - 8.0

    # Interleave low,high → [N, K]
    w_full = np.empty((N, K), dtype=np.float32)
    w_full[:, 0::2] = low
    w_full[:, 1::2] = high

    # Apply per-group scales
    scales_expanded = np.repeat(scales, group_size, axis=1)[:, :K]
    w_dequant = w_full * scales_expanded

    return w_dequant @ x


def fast_numpy_matvec_int8(x, w_int8, scales, group_size=32):
    """Vectorized NumPy INT8 dequant matvec (for benchmarking)."""
    N, K = w_int8.shape
    w_float = w_int8.astype(np.float32)
    scales_expanded = np.repeat(scales, group_size, axis=1)[:, :K]
    w_dequant = w_float * scales_expanded
    return w_dequant @ x


# =============================================================================
# Quantization helpers
# =============================================================================

def quantize_int4(weights, group_size=32):
    """Quantize float32 weights to INT4 symmetric. Returns (packed_uint8, scales)."""
    N, K = weights.shape
    assert K % group_size == 0
    num_groups = K // group_size

    scales = np.zeros((N, num_groups), dtype=np.float32)
    packed = np.zeros((N, K // 2), dtype=np.uint8)

    for n_idx in range(N):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = weights[n_idx, start:end]
            amax = np.max(np.abs(group))
            scale = amax / 7.0 if amax > 0 else 1.0
            scales[n_idx, g] = scale
            quantized = np.clip(np.round(group / scale) + 8, 0, 15).astype(np.uint8)

            for j in range(0, group_size, 2):
                byte_idx = (start + j) // 2
                packed[n_idx, byte_idx] = quantized[j] | (quantized[j + 1] << 4)

    return packed, scales


def quantize_int8(weights, group_size=32):
    """Quantize float32 weights to INT8 symmetric. Returns (int8_weights, scales)."""
    N, K = weights.shape
    assert K % group_size == 0
    num_groups = K // group_size

    scales = np.zeros((N, num_groups), dtype=np.float32)
    w_int8 = np.zeros((N, K), dtype=np.int8)

    for n_idx in range(N):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = weights[n_idx, start:end]
            amax = np.max(np.abs(group))
            scale = amax / 127.0 if amax > 0 else 1.0
            scales[n_idx, g] = scale
            w_int8[n_idx, start:end] = np.clip(np.round(group / scale), -128, 127).astype(np.int8)

    return w_int8, scales


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    WARMUP = 5
    RUNS = 15
    GROUP_SIZE = 32

    print(f"\n{'='*75}")
    print("Quantized Matvec: INT4 and INT8 (dequantizing on-the-fly)")
    print(f"{'='*75}")
    print(f"{'Config':>35} | {'GPU':>8} | {'NumPy':>8} | GPU/NP | {'Error':>10}")
    print("-" * 78)

    # --- INT4 tests ---
    for N, K in [(256, 256), (512, 512), (1024, 1024), (4096, 4096)]:
        np.random.seed(42)
        w_float = np.random.randn(N, K).astype(np.float32) * 0.1
        x = np.random.randn(K).astype(np.float32) * 0.1
        w_packed, scales = quantize_int4(w_float, GROUP_SIZE)

        # Correctness: element-wise reference
        ref = numpy_matvec_int4(x, w_packed, scales, GROUP_SIZE)
        gpu_out = gpu_matvec_int4(x, w_packed, scales, GROUP_SIZE)
        err = np.max(np.abs(gpu_out - ref))

        # Benchmark GPU vs vectorized NumPy
        for _ in range(WARMUP):
            gpu_matvec_int4(x, w_packed, scales, GROUP_SIZE)
        times_gpu = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_matvec_int4(x, w_packed, scales, GROUP_SIZE)
            times_gpu.append((time.perf_counter() - t0) * 1000)
        t_gpu = sorted(times_gpu)[RUNS // 2]

        for _ in range(WARMUP):
            fast_numpy_matvec_int4(x, w_packed, scales, GROUP_SIZE)
        times_np = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            fast_numpy_matvec_int4(x, w_packed, scales, GROUP_SIZE)
            times_np.append((time.perf_counter() - t0) * 1000)
        t_np = sorted(times_np)[RUNS // 2]

        r = t_gpu / t_np
        print(f"{'INT4 N='+str(N)+' K='+str(K):>35} | {t_gpu:>6.3f}ms | {t_np:>6.3f}ms | {r:>5.2f}x | {err:.2e}")

    # --- INT8 tests ---
    for N, K in [(256, 256), (512, 512), (1024, 1024), (4096, 4096)]:
        np.random.seed(42)
        w_float = np.random.randn(N, K).astype(np.float32) * 0.1
        x = np.random.randn(K).astype(np.float32) * 0.1
        w_int8, scales = quantize_int8(w_float, GROUP_SIZE)

        ref = numpy_matvec_int8(x, w_int8, scales, GROUP_SIZE)
        gpu_out = gpu_matvec_int8(x, w_int8, scales, GROUP_SIZE)
        err = np.max(np.abs(gpu_out - ref))

        for _ in range(WARMUP):
            gpu_matvec_int8(x, w_int8, scales, GROUP_SIZE)
        times_gpu = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_matvec_int8(x, w_int8, scales, GROUP_SIZE)
            times_gpu.append((time.perf_counter() - t0) * 1000)
        t_gpu = sorted(times_gpu)[RUNS // 2]

        for _ in range(WARMUP):
            fast_numpy_matvec_int8(x, w_int8, scales, GROUP_SIZE)
        times_np = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            fast_numpy_matvec_int8(x, w_int8, scales, GROUP_SIZE)
            times_np.append((time.perf_counter() - t0) * 1000)
        t_np = sorted(times_np)[RUNS // 2]

        r = t_gpu / t_np
        print(f"{'INT8 N='+str(N)+' K='+str(K):>35} | {t_gpu:>6.3f}ms | {t_np:>6.3f}ms | {r:>5.2f}x | {err:.2e}")
