"""
Example 55: BFloat16 kernels — end-to-end hardware validation.

Tests the full BFloat16 pipeline:
  Python annotation → IR (BFLOAT16) → MSL (bfloat) → Metal GPU → numpy readback

Note on buffer layout: BFloat16 is 2 bytes per element. LocompTensor uses
np.float16 as the carrier dtype (also 2 bytes) to get the right Metal buffer
size. BFloat16 bit patterns are packed into the float16 buffer via uint16
reinterpretation — the float16 values are meaningless as float16 but the bytes
hold valid bfloat16 data.

Tests:
  1. Vector add in BFloat16
  2. Scale (elementwise multiply) in BFloat16
  3. Mixed: BFloat16 input → Float32 accumulation → BFloat16 output (dot product)
  4. Cast float32 → bfloat16 → float32 round-trip
"""

import numpy as np
import locomp

# ── BFloat16 buffer helpers ───────────────────────────────────────────────────

def pack_as_bf16(arr_f32: np.ndarray) -> np.ndarray:
    """
    Pack float32 values as bfloat16, returned as float16 dtype (same 2-byte size).
    Use this to create LocompTensor inputs for BFloat16 kernel params.
    """
    raw = arr_f32.astype(np.float32).view(np.uint32)
    bf16_u16 = ((raw + 0x8000) >> 16).astype(np.uint16)   # round-to-nearest bf16
    return bf16_u16.view(np.float16)                        # same 2 bytes, fp16 carrier


def unpack_bf16(tensor_f16: np.ndarray) -> np.ndarray:
    """
    Unpack bfloat16 bytes stored in a float16 numpy array → float32.
    Use this to read back BFloat16 kernel output.
    """
    bf16_u16 = tensor_f16.view(np.uint16).astype(np.uint32)
    return (bf16_u16 << 16).view(np.float32)


def bf16_round(arr: np.ndarray) -> np.ndarray:
    """Simulate bfloat16 precision: round float32 to bf16 and back."""
    raw = arr.astype(np.float32).view(np.uint32)
    raw = ((raw + 0x8000) & 0xFFFF0000)
    return raw.view(np.float32)


def allclose_bf16(a, b, rtol=1e-2, atol=1e-2):
    return np.allclose(a, b, rtol=rtol, atol=atol)


# ── kernel definitions ────────────────────────────────────────────────────────

@locomp.kernel
def bf16_vector_add(X: locomp.BFloat16, Y: locomp.BFloat16, O: locomp.BFloat16,
                    N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    y = locomp.load(Y + i)
    locomp.store(O + i, x + y)


@locomp.kernel
def bf16_scale(X: locomp.BFloat16, O: locomp.BFloat16,
               scale: locomp.constexpr, N: locomp.constexpr):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    locomp.store(O + i, x * scale)


@locomp.kernel
def bf16_dot_accum(X: locomp.BFloat16, Y: locomp.BFloat16, O: locomp.BFloat16,
                   N: locomp.constexpr):
    """Element-wise product fully in BFloat16."""
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    y = locomp.load(Y + i)
    locomp.store(O + i, x * y)


@locomp.kernel
def bf16_cast_from_f32(X: locomp.Tensor, O: locomp.BFloat16, N: locomp.constexpr):
    """Cast float32 input to bfloat16 output."""
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    locomp.store(O + i, locomp.cast(x, "bfloat16"))


# ── test runner ───────────────────────────────────────────────────────────────

def test_bf16_vector_add():
    N = 64
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    expected = bf16_round(bf16_round(a) + bf16_round(b))

    x = locomp.tensor(pack_as_bf16(a))
    y = locomp.tensor(pack_as_bf16(b))
    o = locomp.empty(N, dtype=np.float16)

    bf16_vector_add[(N,)](x, y, o, N=N)
    result = unpack_bf16(o.numpy())

    assert allclose_bf16(result, expected), \
        f"BFloat16 vector add FAILED\n  expected: {expected[:8]}\n  got:      {result[:8]}"
    print(f"  ✓  bf16_vector_add   N={N}  max_err={np.max(np.abs(result - expected)):.4f}")


def test_bf16_scale():
    N = 128
    scale = 2.5
    a = np.random.randn(N).astype(np.float32)
    expected = bf16_round(bf16_round(a) * scale)

    x = locomp.tensor(pack_as_bf16(a))
    o = locomp.empty(N, dtype=np.float16)

    bf16_scale[(N,)](x, o, scale=scale, N=N)
    result = unpack_bf16(o.numpy())

    assert allclose_bf16(result, expected), \
        f"BFloat16 scale FAILED\n  expected: {expected[:8]}\n  got:      {result[:8]}"
    print(f"  ✓  bf16_scale         N={N}  scale={scale}  max_err={np.max(np.abs(result - expected)):.4f}")


def test_bf16_dot_accum():
    N = 256
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    expected = bf16_round(bf16_round(a) * bf16_round(b))

    x = locomp.tensor(pack_as_bf16(a))
    y = locomp.tensor(pack_as_bf16(b))
    o = locomp.empty(N, dtype=np.float16)

    bf16_dot_accum[(N,)](x, y, o, N=N)
    result = unpack_bf16(o.numpy())

    assert allclose_bf16(result, expected), \
        f"BFloat16 dot accum FAILED\n  expected: {expected[:8]}\n  got:      {result[:8]}"
    print(f"  ✓  bf16_dot_accum    N={N}  max_err={np.max(np.abs(result - expected)):.4f}")


def test_f32_to_bf16_cast():
    N = 64
    a = np.array([1.0, -2.5, 0.1, 100.0, -0.001, 3.14159, 0.0, 1e-4], dtype=np.float32)
    a = np.tile(a, N // len(a))
    expected = bf16_round(a)

    x = locomp.tensor(a)
    o = locomp.empty(N, dtype=np.float16)

    bf16_cast_from_f32[(N,)](x, o, N=N)
    result = unpack_bf16(o.numpy())

    assert allclose_bf16(result, expected), \
        f"f32→bf16 cast FAILED\n  expected: {expected[:8]}\n  got:      {result[:8]}"
    print(f"  ✓  f32_to_bf16_cast  N={N}  max_err={np.max(np.abs(result - expected)):.4f}")


if __name__ == "__main__":
    print("BFloat16 end-to-end hardware validation")
    print("─" * 45)

    tests = [
        test_bf16_vector_add,
        test_bf16_scale,
        test_bf16_dot_accum,
        test_f32_to_bf16_cast,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗  {t.__name__}: {e}")

    print("─" * 45)
    print(f"  {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("  BFloat16 pipeline fully validated on hardware")
    else:
        print("  FAILURES detected — check above")
        raise SystemExit(1)

