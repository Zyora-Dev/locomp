"""Tests for compiler hardening — Float16, shared memory, barrier, Conv2D edge cases."""

import numpy as np
import pytest
import locomp
from locomp.frontend import compile_kernel, constexpr, Tensor
from locomp.ir import OpCode, IRType, IRValue


# --- Float16 support ---

def test_float16_shared_memory_ir():
    """shared_memory with Float16 dtype produces FLOAT16 in IR."""
    def k(X: Tensor):
        smem = locomp.shared_memory(64, locomp.Float16)

    ir = compile_kernel(k)
    assert "smem_0" in ir.shared_mem
    dtype, size = ir.shared_mem["smem_0"]
    assert dtype == IRType.FLOAT16
    assert size == 64


def test_float32_shared_memory_ir():
    """shared_memory without dtype defaults to FLOAT32."""
    def k(X: Tensor):
        smem = locomp.shared_memory(128)

    ir = compile_kernel(k)
    dtype, size = ir.shared_mem["smem_0"]
    assert dtype == IRType.FLOAT32
    assert size == 128


def test_float16_shared_memory_codegen():
    """Float16 shared memory emits 'threadgroup half' in MSL."""
    @locomp.kernel
    def k(X: locomp.Tensor):
        smem = locomp.shared_memory(32, locomp.Float16)

    msl = k.msl
    assert "threadgroup half smem_0[32];" in msl


def test_float32_shared_memory_codegen():
    """Float32 shared memory emits 'threadgroup float' in MSL."""
    @locomp.kernel
    def k(X: locomp.Tensor):
        smem = locomp.shared_memory(64)

    msl = k.msl
    assert "threadgroup float smem_0[64];" in msl


# --- Constexpr-dependent shared memory ---

def test_constexpr_shared_memory_ir():
    """shared_memory(N) where N is constexpr stores symbolic size in IR."""
    def k(X: Tensor, N: constexpr):
        smem = locomp.shared_memory(N)

    ir = compile_kernel(k)
    dtype, size = ir.shared_mem["smem_0"]
    assert dtype == IRType.FLOAT32
    # Size should be an IRValue (symbolic), not an int
    assert isinstance(size, IRValue)


def test_constexpr_shared_memory_codegen():
    """shared_memory(N) with constexpr N inlines value in MSL."""
    from locomp.backends.metal_codegen import compile_to_metal

    def k(X: Tensor, N: constexpr):
        smem = locomp.shared_memory(N)
        locomp.shared_store(smem, 0, locomp.load(X))

    ir = compile_kernel(k)
    # Get actual IR param name for N (mangled to N_<id>)
    n_param = [p for p in ir.params if not p.is_pointer][0]
    msl, _ = compile_to_metal(ir, constexpr_values={n_param.name: 256})
    assert "smem_0[256]" in msl


def test_constexpr_shared_memory_different_values():
    """Different constexpr values produce different shared memory sizes."""
    from locomp.backends.metal_codegen import compile_to_metal

    def k(X: Tensor, N: constexpr):
        smem = locomp.shared_memory(N)

    ir = compile_kernel(k)
    n_param = [p for p in ir.params if not p.is_pointer][0]
    msl1, _ = compile_to_metal(ir, constexpr_values={n_param.name: 64})
    msl2, _ = compile_to_metal(ir, constexpr_values={n_param.name: 512})
    assert "smem_0[64]" in msl1
    assert "smem_0[512]" in msl2


def test_constexpr_expr_shared_memory():
    """shared_memory(A * B) where A, B are constexpr handles symbolically."""
    def k(X: Tensor, A: constexpr, B: constexpr):
        smem = locomp.shared_memory(A * B)

    ir = compile_kernel(k)
    dtype, size = ir.shared_mem["smem_0"]
    # Size is an IRValue (result of MUL of two constexprs)
    assert isinstance(size, IRValue)


# --- Barrier ---

def test_barrier_ir():
    """locomp.barrier() produces BARRIER opcode."""
    def k(X: Tensor):
        locomp.barrier()

    ir = compile_kernel(k)
    opcodes = [op.opcode for op in ir.ops]
    assert OpCode.BARRIER in opcodes


def test_barrier_codegen():
    """barrier() emits threadgroup_barrier in MSL."""
    @locomp.kernel
    def k(X: locomp.Tensor):
        locomp.barrier()

    msl = k.msl
    assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in msl


# --- Multi-dimensional local_id ---

def test_local_id_axes():
    """local_id(0/1/2) maps to lid.x/y/z in MSL."""
    @locomp.kernel
    def k(X: locomp.Tensor):
        a = locomp.local_id(0)
        b = locomp.local_id(1)
        c = locomp.local_id(2)
        # Use them so they don't get optimized away
        locomp.store(X + a, 0.0)
        locomp.store(X + b, 0.0)
        locomp.store(X + c, 0.0)

    msl = k.msl
    assert "lid.x" in msl
    assert "lid.y" in msl
    assert "lid.z" in msl


# --- Range with 3 args ---

def test_range_3_args_codegen():
    """range(start, end, step) generates correct MSL for-loop."""
    @locomp.kernel
    def k(X: locomp.Tensor, N: locomp.constexpr):
        tid = locomp.local_id(0)
        for i in range(tid, N, 32):
            locomp.store(X + i, 0.0)

    msl = k.msl
    # Should have a for-loop with lid as start and step
    assert "lid.x" in msl
    assert "+= 32" in msl or "+= const" in msl


# --- End-to-end GPU tests ---

def test_shared_memory_store_load_roundtrip():
    """Write to shared memory and read back — values must match."""
    @locomp.kernel
    def k(Input: locomp.Tensor, Output: locomp.Tensor, N: locomp.constexpr):
        tid = locomp.local_id(0)
        smem = locomp.shared_memory(N)
        val = locomp.load(Input + tid)
        locomp.shared_store(smem, tid, val)
        locomp.barrier()
        out_val = locomp.shared_load(smem, tid)
        locomp.store(Output + tid, out_val)

    N = 32
    x = np.arange(N, dtype=np.float32)
    x_t = locomp.tensor(x)
    o_t = locomp.empty(N)
    k[(1,), (N,)](x_t, o_t, N)
    result = o_t.numpy()
    np.testing.assert_allclose(result, x, atol=1e-6)


def test_cooperative_smem_load():
    """Threads cooperatively load into shared memory, then read each other's values."""
    @locomp.kernel
    def k(Input: locomp.Tensor, Output: locomp.Tensor, N: locomp.constexpr):
        tid = locomp.local_id(0)
        smem = locomp.shared_memory(N)
        # Each thread stores its value
        val = locomp.load(Input + tid)
        locomp.shared_store(smem, tid, val)
        locomp.barrier()
        # Each thread reads the value from the reversed position
        read_idx = N - 1 - tid
        out_val = locomp.shared_load(smem, read_idx)
        locomp.store(Output + tid, out_val)

    N = 64
    x = np.arange(N, dtype=np.float32)
    x_t = locomp.tensor(x)
    o_t = locomp.empty(N)
    k[(1,), (N,)](x_t, o_t, N)
    result = o_t.numpy()
    # Thread i should have value (N-1-i)
    expected = x[::-1]
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_conv2d_small():
    """Small Conv2D correctness test — 1×1×4×4 → 2×3×3."""
    @locomp.kernel
    def conv2d(Input: locomp.Tensor, Weight: locomp.Tensor, Output: locomp.Tensor,
               CI: locomp.constexpr, H: locomp.constexpr, W: locomp.constexpr,
               CO: locomp.constexpr, KH: locomp.constexpr, KW: locomp.constexpr,
               OH: locomp.constexpr, OW: locomp.constexpr,
               PATCH: locomp.constexpr):
        tid = locomp.local_id(0)
        spatial = locomp.program_id(0)
        oh = spatial // OW
        ow = spatial % OW
        patch = locomp.shared_memory(PATCH)
        for i in range(tid, PATCH, CO):
            ci = i // (KH * KW)
            rem = i % (KH * KW)
            kh = rem // KW
            kw = rem % KW
            ih = oh + kh
            iw = ow + kw
            val = locomp.load(Input + (ci * H * W + ih * W + iw))
            locomp.shared_store(patch, i, val)
        locomp.barrier()
        co = tid
        acc = 0.0
        for i in range(PATCH):
            acc = acc + locomp.shared_load(patch, i) * locomp.load(Weight + (co * PATCH + i))
        locomp.store(Output + (co * OH * OW + oh * OW + ow), acc)

    CI, H, W, CO, KH, KW = 1, 4, 4, 2, 3, 3
    OH, OW = H - KH + 1, W - KW + 1
    PATCH = CI * KH * KW

    np.random.seed(123)
    x_np = np.random.randn(CI, H, W).astype(np.float32)
    w_np = np.random.randn(CO, CI, KH, KW).astype(np.float32)
    expected = np.zeros((CO, OH, OW), dtype=np.float32)
    for co in range(CO):
        for ci in range(CI):
            for kh in range(KH):
                for kw in range(KW):
                    expected[co] += x_np[ci, kh:kh + OH, kw:kw + OW] * w_np[co, ci, kh, kw]

    x_t = locomp.tensor(x_np.flatten())
    w_t = locomp.tensor(w_np.flatten())
    o_t = locomp.empty(CO * OH * OW)
    conv2d[(OH * OW,), (CO,)](x_t, w_t, o_t, CI, H, W, CO, KH, KW, OH, OW, PATCH)
    result = o_t.numpy().reshape(CO, OH, OW)
    np.testing.assert_allclose(result, expected, atol=1e-5)
