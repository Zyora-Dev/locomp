"""Tests for the Locust IR module."""

from locomp.ir import IRKernel, IRType, IRValue, IROp, OpCode


def test_ir_type_bytewidth():
    assert IRType.FLOAT32.bytewidth == 4
    assert IRType.FLOAT16.bytewidth == 2
    assert IRType.INT32.bytewidth == 4
    assert IRType.INT8.bytewidth == 1


def test_ir_type_msl():
    assert IRType.FLOAT32.to_msl() == "float"
    assert IRType.FLOAT16.to_msl() == "half"
    assert IRType.INT32.to_msl() == "int"
    assert IRType.UINT32.to_msl() == "uint"
    assert IRType.BOOL.to_msl() == "bool"


def test_ir_type_properties():
    assert IRType.FLOAT32.is_float is True
    assert IRType.INT32.is_float is False
    assert IRType.INT32.is_int is True
    assert IRType.BOOL.is_int is False


def test_ir_kernel_new_value():
    kernel = IRKernel(name="test", params=[])
    v1 = kernel.new_value("x", IRType.FLOAT32)
    v2 = kernel.new_value("y", IRType.INT32)
    assert v1.id == 0
    assert v2.id == 1
    assert v1.dtype == IRType.FLOAT32
    assert v2.dtype == IRType.INT32


def test_ir_kernel_add_op():
    kernel = IRKernel(name="test", params=[])
    v1 = kernel.new_value("a", IRType.FLOAT32)
    v2 = kernel.new_value("b", IRType.FLOAT32)
    result = kernel.new_value("c", IRType.FLOAT32)
    kernel.add_op(OpCode.ADD, result, [v1, v2])
    assert len(kernel.ops) == 1
    assert kernel.ops[0].opcode == OpCode.ADD


def test_ir_kernel_dump():
    kernel = IRKernel(name="my_kernel", params=[])
    v = kernel.new_value("pid", IRType.INT32)
    kernel.add_op(OpCode.PROGRAM_ID, v, attrs={"axis": 0})
    dump = kernel.dump()
    assert "my_kernel" in dump
    assert "PROGRAM_ID" in dump


def test_ir_value_repr():
    v = IRValue(id=0, name="x", dtype=IRType.FLOAT32, shape=(256,))
    r = repr(v)
    assert "x" in r
    assert "FLOAT32" in r
    assert "256" in r

    v_ptr = IRValue(id=1, name="ptr", dtype=IRType.FLOAT32, is_pointer=True)
    r2 = repr(v_ptr)
    assert "*" in r2
