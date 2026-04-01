"""
Locust Frontend — Traces Python kernel functions into IR.

Uses Python AST inspection to convert @locomp.kernel decorated functions
into Locust IR. The tracer walks the function body and builds the IR graph.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable

from locomp.ir import IRKernel, IRType, IRValue, OpCode


class constexpr:
    """Marker type for compile-time constant kernel parameters."""
    pass


class Tensor:
    """Type annotation for tensor/pointer kernel parameters (float32)."""
    pass


class Float16:
    """Type annotation for float16 tensor/pointer kernel parameters."""
    pass


class UInt8:
    """Type annotation for uint8 tensor/pointer kernel parameters."""
    pass


class Int8:
    """Type annotation for int8 tensor/pointer kernel parameters."""
    pass


class Int32:
    """Type annotation for int32 tensor/pointer kernel parameters."""
    pass


class Bool:
    """Type annotation for bool tensor/pointer kernel parameters."""
    pass


# --- AST → IR Compiler ---

_BINOP_MAP = {
    ast.Add: OpCode.ADD,
    ast.Sub: OpCode.SUB,
    ast.Mult: OpCode.MUL,
    ast.Div: OpCode.DIV,
    ast.FloorDiv: OpCode.DIV,
    ast.Mod: OpCode.MOD,
    ast.BitAnd: OpCode.BIT_AND,
    ast.BitOr: OpCode.BIT_OR,
    ast.BitXor: OpCode.BIT_XOR,
    ast.LShift: OpCode.LSHIFT,
    ast.RShift: OpCode.RSHIFT,
}

_CMPOP_MAP = {
    ast.Lt: OpCode.CMP_LT,
    ast.LtE: OpCode.CMP_LE,
    ast.Gt: OpCode.CMP_GT,
    ast.GtE: OpCode.CMP_GE,
    ast.Eq: OpCode.CMP_EQ,
    ast.NotEq: OpCode.CMP_NE,
}

_FUNC_MAP = {
    "sqrt": OpCode.SQRT,
    "exp": OpCode.EXP,
    "log": OpCode.LOG,
    "abs": OpCode.ABS,
    "max": OpCode.MAX,
    "min": OpCode.MIN,
    "sum": OpCode.REDUCE_SUM,
    "tanh": OpCode.TANH,
    "sin": OpCode.SIN,
    "cos": OpCode.COS,
    "asin": OpCode.ASIN,
    "acos": OpCode.ACOS,
    "atan": OpCode.ATAN,
    "sinh": OpCode.SINH,
    "cosh": OpCode.COSH,
    "exp2": OpCode.EXP2,
    "log2": OpCode.LOG2,
    "log10": OpCode.LOG10,
    "rsqrt": OpCode.RSQRT,
    "ceil": OpCode.CEIL,
    "floor": OpCode.FLOOR,
    "round": OpCode.ROUND,
    "sigmoid": OpCode.SIGMOID,
}


class KernelCompiler(ast.NodeVisitor):
    """Compiles a Python AST function into Locust IR."""

    def __init__(self, func_name: str, params: list[str], param_types: dict[str, Any]):
        self.kernel = IRKernel(name=func_name, params=[])
        self.scope: dict[str, IRValue] = {}
        self._loop_depth = 0
        self._pre_loop_vars: list[set[str]] = []  # stack of var sets defined before each loop
        self._var_depth: dict[str, int] = {}  # variable name → loop depth when defined
        self._build_params(params, param_types)

    def _build_params(self, params: list[str], param_types: dict[str, Any]):
        for name in params:
            ann = param_types.get(name)
            if ann is constexpr or (isinstance(ann, type) and issubclass(ann, constexpr)):
                val = self.kernel.new_value(name, IRType.INT32)
                self.kernel.params.append(val)
                self.scope[name] = val
                self._var_depth[name] = 0
            elif ann is Float16 or (isinstance(ann, type) and issubclass(ann, Float16)):
                val = self.kernel.new_value(name, IRType.FLOAT16, is_pointer=True)
                self.kernel.params.append(val)
                self.scope[name] = val
                self._var_depth[name] = 0
            elif ann is UInt8 or (isinstance(ann, type) and issubclass(ann, UInt8)):
                val = self.kernel.new_value(name, IRType.UINT8, is_pointer=True)
                self.kernel.params.append(val)
                self.scope[name] = val
                self._var_depth[name] = 0
            elif ann is Int8 or (isinstance(ann, type) and issubclass(ann, Int8)):
                val = self.kernel.new_value(name, IRType.INT8, is_pointer=True)
                self.kernel.params.append(val)
                self.scope[name] = val
                self._var_depth[name] = 0
            elif ann is Int32 or (isinstance(ann, type) and issubclass(ann, Int32)):
                val = self.kernel.new_value(name, IRType.INT32, is_pointer=True)
                self.kernel.params.append(val)
                self.scope[name] = val
                self._var_depth[name] = 0
            elif ann is Bool or (isinstance(ann, type) and issubclass(ann, Bool)):
                val = self.kernel.new_value(name, IRType.BOOL, is_pointer=True)
                self.kernel.params.append(val)
                self.scope[name] = val
                self._var_depth[name] = 0
            else:
                val = self.kernel.new_value(name, IRType.FLOAT32, is_pointer=True)
                self.kernel.params.append(val)
                self.scope[name] = val
                self._var_depth[name] = 0

    def compile(self, tree: ast.FunctionDef) -> IRKernel:
        for stmt in tree.body:
            self._visit_stmt(stmt)
        return self.kernel

    def _visit_stmt(self, node: ast.stmt):
        if isinstance(node, ast.Assign):
            self._visit_assign(node)
        elif isinstance(node, ast.Expr):
            self._visit_expr_stmt(node)
        elif isinstance(node, ast.AugAssign):
            self._visit_aug_assign(node)
        elif isinstance(node, ast.For):
            self._visit_for(node)
        elif isinstance(node, ast.While):
            self._visit_while(node)
        elif isinstance(node, ast.If):
            self._visit_if(node)
        elif isinstance(node, ast.Break):
            self._visit_break(node)
        elif isinstance(node, ast.Continue):
            self._visit_continue(node)
        else:
            lineno = getattr(node, 'lineno', '?')
            raise NotImplementedError(
                f"Statement not supported: {type(node).__name__} (line {lineno})"
            )

    def _visit_assign(self, node: ast.Assign):
        assert len(node.targets) == 1, "Only single-target assignment supported"
        target = node.targets[0]

        # Tuple unpacking: a, b = expr1, expr2
        if isinstance(target, ast.Tuple):
            if isinstance(node.value, ast.Tuple):
                assert len(target.elts) == len(node.value.elts), \
                    f"Tuple unpack mismatch: {len(target.elts)} targets, {len(node.value.elts)} values"
                for t, v in zip(target.elts, node.value.elts):
                    assert isinstance(t, ast.Name), "Only simple name targets in tuple unpack"
                    value = self._visit_expr(v)
                    self.scope[t.id] = value
                    self._var_depth[t.id] = self._loop_depth
                return
            else:
                lineno = getattr(node, 'lineno', '?')
                raise NotImplementedError(
                    f"Tuple unpacking only supported with tuple RHS (line {lineno})"
                )

        assert isinstance(target, ast.Name), "Only simple name assignment supported"
        value = self._visit_expr(node.value)

        name = target.id
        # Inside a for-loop and reassigning a variable defined at a LOWER depth → mutable accumulator
        if self._loop_depth > 0 and name in self.scope:
            original_depth = self._var_depth.get(name, 0)
            if original_depth < self._loop_depth:
                original = self.scope[name]
                original.is_mutable = True

                # Emit an explicit COPY that writes to the mutable variable.
                # This prevents retroactive aliasing from poisoning the op that
                # originally produced `value` (e.g. a WHERE or ADD result that
                # other ops also reference with the OLD value of the mutable).
                copy_val = self.kernel.new_value("mut", value.dtype, shape=value.shape)
                self.kernel.add_op(OpCode.COPY, copy_val, [value])
                copy_val.is_mutable = True
                copy_val.aliases = original.id
                value = copy_val

        self.scope[name] = value
        self._var_depth[name] = self._loop_depth

    def _visit_aug_assign(self, node: ast.AugAssign):
        assert isinstance(node.target, ast.Name)
        name = node.target.id
        lhs = self.scope[name]
        rhs = self._visit_expr(node.value)
        opcode = _BINOP_MAP.get(type(node.op))
        if opcode is None:
            raise NotImplementedError(f"AugAssign op not supported: {type(node.op).__name__}")
        result = self.kernel.new_value("tmp", lhs.dtype, shape=lhs.shape)
        self.kernel.add_op(opcode, result, [lhs, rhs])

        # Inside a for-loop and reassigning a variable defined at a LOWER depth → mutable
        if self._loop_depth > 0:
            original_depth = self._var_depth.get(name, 0)
            if original_depth < self._loop_depth:
                original = self.scope[name]
                original.is_mutable = True
                result.is_mutable = True
                result.aliases = original.id

        self.scope[name] = result

    def _visit_for(self, node: ast.For):
        """Handle: for i in range(start, end, step)"""
        assert isinstance(node.target, ast.Name), "Only simple loop variable supported"
        assert isinstance(node.iter, ast.Call), "Only range() iterator supported"
        call_name = self._resolve_call_name(node.iter.func)
        assert call_name == "range", f"Only range() supported in for loops, got {call_name}"

        # Parse range(start, end, step) — all forms
        args = [self._visit_expr(a) for a in node.iter.args]
        if len(args) == 1:
            start = self.kernel.new_value("const", IRType.INT32)
            self.kernel.add_op(OpCode.CONSTANT, start, attrs={"value": 0})
            end = args[0]
            step = self.kernel.new_value("const", IRType.INT32)
            self.kernel.add_op(OpCode.CONSTANT, step, attrs={"value": 1})
        elif len(args) == 2:
            start, end = args
            step = self.kernel.new_value("const", IRType.INT32)
            self.kernel.add_op(OpCode.CONSTANT, step, attrs={"value": 1})
        elif len(args) == 3:
            start, end, step = args
        else:
            raise ValueError(f"range() takes 1-3 arguments, got {len(args)}")

        # Create the loop variable (scalar int)
        loop_var = self.kernel.new_value("i", IRType.INT32)
        self.kernel.add_op(OpCode.FOR_LOOP_START, loop_var, [start, end, step])

        # Track variables that existed before this loop
        self._pre_loop_vars.append(set(self.scope.keys()))
        self._loop_depth += 1
        self.scope[node.target.id] = loop_var

        # Compile body
        for stmt in node.body:
            self._visit_stmt(stmt)

        # End marker
        end_marker = self.kernel.new_value("endfor", IRType.BOOL)
        self.kernel.add_op(OpCode.FOR_LOOP_END, end_marker)
        self._loop_depth -= 1
        self._pre_loop_vars.pop()

    def _visit_if(self, node: ast.If):
        """Handle: if condition: body [elif ...: body] [else: body]"""
        cond = self._visit_expr(node.test)
        start_marker = self.kernel.new_value("if", IRType.BOOL)
        self.kernel.add_op(OpCode.IF_START, start_marker, [cond])

        for stmt in node.body:
            self._visit_stmt(stmt)

        if node.orelse:
            else_marker = self.kernel.new_value("else", IRType.BOOL)
            self.kernel.add_op(OpCode.ELSE_START, else_marker)
            for stmt in node.orelse:
                self._visit_stmt(stmt)

        end_marker = self.kernel.new_value("endif", IRType.BOOL)
        self.kernel.add_op(OpCode.IF_END, end_marker)

    def _visit_while(self, node: ast.While):
        """Handle: while condition: body"""
        start_marker = self.kernel.new_value("while", IRType.BOOL)
        # Emit a dummy cond first — the real cond is re-evaluated each iteration
        self.kernel.add_op(OpCode.WHILE_START, start_marker, attrs={"has_cond": True})

        self._pre_loop_vars.append(set(self.scope.keys()))
        self._loop_depth += 1

        # The condition is evaluated inside the loop body as the break test
        cond = self._visit_expr(node.test)
        # Negate: if NOT cond → break
        neg_cond = self.kernel.new_value("ncond", IRType.BOOL)
        zero = self.kernel.new_value("const", IRType.BOOL)
        self.kernel.add_op(OpCode.CONSTANT, zero, attrs={"value": 0})
        self.kernel.add_op(OpCode.CMP_EQ, neg_cond, [cond, zero])
        break_if = self.kernel.new_value("if", IRType.BOOL)
        self.kernel.add_op(OpCode.IF_START, break_if, [neg_cond])
        break_marker = self.kernel.new_value("break", IRType.BOOL)
        self.kernel.add_op(OpCode.BREAK, break_marker)
        break_endif = self.kernel.new_value("endif", IRType.BOOL)
        self.kernel.add_op(OpCode.IF_END, break_endif)

        for stmt in node.body:
            self._visit_stmt(stmt)

        end_marker = self.kernel.new_value("endwhile", IRType.BOOL)
        self.kernel.add_op(OpCode.WHILE_END, end_marker)
        self._loop_depth -= 1
        self._pre_loop_vars.pop()

    def _visit_break(self, node: ast.Break):
        marker = self.kernel.new_value("break", IRType.BOOL)
        self.kernel.add_op(OpCode.BREAK, marker)

    def _visit_continue(self, node: ast.Continue):
        marker = self.kernel.new_value("continue", IRType.BOOL)
        self.kernel.add_op(OpCode.CONTINUE, marker)

    def _visit_expr_stmt(self, node: ast.Expr):
        self._visit_expr(node.value)

    def _visit_expr(self, node: ast.expr) -> IRValue:
        if isinstance(node, ast.Name):
            return self._visit_name(node)
        elif isinstance(node, ast.Constant):
            return self._visit_constant(node)
        elif isinstance(node, ast.BinOp):
            return self._visit_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._visit_unaryop(node)
        elif isinstance(node, ast.Compare):
            return self._visit_compare(node)
        elif isinstance(node, ast.Call):
            return self._visit_call(node)
        else:
            lineno = getattr(node, 'lineno', '?')
            raise NotImplementedError(
                f"Expression not supported: {type(node).__name__} (line {lineno})"
            )

    def _visit_name(self, node: ast.Name) -> IRValue:
        if node.id in self.scope:
            return self.scope[node.id]
        # Resolve module-level constants (e.g., TILE = 8)
        if hasattr(self, '_func_globals') and node.id in self._func_globals:
            val = self._func_globals[node.id]
            if isinstance(val, (int, float)):
                dtype = IRType.FLOAT32 if isinstance(val, float) else IRType.INT32
                result = self.kernel.new_value("const", dtype)
                self.kernel.add_op(OpCode.CONSTANT, result, attrs={"value": val})
                return result
        raise NameError(f"Undefined variable in kernel: '{node.id}'")

    def _visit_constant(self, node: ast.Constant) -> IRValue:
        if isinstance(node.value, float):
            dtype = IRType.FLOAT32
        elif isinstance(node.value, int):
            dtype = IRType.INT32
        elif isinstance(node.value, bool):
            dtype = IRType.BOOL
        else:
            raise TypeError(f"Unsupported constant type: {type(node.value)}")
        result = self.kernel.new_value("const", dtype)
        self.kernel.add_op(OpCode.CONSTANT, result, attrs={"value": node.value})
        return result

    def _visit_binop(self, node: ast.BinOp) -> IRValue:
        lhs = self._visit_expr(node.left)
        rhs = self._visit_expr(node.right)
        opcode = _BINOP_MAP.get(type(node.op))
        if opcode is None:
            raise NotImplementedError(f"Binary op not supported: {type(node.op).__name__}")
        out_shape = lhs.shape if lhs.shape else rhs.shape
        out_dtype = IRType.FLOAT32 if lhs.dtype.is_float or rhs.dtype.is_float else IRType.INT32
        result = self.kernel.new_value("tmp", out_dtype, shape=out_shape)
        self.kernel.add_op(opcode, result, [lhs, rhs])
        return result

    def _visit_unaryop(self, node: ast.UnaryOp) -> IRValue:
        operand = self._visit_expr(node.operand)
        if isinstance(node.op, ast.USub):
            result = self.kernel.new_value("tmp", operand.dtype, shape=operand.shape)
            self.kernel.add_op(OpCode.NEG, result, [operand])
            return result
        raise NotImplementedError(f"Unary op not supported: {type(node.op).__name__}")

    def _visit_compare(self, node: ast.Compare) -> IRValue:
        assert len(node.ops) == 1 and len(node.comparators) == 1
        lhs = self._visit_expr(node.left)
        rhs = self._visit_expr(node.comparators[0])
        opcode = _CMPOP_MAP.get(type(node.ops[0]))
        if opcode is None:
            raise NotImplementedError(f"Compare op: {type(node.ops[0]).__name__}")
        out_shape = lhs.shape if lhs.shape else rhs.shape
        result = self.kernel.new_value("cmp", IRType.BOOL, shape=out_shape)
        self.kernel.add_op(opcode, result, [lhs, rhs])
        return result

    def _visit_call(self, node: ast.Call) -> IRValue:
        func_name = self._resolve_call_name(node.func)

        # locomp.program_id(axis)
        if func_name == "program_id":
            axis = node.args[0].value if node.args else 0
            result = self.kernel.new_value("pid", IRType.INT32)
            self.kernel.add_op(OpCode.PROGRAM_ID, result, attrs={"axis": axis})
            return result

        # locomp.thread_id(axis) — global thread index
        if func_name == "thread_id":
            axis = node.args[0].value if node.args else 0
            result = self.kernel.new_value("tid", IRType.INT32)
            self.kernel.add_op(OpCode.THREAD_ID, result, attrs={"axis": axis})
            return result

        # locomp.local_id(axis) — thread index within threadgroup
        if func_name == "local_id":
            axis = node.args[0].value if node.args else 0
            result = self.kernel.new_value("lid", IRType.INT32)
            self.kernel.add_op(OpCode.LOCAL_ID, result, attrs={"axis": axis})
            return result

        # locomp.group_size(axis) — threadgroup size
        if func_name == "group_size":
            axis = node.args[0].value if node.args else 0
            result = self.kernel.new_value("gsz", IRType.INT32)
            self.kernel.add_op(OpCode.GROUP_SIZE, result, attrs={"axis": axis})
            return result

        # locomp.num_groups(axis) — number of threadgroups
        if func_name == "num_groups":
            axis = node.args[0].value if node.args else 0
            result = self.kernel.new_value("ngrp", IRType.INT32)
            self.kernel.add_op(OpCode.NUM_GROUPS, result, attrs={"axis": axis})
            return result

        # locomp.barrier() — threadgroup synchronization
        if func_name == "barrier":
            result = self.kernel.new_value("bar", IRType.BOOL)
            self.kernel.add_op(OpCode.BARRIER, result)
            return result

        # --- SIMD group operations ---
        if func_name == "simd_sum":
            arg = self._visit_expr(node.args[0])
            result = self.kernel.new_value("ssum", arg.dtype)
            self.kernel.add_op(OpCode.SIMD_SUM, result, [arg])
            return result

        if func_name == "simd_max":
            arg = self._visit_expr(node.args[0])
            result = self.kernel.new_value("smax", arg.dtype)
            self.kernel.add_op(OpCode.SIMD_MAX, result, [arg])
            return result

        if func_name == "simd_min":
            arg = self._visit_expr(node.args[0])
            result = self.kernel.new_value("smin", arg.dtype)
            self.kernel.add_op(OpCode.SIMD_MIN, result, [arg])
            return result

        if func_name == "simd_broadcast":
            arg = self._visit_expr(node.args[0])
            lane = self._visit_expr(node.args[1])
            result = self.kernel.new_value("sbcast", arg.dtype)
            self.kernel.add_op(OpCode.SIMD_BROADCAST, result, [arg, lane])
            return result

        if func_name == "simd_shuffle_down":
            arg = self._visit_expr(node.args[0])
            delta = self._visit_expr(node.args[1])
            result = self.kernel.new_value("sshuf", arg.dtype)
            self.kernel.add_op(OpCode.SIMD_SHUFFLE_DOWN, result, [arg, delta])
            return result

        if func_name == "simd_lane_id":
            result = self.kernel.new_value("slane", IRType.INT32)
            self.kernel.add_op(OpCode.SIMD_LANE_ID, result)
            return result

        if func_name == "simd_group_id":
            result = self.kernel.new_value("sgrp", IRType.INT32)
            self.kernel.add_op(OpCode.SIMD_GROUP_ID, result)
            return result

        # --- Simdgroup matrix operations (hardware 8×8 matmul) ---

        # locomp.simdgroup_matrix_load(shared_arr, offset, stride)
        # — loads 8×8 block from shared memory at shared_arr + offset with given stride
        if func_name == "simdgroup_matrix_load":
            arr = self._visit_expr(node.args[0])
            offset = self._visit_expr(node.args[1])
            stride = self._visit_expr(node.args[2])
            result = self.kernel.new_value("smat", arr.dtype)
            result.is_simdgroup_matrix = True
            self.kernel.add_op(OpCode.SIMDGROUP_MATRIX_LOAD, result, [arr, offset, stride],
                               attrs={"source": "shared"})
            return result

        # locomp.simdgroup_matrix_load_device(ptr, stride)
        # — loads 8×8 block from device pointer with given stride
        if func_name == "simdgroup_matrix_load_device":
            ptr = self._visit_expr(node.args[0])
            stride = self._visit_expr(node.args[1])
            result = self.kernel.new_value("smat", self._resolve_ptr_dtype(ptr))
            result.is_simdgroup_matrix = True
            self.kernel.add_op(OpCode.SIMDGROUP_MATRIX_LOAD, result, [ptr, stride],
                               attrs={"source": "device"})
            return result

        # locomp.simdgroup_matrix_store(mat, shared_arr, offset, stride)
        if func_name == "simdgroup_matrix_store":
            mat = self._visit_expr(node.args[0])
            arr = self._visit_expr(node.args[1])
            offset = self._visit_expr(node.args[2])
            stride = self._visit_expr(node.args[3])
            result = self.kernel.new_value("smst", IRType.BOOL)
            self.kernel.add_op(OpCode.SIMDGROUP_MATRIX_STORE, result, [mat, arr, offset, stride],
                               attrs={"dest": "shared"})
            return result

        # locomp.simdgroup_matrix_store_device(mat, ptr, stride)
        if func_name == "simdgroup_matrix_store_device":
            mat = self._visit_expr(node.args[0])
            ptr = self._visit_expr(node.args[1])
            stride = self._visit_expr(node.args[2])
            result = self.kernel.new_value("smst", IRType.BOOL)
            self.kernel.add_op(OpCode.SIMDGROUP_MATRIX_STORE, result, [mat, ptr, stride],
                               attrs={"dest": "device"})
            return result

        # locomp.simdgroup_mac(acc, a, b) — D = A * B + C
        if func_name == "simdgroup_mac":
            acc = self._visit_expr(node.args[0])
            a = self._visit_expr(node.args[1])
            b = self._visit_expr(node.args[2])
            # Result dtype follows input matrices (a, b), not the accumulator
            # This handles mixed cases like float32 acc + half8x8 inputs
            result_dtype = a.dtype
            result = self.kernel.new_value("smat", result_dtype)
            result.is_simdgroup_matrix = True
            self.kernel.add_op(OpCode.SIMDGROUP_MATRIX_MAC, result, [acc, a, b])
            return result

        # locomp.simdgroup_matrix(fill_value, dtype=None) — create matrix filled with constant
        if func_name == "simdgroup_matrix":
            fill = self._visit_expr(node.args[0])
            mat_dtype = fill.dtype
            # Optional second arg: locomp.Float16 → FLOAT16
            if len(node.args) > 1:
                dtype_node = node.args[1]
                if isinstance(dtype_node, ast.Attribute) and dtype_node.attr in ("Float16", "float16"):
                    mat_dtype = IRType.FLOAT16
            result = self.kernel.new_value("smat", mat_dtype)
            result.is_simdgroup_matrix = True
            self.kernel.add_op(OpCode.SIMDGROUP_MATRIX_FILL, result, [fill])
            return result

        # locomp.shared_memory(size, dtype) — allocate threadgroup shared memory
        if func_name == "shared_memory":
            try:
                size = self._eval_const_size(node.args[0])
            except ValueError:
                # Size depends on constexpr params — store as symbolic IRValue
                size = self._visit_expr(node.args[0])
            # Determine dtype: default FLOAT32, but locomp.Float16 → FLOAT16
            smem_dtype = IRType.FLOAT32
            if len(node.args) > 1:
                dtype_node = node.args[1]
                if isinstance(dtype_node, ast.Attribute):
                    _smem_dtype_map = {
                        "Float16": IRType.FLOAT16, "float16": IRType.FLOAT16,
                        "Int32": IRType.INT32, "int32": IRType.INT32,
                        "UInt8": IRType.UINT8, "uint8": IRType.UINT8,
                    }
                    if dtype_node.attr in _smem_dtype_map:
                        smem_dtype = _smem_dtype_map[dtype_node.attr]
            name = f"smem_{len(self.kernel.shared_mem)}"
            self.kernel.shared_mem[name] = (smem_dtype, size)
            shape = (size,) if isinstance(size, int) else None
            result = self.kernel.new_value(name, smem_dtype, shape=shape)
            self.kernel.add_op(OpCode.CONSTANT, result, attrs={"shared_mem": name, "value": 0})
            return result

        # locomp.arange(start, end)
        if func_name == "arange":
            start = node.args[0].value if len(node.args) > 1 else 0
            end = node.args[-1].value
            size = end - start
            result = self.kernel.new_value("range", IRType.INT32, shape=(size,))
            self.kernel.add_op(OpCode.ARANGE, result, attrs={"start": start, "end": end})
            return result

        # locomp.shared_load(arr, idx)
        if func_name == "shared_load":
            arr = self._visit_expr(node.args[0])
            idx = self._visit_expr(node.args[1])
            # Determine dtype from the shared memory array
            load_dtype = arr.dtype
            result = self.kernel.new_value("sloaded", load_dtype)
            self.kernel.add_op(OpCode.SHARED_LOAD, result, [arr, idx])
            return result

        # locomp.shared_store(arr, idx, val)
        if func_name == "shared_store":
            arr = self._visit_expr(node.args[0])
            idx = self._visit_expr(node.args[1])
            val = self._visit_expr(node.args[2])
            result = self.kernel.new_value("sstore", IRType.BOOL)
            self.kernel.add_op(OpCode.SHARED_STORE, result, [arr, idx, val])
            return result

        # locomp.cast(val, "dtype_string") — explicit type cast
        if func_name == "cast":
            arg = self._visit_expr(node.args[0])
            dtype_node = node.args[1]
            _cast_dtype_map = {
                "float32": IRType.FLOAT32, "float16": IRType.FLOAT16,
                "int32": IRType.INT32, "int8": IRType.INT8,
                "uint8": IRType.UINT8, "uint32": IRType.UINT32,
            }
            if isinstance(dtype_node, ast.Constant):
                type_name = dtype_node.value
            elif isinstance(dtype_node, ast.Attribute):
                type_name = dtype_node.attr
            else:
                raise ValueError(f"Cast type must be a string, got {ast.dump(dtype_node)}")
            target_dtype = _cast_dtype_map[type_name]
            result = self.kernel.new_value("cast", target_dtype)
            self.kernel.add_op(OpCode.CAST, result, [arg])
            return result

        # locomp.load(ptr, mask=...)
        if func_name == "load":
            ptr = self._visit_expr(node.args[0])
            mask = None
            for kw in node.keywords:
                if kw.arg == "mask":
                    mask = self._visit_expr(kw.value)
            # Infer load dtype from pointer's base type
            load_dtype = self._resolve_ptr_dtype(ptr)
            result = self.kernel.new_value("loaded", load_dtype, shape=ptr.shape)
            operands = [ptr] if mask is None else [ptr, mask]
            self.kernel.add_op(OpCode.LOAD, result, operands)
            return result

        # locomp.store(ptr, value, mask=...)
        if func_name == "store":
            ptr = self._visit_expr(node.args[0])
            value = self._visit_expr(node.args[1])
            mask = None
            for kw in node.keywords:
                if kw.arg == "mask":
                    mask = self._visit_expr(kw.value)
            result = self.kernel.new_value("store", IRType.BOOL)
            operands = [ptr, value] if mask is None else [ptr, value, mask]
            self.kernel.add_op(OpCode.STORE, result, operands)
            return result

        # locomp.where(cond, a, b) — ternary select
        if func_name == "where":
            cond = self._visit_expr(node.args[0])
            a = self._visit_expr(node.args[1])
            b = self._visit_expr(node.args[2])
            out_shape = a.shape if a.shape else b.shape
            result = self.kernel.new_value("sel", a.dtype, shape=out_shape)
            self.kernel.add_op(OpCode.WHERE, result, [cond, a, b])
            return result

        # Multi-argument math functions
        if func_name == "fma":
            a = self._visit_expr(node.args[0])
            b = self._visit_expr(node.args[1])
            c = self._visit_expr(node.args[2])
            result = self.kernel.new_value("math", a.dtype, shape=a.shape)
            self.kernel.add_op(OpCode.FMA, result, [a, b, c])
            return result

        if func_name == "clamp":
            x = self._visit_expr(node.args[0])
            lo = self._visit_expr(node.args[1])
            hi = self._visit_expr(node.args[2])
            result = self.kernel.new_value("math", x.dtype, shape=x.shape)
            self.kernel.add_op(OpCode.CLAMP, result, [x, lo, hi])
            return result

        if func_name in ("pow", "atan2", "copysign", "fmod", "step"):
            _two_arg_ops = {
                "pow": OpCode.POW, "atan2": OpCode.ATAN2,
                "copysign": OpCode.COPYSIGN, "fmod": OpCode.FMOD,
                "step": OpCode.STEP,
            }
            a = self._visit_expr(node.args[0])
            b = self._visit_expr(node.args[1])
            result = self.kernel.new_value("math", a.dtype, shape=a.shape)
            self.kernel.add_op(_two_arg_ops[func_name], result, [a, b])
            return result

        # Math functions: locomp.sqrt, locomp.exp, locomp.log, etc. (1-arg)
        if func_name in _FUNC_MAP:
            opcode = _FUNC_MAP[func_name]
            arg = self._visit_expr(node.args[0])
            # Reductions collapse shape to scalar
            if opcode in (OpCode.REDUCE_SUM, OpCode.REDUCE_MAX, OpCode.REDUCE_MIN):
                result = self.kernel.new_value("reduced", arg.dtype)
            else:
                result = self.kernel.new_value("math", arg.dtype, shape=arg.shape)
            self.kernel.add_op(opcode, result, [arg])
            return result

        # locomp.atomic_add(ptr, val) — returns old value
        if func_name == "atomic_add":
            ptr = self._visit_expr(node.args[0])
            val = self._visit_expr(node.args[1])
            load_dtype = self._resolve_ptr_dtype(ptr)
            result = self.kernel.new_value("aold", load_dtype)
            self.kernel.add_op(OpCode.ATOMIC_ADD, result, [ptr, val])
            return result

        # locomp.atomic_max(ptr, val) — returns old value
        if func_name == "atomic_max":
            ptr = self._visit_expr(node.args[0])
            val = self._visit_expr(node.args[1])
            load_dtype = self._resolve_ptr_dtype(ptr)
            result = self.kernel.new_value("aold", load_dtype)
            self.kernel.add_op(OpCode.ATOMIC_MAX, result, [ptr, val])
            return result

        # locomp.atomic_min(ptr, val) — returns old value
        if func_name == "atomic_min":
            ptr = self._visit_expr(node.args[0])
            val = self._visit_expr(node.args[1])
            load_dtype = self._resolve_ptr_dtype(ptr)
            result = self.kernel.new_value("aold", load_dtype)
            self.kernel.add_op(OpCode.ATOMIC_MIN, result, [ptr, val])
            return result

        lineno = getattr(node, 'lineno', '?')
        raise NotImplementedError(f"Unknown function call: {func_name} (line {lineno})")

    def _resolve_call_name(self, node: ast.expr) -> str:
        """Resolve function name from call node — handles locomp.foo and bare foo."""
        if isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        raise NotImplementedError(f"Cannot resolve call: {ast.dump(node)}")

    def _eval_const_size(self, node: ast.expr) -> int:
        """Evaluate a constant AST expression to an integer (for shared_memory sizes, etc.)."""
        if isinstance(node, ast.Constant):
            return int(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._eval_const_size(node.left)
            right = self._eval_const_size(node.right)
            if isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.FloorDiv):
                return left // right
        elif isinstance(node, ast.Name):
            # Look up in kernel scope — must be a constexpr param
            if node.id in self.scope:
                val = self.scope[node.id]
                # Check if it was folded to a constant
                for op in self.kernel.ops:
                    if op.result.id == val.id and op.opcode == OpCode.CONSTANT:
                        return int(op.attrs["value"])
                # If it's a constexpr param (not a constant), DON'T use globals — value is unknown at IR time
                constexpr_param_ids = {p.id for p in self.kernel.params if not p.is_pointer}
                if val.id in constexpr_param_ids:
                    raise ValueError(f"Constexpr parameter '{node.id}' — cannot evaluate at IR time")
            # Try the function's closure/globals (module-level constants like TILE=16)
            if hasattr(self, '_func_globals') and node.id in self._func_globals:
                return int(self._func_globals[node.id])
        raise ValueError(f"Cannot evaluate constant expression: {ast.dump(node)}")

    def _resolve_ptr_dtype(self, ptr: IRValue) -> IRType:
        """Trace a pointer expression back to its base pointer's dtype."""
        # Walk backwards through ops to find the base pointer
        for op in reversed(self.kernel.ops):
            if op.result.id == ptr.id and op.opcode == OpCode.ADD:
                for operand in op.operands:
                    if operand.is_pointer:
                        return operand.dtype
        # Direct pointer
        if ptr.is_pointer:
            return ptr.dtype
        # Default fallback
        return IRType.FLOAT32


def compile_kernel(func: Callable) -> IRKernel:
    """Compile a Python function into Locust IR."""
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # Find the function definition (skip decorators)
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            func_def = node
            break
    assert func_def is not None, f"Could not find function definition for {func.__name__}"

    # Extract parameter names and type annotations
    params = [arg.arg for arg in func_def.args.args]
    annotations = func.__annotations__ if hasattr(func, "__annotations__") else {}

    compiler = KernelCompiler(func.__name__, params, annotations)
    func_globals = dict(getattr(func, '__globals__', {}))
    # Also capture closure variables (from enclosing scopes)
    if hasattr(func, '__code__') and hasattr(func, '__closure__') and func.__closure__:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            try:
                func_globals[name] = cell.cell_contents
            except ValueError:
                pass
    compiler._func_globals = func_globals
    return compiler.compile(func_def)
