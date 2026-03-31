"""
Locust Metal Backend — Compiles IR to Metal Shading Language (MSL).

This is the core of Locust: takes optimized IR and generates MSL source code
that can be compiled and run on Apple Silicon GPUs (M1/M2/M3/M4).

The generated MSL uses:
- Threadgroup-based parallelism (maps to IR tiles)
- Device memory buffers (maps to IR pointers)
- Thread position in grid (maps to program_id)
- SIMD group operations for reductions
"""

from __future__ import annotations

from locomp.ir import IRKernel, IROp, IRType, IRValue, OpCode


class MetalCodegen:
    """Generates Metal Shading Language from Locust IR."""

    def __init__(self, kernel: IRKernel, constexpr_values: dict[str, int] | None = None):
        self.kernel = kernel
        self.indent = "    "
        self._var_names: dict[int, str] = {}
        self._next_var = 0
        self._buffer_index = 0
        self._param_buffer_map: dict[int, int] = {}
        # Track pointer expressions: value_id → (base_ptr_var, index_var, is_tiled)
        self._ptr_exprs: dict[int, tuple[str, str, bool]] = {}
        # Constexpr values to inline as literals (param_name → value)
        self._constexpr_values = constexpr_values or {}

    def generate(self) -> str:
        """Generate complete MSL source code."""
        lines: list[str] = []
        lines.append("#include <metal_stdlib>")
        # Include simdgroup_matrix header if kernel uses matrix ops
        has_matrix_ops = any(op.opcode in (
            OpCode.SIMDGROUP_MATRIX_LOAD, OpCode.SIMDGROUP_MATRIX_STORE,
            OpCode.SIMDGROUP_MATRIX_MAC, OpCode.SIMDGROUP_MATRIX_FILL
        ) for op in self.kernel.ops)
        if has_matrix_ops:
            lines.append("#include <metal_simdgroup_matrix>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(self._gen_kernel_signature())
        lines.append("{")
        lines.extend(self._gen_body())
        lines.append("}")
        return "\n".join(lines)

    def _var(self, value: IRValue) -> str:
        """Get or create a variable name for an IR value."""
        # If this value aliases another (mutable accumulator), use the original's name
        if value.aliases is not None and value.aliases in self._var_names:
            self._var_names[value.id] = self._var_names[value.aliases]
            return self._var_names[value.id]
        if value.id not in self._var_names:
            base = value.name.replace(".", "_")
            self._var_names[value.id] = base
        return self._var_names[value.id]

    def _msl_type(self, dtype: IRType) -> str:
        """Convert IR type to MSL type."""
        return dtype.to_msl()

    def _simdgroup_type(self, dtype: IRType) -> str:
        """Convert IR type to simdgroup matrix MSL type."""
        if dtype == IRType.FLOAT16:
            return "simdgroup_half8x8"
        return "simdgroup_float8x8"

    def _resolve_ptr(self, value: IRValue) -> str:
        """Resolve a pointer value to its MSL expression (base + offset)."""
        if value.id in self._ptr_exprs:
            base, idx, _ = self._ptr_exprs[value.id]
            return f"{base} + {idx}"
        return self._var(value)

    def _is_ptr(self, value: IRValue) -> bool:
        """Check if a value is a pointer or pointer expression."""
        return value.is_pointer or value.id in self._ptr_exprs

    def _gen_kernel_signature(self) -> str:
        """Generate the kernel function signature."""
        params = []
        buffer_idx = 0

        for param in self.kernel.params:
            if param.is_pointer:
                params.append(
                    f"device {param.dtype.to_msl()}* {self._var(param)} "
                    f"[[buffer({buffer_idx})]]"
                )
                self._param_buffer_map[param.id] = buffer_idx
                buffer_idx += 1
            else:
                # Constexpr param — check if we can inline it
                if param.name in self._constexpr_values:
                    # Inline as literal: set var name to the literal value
                    self._var_names[param.id] = str(self._constexpr_values[param.name])
                    # Don't add to signature or allocate a buffer slot
                else:
                    params.append(
                        f"constant int& {self._var(param)} "
                        f"[[buffer({buffer_idx})]]"
                    )
                    self._param_buffer_map[param.id] = buffer_idx
                    buffer_idx += 1

        params.append("uint3 tid [[thread_position_in_grid]]")
        params.append("uint3 tgid [[threadgroup_position_in_grid]]")
        params.append("uint3 lid [[thread_position_in_threadgroup]]")
        params.append("uint3 tsize [[threads_per_threadgroup]]")
        params.append("uint3 ngroups [[threadgroups_per_grid]]")
        params.append("uint simd_lid [[thread_index_in_simdgroup]]")
        params.append("uint simd_gid [[simdgroup_index_in_threadgroup]]")

        params_str = ",\n    ".join(params)
        return f"kernel void {self.kernel.name}(\n    {params_str}\n)"

    def _gen_body(self) -> list[str]:
        """Generate the kernel body from IR ops."""
        lines = []
        self._nesting = 0  # Track for/if nesting depth

        # Declare shared memory
        for name, (dtype, size) in self.kernel.shared_mem.items():
            msl_type = dtype.to_msl()
            lines.append(f"{self.indent}threadgroup {msl_type} {name}[{size}];")
        if self.kernel.shared_mem:
            lines.append("")

        for op in self.kernel.ops:
            # FOR_LOOP_END and IF_END decrement nesting BEFORE emitting the closing brace
            if op.opcode in (OpCode.FOR_LOOP_END, OpCode.IF_END):
                self._nesting -= 1
                cur_indent = self.indent * (1 + self._nesting)
                lines.append(f"{cur_indent}" + "}")
                continue

            cur_indent = self.indent * (1 + self._nesting)
            line = self._gen_op(op)
            if line:
                if isinstance(line, list):
                    lines.extend(f"{cur_indent}{l}" for l in line)
                else:
                    lines.append(f"{cur_indent}{line}")

            # FOR_LOOP_START and IF_START increment nesting AFTER emitting the opening line
            if op.opcode in (OpCode.FOR_LOOP_START, OpCode.IF_START):
                self._nesting += 1

        return lines

    def _gen_op(self, op: IROp) -> str | list[str] | None:
        """Generate MSL code for a single IR operation."""
        result_var = self._var(op.result)

        if op.opcode == OpCode.PROGRAM_ID:
            axis = op.attrs.get("axis", 0)
            axis_component = ["x", "y", "z"][axis]
            return f"int {result_var} = tgid.{axis_component};"

        elif op.opcode == OpCode.THREAD_ID:
            axis = op.attrs.get("axis", 0)
            axis_component = ["x", "y", "z"][axis]
            return f"int {result_var} = tid.{axis_component};"

        elif op.opcode == OpCode.LOCAL_ID:
            axis = op.attrs.get("axis", 0)
            axis_component = ["x", "y", "z"][axis]
            return f"int {result_var} = lid.{axis_component};"

        elif op.opcode == OpCode.GROUP_SIZE:
            axis = op.attrs.get("axis", 0)
            axis_component = ["x", "y", "z"][axis]
            return f"int {result_var} = tsize.{axis_component};"

        elif op.opcode == OpCode.NUM_GROUPS:
            axis = op.attrs.get("axis", 0)
            axis_component = ["x", "y", "z"][axis]
            return f"int {result_var} = ngroups.{axis_component};"

        elif op.opcode == OpCode.BARRIER:
            return "threadgroup_barrier(mem_flags::mem_threadgroup);"

        elif op.opcode == OpCode.SIMD_LANE_ID:
            return f"int {result_var} = simd_lid;"

        elif op.opcode == OpCode.SIMD_GROUP_ID:
            return f"int {result_var} = simd_gid;"

        elif op.opcode == OpCode.SIMD_SUM:
            arg = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = simd_sum({arg});"

        elif op.opcode == OpCode.SIMD_MAX:
            arg = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = simd_max({arg});"

        elif op.opcode == OpCode.SIMD_MIN:
            arg = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = simd_min({arg});"

        elif op.opcode == OpCode.SIMD_BROADCAST:
            arg = self._var(op.operands[0])
            lane = self._var(op.operands[1])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = simd_broadcast({arg}, {lane});"

        elif op.opcode == OpCode.SIMD_SHUFFLE_DOWN:
            arg = self._var(op.operands[0])
            delta = self._var(op.operands[1])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = simd_shuffle_down({arg}, {delta});"

        elif op.opcode == OpCode.FOR_LOOP_START:
            start_var = self._var(op.operands[0])
            end_var = self._var(op.operands[1])
            step_var = self._var(op.operands[2])
            loop_var = self._var(op.result)
            return f"for (int {loop_var} = {start_var}; {loop_var} < {end_var}; {loop_var} += {step_var}) {{"

        elif op.opcode == OpCode.FOR_LOOP_END:
            return None  # handled in _gen_body

        elif op.opcode == OpCode.IF_START:
            cond_var = self._var(op.operands[0])
            return f"if ({cond_var}) {{"

        elif op.opcode == OpCode.IF_END:
            return None  # handled in _gen_body

        elif op.opcode == OpCode.ARANGE:
            start = op.attrs["start"]
            end = op.attrs["end"]
            size = end - start
            lines = [
                f"int {result_var}[{size}];",
                f"for (int _i = 0; _i < {size}; _i++) {{",
                f"    {result_var}[_i] = {start} + _i;",
                f"}}",
            ]
            return lines

        elif op.opcode == OpCode.CONSTANT:
            val = op.attrs["value"]
            # Shared memory reference — variable already declared in body preamble
            if "shared_mem" in op.attrs:
                smem_name = op.attrs["shared_mem"]
                # Alias the result variable to the shared memory name
                self._var_names[op.result.id] = smem_name
                return None
            msl_type = self._msl_type(op.result.dtype)
            if isinstance(val, float):
                suffix = "h" if op.result.dtype == IRType.FLOAT16 else "f"
                return f"{msl_type} {result_var} = {val}{suffix};"
            elif isinstance(val, bool):
                return f"bool {result_var} = {'true' if val else 'false'};"
            else:
                return f"{msl_type} {result_var} = {val};"

        elif op.opcode in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD):
            return self._gen_arithmetic(op)

        elif op.opcode == OpCode.COPY:
            src = self._var(op.operands[0])
            # If this COPY aliases a mutable, it's an update — emit assignment, not declaration
            if op.result.aliases is not None:
                return f"{result_var} = {src};"
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = {src};"

        # --- Simdgroup matrix operations ---

        elif op.opcode == OpCode.SIMDGROUP_MATRIX_FILL:
            fill = self._var(op.operands[0])
            mat_type = self._simdgroup_type(op.result.dtype)
            if op.result.aliases is not None:
                return f"{result_var} = {mat_type}({fill});"
            return f"{mat_type} {result_var} = {mat_type}({fill});"

        elif op.opcode == OpCode.SIMDGROUP_MATRIX_LOAD:
            mat_type = self._simdgroup_type(op.result.dtype)
            source = op.attrs.get("source", "shared")
            if source == "shared":
                arr = self._var(op.operands[0])
                offset = self._var(op.operands[1])
                stride = self._var(op.operands[2])
                lines = [f"{mat_type} {result_var};"]
                lines.append(f"simdgroup_load({result_var}, {arr} + {offset}, {stride});")
                return lines
            else:  # device
                ptr = self._resolve_ptr(op.operands[0])
                stride = self._var(op.operands[1])
                lines = [f"{mat_type} {result_var};"]
                lines.append(f"simdgroup_load({result_var}, {ptr}, {stride});")
                return lines

        elif op.opcode == OpCode.SIMDGROUP_MATRIX_STORE:
            dest = op.attrs.get("dest", "shared")
            mat = self._var(op.operands[0])
            if dest == "shared":
                arr = self._var(op.operands[1])
                offset = self._var(op.operands[2])
                stride = self._var(op.operands[3])
                return f"simdgroup_store({mat}, {arr} + {offset}, {stride});"
            else:  # device
                ptr = self._resolve_ptr(op.operands[1])
                stride = self._var(op.operands[2])
                return f"simdgroup_store({mat}, {ptr}, {stride});"

        elif op.opcode == OpCode.SIMDGROUP_MATRIX_MAC:
            acc = self._var(op.operands[0])
            a = self._var(op.operands[1])
            b = self._var(op.operands[2])
            mat_type = self._simdgroup_type(op.result.dtype)
            if op.result.aliases is not None:
                return f"simdgroup_multiply_accumulate({result_var}, {a}, {b}, {acc});"
            lines = [f"{mat_type} {result_var};"]
            lines.append(f"simdgroup_multiply_accumulate({result_var}, {a}, {b}, {acc});")
            return lines

        elif op.opcode == OpCode.NEG:
            operand = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = -{operand};"

        elif op.opcode == OpCode.LOAD:
            return self._gen_load(op)

        elif op.opcode == OpCode.STORE:
            return self._gen_store(op)

        elif op.opcode == OpCode.SHARED_LOAD:
            arr = self._var(op.operands[0])
            idx = self._var(op.operands[1])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = {arr}[{idx}];"

        elif op.opcode == OpCode.SHARED_STORE:
            arr = self._var(op.operands[0])
            idx = self._var(op.operands[1])
            val = self._var(op.operands[2])
            return f"{arr}[{idx}] = {val};"

        elif op.opcode in (OpCode.SQRT, OpCode.EXP, OpCode.LOG, OpCode.ABS,
                           OpCode.TANH, OpCode.SIN, OpCode.COS,
                           OpCode.ASIN, OpCode.ACOS, OpCode.ATAN,
                           OpCode.SINH, OpCode.COSH,
                           OpCode.EXP2, OpCode.LOG2, OpCode.LOG10,
                           OpCode.RSQRT, OpCode.CEIL, OpCode.FLOOR,
                           OpCode.ROUND):
            func_name = {
                OpCode.SQRT: "sqrt", OpCode.EXP: "exp",
                OpCode.LOG: "log", OpCode.ABS: "abs",
                OpCode.TANH: "tanh", OpCode.SIN: "sin",
                OpCode.COS: "cos", OpCode.ASIN: "asin",
                OpCode.ACOS: "acos", OpCode.ATAN: "atan",
                OpCode.SINH: "sinh", OpCode.COSH: "cosh",
                OpCode.EXP2: "exp2", OpCode.LOG2: "log2",
                OpCode.LOG10: "log10", OpCode.RSQRT: "rsqrt",
                OpCode.CEIL: "ceil", OpCode.FLOOR: "floor",
                OpCode.ROUND: "rint",
            }[op.opcode]
            arg = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            if op.result.shape:
                size = op.result.shape[0]
                lines = [f"{msl_type} {result_var}[{size}];"]
                lines.append(f"for (int _i = 0; _i < {size}; _i++) {{")
                lines.append(f"    {result_var}[_i] = {func_name}({arg}[_i]);")
                lines.append("}")
                return lines
            return f"{msl_type} {result_var} = {func_name}({arg});"

        # sigmoid(x) = 1 / (1 + exp(-x))
        elif op.opcode == OpCode.SIGMOID:
            arg = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = 1.0 / (1.0 + exp(-{arg}));"

        # 2-arg math: pow, atan2, copysign, fmod, step
        elif op.opcode in (OpCode.POW, OpCode.ATAN2, OpCode.COPYSIGN,
                           OpCode.FMOD, OpCode.STEP):
            func_name = {
                OpCode.POW: "pow", OpCode.ATAN2: "atan2",
                OpCode.COPYSIGN: "copysign", OpCode.FMOD: "fmod",
                OpCode.STEP: "step",
            }[op.opcode]
            a = self._var(op.operands[0])
            b = self._var(op.operands[1])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = {func_name}({a}, {b});"

        # 3-arg math: fma(a,b,c) = a*b+c, clamp(x,lo,hi)
        elif op.opcode in (OpCode.FMA, OpCode.CLAMP):
            func_name = "fma" if op.opcode == OpCode.FMA else "clamp"
            a = self._var(op.operands[0])
            b = self._var(op.operands[1])
            c = self._var(op.operands[2])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = {func_name}({a}, {b}, {c});"

        elif op.opcode in (OpCode.CMP_LT, OpCode.CMP_LE, OpCode.CMP_GT,
                           OpCode.CMP_GE, OpCode.CMP_EQ, OpCode.CMP_NE):
            cmp_str = {
                OpCode.CMP_LT: "<", OpCode.CMP_LE: "<=", OpCode.CMP_GT: ">",
                OpCode.CMP_GE: ">=", OpCode.CMP_EQ: "==", OpCode.CMP_NE: "!="
            }[op.opcode]
            lhs = self._var(op.operands[0])
            rhs = self._var(op.operands[1])
            if op.result.shape:
                size = op.result.shape[0]
                lhs_tile = bool(op.operands[0].shape)
                rhs_tile = bool(op.operands[1].shape)
                lines = [f"bool {result_var}[{size}];"]
                lines.append(f"for (int _i = 0; _i < {size}; _i++) {{")
                l = f"{lhs}[_i]" if lhs_tile else lhs
                r = f"{rhs}[_i]" if rhs_tile else rhs
                lines.append(f"    {result_var}[_i] = {l} {cmp_str} {r};")
                lines.append("}")
                return lines
            return f"bool {result_var} = {lhs} {cmp_str} {rhs};"

        elif op.opcode == OpCode.REDUCE_SUM:
            arg = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            if op.operands[0].shape:
                size = op.operands[0].shape[0]
                return [
                    f"{msl_type} {result_var} = 0;",
                    f"for (int _i = 0; _i < {size}; _i++) {{",
                    f"    {result_var} += {arg}[_i];",
                    "}",
                ]
            return f"{msl_type} {result_var} = {arg};"

        elif op.opcode == OpCode.REDUCE_MAX:
            arg = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            if op.operands[0].shape:
                size = op.operands[0].shape[0]
                return [
                    f"{msl_type} {result_var} = {arg}[0];",
                    f"for (int _i = 1; _i < {size}; _i++) {{",
                    f"    {result_var} = max({result_var}, {arg}[_i]);",
                    "}",
                ]
            return f"{msl_type} {result_var} = {arg};"

        elif op.opcode == OpCode.REDUCE_MIN:
            arg = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            if op.operands[0].shape:
                size = op.operands[0].shape[0]
                return [
                    f"{msl_type} {result_var} = {arg}[0];",
                    f"for (int _i = 1; _i < {size}; _i++) {{",
                    f"    {result_var} = min({result_var}, {arg}[_i]);",
                    "}",
                ]
            return f"{msl_type} {result_var} = {arg};"

        elif op.opcode == OpCode.CAST:
            arg = self._var(op.operands[0])
            msl_type = self._msl_type(op.result.dtype)
            return f"{msl_type} {result_var} = ({msl_type}){arg};"

        elif op.opcode == OpCode.WHERE:
            cond = self._var(op.operands[0])
            a = self._var(op.operands[1])
            b = self._var(op.operands[2])
            msl_type = self._msl_type(op.result.dtype)
            is_alias = op.result.aliases is not None
            if op.result.shape:
                size = op.result.shape[0]
                cond_tile = bool(op.operands[0].shape)
                a_tile = bool(op.operands[1].shape)
                b_tile = bool(op.operands[2].shape)
                lines = [f"{msl_type} {result_var}[{size}];"]
                lines.append(f"for (int _i = 0; _i < {size}; _i++) {{")
                c = f"{cond}[_i]" if cond_tile else cond
                av = f"{a}[_i]" if a_tile else a
                bv = f"{b}[_i]" if b_tile else b
                lines.append(f"    {result_var}[_i] = {c} ? {av} : {bv};")
                lines.append("}")
                return lines
            elif is_alias:
                return f"{result_var} = {cond} ? {a} : {b};"
            else:
                return f"{msl_type} {result_var} = {cond} ? {a} : {b};"

        return None

    def _gen_arithmetic(self, op: IROp) -> str | list[str] | None:
        """Generate arithmetic, with special handling for pointer + offset."""
        op_sym = {
            OpCode.ADD: "+", OpCode.SUB: "-", OpCode.MUL: "*",
            OpCode.DIV: "/", OpCode.MOD: "%"
        }[op.opcode]
        result_var = self._var(op.result)
        lhs, rhs = op.operands[0], op.operands[1]
        lhs_var, rhs_var = self._var(lhs), self._var(rhs)

        # Detect pointer arithmetic: ptr + offset → store as ptr expression, emit nothing
        if op.opcode == OpCode.ADD and (self._is_ptr(lhs) or self._is_ptr(rhs)):
            if self._is_ptr(lhs):
                ptr_op, idx_op = lhs, rhs
            else:
                ptr_op, idx_op = rhs, lhs

            # Resolve base pointer
            if ptr_op.id in self._ptr_exprs:
                base_ptr = self._ptr_exprs[ptr_op.id][0]
            else:
                base_ptr = self._var(ptr_op)

            idx_var = self._var(idx_op)
            is_tiled = bool(idx_op.shape)
            self._ptr_exprs[op.result.id] = (base_ptr, idx_var, is_tiled)
            return None  # No MSL emitted — LOAD/STORE will use the ptr expression

        # Regular arithmetic
        msl_type = self._msl_type(op.result.dtype)
        is_alias = op.result.aliases is not None
        if op.result.shape:
            size = op.result.shape[0]
            lhs_is_tile = bool(lhs.shape)
            rhs_is_tile = bool(rhs.shape)
            lines = [f"{msl_type} {result_var}[{size}];"]
            lines.append(f"for (int _i = 0; _i < {size}; _i++) {{")
            l = f"{lhs_var}[_i]" if lhs_is_tile else lhs_var
            r = f"{rhs_var}[_i]" if rhs_is_tile else rhs_var
            lines.append(f"    {result_var}[_i] = {l} {op_sym} {r};")
            lines.append("}")
            return lines
        elif is_alias:
            return f"{result_var} = {lhs_var} {op_sym} {rhs_var};"
        else:
            return f"{msl_type} {result_var} = {lhs_var} {op_sym} {rhs_var};"

    def _gen_load(self, op: IROp) -> str | list[str]:
        """Generate load — resolves ptr expressions to base[index] access."""
        result_var = self._var(op.result)
        ptr = op.operands[0]
        mask = op.operands[1] if len(op.operands) > 1 else None
        mask_var = self._var(mask) if mask else None
        msl_type = self._msl_type(op.result.dtype)

        if ptr.id in self._ptr_exprs:
            base_ptr, idx_var, is_tiled = self._ptr_exprs[ptr.id]
        elif ptr.is_pointer:
            base_ptr, idx_var, is_tiled = self._var(ptr), None, False
        else:
            base_ptr, idx_var, is_tiled = self._var(ptr), None, False

        if is_tiled and op.result.shape:
            size = op.result.shape[0]
            lines = [f"{msl_type} {result_var}[{size}];"]
            lines.append(f"for (int _i = 0; _i < {size}; _i++) {{")
            access = f"{base_ptr}[{idx_var}[_i]]"
            if mask_var:
                lines.append(f"    {result_var}[_i] = {mask_var}[_i] ? {access} : 0;")
            else:
                lines.append(f"    {result_var}[_i] = {access};")
            lines.append("}")
            return lines
        elif idx_var:
            access = f"{base_ptr}[{idx_var}]"
            if mask_var:
                return f"{msl_type} {result_var} = {mask_var} ? {access} : 0;"
            return f"{msl_type} {result_var} = {access};"
        else:
            if mask_var:
                return f"{msl_type} {result_var} = {mask_var} ? {base_ptr}[0] : 0;"
            return f"{msl_type} {result_var} = {base_ptr}[0];"

    def _gen_store(self, op: IROp) -> str | list[str]:
        """Generate store — resolves ptr expressions to base[index] access."""
        ptr = op.operands[0]
        val = op.operands[1]
        mask = op.operands[2] if len(op.operands) > 2 else None
        val_var = self._var(val)
        mask_var = self._var(mask) if mask else None

        if ptr.id in self._ptr_exprs:
            base_ptr, idx_var, is_tiled = self._ptr_exprs[ptr.id]
        elif ptr.is_pointer:
            base_ptr, idx_var, is_tiled = self._var(ptr), None, False
        else:
            base_ptr, idx_var, is_tiled = self._var(ptr), None, False

        if is_tiled and val.shape:
            size = val.shape[0]
            lines = []
            lines.append(f"for (int _i = 0; _i < {size}; _i++) {{")
            access = f"{base_ptr}[{idx_var}[_i]]"
            if mask_var:
                lines.append(f"    if ({mask_var}[_i]) {access} = {val_var}[_i];")
            else:
                lines.append(f"    {access} = {val_var}[_i];")
            lines.append("}")
            return lines
        elif idx_var:
            if mask_var:
                return f"if ({mask_var}) {base_ptr}[{idx_var}] = {val_var};"
            return f"{base_ptr}[{idx_var}] = {val_var};"
        else:
            if mask_var:
                return f"if ({mask_var}) {base_ptr}[0] = {val_var};"
            return f"{base_ptr}[0] = {val_var};"

    @property
    def buffer_map(self) -> dict[int, int]:
        """Mapping of param IR value IDs to Metal buffer indices."""
        return self._param_buffer_map


def compile_to_metal(kernel: IRKernel, constexpr_values: dict[str, int] | None = None) -> tuple[str, dict[int, int]]:
    """Compile IR kernel to MSL source code. Returns (msl_source, buffer_map)."""
    codegen = MetalCodegen(kernel, constexpr_values=constexpr_values)
    source = codegen.generate()
    return source, codegen.buffer_map
