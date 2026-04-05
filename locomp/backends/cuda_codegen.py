"""
Locust CUDA Backend — Compiles IR to CUDA C kernels.

Generates standard CUDA C (.cu) source that can be compiled with nvcc:
  nvcc -arch=sm_80 -O2 -shared -Xcompiler -fPIC -o kernel.so kernel.cu

The generated CUDA C uses:
- blockIdx.x  = program_id(0), blockIdx.y = program_id(1)
- threadIdx.x = local_id(0), threadIdx.y = local_id(1)
- blockDim.x  = group size
- __shared__  = locomp shared_memory()
- __syncthreads() = barrier()
- atomicAdd / atomicMax / atomicMin for atomics
- Warp shuffle intrinsics for SIMD ops (__shfl_down_sync, __reduce_add_sync)

IR → CUDA mapping:
  PROGRAM_ID   → blockIdx.x / .y / .z
  LOCAL_ID     → threadIdx.x / .y / .z
  ARANGE       → blockIdx.x*blockDim.x + threadIdx.x  (or loop)
  LOAD         → pointer dereference / vectorised __ldg
  STORE        → pointer dereference
  ADD/MUL/...  → scalar C arithmetic (CUDA handles SIMT implicitly)
  REDUCE_SUM   → warp shuffle reduction + atomicAdd
  SIMD_SUM     → __shfl_down_sync warp-reduce
  BARRIER      → __syncthreads()
  ATOMIC_ADD   → atomicAdd()
  SHARED_*     → __shared__ array
"""

from __future__ import annotations

from locomp.ir import IRKernel, IROp, IRType, IRValue, OpCode


# ──────────────────────────────────────────────────────────────────────────────
# Type helpers
# ──────────────────────────────────────────────────────────────────────────────

def _c_type(dtype: IRType) -> str:
    return {
        IRType.FLOAT16:  "__half",
        IRType.BFLOAT16: "__nv_bfloat16",
        IRType.FLOAT32:  "float",
        IRType.FLOAT64:  "double",
        IRType.INT8:     "int8_t",
        IRType.INT16:    "int16_t",
        IRType.INT32:    "int32_t",
        IRType.INT64:    "int64_t",
        IRType.UINT8:    "uint8_t",
        IRType.UINT16:   "uint16_t",
        IRType.UINT32:   "uint32_t",
        IRType.UINT64:   "uint64_t",
        IRType.BOOL:     "int",
    }[dtype]


def _math_fn(op: OpCode, dtype: IRType) -> str:
    """Return CUDA math function name for a unary op."""
    use_f = dtype == IRType.FLOAT32
    use_h = dtype in (IRType.FLOAT16, IRType.BFLOAT16)
    # float16/bfloat16 use CUDA half-precision intrinsics (hsqrt, hexp, etc.)
    if use_h:
        return {
            OpCode.SQRT:  "hsqrt",
            OpCode.EXP:   "hexp",
            OpCode.LOG:   "hlog",
            OpCode.ABS:   "__habs",
            OpCode.EXP2:  "hexp2",
            OpCode.LOG2:  "hlog2",
            OpCode.CEIL:  "hceil",
            OpCode.FLOOR: "hfloor",
            OpCode.ROUND: "hrint",
            OpCode.RSQRT: "hrsqrt",
            # No tanh/sin/cos half intrinsics — fall back to float promote
            OpCode.TANH:  "tanhf",
            OpCode.SIN:   "sinf",
            OpCode.COS:   "cosf",
            OpCode.ASIN:  "asinf",
            OpCode.ACOS:  "acosf",
            OpCode.ATAN:  "atanf",
            OpCode.SINH:  "sinhf",
            OpCode.COSH:  "coshf",
            OpCode.LOG10: "log10f",
        }[op]
    return {
        OpCode.SQRT:  "sqrtf" if use_f else "sqrt",
        OpCode.EXP:   "expf"  if use_f else "exp",
        OpCode.LOG:   "logf"  if use_f else "log",
        OpCode.ABS:   "fabsf" if use_f else "fabs",
        OpCode.TANH:  "tanhf" if use_f else "tanh",
        OpCode.SIN:   "sinf"  if use_f else "sin",
        OpCode.COS:   "cosf"  if use_f else "cos",
        OpCode.ASIN:  "asinf" if use_f else "asin",
        OpCode.ACOS:  "acosf" if use_f else "acos",
        OpCode.ATAN:  "atanf" if use_f else "atan",
        OpCode.SINH:  "sinhf" if use_f else "sinh",
        OpCode.COSH:  "coshf" if use_f else "cosh",
        OpCode.EXP2:  "exp2f" if use_f else "exp2",
        OpCode.LOG2:  "log2f" if use_f else "log2",
        OpCode.LOG10: "log10f" if use_f else "log10",
        OpCode.RSQRT: "rsqrtf" if use_f else "rsqrt",
        OpCode.CEIL:  "ceilf" if use_f else "ceil",
        OpCode.FLOOR: "floorf" if use_f else "floor",
        OpCode.ROUND: "roundf" if use_f else "round",
    }[op]


# ──────────────────────────────────────────────────────────────────────────────
# Codegen class
# ──────────────────────────────────────────────────────────────────────────────

class CUDACodegen:
    """Generates CUDA C source from Locust IR.

    The output is a self-contained .cu file exposing:
      extern "C" void locomp_launch_<name>(int grid_x, int grid_y, int block_x,
                                           int block_y, void** args)
    which caller can load via ctypes after compiling to a .so with nvcc.
    """

    def __init__(self, kernel: IRKernel,
                 constexpr_values: dict[str, int | float] | None = None):
        self.kernel = kernel
        self.indent = "    "
        self._var_names: dict[int, str] = {}
        raw = constexpr_values or {}
        self._constexpr_values: dict[str, int | float] = {}
        for k, v in raw.items():
            self._constexpr_values[k] = v
        self._ptr_exprs: dict[int, tuple[str, str, bool]] = {}
        self._nesting: int = 0
        self._predeclared: set[int] = set()

        self._base_names: dict[int, str] = {}
        for p in kernel.params:
            base = p.name.rsplit("_", 1)[0] if "_" in p.name else p.name
            self._base_names[p.id] = base
            if base in raw:
                self._constexpr_values[p.name] = raw[base]
            elif p.name in raw:
                self._constexpr_values[base] = raw[p.name]

    # ── public entry point ────────────────────────────────────────────────────

    def generate(self) -> tuple[str, dict[str, int]]:
        """Return (cuda_source, param_map)."""
        param_map: dict[str, int] = {}
        idx = 0
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                val = (self._constexpr_values.get(base)
                       or self._constexpr_values.get(p.name))
                self._var_names[p.id] = str(val)
            else:
                self._var_names[p.id] = base
                param_map[base] = idx
                idx += 1

        src = self._build_source(param_map)
        return src, param_map

    # ── source builder ────────────────────────────────────────────────────────

    def _build_source(self, param_map: dict[str, int]) -> str:
        lines: list[str] = []

        lines += [
            "/* Generated by locomp — CUDA backend */",
            "#include <stdint.h>",
            "#include <stddef.h>",
            "#include <math.h>",
            "#include <cuda_fp16.h>",
            "#include <cuda_bf16.h>",
        ]

        # Include wmma headers if this kernel uses tensor core ops
        if self._uses_wmma():
            lines += [
                "#include <mma.h>",
                "using namespace nvcuda;",
            ]
        lines.append("")

        # The __global__ kernel
        lines += self._gen_kernel_fn()
        lines.append("")

        # Host-callable launch wrapper
        lines += self._gen_launch_fn(param_map)

        return "\n".join(lines)

    # ── kernel function ───────────────────────────────────────────────────────

    def _gen_kernel_fn(self) -> list[str]:
        # Signature: __global__ void <name>_kernel(params...)
        sig_parts = []
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                continue
            ct = _c_type(p.dtype)
            if p.is_pointer:
                # __restrict__ lets nvcc assume no aliasing → enables auto-vec + LDG
                sig_parts.append(f"{ct}* __restrict__ {base}")
            else:
                sig_parts.append(f"{ct} {base}")

        lines = [
            f"__global__ void {self.kernel.name}_kernel(",
            "    " + ", ".join(sig_parts) if sig_parts else "    void",
            ") {",
        ]
        lines.extend(self._gen_body())
        lines.append("}")
        return lines

    def _gen_body(self) -> list[str]:
        lines: list[str] = []
        self._nesting = 0
        self._ptr_exprs = {}

        # Pre-declare variables defined inside nested scopes (same as riscv backend)
        self._predeclared = set()
        depth = 0
        defined_at_depth: dict[int, int] = {}
        NO_RESULT = {OpCode.STORE, OpCode.SHARED_STORE, OpCode.BARRIER,
                     OpCode.FOR_LOOP_END, OpCode.IF_END, OpCode.WHILE_END,
                     OpCode.ELSE_START, OpCode.BREAK, OpCode.CONTINUE,
                     OpCode.ATOMIC_ADD, OpCode.ATOMIC_MAX, OpCode.ATOMIC_MIN}
        for op in self.kernel.ops:
            if op.opcode in (OpCode.FOR_LOOP_END, OpCode.IF_END, OpCode.WHILE_END):
                depth -= 1
            if op.opcode not in NO_RESULT and op.result is not None:
                defined_at_depth[op.result.id] = depth
            if op.opcode in (OpCode.FOR_LOOP_START, OpCode.IF_START, OpCode.WHILE_START):
                depth += 1

        referenced: set[int] = set()
        for op in self.kernel.ops:
            for o in op.operands:
                referenced.add(o.id)

        # Pre-identify pointer-arithmetic ADD results: these are emitted as `T*`
        # by _gen_arith_binop and must NOT be predeclared as plain `T`.
        # A value is a pointer result if it is a kernel param OR is produced by
        # PTR_ADD or by ADD(pointer, int).  Fixed-point propagation handles chains.
        _ptr_result_ids: set[int] = {p.id for p in self.kernel.params}
        _changed = True
        while _changed:
            _changed = False
            for op in self.kernel.ops:
                if op.result is None:
                    continue
                if op.opcode in (OpCode.PTR_ADD, OpCode.ADD):
                    if any(o.id in _ptr_result_ids or getattr(o, 'is_pointer', False)
                           for o in op.operands):
                        if op.result.id not in _ptr_result_ids:
                            _ptr_result_ids.add(op.result.id)
                            _changed = True

        for op in self.kernel.ops:
            if op.result is None:
                continue
            rid = op.result.id
            if (defined_at_depth.get(rid, 0) > 0
                    and rid in referenced
                    and op.result.aliases is None
                    and op.opcode not in (OpCode.FOR_LOOP_START, OpCode.CONSTANT)
                    and not op.result.is_pointer
                    and op.opcode != OpCode.PTR_ADD
                    and rid not in _ptr_result_ids
                    and not getattr(op.result, 'is_simdgroup_matrix', False)):
                var = self._vname(op.result)
                ct = _c_type(op.result.dtype)
                lines.append(f"{self.indent}{ct} {var};")
                self._predeclared.add(rid)

        if self._predeclared:
            lines.append("")

        # Emit __shared__ array declarations (must be at top of kernel body)
        for smem_name, (smem_dtype, smem_size) in self.kernel.shared_mem.items():
            ct = _c_type(smem_dtype)
            lines.append(f"{self.indent}__shared__ {ct} {smem_name}[{smem_size}];")
        if self.kernel.shared_mem:
            lines.append("")

        for op in self.kernel.ops:
            if op.opcode in (OpCode.FOR_LOOP_END, OpCode.IF_END, OpCode.WHILE_END):
                self._nesting -= 1
                lines.append(self.indent * (1 + self._nesting) + "}")
                continue
            if op.opcode == OpCode.ELSE_START:
                self._nesting -= 1
                lines.append(self.indent * (1 + self._nesting) + "} else {")
                self._nesting += 1
                continue

            ind = self.indent * (1 + self._nesting)
            emitted = self._gen_op(op)
            if emitted is None:
                pass
            elif isinstance(emitted, list):
                for l in emitted:
                    full = f"{ind}{l}"
                    if op.result is not None:
                        full = self._strip_type(full, op.result.id)
                    lines.append(full)
            else:
                full = f"{ind}{emitted}"
                if op.result is not None:
                    full = self._strip_type(full, op.result.id)
                lines.append(full)

            if op.opcode in (OpCode.FOR_LOOP_START, OpCode.IF_START, OpCode.WHILE_START):
                self._nesting += 1

        return lines

    # ── launch function ───────────────────────────────────────────────────────

    def _gen_launch_fn(self, param_map: dict[str, int]) -> list[str]:
        """extern "C" launch wrapper callable from Python ctypes."""
        lines = [
            'extern "C" void locomp_launch_' + self.kernel.name + '(',
            "    int _grid_x, int _grid_y, int _block_x, int _block_y,",
            "    void** _args",
            ") {",
        ]

        # Unpack void** args
        idx = 0
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if is_constexpr:
                continue
            ct = _c_type(p.dtype)
            if p.is_pointer:
                lines.append(f"    {ct}* {base} = ({ct}*)_args[{idx}];")
            else:
                lines.append(f"    {ct} {base} = *({ct}*)_args[{idx}];")
            idx += 1

        # Build kernel call args
        call_args = []
        for p in self.kernel.params:
            base = self._base_names[p.id]
            is_constexpr = (base in self._constexpr_values or
                            p.name in self._constexpr_values)
            if not is_constexpr:
                call_args.append(base)

        lines += [
            "    dim3 _grid(_grid_x, _grid_y, 1);",
            "    dim3 _block(_block_x, _block_y, 1);",
            f"    {self.kernel.name}_kernel<<<_grid, _block>>>({', '.join(call_args)});",
            "    // caller is responsible for synchronization",
            "}",
        ]
        return lines

    # ── op codegen ────────────────────────────────────────────────────────────

    def _gen_op(self, op: IROp) -> str | list[str] | None:
        rv = self._vname(op.result) if op.result else "_"

        # ── thread indexing ──────────────────────────────────────────────────
        if op.opcode == OpCode.PROGRAM_ID:
            axis = op.attrs.get("axis", 0)
            dim = ["x", "y", "z"][axis]
            return f"int {rv} = blockIdx.{dim};"

        elif op.opcode == OpCode.THREAD_ID:
            return f"int {rv} = blockIdx.x * blockDim.x + threadIdx.x;"

        elif op.opcode == OpCode.LOCAL_ID:
            axis = op.attrs.get("axis", 0)
            dim = ["x", "y", "z"][axis]
            return f"int {rv} = threadIdx.{dim};"

        elif op.opcode == OpCode.GROUP_SIZE:
            return f"int {rv} = blockDim.x;"

        elif op.opcode == OpCode.NUM_GROUPS:
            return f"int {rv} = gridDim.x;"

        # ── arange ──────────────────────────────────────────────────────────
        elif op.opcode == OpCode.ARANGE:
            start = op.attrs["start"]
            end = op.attrs["end"]
            size = end - start
            ct = _c_type(op.result.dtype)
            # In CUDA: each block thread handles one element — arange is
            # the per-thread offset array. We materialise as a local C array.
            return [
                f"{ct} {rv}[{size}];",
                f"for (int _i = 0; _i < {size}; _i++) {{ {rv}[_i] = {start} + _i; }}",
            ]

        # ── constants ────────────────────────────────────────────────────────
        elif op.opcode == OpCode.CONSTANT:
            if "shared_mem" in op.attrs:
                smem_name = op.attrs["shared_mem"]
                self._var_names[op.result.id] = smem_name
                return None
            val = op.attrs["value"]
            ct = _c_type(op.result.dtype)
            if isinstance(val, bool):
                return f"{ct} {rv} = {1 if val else 0};"
            elif isinstance(val, float):
                return f"{ct} {rv} = {val}f;"
            else:
                return f"{ct} {rv} = {val};"

        elif op.opcode == OpCode.CONSTEXPR:
            val = op.attrs.get("value", 0)
            return f"int {rv} = {val};"

        # ── pointer arithmetic ───────────────────────────────────────────────
        elif op.opcode == OpCode.PTR_ADD:
            base = self._vname(op.operands[0])
            offset = self._vname(op.operands[1])
            self._ptr_exprs[op.result.id] = (base, offset, False)
            ct = _c_type(op.result.dtype)
            return f"{ct}* {rv} = {base} + {offset};"

        # ── memory ───────────────────────────────────────────────────────────
        elif op.opcode == OpCode.LOAD:
            return self._gen_load(op)

        elif op.opcode == OpCode.STORE:
            return self._gen_store(op)

        elif op.opcode == OpCode.SHARED_LOAD:
            arr = self._vname(op.operands[0])
            idx = self._vname(op.operands[1])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {arr}[{idx}];"

        elif op.opcode == OpCode.SHARED_STORE:
            arr = self._vname(op.operands[0])
            idx = self._vname(op.operands[1])
            val = self._vname(op.operands[2])
            return f"{arr}[{idx}] = {val};"

        elif op.opcode == OpCode.BARRIER:
            return "__syncthreads();"

        # ── arithmetic ───────────────────────────────────────────────────────
        elif op.opcode in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD):
            return self._gen_arith_binop(op)

        elif op.opcode in (OpCode.BIT_AND, OpCode.BIT_OR, OpCode.BIT_XOR,
                           OpCode.LSHIFT, OpCode.RSHIFT):
            sym = {OpCode.BIT_AND: "&", OpCode.BIT_OR: "|", OpCode.BIT_XOR: "^",
                   OpCode.LSHIFT: "<<", OpCode.RSHIFT: ">>"}[op.opcode]
            a = self._vname(op.operands[0])
            b = self._vname(op.operands[1])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a} {sym} {b};"

        elif op.opcode == OpCode.NEG:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = -{a};"

        # ── math functions ────────────────────────────────────────────────────
        elif op.opcode in (OpCode.SQRT, OpCode.EXP, OpCode.LOG, OpCode.ABS,
                           OpCode.TANH, OpCode.SIN, OpCode.COS,
                           OpCode.ASIN, OpCode.ACOS, OpCode.ATAN,
                           OpCode.SINH, OpCode.COSH,
                           OpCode.EXP2, OpCode.LOG2, OpCode.LOG10,
                           OpCode.RSQRT, OpCode.CEIL, OpCode.FLOOR, OpCode.ROUND):
            fn = _math_fn(op.opcode, op.result.dtype)
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            if op.result.shape:
                size = op.result.shape[0]
                return [
                    f"{ct} {rv}[{size}];",
                    f"for (int _i = 0; _i < {size}; _i++) {{ {rv}[_i] = {fn}({a}[_i]); }}",
                ]
            return f"{ct} {rv} = {fn}({a});"

        elif op.opcode == OpCode.SIGMOID:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = 1.0f / (1.0f + expf(-{a}));"

        elif op.opcode in (OpCode.POW, OpCode.ATAN2, OpCode.COPYSIGN,
                           OpCode.FMOD, OpCode.STEP, OpCode.MAX, OpCode.MIN):
            fn_map = {OpCode.POW: "powf", OpCode.ATAN2: "atan2f",
                      OpCode.COPYSIGN: "copysignf", OpCode.FMOD: "fmodf",
                      OpCode.MAX: "fmaxf", OpCode.MIN: "fminf"}
            a = self._vname(op.operands[0])
            b = self._vname(op.operands[1])
            ct = _c_type(op.result.dtype)
            if op.opcode == OpCode.STEP:
                return f"{ct} {rv} = ({a} < {b}) ? 0.0f : 1.0f;"
            return f"{ct} {rv} = {fn_map[op.opcode]}({a}, {b});"

        elif op.opcode == OpCode.FMA:
            a = self._vname(op.operands[0])
            b = self._vname(op.operands[1])
            c = self._vname(op.operands[2])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = fmaf({a}, {b}, {c});"

        elif op.opcode == OpCode.CLAMP:
            a = self._vname(op.operands[0])
            lo = self._vname(op.operands[1])
            hi = self._vname(op.operands[2])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = fmaxf({lo}, fminf({hi}, {a}));"

        # ── comparisons ───────────────────────────────────────────────────────
        elif op.opcode in (OpCode.CMP_LT, OpCode.CMP_LE, OpCode.CMP_GT,
                           OpCode.CMP_GE, OpCode.CMP_EQ, OpCode.CMP_NE):
            sym = {OpCode.CMP_LT: "<", OpCode.CMP_LE: "<=", OpCode.CMP_GT: ">",
                   OpCode.CMP_GE: ">=", OpCode.CMP_EQ: "==", OpCode.CMP_NE: "!="}[op.opcode]
            a = self._vname(op.operands[0])
            b = self._vname(op.operands[1])
            if op.result.shape:
                size = op.result.shape[0]
                a_tile = bool(op.operands[0].shape)
                b_tile = bool(op.operands[1].shape)
                return [
                    f"int {rv}[{size}];",
                    f"for (int _i = 0; _i < {size}; _i++) {{",
                    f"    {rv}[_i] = ({a if not a_tile else a+'[_i]'} {sym} {b if not b_tile else b+'[_i]'}) ? 1 : 0;",
                    "}",
                ]
            return f"int {rv} = ({a} {sym} {b}) ? 1 : 0;"

        # ── reductions ────────────────────────────────────────────────────────
        elif op.opcode == OpCode.REDUCE_SUM:
            return self._gen_reduce(op, "sum")

        elif op.opcode == OpCode.REDUCE_MAX:
            return self._gen_reduce(op, "max")

        elif op.opcode == OpCode.REDUCE_MIN:
            return self._gen_reduce(op, "min")

        # ── SIMD / warp ops ───────────────────────────────────────────────────
        elif op.opcode == OpCode.SIMD_SUM:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            # Warp reduction using shuffle
            return [
                f"{ct} {rv} = {a};",
                f"#pragma unroll",
                f"for (int _off = 16; _off > 0; _off >>= 1)",
                f"    {rv} += __shfl_down_sync(0xffffffff, {rv}, _off);",
            ]

        elif op.opcode == OpCode.SIMD_MAX:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return [
                f"{ct} {rv} = {a};",
                f"#pragma unroll",
                f"for (int _off = 16; _off > 0; _off >>= 1)",
                f"    {rv} = fmaxf({rv}, __shfl_down_sync(0xffffffff, {rv}, _off));",
            ]

        elif op.opcode == OpCode.SIMD_MIN:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return [
                f"{ct} {rv} = {a};",
                f"#pragma unroll",
                f"for (int _off = 16; _off > 0; _off >>= 1)",
                f"    {rv} = fminf({rv}, __shfl_down_sync(0xffffffff, {rv}, _off));",
            ]

        elif op.opcode == OpCode.SIMD_BROADCAST:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            # broadcast lane 0 to all lanes
            return f"{ct} {rv} = __shfl_sync(0xffffffff, {a}, 0);"

        elif op.opcode == OpCode.SIMD_SHUFFLE_DOWN:
            a = self._vname(op.operands[0])
            delta = self._vname(op.operands[1]) if len(op.operands) > 1 else "1"
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = __shfl_down_sync(0xffffffff, {a}, {delta});"

        elif op.opcode == OpCode.SIMD_LANE_ID:
            return f"int {rv} = threadIdx.x % 32;"

        elif op.opcode == OpCode.SIMD_GROUP_ID:
            return f"int {rv} = threadIdx.x / 32;"

        # ── cast ──────────────────────────────────────────────────────────────
        elif op.opcode == OpCode.CAST:
            a = self._vname(op.operands[0])
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = ({ct}){a};"

        # ── where / select ────────────────────────────────────────────────────
        elif op.opcode == OpCode.WHERE:
            cond = self._vname(op.operands[0])
            a = self._vname(op.operands[1])
            b = self._vname(op.operands[2])
            ct = _c_type(op.result.dtype)
            if op.result.shape:
                size = op.result.shape[0]
                cond_tile = bool(op.operands[0].shape)
                a_tile = bool(op.operands[1].shape)
                b_tile = bool(op.operands[2].shape)
                return [
                    f"{ct} {rv}[{size}];",
                    f"for (int _i = 0; _i < {size}; _i++) {{",
                    f"    {rv}[_i] = ({cond if not cond_tile else cond+'[_i]'}) ? "
                    f"({a if not a_tile else a+'[_i]'}) : ({b if not b_tile else b+'[_i]'});",
                    "}",
                ]
            return f"{ct} {rv} = {cond} ? {a} : {b};"

        # ── copy ──────────────────────────────────────────────────────────────
        elif op.opcode == OpCode.COPY:
            a = self._vname(op.operands[0])
            if op.result.aliases is not None:
                return f"{rv} = {a};"
            ct = _c_type(op.result.dtype)
            return f"{ct} {rv} = {a};"

        # ── atomics ───────────────────────────────────────────────────────────
        elif op.opcode in (OpCode.ATOMIC_ADD, OpCode.ATOMIC_MAX, OpCode.ATOMIC_MIN):
            return self._gen_atomic(op)

        # ── control flow ─────────────────────────────────────────────────────
        elif op.opcode == OpCode.FOR_LOOP_START:
            start = self._vname(op.operands[0])
            end = self._vname(op.operands[1])
            step = self._vname(op.operands[2])
            loop_var = self._vname(op.result)
            return f"for (int {loop_var} = {start}; {loop_var} < {end}; {loop_var} += {step}) {{"

        elif op.opcode == OpCode.WHILE_START:
            return "while (1) {"

        elif op.opcode == OpCode.IF_START:
            cond = self._vname(op.operands[0])
            return f"if ({cond}) {{"

        elif op.opcode in (OpCode.FOR_LOOP_END, OpCode.IF_END, OpCode.WHILE_END,
                           OpCode.ELSE_START):
            return None

        elif op.opcode == OpCode.BREAK:
            return "break;"

        elif op.opcode == OpCode.CONTINUE:
            return "continue;"

        # ── simdgroup matrix → CUDA wmma tensor core ops ─────────────────────
        elif op.opcode == OpCode.SIMDGROUP_MATRIX_FILL:
            rv = self._vname(op.result)
            fill = self._vname(op.operands[0])
            frag_type = self._wmma_fragment_type(op.result.dtype, role="acc")
            lines = [f"wmma::fragment<{frag_type}> {rv};"]
            lines.append(f"wmma::fill_fragment({rv}, ({_c_type(op.result.dtype)}){fill});")
            return lines

        elif op.opcode == OpCode.SIMDGROUP_MATRIX_LOAD:
            rv = self._vname(op.result)
            source = op.attrs.get("source", "shared")
            role   = op.attrs.get("role", "a")   # "a", "b", or "acc"
            layout = "wmma::row_major" if op.attrs.get("row_major", True) else "wmma::col_major"
            frag_type = self._wmma_fragment_type(op.result.dtype, role=role, layout=layout)
            # Fragment input tiles always need __half* — cast inline to avoid
            # both the type-mismatch compile error AND the float/float* redeclaration.
            input_cast = "(const __half*)" if role in ("a", "b") else ""
            if source == "shared":
                arr    = self._vname(op.operands[0])
                offset = self._vname(op.operands[1])
                stride = self._vname(op.operands[2])
                lines = [f"wmma::fragment<{frag_type}> {rv};"]
                lines.append(f"wmma::load_matrix_sync({rv}, {input_cast}({arr} + {offset}), {stride});")
            else:  # device pointer — resolve pointer expression inline (no temp variable)
                ptr_op = op.operands[0]
                if ptr_op.id in self._ptr_exprs:
                    pb, pi, _ = self._ptr_exprs[ptr_op.id]
                    ptr_expr = f"({pb} + {pi})"
                else:
                    ptr_expr = self._vname(ptr_op)
                stride = self._vname(op.operands[1])
                lines = [f"wmma::fragment<{frag_type}> {rv};"]
                lines.append(f"wmma::load_matrix_sync({rv}, {input_cast}{ptr_expr}, {stride});")
            return lines

        elif op.opcode == OpCode.SIMDGROUP_MATRIX_STORE:
            dest   = op.attrs.get("dest", "shared")
            mat    = self._vname(op.operands[0])
            layout = "wmma::mem_row_major" if op.attrs.get("row_major", True) else "wmma::mem_col_major"
            if dest == "shared":
                arr    = self._vname(op.operands[1])
                offset = self._vname(op.operands[2])
                stride = self._vname(op.operands[3])
                return f"wmma::store_matrix_sync({arr} + {offset}, {mat}, {stride}, {layout});"
            else:  # device pointer — resolve inline to avoid float/float* redeclaration
                ptr_op = op.operands[1]
                if ptr_op.id in self._ptr_exprs:
                    pb, pi, _ = self._ptr_exprs[ptr_op.id]
                    ptr_expr = f"({pb} + {pi})"
                else:
                    ptr_expr = self._vname(ptr_op)
                stride = self._vname(op.operands[2])
                return f"wmma::store_matrix_sync({ptr_expr}, {mat}, {stride}, {layout});"

        elif op.opcode == OpCode.SIMDGROUP_MATRIX_MAC:
            rv  = self._vname(op.result)
            acc = self._vname(op.operands[0])
            a   = self._vname(op.operands[1])
            b   = self._vname(op.operands[2])
            # wmma::mma_sync(d, a, b, c)  — d = a*b + c
            if op.result.aliases is not None:
                return f"wmma::mma_sync({rv}, {a}, {b}, {acc});"
            # new accumulator fragment — copy c into d first then accumulate
            frag_type = self._wmma_fragment_type(op.result.dtype, role="acc")
            return [
                f"wmma::fragment<{frag_type}> {rv};",
                f"wmma::mma_sync({rv}, {a}, {b}, {acc});",
            ]

        else:
            return f"/* unhandled op: {op.opcode.name} */"

    # ── wmma / tensor core helpers ───────────────────────────────────────────

    def _uses_wmma(self) -> bool:
        """Return True if any op in this kernel uses simdgroup/wmma matrix ops."""
        wmma_ops = {OpCode.SIMDGROUP_MATRIX_LOAD, OpCode.SIMDGROUP_MATRIX_STORE,
                    OpCode.SIMDGROUP_MATRIX_MAC, OpCode.SIMDGROUP_MATRIX_FILL}
        return any(op.opcode in wmma_ops for op in self.kernel.ops)

    def _wmma_fragment_type(self, dtype: IRType, role: str = "a",
                             layout: str = "wmma::row_major") -> str:
        """Return the wmma::fragment<...> template arguments string.

        CUDA wmma uses 16x16x16 tiles for fp16, 16x16x8 for tf32.
        We always use the 16x16x16 fp16 variant since locomp maps
        Metal's 8x8 simdgroup ops to the smallest CUDA tensor core tile.

        role: "a" | "b" | "acc"
        """
        if role == "acc":
            # Accumulator is always float32
            return "wmma::accumulator, 16, 16, 16, float"
        ct = "__half" if dtype == IRType.FLOAT16 else "__half"  # always fp16 input tiles
        if role == "b":
            return f"wmma::matrix_b, 16, 16, 16, {ct}, {layout}"
        return f"wmma::matrix_a, 16, 16, 16, {ct}, {layout}"

    # ── load / store helpers ─────────────────────────────────────────────────

    def _gen_load(self, op: IROp) -> str | list[str]:
        rv = self._vname(op.result)
        ptr_val = op.operands[0]
        ct = _c_type(op.result.dtype)

        is_tiled = False
        pb = None
        pi = None
        if ptr_val.id in self._ptr_exprs:
            pb, pi, is_tiled = self._ptr_exprs[ptr_val.id]
            ptr_expr = pb if is_tiled else f"({pb} + {pi})"
        else:
            ptr_expr = self._vname(ptr_val)

        if op.result.shape:
            size = op.result.shape[0]
            if is_tiled and pi is not None:
                # Indexed tiled load: rv[_i] = pb[ pi[_i] ]
                # float4 path: assume pi is a linear index (pi[_i] = pi[0] + _i)
                # which holds for all ARANGE-derived index arrays.
                if ct == "float" and size % 4 == 0:
                    n4 = size // 4
                    # Use .x/.y/.z/.w to keep data in registers — avoids
                    # unaligned local-memory float4* cast of stack float[].
                    ls = [f"float {rv}[{size}];",
                          f"{{ float4* _src4 = (float4*)({pb} + {pi}[0]);"]
                    ls.append(f"  for (int _j = 0; _j < {n4}; _j++) {{")
                    ls.append(f"    float4 _v = __ldg(_src4 + _j);")
                    ls.append(f"    {rv}[_j*4+0]=_v.x; {rv}[_j*4+1]=_v.y;")
                    ls.append(f"    {rv}[_j*4+2]=_v.z; {rv}[_j*4+3]=_v.w; }}")
                    ls.append(f"}}")
                    return ls
                return [
                    f"{ct} {rv}[{size}];",
                    f"for (int _i = 0; _i < {size}; _i++) {{ {rv}[_i] = __ldg({pb} + {pi}[_i]); }}",
                ]
            # Non-indexed tiled load (ptr_expr already has embedded offset)
            if ct == "float" and size % 4 == 0:
                n4 = size // 4
                ls = [f"float {rv}[{size}];",
                      f"{{ float4* _src4 = (float4*)({ptr_expr});"]
                ls.append(f"  for (int _j = 0; _j < {n4}; _j++) {{")
                ls.append(f"    float4 _v = __ldg(_src4 + _j);")
                ls.append(f"    {rv}[_j*4+0]=_v.x; {rv}[_j*4+1]=_v.y;")
                ls.append(f"    {rv}[_j*4+2]=_v.z; {rv}[_j*4+3]=_v.w; }}")
                ls.append(f"}}")
                return ls
            return [
                f"{ct} {rv}[{size}];",
                f"for (int _i = 0; _i < {size}; _i++) {{ {rv}[_i] = __ldg({ptr_expr} + _i); }}",
            ]

        mask = op.attrs.get("mask")
        other = op.attrs.get("other")
        if mask is not None:
            from locomp.ir import IRValue as _IRV
            mask_var = self._vname(mask) if isinstance(mask, _IRV) else str(mask)
            other_val = self._vname(other) if isinstance(other, _IRV) else str(other)
            return [
                f"{ct} {rv};",
                f"if ({mask_var}) {{ {rv} = __ldg({ptr_expr}); }}",
                f"else {{ {rv} = ({ct}){other_val}; }}",
            ]
        # __ldg: L1 read-only cache (non-coherent load) — optimal for streaming reads
        return f"{ct} {rv} = __ldg({ptr_expr});"

    def _gen_store(self, op: IROp) -> str | list[str]:
        ptr_val = op.operands[0]
        val_val = op.operands[1]

        is_tiled = False
        pb = None
        pi = None
        if ptr_val.id in self._ptr_exprs:
            pb, pi, is_tiled = self._ptr_exprs[ptr_val.id]
            ptr_expr = pb if is_tiled else f"({pb} + {pi})"
        else:
            ptr_expr = self._vname(ptr_val)

        val = self._vname(val_val)

        if val_val.shape:
            size = val_val.shape[0]
            ct = _c_type(val_val.dtype)
            if is_tiled and pi is not None:
                # Indexed tiled store: pb[ pi[_i] ] = val[_i]
                if ct == "float" and size % 4 == 0:
                    n4 = size // 4
                    # make_float4 keeps data in registers — avoids unaligned
                    # local-memory cast of stack float[] to float4*.
                    ls = [f"{{ float4* _dst4 = (float4*)({pb} + {pi}[0]);"]
                    ls.append(f"  for (int _j = 0; _j < {n4}; _j++)")
                    ls.append(f"    _dst4[_j] = make_float4({val}[_j*4+0],{val}[_j*4+1],"
                              f"{val}[_j*4+2],{val}[_j*4+3]);")
                    ls.append(f"}}")
                    return ls
                return [
                    f"for (int _i = 0; _i < {size}; _i++) {{ {pb}[{pi}[_i]] = {val}[_i]; }}",
                ]
            # Non-indexed tiled store (ptr_expr has embedded offset)
            if ct == "float" and size % 4 == 0:
                n4 = size // 4
                ls = [f"{{ float4* _dst4 = (float4*)({ptr_expr});"]
                ls.append(f"  for (int _j = 0; _j < {n4}; _j++)")
                ls.append(f"    _dst4[_j] = make_float4({val}[_j*4+0],{val}[_j*4+1],"
                          f"{val}[_j*4+2],{val}[_j*4+3]);")
                ls.append(f"}}")
                return ls
            return [
                f"for (int _i = 0; _i < {size}; _i++) {{ {ptr_expr}[_i] = {val}[_i]; }}",
            ]

        mask = op.attrs.get("mask")
        if mask is not None:
            from locomp.ir import IRValue as _IRV
            mask_var = self._vname(mask) if isinstance(mask, _IRV) else str(mask)
            return [f"if ({mask_var}) {{ *{ptr_expr} = {val}; }}"]
        return f"*{ptr_expr} = {val};"

    # ── arithmetic helpers ───────────────────────────────────────────────────

    def _gen_arith_binop(self, op: IROp) -> str | list[str]:
        rv = self._vname(op.result)
        a = self._vname(op.operands[0])
        b = self._vname(op.operands[1])
        ct = _c_type(op.result.dtype)

        # Pointer arithmetic
        a_ptr = op.operands[0].is_pointer or (op.operands[0].id in self._ptr_exprs)
        b_ptr = op.operands[1].is_pointer or (op.operands[1].id in self._ptr_exprs)
        if (op.opcode == OpCode.ADD) and (a_ptr or b_ptr):
            ptr_op = op.operands[0] if a_ptr else op.operands[1]
            idx_op = op.operands[1] if a_ptr else op.operands[0]
            ptr_var = a if a_ptr else b
            idx_var = b if a_ptr else a
            if ptr_op.id in self._ptr_exprs:
                pb, pi, pt = self._ptr_exprs[ptr_op.id]
                base_expr = pb if pt else f"({pb} + {pi})"
            else:
                base_expr = ptr_var
            if idx_op.shape:
                self._ptr_exprs[op.result.id] = (base_expr, idx_var, True)
                return None
            else:
                self._ptr_exprs[op.result.id] = (base_expr, idx_var, False)
                return f"{ct}* {rv} = {base_expr} + {idx_var};"

        sym = {OpCode.ADD: "+", OpCode.SUB: "-", OpCode.MUL: "*",
               OpCode.DIV: "/", OpCode.MOD: "%"}.get(op.opcode, "+")

        if op.result.shape:
            size = op.result.shape[0]
            a_tile = bool(op.operands[0].shape)
            b_tile = bool(op.operands[1].shape)
            return [
                f"{ct} {rv}[{size}];",
                f"for (int _i = 0; _i < {size}; _i++) {{",
                f"    {rv}[_i] = {a if not a_tile else a+'[_i]'} {sym} {b if not b_tile else b+'[_i]'};",
                "}",
            ]

        return f"{ct} {rv} = {a} {sym} {b};"

    # ── reduce helpers ───────────────────────────────────────────────────────

    def _gen_reduce(self, op: IROp, kind: str) -> str | list[str]:
        rv = self._vname(op.result)
        arg = self._vname(op.operands[0])
        ct = _c_type(op.result.dtype)

        if op.operands[0].shape:
            # Reduce a local tiled array (not a cross-thread reduction)
            size = op.operands[0].shape[0]
            init = {"sum": "0.0f", "max": "-1e38f", "min": "1e38f"}[kind]
            op_expr = {"sum": f"{rv} += {arg}[_i]",
                       "max": f"{rv} = fmaxf({rv}, {arg}[_i])",
                       "min": f"{rv} = fminf({rv}, {arg}[_i])"}[kind]
            return [
                f"{ct} {rv} = {init};",
                f"for (int _i = 0; _i < {size}; _i++) {{ {op_expr}; }}",
            ]

        # 2-operand form: REDUCE_SUM(val, out_ptr) — atomicAdd to global output
        if len(op.operands) == 2:
            val = arg
            out_ptr = self._vname(op.operands[1])
            if kind == "sum":
                return f"atomicAdd({out_ptr}, {val});"
            elif kind == "max":
                # No atomicMax for float — use CAS loop
                return [
                    f"{{ int* _addr = (int*){out_ptr};",
                    f"  int _old = *_addr, _assumed;",
                    f"  float _val = {val};",
                    f"  do {{ _assumed = _old;",
                    f"       _old = atomicCAS(_addr, _assumed,",
                    f"           __float_as_int(fmaxf(__int_as_float(_assumed), _val)));",
                    f"  }} while (_assumed != _old); }}",
                ]
            elif kind == "min":
                return [
                    f"{{ int* _addr = (int*){out_ptr};",
                    f"  int _old = *_addr, _assumed;",
                    f"  float _val = {val};",
                    f"  do {{ _assumed = _old;",
                    f"       _old = atomicCAS(_addr, _assumed,",
                    f"           __float_as_int(fminf(__int_as_float(_assumed), _val)));",
                    f"  }} while (_assumed != _old); }}",
                ]

        return f"{ct} {rv} = {arg};"

    # ── atomic helpers ───────────────────────────────────────────────────────

    def _gen_atomic(self, op: IROp) -> str | list[str]:
        ptr_val = op.operands[0]
        val_val = op.operands[1]
        if ptr_val.id in self._ptr_exprs:
            pb, pi, _ = self._ptr_exprs[ptr_val.id]
            ptr_expr = f"({pb} + {pi})"
        else:
            ptr_expr = self._vname(ptr_val)
        val = self._vname(val_val)

        if op.opcode == OpCode.ATOMIC_ADD:
            return f"atomicAdd({ptr_expr}, {val});"
        elif op.opcode == OpCode.ATOMIC_MAX:
            ct = _c_type(val_val.dtype)
            if val_val.dtype == IRType.INT32:
                return f"atomicMax((int*){ptr_expr}, (int){val});"
            # Float atomicMax via CAS
            return [
                f"{{ int* _addr = (int*){ptr_expr};",
                f"  int _old = *_addr, _assumed;",
                f"  do {{ _assumed = _old;",
                f"       _old = atomicCAS(_addr, _assumed,",
                f"           __float_as_int(fmaxf(__int_as_float(_assumed), {val})));",
                f"  }} while (_assumed != _old); }}",
            ]
        elif op.opcode == OpCode.ATOMIC_MIN:
            if val_val.dtype == IRType.INT32:
                return f"atomicMin((int*){ptr_expr}, (int){val});"
            return [
                f"{{ int* _addr = (int*){ptr_expr};",
                f"  int _old = *_addr, _assumed;",
                f"  do {{ _assumed = _old;",
                f"       _old = atomicCAS(_addr, _assumed,",
                f"           __float_as_int(fminf(__int_as_float(_assumed), {val})));",
                f"  }} while (_assumed != _old); }}",
            ]
        return "/* atomic op unhandled */;"

    # ── variable name helpers ────────────────────────────────────────────────

    def _vname(self, value: IRValue) -> str:
        if value is None:
            return "_"
        if value.aliases is not None and value.aliases in self._var_names:
            self._var_names[value.id] = self._var_names[value.aliases]
            return self._var_names[value.id]
        if value.id not in self._var_names:
            base = value.name.replace(".", "_")
            self._var_names[value.id] = base
        return self._var_names[value.id]

    def _strip_type(self, line: str, result_id: int) -> str:
        if result_id not in self._predeclared:
            return line
        stripped = line.lstrip()
        if '* ' in stripped.split('=')[0] or stripped.split('=')[0].rstrip().endswith('*'):
            return line
        for prefix in ("float ", "int ", "double ", "int8_t ", "int16_t ",
                       "int32_t ", "int64_t ", "uint8_t ", "uint16_t ",
                       "uint32_t ", "uint64_t ", "__half ", "__nv_bfloat16 "):
            if stripped.startswith(prefix):
                indent = line[:len(line) - len(stripped)]
                return indent + stripped[len(prefix):]
        return line


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def compile_to_cuda(kernel: IRKernel,
                    constexpr_values: dict[str, int | float] | None = None
                    ) -> tuple[str, dict[str, int]]:
    """Compile IR kernel to CUDA C source.

    Returns:
        (cuda_source, param_map)  — param_map: {param_name → arg_index}
    """
    cg = CUDACodegen(kernel, constexpr_values=constexpr_values)
    return cg.generate()
