"""
Locust IR — Intermediate Representation for GPU compute kernels.

The IR represents tiled parallel computations as a graph of operations.
Each operation maps to GPU concepts: thread indexing, memory loads/stores,
arithmetic, reductions, and control flow.

IR Design Principles:
1. Tiled — operations work on blocks/tiles, not individual elements
2. Explicit memory — loads and stores are explicit (not implicit like NumPy)
3. Backend-agnostic — no Metal-specific details leak into the IR
4. SSA form — each value defined exactly once (Static Single Assignment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class IRType(Enum):
    """Data types supported in the IR."""
    FLOAT16 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    UINT64 = auto()
    BOOL = auto()

    @property
    def bytewidth(self) -> int:
        _widths = {
            IRType.FLOAT16: 2, IRType.FLOAT32: 4, IRType.FLOAT64: 8,
            IRType.INT8: 1, IRType.INT16: 2, IRType.INT32: 4, IRType.INT64: 8,
            IRType.UINT8: 1, IRType.UINT16: 2, IRType.UINT32: 4, IRType.UINT64: 8,
            IRType.BOOL: 1,
        }
        return _widths[self]

    @property
    def is_float(self) -> bool:
        return self in (IRType.FLOAT16, IRType.FLOAT32, IRType.FLOAT64)

    @property
    def is_int(self) -> bool:
        return not self.is_float and self != IRType.BOOL

    def to_msl(self) -> str:
        """Convert to Metal Shading Language type string."""
        _msl = {
            IRType.FLOAT16: "half", IRType.FLOAT32: "float", IRType.FLOAT64: "float",
            IRType.INT8: "char", IRType.INT16: "short", IRType.INT32: "int", IRType.INT64: "long",
            IRType.UINT8: "uchar", IRType.UINT16: "ushort", IRType.UINT32: "uint",
            IRType.UINT64: "ulong", IRType.BOOL: "bool",
        }
        return _msl[self]


class OpCode(Enum):
    """IR operation codes."""
    # Thread indexing
    PROGRAM_ID = auto()       # Get block/threadgroup index
    THREAD_ID = auto()        # Get global thread index in grid
    LOCAL_ID = auto()         # Get thread index within threadgroup
    GROUP_SIZE = auto()       # Get threadgroup size
    NUM_GROUPS = auto()       # Get number of threadgroups
    ARANGE = auto()           # Generate range [0, N) within a tile

    # Memory
    LOAD = auto()             # Load from global memory
    STORE = auto()            # Store to global memory
    SHARED_LOAD = auto()      # Load from threadgroup shared memory
    SHARED_STORE = auto()     # Store to threadgroup shared memory

    # Synchronization
    BARRIER = auto()          # Threadgroup barrier (sync all threads in group)

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()

    # Math functions
    SQRT = auto()
    EXP = auto()
    LOG = auto()
    ABS = auto()
    MAX = auto()
    MIN = auto()
    # Math functions — extended
    TANH = auto()
    SIN = auto()
    COS = auto()
    ASIN = auto()
    ACOS = auto()
    ATAN = auto()
    ATAN2 = auto()
    SINH = auto()
    COSH = auto()
    EXP2 = auto()
    LOG2 = auto()
    LOG10 = auto()
    RSQRT = auto()
    CEIL = auto()
    FLOOR = auto()
    ROUND = auto()
    FMA = auto()           # fma(a, b, c) = a*b + c
    POW = auto()           # pow(x, y)
    CLAMP = auto()         # clamp(x, lo, hi)
    COPYSIGN = auto()      # copysign(x, y)
    FMOD = auto()          # fmod(x, y)
    STEP = auto()          # step(edge, x) → 0.0 if x < edge, else 1.0
    SIGMOID = auto()       # 1 / (1 + exp(-x))

    # Comparison
    CMP_LT = auto()
    CMP_LE = auto()
    CMP_GT = auto()
    CMP_GE = auto()
    CMP_EQ = auto()
    CMP_NE = auto()

    # Reductions
    REDUCE_SUM = auto()
    REDUCE_MAX = auto()
    REDUCE_MIN = auto()

    # Cast
    CAST = auto()

    # Select / ternary
    WHERE = auto()            # where(cond, a, b) → cond ? a : b

    # Constants
    CONSTANT = auto()
    CONSTEXPR = auto()

    # Pointer arithmetic
    PTR_ADD = auto()

    # Control flow
    FOR_LOOP_START = auto()   # Start of for loop: attrs={start, end, step}
    FOR_LOOP_END = auto()     # End of for loop body
    IF_START = auto()         # Start of if block
    IF_END = auto()           # End of if block

    # SIMD group operations (warp-level on Apple Silicon, 32 threads)
    SIMD_SUM = auto()         # simd_sum(x) — sum across SIMD group
    SIMD_MAX = auto()         # simd_max(x) — max across SIMD group
    SIMD_MIN = auto()         # simd_min(x) — min across SIMD group
    SIMD_BROADCAST = auto()   # simd_broadcast(x, lane) — broadcast from lane
    SIMD_SHUFFLE_DOWN = auto()  # simd_shuffle_down(x, delta)
    SIMD_LANE_ID = auto()     # get lane index within SIMD group (0-31)
    SIMD_GROUP_ID = auto()    # get SIMD group index within threadgroup

    # Value copy (snapshot for mutable aliasing)
    COPY = auto()              # copy(x) — snapshot value before mutable update

    # Simdgroup matrix operations (hardware 8×8 matmul on Apple Silicon)
    SIMDGROUP_MATRIX_LOAD = auto()     # load 8x8 from device/shared memory
    SIMDGROUP_MATRIX_STORE = auto()    # store 8x8 to device/shared memory
    SIMDGROUP_MATRIX_MAC = auto()      # D = A * B + C (multiply-accumulate)
    SIMDGROUP_MATRIX_FILL = auto()     # fill matrix with constant value


@dataclass
class IRValue:
    """A value in the IR graph (SSA form)."""
    id: int
    name: str
    dtype: IRType
    shape: tuple[int, ...] = ()  # empty = scalar, (N,) = 1D tile
    is_pointer: bool = False
    is_mutable: bool = False     # True for loop accumulators (re-assigned in loop body)
    aliases: int | None = None   # If set, this value is a re-assignment of the value with this id
    is_simdgroup_matrix: bool = False  # True for simdgroup_float8x8 values

    def __repr__(self) -> str:
        shape_str = f"<{','.join(str(s) for s in self.shape)}>" if self.shape else ""
        ptr_str = "*" if self.is_pointer else ""
        return f"%{self.name}:{ptr_str}{self.dtype.name}{shape_str}"


@dataclass
class IROp:
    """A single IR operation."""
    opcode: OpCode
    result: IRValue
    operands: list[IRValue] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        ops_str = ", ".join(repr(o) for o in self.operands)
        attrs_str = ""
        if self.attrs:
            attrs_str = " {" + ", ".join(f"{k}={v}" for k, v in self.attrs.items()) + "}"
        return f"  {self.result!r} = {self.opcode.name}({ops_str}){attrs_str}"


@dataclass
class IRKernel:
    """A complete kernel in IR form."""
    name: str
    params: list[IRValue]           # kernel parameters (pointers + constexprs)
    ops: list[IROp] = field(default_factory=list)
    grid_dim: int = 1               # number of grid dimensions
    shared_mem: dict[str, tuple[IRType, int | 'IRValue']] = field(default_factory=dict)  # name → (dtype, size or IRValue)
    _next_id: int = field(default=0, repr=False)

    def new_value(self, name: str, dtype: IRType, shape: tuple = (),
                  is_pointer: bool = False) -> IRValue:
        """Create a new SSA value."""
        val = IRValue(id=self._next_id, name=f"{name}_{self._next_id}",
                      dtype=dtype, shape=shape, is_pointer=is_pointer)
        self._next_id += 1
        return val

    def add_op(self, opcode: OpCode, result: IRValue,
               operands: list[IRValue] = None, attrs: dict = None) -> IROp:
        """Add an operation to the kernel."""
        op = IROp(opcode=opcode, result=result,
                  operands=operands or [], attrs=attrs or {})
        self.ops.append(op)
        return op

    def dump(self) -> str:
        """Pretty-print the IR."""
        lines = [f"kernel @{self.name}({', '.join(repr(p) for p in self.params)}) {{"]
        for op in self.ops:
            lines.append(repr(op))
        lines.append("}")
        return "\n".join(lines)
