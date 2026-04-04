"""
Built-in GPU embedding lookup kernels for locomp.embed_lookup().
Defined in a separate module so inspect.getsource() can find them.
"""
import locomp as _locomp


@_locomp.kernel
def _embed_fp32(token_ids: _locomp.Int32, table: _locomp.Tensor, out: _locomp.Tensor,
                DIM: _locomp.constexpr):
    seq = _locomp.program_id(0)
    d   = _locomp.program_id(1)
    tok = _locomp.load(token_ids + seq)
    val = _locomp.load(table + tok * DIM + d)
    _locomp.store(out + seq * DIM + d, val)


@_locomp.kernel
def _embed_fp16(token_ids: _locomp.Int32, table: _locomp.Float16, out: _locomp.Float16,
                DIM: _locomp.constexpr):
    seq = _locomp.program_id(0)
    d   = _locomp.program_id(1)
    tok = _locomp.load(token_ids + seq)
    val = _locomp.load(table + tok * DIM + d)
    _locomp.store(out + seq * DIM + d, val)
