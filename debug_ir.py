import ast, textwrap
from locomp.frontend import KernelCompiler, Tensor, constexpr

code = '''
def matmul(A, B, C, M, N, K):
    tid = locomp.thread_id(0)
    row = tid / N
    col = tid % N
    acc = 0.0
    for k in range(K):
        a_val = locomp.load(A + (row * K + k))
        b_val = locomp.load(B + (k * N + col))
        acc = acc + a_val * b_val
    total = M * N
    mask = tid < total
    locomp.store(C + tid, acc, mask=mask)
'''

tree = ast.parse(textwrap.dedent(code))
func_def = tree.body[0]
params = ['A','B','C','M','N','K']
types = {'A': Tensor, 'B': Tensor, 'C': Tensor, 'M': constexpr, 'N': constexpr, 'K': constexpr}
compiler = KernelCompiler('matmul', params, types)
ir = compiler.compile(func_def)
for i, op in enumerate(ir.ops):
    a = op.result.aliases
    m = op.result.is_mutable
    print(f"{i:3d} id={op.result.id:3d} {op.opcode.name:20s} aliases={a} mutable={m}")
