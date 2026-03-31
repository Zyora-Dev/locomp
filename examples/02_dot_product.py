"""
Example: Dot Product / Cosine Similarity on Apple GPU.

Demonstrates reduction operations — computing dot products for vector search.
This is the core operation for a GPU-accelerated vector database.
"""

import locomp
import numpy as np


@locomp.kernel
def dot_product(A: locomp.Tensor, B: locomp.Tensor, OUT: locomp.Tensor, N: locomp.constexpr):
    pid = locomp.program_id(0)
    offsets = locomp.arange(0, 256)
    mask = offsets < N
    a = locomp.load(A + offsets, mask=mask)
    b = locomp.load(B + offsets, mask=mask)
    dot = locomp.sum(a * b)
    locomp.store(OUT + pid, dot)


def main():
    # Show generated code
    print("=" * 60)
    print("LOCUST IR (dot product):")
    print("=" * 60)
    print(dot_product.ir.dump())

    print("\n" + "=" * 60)
    print("GENERATED MSL:")
    print("=" * 60)
    print(dot_product.msl)

    # Execute
    print("\n" + "=" * 60)
    print("EXECUTING DOT PRODUCT ON GPU:")
    print("=" * 60)

    N = 256
    a = locomp.tensor(np.random.randn(N).astype(np.float32))
    b = locomp.tensor(np.random.randn(N).astype(np.float32))
    out = locomp.empty(1)

    dot_product[(1,)](a, b, out, N)

    expected = np.dot(a.numpy(), b.numpy())
    actual = out.numpy()[0]
    print(f"GPU result:  {actual:.6f}")
    print(f"CPU result:  {expected:.6f}")
    print(f"Difference:  {abs(expected - actual):.8f}")


if __name__ == "__main__":
    main()
