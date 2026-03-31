"""
Example: Matrix Multiplication on Apple GPU using Locust.

This is the key benchmark for GPU compilers. Each thread computes one element
of the output matrix C[row, col] = sum(A[row, k] * B[k, col]) for k in range(K).

Demonstrates: thread_id(), for-loops, pointer arithmetic with loop variables.
"""

import locomp
import numpy as np


@locomp.kernel
def matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
           M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr):
    # Each thread computes one element of C
    tid = locomp.thread_id(0)

    # Map 1D thread index → (row, col)
    row = tid / N      # integer division
    col = tid % N

    # Accumulate dot product
    acc = 0.0
    for k in range(K):
        a_val = locomp.load(A + (row * K + k))
        b_val = locomp.load(B + (k * N + col))
        acc = acc + a_val * b_val

    # Guard against out-of-bounds threads
    total = M * N
    mask = tid < total
    locomp.store(C + tid, acc, mask=mask)


def main():
    print("=" * 60)
    print("LOCUST IR (matmul):")
    print("=" * 60)
    print(matmul.ir.dump())

    print("\n" + "=" * 60)
    print("GENERATED MSL (Metal Shading Language):")
    print("=" * 60)
    print(matmul.msl)

    print("\n" + "=" * 60)
    print("EXECUTING MATMUL ON GPU:")
    print("=" * 60)

    M, N, K = 16, 16, 16
    a = locomp.tensor(np.random.randn(M * K).astype(np.float32))
    b = locomp.tensor(np.random.randn(K * N).astype(np.float32))
    c = locomp.empty(M * N)

    total_threads = M * N  # 256
    threads_per_group = 256
    num_groups = (total_threads + threads_per_group - 1) // threads_per_group
    matmul[(num_groups,), (threads_per_group,)](a, b, c, M, N, K)

    # Verify against NumPy
    a_mat = a.numpy().reshape(M, K)
    b_mat = b.numpy().reshape(K, N)
    expected = (a_mat @ b_mat).flatten()
    result = c.numpy()
    max_error = np.max(np.abs(result - expected))

    print(f"Matrix size: {M}x{K} @ {K}x{N} = {M}x{N}")
    print(f"Max error vs NumPy: {max_error}")
    print(f"Result (first 8): {result[:8]}")
    print(f"Expected (first 8): {expected[:8]}")
    assert max_error < 1e-3, f"Error too large: {max_error}"
    print("PASSED!")


if __name__ == "__main__":
    main()
