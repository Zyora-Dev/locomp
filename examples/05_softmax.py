"""
Example: Softmax on Apple GPU using Locust.

Softmax is THE operation in transformer attention:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

Each thread computes softmax for one row of a (ROWS x D) matrix.
Three passes per row: find max, compute exp sum, normalize.

Demonstrates: for-loops, exp(), load/store, mutable accumulators, guards.
"""

import locomp
import numpy as np


@locomp.kernel
def softmax(X: locomp.Tensor, OUT: locomp.Tensor,
            ROWS: locomp.constexpr, D: locomp.constexpr):
    # Each thread handles one row
    row = locomp.thread_id(0)
    guard = row < ROWS

    # Pass 1: find row max (numerical stability)
    row_max = locomp.load(X + (row * D + 0))
    for j in range(1, D):
        val = locomp.load(X + (row * D + j))
        row_max = locomp.where(val > row_max, val, row_max)

    # Pass 2: sum of exp(x - max)
    exp_sum = 0.0
    for j in range(D):
        val = locomp.load(X + (row * D + j))
        e = locomp.exp(val - row_max)
        exp_sum = exp_sum + e

    # Pass 3: write normalized values
    for j in range(D):
        val = locomp.load(X + (row * D + j))
        e = locomp.exp(val - row_max)
        result = e / exp_sum
        locomp.store(OUT + (row * D + j), result, mask=guard)


def main():
    print("=" * 60)
    print("LOCUST IR (softmax):")
    print("=" * 60)
    print(softmax.ir.dump())

    print("\n" + "=" * 60)
    print("GENERATED MSL:")
    print("=" * 60)
    print(softmax.msl)

    print("\n" + "=" * 60)
    print("EXECUTING SOFTMAX ON GPU:")
    print("=" * 60)

    ROWS, D = 8, 16
    x = locomp.tensor(np.random.randn(ROWS * D).astype(np.float32))
    out = locomp.empty(ROWS * D)

    threads_per_group = 256
    num_groups = (ROWS + threads_per_group - 1) // threads_per_group
    softmax[(num_groups,), (threads_per_group,)](x, out, ROWS, D)

    # Verify against NumPy softmax
    x_np = x.numpy().reshape(ROWS, D)
    x_shifted = x_np - x_np.max(axis=1, keepdims=True)
    expected = np.exp(x_shifted) / np.exp(x_shifted).sum(axis=1, keepdims=True)
    expected = expected.flatten()
    result = out.numpy()

    max_error = np.max(np.abs(result - expected))
    print(f"Shape: {ROWS} rows x {D} cols")
    print(f"Max error vs NumPy: {max_error}")
    print(f"Result row 0: {result[:D]}")
    print(f"Expected row 0: {expected[:D]}")
    print(f"Row sums (should be ~1.0): {result.reshape(ROWS, D).sum(axis=1)}")
    assert max_error < 1e-5, f"Error too large: {max_error}"
    print("PASSED!")


if __name__ == "__main__":
    main()
