"""Quick test for enhanced Tensor operations."""
import locomp
import numpy as np

# Test tensor creation and reshape
t = locomp.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
print(f'Original: shape={t.shape}, ndim={t.ndim}, contiguous={t.is_contiguous()}')

# Reshape (zero-copy)
t2 = t.reshape(6, 4)
print(f'Reshape(6,4): shape={t2.shape}, contiguous={t2.is_contiguous()}')

# Flatten
t3 = t.flatten()
print(f'Flatten: shape={t3.shape}')

# Transpose (zero-copy)
t4 = t.transpose(0, 2)
print(f'Transpose(0,2): shape={t4.shape}, contiguous={t4.is_contiguous()}')

# Permute (zero-copy)
t5 = t.permute(2, 0, 1)
print(f'Permute(2,0,1): shape={t5.shape}, contiguous={t5.is_contiguous()}')

# Unsqueeze
t6 = t.unsqueeze(0)
print(f'Unsqueeze(0): shape={t6.shape}')

# Squeeze
t7 = t6.squeeze(0)
print(f'Squeeze(0): shape={t7.shape}')

# Reshape with -1
t8 = t.reshape(-1, 4)
print(f'Reshape(-1,4): shape={t8.shape}')

# View shares base reference
print(f'Reshape is view (shares base): {t2._base is t}')

# Verify data correctness
np.testing.assert_array_equal(t.numpy(), np.arange(24).reshape(2, 3, 4))
np.testing.assert_array_equal(t2.numpy(), np.arange(24).reshape(6, 4))
np.testing.assert_array_equal(t3.numpy(), np.arange(24))
np.testing.assert_array_equal(t8.numpy(), np.arange(24).reshape(6, 4))

# Test that kernel dispatch still works with tensors
data = np.random.randn(256).astype(np.float32)
a = locomp.tensor(data)
b = locomp.tensor(data)
out = locomp.empty(256)

@locomp.kernel
def add_kernel(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor, N: locomp.constexpr):
    i = locomp.program_id(0)
    val_a = locomp.load(A + i)
    val_b = locomp.load(B + i)
    locomp.store(C + i, val_a + val_b)

add_kernel[(256,)](a, b, out, 256)
result = out.numpy()
np.testing.assert_allclose(result, data + data, rtol=1e-6)
print(f'Kernel dispatch with tensor: ✓')

a.free(); b.free(); out.free()
print('\nAll tensor ops verified ✓')
