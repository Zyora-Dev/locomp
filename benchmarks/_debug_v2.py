import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, locust
from examples.flash_attention_v1 import naive_attention_np, Br, Bc, D_HEAD
import importlib
v2mod = importlib.import_module('examples.16_flash_attention_v2')

N, d = 64, 32
np.random.seed(42)
Q = np.random.randn(N, d).astype(np.float32) * 0.1
K = np.random.randn(N, d).astype(np.float32) * 0.1
V = np.random.randn(N, d).astype(np.float32) * 0.1
expected = naive_attention_np(Q, K, V)

Qt = locomp.tensor(Q.flatten())
Kt = locomp.tensor(K.flatten())
Vt = locomp.tensor(V.flatten())
Ot = locomp.empty(N * d)
nkv = N // Bc

v2mod.flash_attn_v2[(N // Br,), (Bc, Br)](Qt, Kt, Vt, Ot, N, d, nkv, Br, Bc)
result = Ot.numpy().reshape(N, d)
diff = np.abs(result - expected)
print('Max err:', diff.max())
print('Row with max err:', np.unravel_index(diff.argmax(), diff.shape))
print('Expected row 0:', expected[0, :8])
print('Got row 0:     ', result[0, :8])
print('Diff row 0:    ', diff[0, :8])
# Check if error is systematic or random
print('Mean abs diff per row:', diff.mean(axis=1)[:8])
