import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import locomp
import importlib
m = importlib.import_module("examples.18_simdgroup_matmul")
A = locomp.tensor([0.0]*64); B = locomp.tensor([0.0]*64); C = locomp.empty(64)
m.matmul_simd[(1,1),(32,)](A,B,C,8,8,8)
print(m.matmul_simd.msl)
