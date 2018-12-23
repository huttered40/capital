import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt

numRows = 200
numColumns = 200
A = np.asarray(np.random.rand(numRows, numColumns),dtype=np.float64)
U,S,V = la.svd(A,full_matrices=0)

# Try tests for all different condition numbers in the range
S = np.logspace(1,10,base=2.,num=numColumns)
S = S[::-1]

# Print singular values
pt.semilogy(S, label='singular values')
pt.legend()
pt.show()

A64 = U.dot(np.diag(S).dot(V))
A32 = A64.astype(dtype=np.float32)
u,s,v = la.svd(A32)
print "SVD low-rank residual (best possible for double precision) - ", la.norm(A64 - U.dot(np.diag(S)).dot(V))
print "SVD low-rank residual (best possible for float precision) - ", la.norm(A32 - u.dot(np.diag(s)).dot(v))

print "Deviation from orthogonality of SVD's left singular vectors - ", la.norm(np.eye(numColumns) - u.T.dot(u))


# Start of SVD Gram
B32 = A32.T.dot(A32)			# Left singular vectors dissapear. Now, eigenvaluedecomposition will give (orthogonal) right singular vectors
s32,v32 = la.eigh(B32)
u32 = A32.dot(v32).dot(la.inv(np.diag(np.sqrt(s32))))

print "Condition number of A - ", la.cond(A32)
print "Condition number of B - ", la.cond(B32)
print "Deviation from orthogonality of (uncovered) left singular vectors (of A32) - ", la.norm(np.eye(numColumns) - u32.T.dot(u32))
print "Deviation from orthogonality of right singular vectors (of A32) - ", la.norm(np.eye(numColumns) - v32.dot(v32.T))
print "Singular value approximant residual error (single precision) - ", la.norm(s32**.5-S[::-1])
print "Gram algorthm's low-rank residual (single precision) - ", la.norm(A64 - u32.dot(np.diag(s32[::-1]**(.5))).dot(v32))
print "Approximate singular values - ", s32**.5
print "Approximate singular values - ", S[::-1]


B64 = A64.T.dot(A64)
s64,v64 = la.eigh(B64)
u64 = A64.dot(v64).dot(la.inv(np.diag(np.sqrt(s64))))

print "Condition number of A - ", la.cond(A64)
print "Condition number of B - ", la.cond(B64)
print "Deviation from orthogonality of (uncovered) left singular vectors (of A64) - ", la.norm(np.eye(numColumns) - u64.T.dot(u64))
print "Deviation from orthogonality of right singular vectors (of A64) - ", la.norm(np.eye(numColumns) - v64.dot(v64.T))
print "Singular value approximant residual error (double precision) - ", la.norm(s64**.5-S[::-1])
print "Gram algorthm's low-rank residual (double precision) - ", la.norm(A64 - u64.dot(np.diag(s64[::-1]**(.5))).dot(v64))
print "Approximate singular values - ", s64**.5
print "Approximate singular values - ", S[::-1]


ts = np.diag(np.sqrt(s32))

#F = u2.dot(np.diag(s2))
FB = ts.dot(u32.T.dot(u32)).dot(ts)
sf32,vf32 = la.eigh(FB)
ut32 = A32.dot(v32).dot(vf32).dot(la.inv(np.diag(np.sqrt(sf32))))

print la.norm(np.eye(numColumns) - ut32.T.dot(ut32))

Aapprox = ut32.dot(np.diag(np.sqrt(sf32))).dot(vf32.T).dot(v32.T)

print la.norm(Aapprox - A64)
