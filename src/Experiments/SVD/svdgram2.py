import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt

numRows = 200
numColumns = 200
Ad = np.asarray(np.random.rand(numRows, numColumns),dtype=np.float64)
U,D,V = la.svd(Ad,full_matrices=0)

# Try tests for all different condition numbers in the range
D = np.logspace(1,10,base=2.,num=numColumns)
D = D[::-1]
#D = np.exp(np.arange(numColumns,0,-1))
#for i in range(numColumns):
#    D[i] = (numColumns-i-1)**(numColumns-i-1)

A64 = U.dot(np.diag(D).dot(V.T))
A32 = A64.astype(dtype=np.float32)
u32,s32,v32 = la.svd(A32)
print la.norm(A32 - u32.dot(np.diag(s32)).dot(v32))

print la.norm(np.eye(numColumns) - u32.T.dot(u32))

u,s,v = la.svd(A64)
pt.semilogy(s, label='singular values')
pt.legend()
pt.show()


B = A32.T.dot(A32)
s2,v2 = la.eigh(B)
u2 = A32.dot(v2).dot(la.inv(np.diag(np.sqrt(s2))))

print la.norm(np.eye(numColumns) - u2.T.dot(u2))

#la.norm(A64 - u2.dot(np.diag(s2**(.5))).dot(v2))
#la.norm(u-u2)


B = A64.T.dot(A64)
s3,v3 = la.eigh(B)
u3 = A64.dot(v3).dot(la.inv(np.diag(np.sqrt(s3))))

#print la.norm(A64 - u2.dot(np.diag(s2**(.5))).dot(v2))
la.norm(u2-u3)

print la.norm(np.eye(numColumns) - u3.T.dot(u3))


ts = np.diag(np.sqrt(s2))

#F = u2.dot(np.diag(s2))
FB = ts.dot(u2.T.dot(u2)).dot(ts)
sf2,vf = la.eigh(FB)
ut2 = A32.dot(v2).dot(vf).dot(la.inv(np.diag(np.sqrt(sf2))))

print la.norm(np.eye(numColumns) - ut2.T.dot(ut2))

Aapprox = ut2.dot(np.diag(np.sqrt(sf2))).dot(vf.T).dot(v2.T)

print la.norm(Aapprox - A64)

#AF = 
