import ctf
import ctf.random
import numpy as np
import numpy.linalg as la

def cholinv(A):
    if (A.shape[0] <= 4):
        B = A.to_nparray()
        #print(B)
        L = la.cholesky(B)
        Linv = la.inv(L)
        return [ctf.astensor(L),ctf.astensor(Linv)]
    n = A.shape[0]
    A11 = A[:(n/2),:(n/2)]
    print(A11)
    [L11, L11inv] = cholinv(A11)
    L21 = ctf.dot(A[(n/2):,:(n/2)],L11inv.T())
    S22 = A[(n/2):,(n/2):] - ctf.dot(L21,L21.T())
    [L22,L22inv] = cholinv(S22)
    L21inv = ctf.dot((-1)*L22inv,ctf.dot(L21,L11inv))
    L = ctf.zeros((n,n))
    Linv = ctf.zeros((n,n))
    L[0:(n/2),0:(n/2)] = L11
    L[(n/2):,0:(n/2)] = L21
    L[(n/2):,(n/2):] = L22
    Linv[0:(n/2),0:(n/2)] = L11inv
    Linv[(n/2):,0:(n/2)] = L21inv
    Linv[(n/2):,(n/2):] = L22inv
    return [L,Linv]


# Start code
n = 32
A = ctf.random.random((n,n))
B = ctf.dot(A.T(),A)
[L,Linv] = cholinv(B)
S1 = (B - ctf.dot(L,L.T())).norm2()
S2 = (ctf.eye(n) - ctf.dot(L,Linv)).norm2()
print(S1)
print(S2)
