#!/usr/bin/env python

'''
Implementation of different Cholesky-QR2 algorithms
'''

import numpy as np
import numpy.linalg as la
import sys
import matplotlib
import matplotlib.pyplot as plt


# Need to check whether random Gaussian matrix is orthogonal

Omega = np.random.normal(size=(100,100))
condNum = la.cond(Omega,2)
devOrthNorm = la.norm(np.eye(100) - np.dot(Omega,Omega.T),2)/la.norm(np.dot(Omega,Omega.T),2)
U,S,V = la.svd(Omega)

print(devOrthNorm)
print(condNum)
print(S)


Omega = np.random.normal(size=(100,30))
condNum = la.cond(Omega,2)
devOrthNorm = la.norm(np.eye(100) - np.dot(Omega,Omega.T),2)/la.norm(np.dot(Omega,Omega.T),2)
U,S,V = la.svd(Omega)

Q,R = la.qr(Omega)
devOrthNorm = la.norm(np.eye(100) - np.dot(Q,Q.T),2)/la.norm(np.dot(Q,Q.T),2)

print(devOrthNorm)
print(condNum)
print(S)
print(np.dot(Q,Q.T))
print(np.dot(Q.T,Q))
devOrthNorm = la.norm(np.eye(30) - np.dot(Q.T,Q),2)/la.norm(np.dot(Q.T,Q),2)
print(devOrthNorm)
