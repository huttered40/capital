#!/usr/bin/env python

"""
Implementations and Notes on the paper:
    Reconstructing Householder vectors from Tall-Skinny QR

"""

import numpy as np
import numpy.linalg as la

# Algorithm 1
# I am having trouble understanding exactly why this algorithm works
# It is formatted differently than I am used to
# For one thing, it normalizes the first element of the reflector to 1, so the entire reflector
#   needs to be divided out by that factor.

def HouseholderQR(A):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    R = np.zeros(A.shape)
    Tau = np.zeros((numColumns,1))
    Y = np.zeros(A.shape)
    for i in range(numColumns):
        # Compute Householder vector
        R[i,i] = (-1)*np.sign(A[i,i])*la.norm(A[i:,i],2)
        Tau[i] = (R[i,i] - A[i,i]) / (R[i,i]**2)
        Y[i,i] = 1
        Y[(i+1):,i] = A[(i+1):,i]/(A[i,i] - R[i,i])
        # Apply the Householder transformation to the trailing matrix
        z = Tau[i] * (A[i,(i+1):] + np.dot(A[(i+1):,i].T, A[(i+1):,(i+1):]))
        R[i,(i+1):] = A[i,(i+1):] - z
        if ((i+1) == numColumns):
            break
        A[(i+1):,(i+1):] = A[(i+1):,(i+1):] - np.dot(Y[(i+1):,i][:,np.newaxis],z[np.newaxis,:])
    return (Y,Tau,R)
        


# Test Algorithm 1

A = np.random.rand(80,20)
A_copy = A.copy()
Y,Tau,R = HouseholderQR(A)
A_temp = np.zeros(A.shape)
numColumns = A.shape[1]
for i in range(numColumns):
    Temp1 = np.dot(Y[:,i][:,np.newaxis].T,R[:,i][:,np.newaxis])
    Temp2 = np.dot(Y[:,i][:,np.newaxis],Temp1)
    Temp3 = Tau[i]*Temp2
    Temp4 = R[:,i][:,np.newaxis] - Temp3
    A_temp[:,i][:,np.newaxis] = Temp4
print A_temp
print la.norm(A_temp-A_copy,2)

# End of code
