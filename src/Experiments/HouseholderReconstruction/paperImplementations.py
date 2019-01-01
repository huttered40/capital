#!/usr/bin/env python

"""
Implementations and Notes on the paper:
    Reconstructing Householder vectors from Tall-Skinny QR

"""

import numpy as np
import numpy.linalg as la

def convertReflectorsToOrthgonal_version_1(Reflectors):
    numRows = Reflectors.shape[0]
    numColumns = Reflectors.shape[1]
    # Q must be square now, instead of reduced
    newQ = np.eye(numRows)
    # Iterate backwards over the reflectors
    for i in range(numColumns-1,-1,-1):
        #print(Reflectors[:,i])
        tau = 2./(np.dot(Reflectors[:,i], Reflectors[:,i]))
        #print "shape of Reflector - ", Reflectors[:,i][:,np.newaxis].T.shape, " and shape of newQ - ", newQ.shape
        tempMatrix = np.dot(Reflectors[:,i][:,np.newaxis].T,newQ)
        newQ = newQ - np.dot(tau*Reflectors[:,i][:,np.newaxis],tempMatrix)
    return newQ

def HouseHolderQR_BLAS2_version_1(A):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    Reflectors = np.zeros((numRows, numColumns))   # will be upper-trapezoidal
    for i in range(numColumns):
        dirLen = la.norm(A[i:,i],2)   # Don't worry about the first i rows.
        # Below: there could probably be a faster way to do this than to create an identity matrix
        #   at each iteration
        # curReflector points from vector a to norm(a)*e_1, just via vector addition
        # Also, we address possible catastrophic cancellation by avoiding subtracting similar values in the
        #   first element of the saxpy
        Reflectors[i:,i] = np.sign(A[i,i])*(-1)*dirLen*np.eye(numRows-i)[0:numRows-i,0] - A[i:,i]
        tau = 2./(np.dot(Reflectors[i:,i],Reflectors[i:,i]))
        # update trailing columns all at once (BLAS-2) to complete the Matrix-Matrix product
        A[i:,i:] = A[i:,i:] - np.dot(tau*Reflectors[i:,i][:,np.newaxis], np.dot(Reflectors[i:,i][:,np.newaxis].T, A[i:,i:]))
    # R will be in A, and A should be upper-trapezoidal
    # No explicit Q is being returned, as is the case with HouseHolder orthogonal triangularization methods
    return (Reflectors,A)

# Algorithm 1
# I am having trouble understanding exactly why this algorithm works
# It is formatted differently than I am used to
# For one thing, it normalizes the first element of the reflector to 1, so the entire reflector
#   needs to be divided out by that factor.

def HouseholderQR_Paper(A):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    R = np.zeros(A.shape)
    Tau = np.zeros(numColumns)
    Y = np.zeros(A.shape)
    for i in range(numColumns):
        # Compute Householder vector
        R[i,i] = (-1)*np.sign(A[i,i])*la.norm(A[i:,i],2)
        Tau[i] = (R[i,i] - A[i,i]) / R[i,i]
        Y[i,i] = 1
        Y[(i+1):,i] = A[(i+1):,i]/(A[i,i] - R[i,i])
        # Apply the Householder transformation to the trailing matrix
        z = Tau[i] * (A[i,(i+1):] + np.dot(A[(i+1):,i][:,np.newaxis].T, A[(i+1):,(i+1):]))
        R[i,(i+1):] = A[i,(i+1):] - z
        print(z.shape)
        A[(i+1):,(i+1):] = A[(i+1):,(i+1):] - np.dot(Y[(i+1):,i][:,np.newaxis],z)
    return (Y,Tau,R)

def HouseholderQR(A):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    Tau = np.zeros(numColumns)
    Y = np.zeros(A.shape)
    for i in range(numColumns):
        Y[i:,i] = (-1)*np.sign(A[i,i])*la.norm(A[i:,i],2)*np.eye(numRows-i)[:,0] - A[i:,i]
        Y[i:,i] = Y[i:,i] / Y[i,i]
        Tau[i] = 2./np.dot(Y[i:,i],Y[i:,i])
        A[i:,i:] = A[i:,i:] - Tau[i]*np.dot(Y[i:,i][:,np.newaxis],np.dot(Y[i:,i][:,np.newaxis].T,A[i:,i:]))
    return (Y,Tau,A)



# Test Algorithm 1

A = np.random.rand(80,20)
A_copy = A.copy()
Y,Tau,R = HouseholderQR_Paper(A)
#Y,R = HouseHolderQR_BLAS2_version_1(A)
print R
#numColumns = A.shape[1]
#for i in range(numColumns):
#    Temp1 = np.dot(Y[:,numColumns-1-i][:,np.newaxis].T,R)
#    Temp2 = np.dot(Y[:,numColumns-1-i][:,np.newaxis],Temp1)
#    Temp3 = Tau[numColumns-1-i]*Temp2
#    R = R - Temp3
#print R

Q = convertReflectorsToOrthgonal_version_1(Y)
print la.norm(np.dot(Q,R)-A_copy,2)/la.norm(A_copy,2)

# End of code
