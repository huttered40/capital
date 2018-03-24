#!/usr/bin/env python

"""
Notebook containing QR and QRCP algorithm implementations

The following algorithms are implemented

The following algorithms are in progress
  1. HouseHolder QR with BLAS-1
      - other version (2-4)
  2. HouseHolder QR with BLAS-2
      - other version (2-4)
  3. HouseHolder QR with BLAS-3 (with WY blocked representation)
  4. HouseHolder QR with BLAS-3 (with compact WY, or YT, blocked representation)
  5. Gram-Schmidt QR with BLAS-1
      - need it to handle rectangular matrices
      - figure out why version 3 is wack
  6. Gram-Schmidt QR with BLAS-2
      - need it to handle rectangular matrices
      - finish version 2
      - finish version 3
  10. HouseHolder QR with column-pivoting with BLAS-1
      - more robust norm-update scheme to avoid cancellation
      - finish other versions
  11. HouseHolder QR with column-pivoting with BLAS-2
      - more robust norm-update scheme to avoid cancellation
      - finish other versions
  12. HouseHolder QR with column-pivoting with BLAS-3
      - other two Connector-matrix representation versions
  16. Single-sampled Randomized QRCP
  17. Repeated-sampled Randomized QRCP
  18. Randomized QRCP
  19. Truncated Randomized QRCP

The following algorithms need to be implemented 

The following algorithms would be fun to implement later but are not important now:
  7. Gram-Schmidt QR with BLAS-3
  8. FLAME notation unblocked HouseHolder QR
  9. FLAME notation blocked HouseHolder QR (with alternate blocked representation)
  13. FLAME notation HouseHolder QR with column-pivoting
  15. Jed/Ming HouseHolder QR with column pivoting BLAS-3

  Extra:
    experiment with different precision
    try using CholeskyQR instead of QRCP for the short and fat panel
"""

# Import all of the necessary libraries
import numpy as np
import numpy.linalg as la
import scipy.linalg as sc
import sys
import time
import matplotlib.pyplot as ppt


"""
5. Gram-Schmidt QR with BLAS-1 implementations

   3 different versions: 1) Iterative
                         2) Recursive
"""

def Gram_Schmidt_QR_BLAS1_version_1(A):
    numColumns = A.shape[1]
    R = np.zeros((numColumns, numColumns))
    
    for i in range(numColumns):
        orthDirLen = la.norm(A[:,i],2)
        A[:,i] = A[:,i]/orthDirLen
        R[i,i] = orthDirLen
        # subtract out components of the remaining pool of vectors that lie in same
        #   direction as A[:,i]
        for j in range(i+1,numColumns):
            subLen = np.dot(A[:,j], A[:,i])
            A[:,j] = A[:,j] - subLen*A[:,i]
            R[i,j] = subLen                 
    
    # A has been transformed into Q, as Gram-Schmidt is a triangular orthogonalization algorithm
    return (A,R)


# Recursive helper function for version 2 below
def Gram_Schmidt_QR_BLAS1_version_2_backtrack(A,R, numColumns, currColumn):
    # recursive base case
    if (currColumn == numColumns):
        return (A,R)
    
    orthDirLen = la.norm(A[:,currColumn],2)
    A[:,currColumn] = A[:,currColumn]/orthDirLen
    R[currColumn,currColumn] = orthDirLen
    # subtract out components of the remaining pool of vectors that lie in same
    #   direction as A[:,i]
    for j in range(currColumn+1,numColumns):
        subLen = np.dot(A[:,j], A[:,currColumn])
        A[:,j] = A[:,j] - subLen*A[:,currColumn]
        R[currColumn,j] = subLen                 
    
    # A has been transformed into Q, as Gram-Schmidt is a triangular orthogonalization algorithm
    return Gram_Schmidt_QR_BLAS1_version_2_backtrack(A,R,numColumns,currColumn+1)

# Python is a bit weird with references, and I like to use references with recursive backtracking algorithms
def Gram_Schmidt_QR_BLAS1_version_2(A):
    numColumns = A.shape[1]
    R = np.zeros((numColumns, numColumns))
    return Gram_Schmidt_QR_BLAS1_version_2_backtrack(A,R,numColumns,0)

def Gram_Schmidt_QR_BLAS1_version_3(A):
    numColumns = A.shape[1]
    R = np.eye(numColumns)
    
    for i in range(numColumns):
        R_current = np.eye(numColumns)
        orthDirLen = la.norm(A[:,i],2)
        R_current[i,i] = orthDirLen
        # subtract out components of the remaining pool of vectors that lie in same
        #   direction as A[:,i]
        for j in range(i+1,numColumns):
            subLen = np.dot(A[:,j], A[:,i])
            R_current[i,j] = (-1)*subLen
        R = np.dot(R,R_current)
    
    # A has been transformed into Q, as Gram-Schmidt is a triangular orthogonalization algorithm
    
    return (np.dot(A,R),la.inv(R))


"""
5. Gram-Schmidt QR with BLAS-1 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
A = np.random.rand(numRows, numColumns)
Q,R = Gram_Schmidt_QR_BLAS1_version_2(A)
#print(Q)
#print(R)
#print(la.norm(np.eye(numColumns) - np.dot(Q.T,Q)))

# Test deviation from orthogonality
# Test residual
"""


"""
6. Gram-Schmidt QR with BLAS-2 implementations

   3 different versions: 1) Iterative
                         2) Recursive
                         3) Matrix-notation
"""

def Gram_Schmidt_QR_BLAS2_version_1(A):
    numColumns = A.shape[1]
    R = np.eye(numColumns)
    
    for i in range(numColumns):
        orthDirLen = la.norm(A[:,i],2)
        R[i,i] = orthDirLen
        A[:,i] = A[:,i]/orthDirLen
        stupidTransposeColumn = np.array([A[:,i]])
        #print stupidTransposeColumn.shape
        #print A[:,(i+1):].shape
        lengthRow = np.dot(stupidTransposeColumn, A[:,(i+1):])
        R[i, (i+1):] = lengthRow
        A[:,(i+1):] = A[:,(i+1):] - np.outer(A[:,i], lengthRow)
    return (A,R)


"""
6. Gram-Schmidt QR with BLAS-2 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
A = np.random.rand(numRows, numColumns)
Q,R = Gram_Schmidt_QR_BLAS2_version_1(A)
print(Q)
print(R)
print(la.norm(np.eye(numColumns) - np.dot(Q.T,Q)))

# Test deviation from orthogonality
# Test residual
"""


"""
FUNCTION: convertReflectorsToOrthgonal_version_1(Reflectors)
This function is needed to reconstruct Q from the reflectors that are stored explicitely,
  since the concatenation of reflectors is NOT the orthogonal matrix

Idea: To explicitely form Q, we need to do: Q*I = Q, where the left Q is defined implicitely using the 
       reflectors. The Q on the right will be defined explicitely via repeated Blas-2 operations
"""
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

"""
FUNCTION: convertReflectorsToOrthgonal_version_2(Reflectors)
This function is needed to reconstruct Q from the reflectors that are stored in the lower-trapezoidal
   part of matrix A->R. We also have a vector that stores the first elements that would need to have lived
   on the diagonals, but were not able to be stored because we also needed to store the diagonal elements of R
"""
def convertReflectorsToOrthgonal_version_2(Reflectors, firstElements):
    return A

"""
Idea: Same as version_1, except we want to form Q-transpose, not Q
"""
def convertReflectorsToOrthgonal_version_3(Reflectors):
    numRows = Reflectors.shape[0]
    numColumns = Reflectors.shape[1]
    # Q must be square now, instead of reduced
    newQ = np.eye(numRows)
    # Iterate backwards over the reflectors
    for i in range(0,numColumns):
        #print(Reflectors[:,i])
        tau = 2./(np.dot(Reflectors[:,i], Reflectors[:,i]))
        #print "shape of Reflector - ", Reflectors[:,i][:,np.newaxis].T.shape, " and shape of newQ - ", newQ.shape
        tempMatrix = np.dot(Reflectors[:,i][:,np.newaxis].T,newQ)
        newQ = newQ - np.dot(tau*Reflectors[:,i][:,np.newaxis],tempMatrix)
    return newQ


"""
1. HouseHolder QR with BLAS-1 implementations

   3 different versions: 1) Iterative with matrix Q not explicitely stored, reflectors are stored separately
                                -- so Q stores the reflectors, and A'=R stores the upper-triangular portion
                         2) Iterative with matrix Q embedded into A (needs an extra array)
                                 -- So the reflectors are stored in the lower-trapezoidal part of A, R is upper
                         3) Recursive
"""

def HouseHolderQR_BLAS1_version_1(A):
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
        # update trailing columns one at a time (BLAS-1) to complete the Matrix-Matrix product
        # Lets include the current column, although we could have done that vector transformation individually
        for j in range(i,numColumns):
            A[i:numRows,j] = A[i:numRows,j] - np.dot(tau*Reflectors[i:,i], np.dot(Reflectors[i:,i],A[i:numRows,j]))
    # R will be in A, and A should be upper-trapezoidal
    # No explicit Q is being returned, as is the case with HouseHolder orthogonal triangularization methods
    return (Reflectors,A)

def HouseHolderQR_BLAS1_version_2(A):
    return (A,A)


"""
1. HouseHolder QR with BLAS-1 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
A = np.random.rand(numRows, numColumns)
# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
print "condition number of input matrix A - ", la.cond(A)
Reflectors,R = HouseHolderQR_BLAS1_version_1(A)
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
print "Residual - ", la.norm(np.dot(Q,R)-A_copy,2)
"""


"""
1. HouseHolder QR with BLAS-2 implementations

   3 different versions: 1) Iterative with matrix Q not explicitely stored, reflectors are stored separately
                                -- so Q stores the reflectors, and A'=R stores the upper-triangular portion
                         2) Iterative with matrix Q embedded into A (needs an extra array)
                                 -- So the reflectors are stored in the lower-trapezoidal part of A, R is upper
                         3) Recursive
"""

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


"""
1. HouseHolder QR with BLAS-2 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
A = np.random.rand(numRows, numColumns)
# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
print "condition number of input matrix A - ", la.cond(A)
Reflectors,R = HouseHolderQR_BLAS2_version_1(A)
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
print "Residual - ", la.norm(np.dot(Q,R)-A_copy,2)
"""


"""
1. HouseHolder QR with column pivoting with BLAS-1 implementations

   3 different versions: 1) Iterative with matrix Q not explicitely stored, reflectors are stored separately
                                -- so Q stores the reflectors, and A'=R stores the upper-triangular portion
                         2) Iterative with matrix Q embedded into A (needs an extra array)
                                 -- So the reflectors are stored in the lower-trapezoidal part of A, R is upper
                         3) Recursive
                         4) Same as version 1 except it can detect rank
"""

def HQRCP_BLAS1_setUp1(A,numPivots):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    permutationIndices = np.arange(numColumns)
    norms = np.zeros((numColumns,1))
    for i in range(numColumns):
        norms[i] = la.norm(A[:,i],2)**2      # power 2 because it helps with update below
    Reflectors = np.zeros((numRows, numPivots))   # will be upper-trapezoidal
    return (numRows,numColumns,permutationIndices,norms,Reflectors)

def HQRCP_BLAS1_SwapColumns(A,permutationIndices,norms,tolerance,currIteration,rankEstimate,rankDetecting):
        pivotIndex = np.argmax(norms[currIteration:])+currIteration
        kingNorm = np.amax(norms[currIteration:])
        
        if (rankDetecting):
            # Detect rank!
            if (kingNorm < tolerance):
                rankEstimate = currIteration
                return (True,rankEstimate)
            
        # Swap out pivot column with current column
        # Key question will be answered right here -- do we need to swap out entire column, or just subcolumn?
        tempColumn = A[:,currIteration].copy()     # avoid overwriting via nasty pass by reference?
        A[:,currIteration] = A[:,pivotIndex].copy() # same thing as above
        A[:,pivotIndex] = tempColumn
        savePermIndex = permutationIndices[pivotIndex].copy()
        permutationIndices[pivotIndex] = permutationIndices[currIteration].copy()
        permutationIndices[currIteration] = savePermIndex
        tempNorm = norms[pivotIndex].copy()
        norms[pivotIndex] = norms[currIteration].copy()
        norms[currIteration] = tempNorm.copy()
        return (False,rankEstimate)

def HQRCP_BLAS1_UpdateMatrixAndNorms(A,Reflectors,norms,numRows,numColumns,currIteration):
    # Continue on with typical Householder QR
    dirLen = la.norm(A[currIteration:,currIteration],2)   # Don't worry about the first i rows.
    # Below: there could probably be a faster way to do this than to create an identity matrix
    #   at each iteration
    # curReflector points from vector a to norm(a)*e_1, just via vector addition
    # Also, we address possible catastrophic cancellation by avoiding subtracting similar values in the
    #   first element of the saxpy
    Reflectors[currIteration:,currIteration] = np.sign(A[currIteration,currIteration])*(-1)*dirLen*np.eye(numRows-currIteration)[0:numRows-currIteration,0] - A[currIteration:,currIteration]
    #print "check this dot product - ", np.dot(Reflectors[i:,i],Reflectors[i:,i])
    tau = 2./(np.dot(Reflectors[currIteration:,currIteration],Reflectors[currIteration:,currIteration]))
    # update trailing columns one at a time (BLAS-1) to complete the Matrix-Matrix product
    # Lets include the current column, although we could have done that vector transformation individually
    for j in range(currIteration,numColumns):
        A[currIteration:,j] = A[currIteration:,j] - np.dot(tau*Reflectors[currIteration:,currIteration], np.dot(Reflectors[currIteration:,currIteration],A[currIteration:,j]))
        # We can merge loop and update the column norm right here
        # But we should find an update scheme that doesn't have cancellation possibility
        norms[j] = norms[j] - A[currIteration,j]**2
    
def HouseHolderQRCP_BLAS1_version_1(A, numPivots):
    numRows,numColumns,permutationIndices,norms,Reflectors = HQRCP_BLAS1_setUp1(A,numPivots)

    # Only tasked to find the first numPivots pivot columns
    for i in range(numPivots):
        # Find next column pivot (lets just use naive scan instead of a heap or something)
        # Note that the hard-coded tolerance is garbage and isn't used anyway.
        lastIteration,rankDetectFunc = HQRCP_BLAS1_SwapColumns(A,permutationIndices,norms,1e-13,i,rankEstimate,False)
        HQRCP_BLAS1_UpdateMatrixAndNorms(A,Reflectors,norms,numRows,numColumns,i)

    # R will be in A, and A should be upper-trapezoidal
    # No explicit Q is being returned, as is the case with HouseHolder orthogonal triangularization methods
    return (Reflectors,A,permutationIndices)

def HouseHolderQRCP_BLAS1_version_4(A, tolerance):
    numRows,numColumns,permutationIndices,norms,Reflectors = HQRCP_BLAS1_setUp1(A,A.shape[1])
    rankEstimate = numColumns
    
    # Only tasked to find the first numPivots pivot columns
    for i in range(numColumns):
        # Find next column pivot (lets just use naive scan instead of a heap or something)
        lastIteration,rankDetectFunc = HQRCP_BLAS1_SwapColumns(A,permutationIndices,norms,tolerance,i,rankEstimate,True)
        if (lastIteration):
            rankEstimate = i
            break
        HQRCP_BLAS1_UpdateMatrixAndNorms(A,Reflectors,norms,numRows,numColumns,i)
    # R will be in A, and A should be upper-trapezoidal
    # No explicit Q is being returned, as is the case with HouseHolder orthogonal triangularization methods
    
    return (Reflectors,A,permutationIndices,rankEstimate)


"""
1. HouseHolder QR with column pivoting with BLAS-1 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
A = np.random.rand(numRows, numColumns)
# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
print "condition number of input matrix A - ", la.cond(A)
Reflectors,R,P = HouseHolderQRCP_BLAS1_version_1(A, numColumns)
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

#print np.eye(numColumns)[:,P]

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
print "Residual - ", la.norm(np.dot(Q,np.dot(R,np.eye(numColumns)[:,P].T))-A_copy,2)
"""


"""
1. Rank-detecting HouseHolder QR with column pivoting with BLAS-1 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
A = np.random.rand(numRows, numColumns)
testRank = input("What rank matrix do you want to test with: ")
U,D,V = la.svd(A,0)

for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0
# Re-form A with rank testRank
tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
print "condition number of input matrix A - ", la.cond(A)
# figure out how to deal  with detecting rank
Reflectors,R,P,rankEstimate = HouseHolderQRCP_BLAS1_version_4(A, 1e-13)
Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:rankEstimate])
# Test deviation from orthogonality
# Test residual
print "Rank estimate - ", rankEstimate
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
print "Residual - ", la.norm(np.dot(Q[:,:rankEstimate],np.dot(R[:rankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)
U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-approxRank):
    D2[numColumns-i-1] = 0
tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
"""


"""
1. HouseHolder QR with column pivoting with BLAS-2 implementations

   3 different versions: 1) Iterative with matrix Q not explicitely stored, reflectors are stored separately
                                -- so Q stores the reflectors, and A'=R stores the upper-triangular portion
                         2) Iterative with matrix Q embedded into A (needs an extra array)
                                 -- So the reflectors are stored in the lower-trapezoidal part of A, R is upper
                         3) Recursive
                         4) Same as version 1 except it can detect rank
"""
def HQRCP_BLAS2_setUp1(A,numPivots):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    permutationIndices = np.arange(numColumns)
    norms = np.zeros((numColumns,1))
    for i in range(numColumns):
        norms[i] = la.norm(A[:,i],2)**2      # power 2 because it helps with update below
    Reflectors = np.zeros((numRows, numPivots))   # will be upper-trapezoidal
    return (numRows,numColumns,permutationIndices,norms,Reflectors)

def HQRCP_BLAS2_SwapColumns(A,permutationIndices,norms,tolerance,currIteration,rankEstimate,rankDetecting):
    pivotIndex = np.argmax(norms[currIteration:])+currIteration
    kingNorm = np.amax(norms[currIteration:])
        
    if (rankDetecting):
        # Detect rank!
        if (kingNorm < tolerance):
            rankEstimate = currIteration
            return (True,rankEstimate)
            
    # Swap out pivot column with current column
    # Key question will be answered right here -- do we need to swap out entire column, or just subcolumn?
    tempColumn = A[:,currIteration].copy()     # avoid overwriting via nasty pass by reference?
    A[:,currIteration] = A[:,pivotIndex].copy() # same thing as above
    A[:,pivotIndex] = tempColumn
    savePermIndex = permutationIndices[pivotIndex].copy()
    permutationIndices[pivotIndex] = permutationIndices[currIteration].copy()
    permutationIndices[currIteration] = savePermIndex
    tempNorm = norms[pivotIndex].copy()
    norms[pivotIndex] = norms[currIteration].copy()
    norms[currIteration] = tempNorm.copy()
    return (False,rankEstimate)

def HQRCP_BLAS2_UpdateMatrixAndNorms(A,Reflectors,norms,numRows,numColumns,currIteration):
    # Continue on with typical Householder QR
    dirLen = la.norm(A[currIteration:,currIteration],2)   # Don't worry about the first i rows.
    # Below: there could probably be a faster way to do this than to create an identity matrix
    #   at each iteration
    # curReflector points from vector a to norm(a)*e_1, just via vector addition
    # Also, we address possible catastrophic cancellation by avoiding subtracting similar values in the
    #   first element of the saxpy
    Reflectors[currIteration:,currIteration] = np.sign(A[currIteration,currIteration])*(-1)*dirLen*np.eye(numRows-currIteration)[0:numRows-currIteration,0] - A[currIteration:,currIteration]
    tau = 2./(np.dot(Reflectors[currIteration:,currIteration],Reflectors[currIteration:,currIteration]))
    # update trailing columns all at once (BLAS-2) to complete the Matrix-Matrix product
    # Lets include the current column, although we could have done that vector transformation individually
    A[currIteration:,currIteration:] = A[currIteration:,currIteration:] - np.dot(tau*Reflectors[currIteration:,currIteration][:,np.newaxis], np.dot(Reflectors[currIteration:,currIteration][:,np.newaxis].T, A[currIteration:,currIteration:]))
    for j in range(currIteration,numColumns):
        # But we should find an update scheme that doesn't have cancellation possibility
        norms[j] = norms[j] - A[currIteration,j]**2


def HouseHolderQRCP_BLAS2_version_1(A,numPivots):
    numRows,numColumns,permutationIndices,norms,Reflectors = HQRCP_BLAS2_setUp1(A,numPivots)
    
    # Only tasked to find the first numPivots pivot columns
    for i in range(numPivots):
        # Find next column pivot (lets just use naive scan instead of a heap or something)
        # Note that the hard-coded tolerance is garbage and isn't used anyway.
        lastIteration,rankDetectFunc = HQRCP_BLAS1_SwapColumns(A,permutationIndices,norms,1e-13,i,rankEstimate,True)   
        HQRCP_BLAS2_UpdateMatrixAndNorms(A,Reflectors,norms,numRows,numColumns,i)
    # R will be in A, and A should be upper-trapezoidal
    # No explicit Q is being returned, as is the case with HouseHolder orthogonal triangularization methods
    return (Reflectors,A,permutationIndices)

def HouseHolderQRCP_BLAS2_version_4(A,tolerance):
    numRows,numColumns,permutationIndices,norms,Reflectors = HQRCP_BLAS2_setUp1(A,A.shape[1])
    rankEstimate = numColumns
    
    # Only tasked to find the first numPivots pivot columns
    for i in range(numColumns):
        # Find next column pivot (lets just use naive scan instead of a heap or something)
        lastIteration,rankDetectFunc = HQRCP_BLAS1_SwapColumns(A,permutationIndices,norms,tolerance,i,rankEstimate,True)
        if (lastIteration):
            rankEstimate = i
            break        
        HQRCP_BLAS2_UpdateMatrixAndNorms(A,Reflectors,norms,numRows,numColumns,i)
    # R will be in A, and A should be upper-trapezoidal
    # No explicit Q is being returned, as is the case with HouseHolder orthogonal triangularization methods
    return (Reflectors,A,permutationIndices,rankEstimate)


"""
1. HouseHolder QR with column pivoting with BLAS-2 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
A = np.random.rand(numRows, numColumns)
# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
print "condition number of input matrix A - ", la.cond(A)
Reflectors,R,P = HouseHolderQRCP_BLAS2_version_1(A, numColumns)
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
print "Residual - ", la.norm(np.dot(Q,np.dot(R,np.eye(numColumns)[:,P].T))-A_copy,2)
"""


"""
1. Rank-detecting HouseHolder QR with column pivoting with BLAS-1 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
A = np.random.rand(numRows, numColumns)
testRank = input("What rank matrix do you want to test with: ")
U,D,V = la.svd(A,0)

for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0
# Re-form A with rank testRank
tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
print "condition number of input matrix A - ", la.cond(A)
# figure out how to deal  with detecting rank
Reflectors,R,P,rankEstimate = HouseHolderQRCP_BLAS2_version_4(A, 1e-13)
Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:rankEstimate])
# Test deviation from orthogonality
# Test residual
print "Rank estimate - ", rankEstimate
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
print "Residual - ", la.norm(np.dot(Q[:,:rankEstimate],np.dot(R[:rankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)
U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-approxRank):
    D2[numColumns-i-1] = 0
tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
"""


"""
1. HouseHolder QR with BLAS-3 implementationS

   3 different versions: 1) Use compact WY connector matrix
                         2) Use UT connector matrix (from FLAME group)
                         3) Use connector matrix proposed by Ming/Duersch paper
"""

def HouseHolderQR_BLAS3_version_1(A, blockSize):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    Reflectors = np.zeros((numRows, numColumns))   # will be upper-trapezoidal
    # Late change: I will make Connector matrix have outside scope so that I can return it
    #  because in TRQRCP, I need the square Connector from the panel
    Connector = np.zeros((blockSize, blockSize))
    
    # Iterate and jump by blockSize
    for i in range(0,numColumns,blockSize):
        # Perform BLAS-2 operations to build up the delayed reflectors before updating
        #  the trailing matrix with the BLAS-3 operation
        
        # We build up the reflector matrix as we iterate over the block
        Connector = np.zeros((min(blockSize,numColumns-i),min(blockSize,numColumns-i)))
        curBlockIndex = -1
        for j in range(i,min(i+blockSize, numColumns)):
            curBlockIndex = curBlockIndex + 1
            # First, before we form our reflector from the pivot vector, we need to accumulate the delayed
            #  reflectors into it. This is a bLAS-2 operation, but the reflector rank update is building up
            #  with each iteration
            
            # Lets break this block up, because we don't need to accumulate reflector updates
            #   into the first column of the block
            if (curBlockIndex > 0):
                # update current column, this will fill out the elements above its diagonal
                #   corresponding to R
                # We must update the ENTIRE column from A[i:,j], not just A[j:,j] -- very important
                tempMatrix1 = np.dot(Reflectors[i:,i:j].T, A[i:,j])
#                tempMatrix2 = np.dot(Connector[0:curBlockIndex,0:curBlockIndex].T, tempMatrix1)
                tempMatrix2 = np.dot(Connector[0:curBlockIndex,0:curBlockIndex].T, tempMatrix1)
                tempMatrix3 = np.dot(Reflectors[i:,i:j], tempMatrix2)
                A[i:,j] = A[i:,j] + tempMatrix3

            # Now, current column will be updated with the rank-(curBlockIndex+1) update and we can calculate reflector
            dirLen = la.norm(A[j:,j],2)   # Don't worry about the first i rows
            # Below: there could probably be a faster way to do this than to create an identity matrix
            #   at each iteration
            # curReflector points from vector a to norm(a)*e_1, just via vector addition
            # Also, we address possible catastrophic cancellation by avoiding subtracting similar values in the
            #   first element of the saxpy
            Reflectors[j:,j] = np.sign(A[j,j])*(-1)*dirLen*np.eye(numRows-j)[0:numRows-j,0] - A[j:,j]
            tau = 2./(np.dot(Reflectors[j:,j],Reflectors[j:,j]))
            
            # Update current sub-column with reflector to clear out below diagonal
            tempMatrix1 = np.dot(Reflectors[j:,j][:,np.newaxis].T, A[j:,j])
            tempMatrix2 = np.dot(tau*Reflectors[j:,j][:,np.newaxis], tempMatrix1)
            A[j:,j] = A[j:,j] - tempMatrix2
            
            # Lets update the Connector matrix
            Connector[curBlockIndex,curBlockIndex] = (-1)*tau
            # Separate this out if we are on first iteration of block
            if (curBlockIndex > 0):
                tempMatrix1 = np.dot(Reflectors[i:,i:j].T, Reflectors[i:,j])
                tempMatrix2 = np.dot(Connector[0:curBlockIndex,0:curBlockIndex], tempMatrix1)
                Connector[0:curBlockIndex, curBlockIndex] = (-1)*tau*tempMatrix2
        # now we can perform BLAS-3 level update with the delayed reflectors on the trailing matrix
        trueBlockSize = min(blockSize, numColumns-i)
        tempMatrix1 = np.dot(Reflectors[i:,i:(i+trueBlockSize)].T, A[i:,(i+trueBlockSize):])
        tempMatrix2 = np.dot(Connector.T, tempMatrix1)
        tempMatrix3 = np.dot(Reflectors[i:,i:(i+trueBlockSize)], tempMatrix2)
        A[i:,(i+trueBlockSize):] = A[i:,(i+trueBlockSize):] + tempMatrix3
    # R will be in A, and A should be upper-trapezoidal
    # No explicit Q is being returned, as is the case with HouseHolder orthogonal triangularization methods
    return (Reflectors,A,Connector)


"""
1. HouseHolder QR with BLAS-3 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
A = np.random.rand(numRows, numColumns)
# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
print "condition number of input matrix A - ", la.cond(A)
Reflectors,R,Connector = HouseHolderQR_BLAS3_version_1(A, blockSize)
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
print "Residual - ", la.norm(np.dot(Q,R)-A_copy,2)
"""


"""
1. HouseHolder QR with column pivoting with BLAS-3 implementationS

   3 different versions: 1) Use compact WY connector matrix
                         2) Use UT connector matrix (from FLAME group)
                         3) Use connector matrix proposed by Ming/Duersch paper
                         4) Same as version 1 except it can detect rank
"""

def HQRCP_BLAS3_setUp1(A,maxPivots,extraRows):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    permutationIndices = np.arange(numColumns,dtype=np.int32)
    norms = np.zeros((numColumns,1))
    for i in range(numColumns):
        norms[i] = la.norm(A[:,i],2)**2      # power 2 because it helps with update below
    Reflectors = np.zeros((numRows, maxPivots))   # will be upper-trapezoidal
    
    # Lets store R explicitely here because of the excruciating pain of dealing with corrupted updates
    # Store in same fashion as A to avoid collateral damage with other functions
    #  Can optimize later
    
    # extraRows is for consideration with downsampling, which needs the bottom rows to update the sample matrix
    R = np.zeros((maxPivots+extraRows, numColumns))
    return (numRows,numColumns,permutationIndices,norms,Reflectors,R)

def HQRCP_BLAS3_OuterLoopSetUp1(blockSize,maxPivots,curIteration):
    trueBlockSize = min(blockSize, maxPivots-curIteration)
    # Perform BLAS-2 operations to build up the delayed reflectors before updating
    #  the trailing matrix with the BLAS-3 operation
    # We build up the reflector matrix as we iterate over the block
    Connector = np.zeros((trueBlockSize,trueBlockSize))
    curBlockIndex = -1
    return (trueBlockSize,Connector,curBlockIndex)

def HQRCP_BLAS3_ColumnSwaps(A,permutationIndices,norms,R,curBlockIndex,curIteration,tolerance,maxPivots,rankDetecting):
    curBlockIndex = curBlockIndex + 1
    # Find next column pivot first (lets just use naive scan instead of a heap or something)
    pivotIndex = np.argmax(norms[curIteration:])+curIteration
    kingNorm = np.amax(norms[curIteration:])
    if (rankDetecting):
        # Detect rank!
        if (kingNorm < tolerance):
            rankEstimate = curIteration
            return (curBlockIndex,True,rankEstimate)
    if (curIteration >= maxPivots):
        rankEstimate = maxPivots
        return (curBlockIndex,True,rankEstimate)

    # Swap out pivot column with current column
    # Key question will be answered right here -- do we need to swap out entire column, or just subcolumn?
    tempColumn = A[:,curIteration].copy()     # avoid overwriting via nasty pass by reference?
    A[:,curIteration] = A[:,pivotIndex].copy() # same thing as above
    A[:,pivotIndex] = tempColumn.copy()   # just for precaution
    savePermIndex = permutationIndices[pivotIndex].copy()
    permutationIndices[pivotIndex] = permutationIndices[curIteration].copy()
    permutationIndices[curIteration] = savePermIndex.copy()
    tempNorm = norms[pivotIndex].copy()
    norms[pivotIndex] = norms[curIteration].copy()
    norms[curIteration] = tempNorm.copy()
    # Need to also permute R
    # Later optimization: wasteful to permute zeros at bottom partition of matrix
    tempColumn1 = R[:,curIteration].copy()
    R[:,curIteration] = R[:,pivotIndex].copy()
    R[:,pivotIndex] = tempColumn1.copy()
    return (curBlockIndex,False,maxPivots)

def HQRCP_BLAS3_TrailingMatrixUpdate(A,Reflectors,Connector,curIteration,trueBlockSize):
        # now we can perform BLAS-3 level update with the delayed reflectors on the trailing matrix
        tempMatrix1 = np.dot(Reflectors[curIteration:,curIteration:(curIteration+trueBlockSize)].T, A[curIteration:,(curIteration+trueBlockSize):])
        # Add in the rest of the terms of the inner-product that were saved during the block iteration
        #  before the data needed was overwritten
        tempMatrix2 = np.dot(Connector.T, tempMatrix1)
        tempMatrix3 = np.dot(Reflectors[(curIteration+trueBlockSize):,curIteration:(curIteration+trueBlockSize)], tempMatrix2)
        # Here, we are supposed to store into A, not R!
        A[(curIteration+trueBlockSize):,(curIteration+trueBlockSize):] = A[(curIteration+trueBlockSize):,(curIteration+trueBlockSize):] + tempMatrix3

def HQRCP_BLAS3_UpdateConnector(Reflectors,Connector,tau,curBlockIndex,curIterationI,curIterationJ):
    Connector[curBlockIndex,curBlockIndex] = (-1)*tau
    # Separate this out if we are on first iteration of block
    if (curBlockIndex > 0):
        tempMatrix1 = np.dot(Reflectors[curIterationI:,curIterationI:curIterationJ].T, Reflectors[curIterationI:,curIterationJ])
        tempMatrix2 = np.dot(Connector[0:curBlockIndex,0:curBlockIndex], tempMatrix1)
        Connector[0:curBlockIndex, curBlockIndex] = (-1)*tau*tempMatrix2
        
def HQRCP_BLAS3_UpdateRow(A,Reflectors,Connector,R,curBlockIndex,curIterationI,curIterationJ):
    # Before we can update the norms, we need to update the current row so that we can make an informed pivot choice
    # Possible problem: we might need the non-updated part of A when performing these updates to get coefficients to scale the reflectors
    # Problem fixed (temporarily) as we are now storing R explicitely so the rows needed of A are not getting corrupted
    tempMatrix1 = np.dot(Reflectors[curIterationI:,curIterationI:(curIterationJ+1)].T, A[curIterationI:,(curIterationJ+1):])
    # we need the fix here too, since if we use the first j-i columns in the product above, we are using overwritten values,
    #   so we need to use the saved partial products and sum them up to complete the inner products
    tempMatrix2 = np.dot(Connector[0:(curBlockIndex+1),0:(curBlockIndex+1)].T, tempMatrix1)
    tempMatrix3 = np.dot(Reflectors[curIterationJ,curIterationI:(curIterationJ+1)], tempMatrix2)
    # Store into R, not A!!!
    R[curIterationJ,(curIterationJ+1):] = A[curIterationJ,(curIterationJ+1):] + tempMatrix3
        
def HQRCP_BLAS3_UpdateColumnPanel(A,Reflectors,Connector,curBlockIndex,curIterationI,curIterationJ):
    # Lets break this block up, because we don't need to accumulate reflector updates
    #   into the first column of the block
    if (curBlockIndex > 0):
        # update current column in submatrix. The elements of R are NOT FOUND HERE. This is just to attain the reflector
        # We must update the SUB column (just A[j:,j], unlike in non-pivoted BLAS level 3 HQR) -- very important
        #    BUT, we still need to use the "silent" columns of the current column of A and the upper-triangular part of the reflectors
        #      and we do this in a special way by saved the partial products in a separate array
        tempMatrix1 = np.dot(Reflectors[curIterationI:,curIterationI:curIterationJ].T, A[curIterationI:,curIterationJ])
        tempMatrix2 = np.dot(Connector[0:curBlockIndex,0:curBlockIndex].T, tempMatrix1)
        tempMatrix3 = np.dot(Reflectors[curIterationJ:,curIterationI:curIterationJ], tempMatrix2)
        # Its ok to modify A here
        A[curIterationJ:,curIterationJ] = A[curIterationJ:,curIterationJ] + tempMatrix3
        
def HQRCP_BLAS3_FactorPanel(A,Reflectors,R,curIteration,numRows,numColumns,maxPivots,extraRows):
    # Now, current column will be updated with the rank-(curBlockIndex+1) update and we can calculate reflector
    dirLen = la.norm(A[curIteration:,curIteration],2)   # Don't worry about the first i rows
    # Below: there could probably be a faster way to do this than to create an identity matrix
    #   at each iteration
    # curReflector points from vector a to norm(a)*e_1, just via vector addition
    # Also, we address possible catastrophic cancellation by avoiding subtracting similar values in the
    #   first element of the saxpy
    Reflectors[curIteration:,curIteration] = np.sign(A[curIteration,curIteration])*(-1)*dirLen*np.eye(numRows-curIteration)[0:numRows-curIteration,0] - A[curIteration:,curIteration]
    tau = 2./(np.dot(Reflectors[curIteration:,curIteration],Reflectors[curIteration:,curIteration]))
    # Update current sub-column with reflector to clear out below diagonal (could just relace this with setting equal to norm*e1)
    tempMatrix1 = np.dot(Reflectors[curIteration:,curIteration][:,np.newaxis].T, A[curIteration:,curIteration])
    tempMatrix2 = np.dot(tau*Reflectors[curIteration:,curIteration][:,np.newaxis], tempMatrix1)
    # Lets modify R instead of A
    R[curIteration:,curIteration] = A[curIteration:(maxPivots+extraRows),curIteration] - tempMatrix2[:(maxPivots+extraRows-curIteration)]
    return tau
        
def HQRCP_BLAS3_UpdateNorms(R,norms,curIteration,numColumns):
    for k in range(curIteration,numColumns):
        # But we should find an update scheme that doesn't have cancellation possibility
        # Use R here, not A!!!
        norms[k] = norms[k] - R[curIteration,k]**2
        
def HouseHolderQRCP_BLAS3_version_1(A,blockSize,maxPivots,extraRows):
    numRows,numColumns,permutationIndices,norms,Reflectors,R = HQRCP_BLAS3_setUp1(A,maxPivots,extraRows)
    
    # Iterate and jump by blockSize
    for i in range(0,maxPivots,blockSize):
        trueBlockSize,Connector,curBlockIndex = HQRCP_BLAS3_OuterLoopSetUp1(blockSize,maxPivots,i)
        
        for j in range(i,i+trueBlockSize):
            curBlockIndex,lastIteration,rankEstimateFunc = HQRCP_BLAS3_ColumnSwaps(A,permutationIndices,norms,R,curBlockIndex,j,maxPivots,False)
            
            # First, before we form our reflector from the pivot vector, we need to accumulate the delayed
            #  reflectors into it. This is a bLAS-2 operation, but the reflector rank update is building up
            #  with each iteration
            HQRCP_BLAS3_UpdateColumnPanel(A,Reflectors,Connector,curBlockIndex,i,j)
            tau = HQRCP_BLAS3_FactorPanel(A,Reflectors,R,j,numRows,numColumns,maxPivots,extraRows)            
            # Lets update the Connector matrix
            HQRCP_BLAS3_UpdateConnector(Reflectors,Connector,tau,curBlockIndex,i,j)
            HQRCP_BLAS3_UpdateRow(A,Reflectors,Connector,R,curBlockIndex,i,j)
            HQRCP_BLAS3_UpdateNorms(R,norms,j,numColumns)
        
        HQRCP_BLAS3_TrailingMatrixUpdate(A,Reflectors,Connector,i,trueBlockSize)

    # R will be in A, and A should be upper-trapezoidal
    # No explicit Q is being returned, as is the case with HouseHolder orthogonal triangularization methods
    return (Reflectors,R,permutationIndices)

def HouseHolderQRCP_BLAS3_version_4(A, blockSize,tolerance,maxPivots,extraRows):
    numRows,numColumns,permutationIndices,norms,Reflectors,R = HQRCP_BLAS3_setUp1(A,maxPivots,extraRows)
    rankEstimate = maxPivots
    
    # Iterate and jump by blockSize
    for i in range(0,maxPivots,blockSize):
        trueBlockSize,Connector,curBlockIndex = HQRCP_BLAS3_OuterLoopSetUp1(blockSize,maxPivots,i)
        foundRank = False   # flag for info once out of inner loop
        
        for j in range(i,i+trueBlockSize):
            curBlockIndex,lastIteration,rankEstimateFunc = HQRCP_BLAS3_ColumnSwaps(A,permutationIndices,norms,R,curBlockIndex,j,tolerance,maxPivots,True)
            if (lastIteration):
                rankEstimate = rankEstimateFunc
                foundRank = True
                break

            # First, before we form our reflector from the pivot vector, we need to accumulate the delayed
            #  reflectors into it. This is a bLAS-2 operation, but the reflector rank update is building up
            #  with each iteration
            HQRCP_BLAS3_UpdateColumnPanel(A,Reflectors,Connector,curBlockIndex,i,j)
            tau = HQRCP_BLAS3_FactorPanel(A,Reflectors,R,j,numRows,numColumns,maxPivots,extraRows)           
            # Lets update the Connector matrix
            HQRCP_BLAS3_UpdateConnector(Reflectors,Connector,tau,curBlockIndex,i,j)
            HQRCP_BLAS3_UpdateRow(A,Reflectors,Connector,R,curBlockIndex,i,j)
            HQRCP_BLAS3_UpdateNorms(R,norms,j,numColumns)
        
        if (foundRank):
            break
        HQRCP_BLAS3_TrailingMatrixUpdate(A,Reflectors,Connector,i,trueBlockSize)

    # R will be in A, and A should be upper-trapezoidal
    # No explicit Q is being returned, as is the case with HouseHolder orthogonal triangularization methods
    return (Reflectors,R,permutationIndices,rankEstimate)


# Short-and-Fat QRCP (python implementation to verify correctness for a parallel implementation)
# Note: downsampling of Sampled matrix does not need itself to update, we can overwrite A with no issue!
# Question: Can we fatten the tree? Yes, I guess whatever allows logP levels, but that should be attained
#  simply by having each processor contribute blockSize columns of A up the tree.

def SFQRCP(A,blockSize,numProcessors):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    # Perform QR factorizations on each segment of blockSize columns
    # Two types of loops. First on numColumnsFirst == numColumns/blockSize
    #  Second on 2*blockSize
    
    P = np.arange(numColumns,dtype=int32)
    size1 = numColumns/numProcessors + int(numColumns % numProcessors)
    size2 = 0
    for i in range(size1):
        trueEnd1 = min((i+1)*numProcessors,numColumns)
        trueEnd2 = min((i+1)*blockSize,numColumns)
        trueBlockSize = min(blockSize,trueEnd2-i*blockSize)
        Q,R,P_local = sc.qr(A[:,i*numProcessors:trueEnd],overwrite_a = False, mode='economic',pivoting=True)
        # Swap the most important columns of the submatrix of A to the front
        A[:,i*blockSize:trueEnd2] = A[:,i*numProcessors:trueEnd][:,P_local[:blockSize]]
        # Get proper global ID on the pivoted columns. Very important later
        P[i*blockSize:trueEnd2] = P_local[:trueBlockSize] + i*numProcessors
        size2 = size2 + min(blockSize,trueBlockSize)
    
    # Reset size1
    size1 = 0
    # Outer loop represents recursion up a tree
    while (size2>blockSize):
        for i in range(0,size2,2*blockSize):
            trueBlockSize1 = min(2*blockSize,size2 - i)
            trueBlockSize2 = min(blockSize,size2 - i)
            Q,R,P[size1:trueBlockSize2] = sc.qr(A[:,i:(i+trueBlockSize1)],mode='economic',pivoting=True)
            # Swap the most important columns of the submatrix of A to the front
            A[:,size1:(size1+trueBlockSize2)] = A[:,P[size1:trueBlockSize2]]
            size1 = size1 + trueBlockSize2
        size2 = size1
        size1 = 0
    # Perform true QR factorization with column pivoting on the most representative columns
    Q,S,P[:blockSize] = sc.qr(A[:,i*numProcessors:trueEnd],overwrite_a = False, mode='economic',pivoting=True)
    
    # We might need to reform the rest of R_12 in order to have it work with downsampling!!!!!!
    return Q,S,P
    


"""
1. HouseHolder QR with column pivoting with BLAS-3 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
A = np.random.rand(numRows, numColumns)
# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
print "condition number of input matrix A - ", la.cond(A)
Reflectors,R,P = HouseHolderQRCP_BLAS3_version_1(A, numColumns, blockSize)
#print R
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
#print "check R - ", R
#print "check P - ", P
#print "check Q - ", Q
#print "check this vector - ", np.dot(Q,np.dot(R,np.eye(numColumns)[:,P].T))
#print "check this vector, a - ", A_copy
print "Residual - ", la.norm(np.dot(Q,np.dot(R,np.eye(numColumns)[:,P].T))-A_copy,2)
"""


"""
1. Rank-detecting HouseHolder QR with column pivoting with BLAS-3 external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
A = np.random.rand(numRows, numColumns)
testRank = input("What rank matrix do you want to test with: ")
U,D,V = la.svd(A,0)

for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0
# Re-form A with rank testRank
tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
print "condition number of input matrix A - ", la.cond(A)
# figure out how to deal  with detecting rank
Reflectors,R,P,rankEstimate = HouseHolderQRCP_BLAS3_version_4(A, blockSize,1e-13,numColumns)
Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:rankEstimate])
# Test deviation from orthogonality
# Test residual
print "Rank estimate - ", rankEstimate
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:rankEstimate].T,Q[:,:rankEstimate])-np.eye(rankEstimate),2)
print "Residual - ", la.norm(np.dot(Q[:,:rankEstimate],np.dot(R[:rankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)
U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-approxRank):
    D2[numColumns-i-1] = 0
tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
"""


# Single-sample randomized QR with column pivoting

# Only 1 version -> not rank detecting. User specifies a rankEstimate that they want to use
# Single-sampled Randomized QRCP does not support rank-detection by its very semantics.
#   That king of support is left to repeated-sampled Randomized QRCP

def SSRQRCP(A,rankEstimate,blockSize,oversamplingParameter):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    sampleRank = rankEstimate+oversamplingParameter
    # really? Why dont I just store in A? Check that out later
    R = np.zeros((rankEstimate, numColumns))
    
    # Generate the random G.I.I.D matrix using numpy random normal
    Omega = np.random.normal(0.0, 1.0, (sampleRank,numRows))
    Sample = np.dot(Omega, A)
    # Perform QR with column-pivoting solely to find the correct pivot order
    # Later optimization: have QRCP call only iterate over rankEstimate columns
    Reflectors,R_temp,Perm = HouseHolderQRCP_BLAS3_version_1(Sample,blockSize,rankEstimate,0)
    # Permute the columns of A so that we can run blocked QR on the sampleRank-size panel of A
    # Note: I might want to only permute with this: Perm[:rankEstimate], but that hit trouble when I tried
    A = np.dot(A,np.eye(numColumns)[:,Perm])
    # Factor to find the upper-left R_11 upper-triangular matrix
    Reflectors,R_temp,Connector = HouseHolderQR_BLAS3_version_1(A[:,0:rankEstimate],blockSize)
    R[:,:rankEstimate] = R_temp[:rankEstimate,:rankEstimate]   # R_temp is A, don't forget
    # We can use the data we have already to solve for submatrix R_12,
    #  which is all we need to be able to create an approximation to original matrix A.

    # But first we need to recreate Q.T from the reflectors we have
    ReflectorsCopy = Reflectors.copy()  # just for saefty and debugging
    # Try version_3 later to see if it works
    Q = convertReflectorsToOrthgonal_version_1(ReflectorsCopy)
    R[:,rankEstimate:] = np.dot(Q[:,:rankEstimate].T, A[:,rankEstimate:])
    return (Reflectors,R,Perm)


"""
3. Single-sample randomized QR with column pivoting external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
testRank = input("What rank matrix do you want to test with: ")
approxRank = input("Enter approximation rank: ")
oversamplingParameter = input("Enter oversampling parameter : ")

# Error checking
if (testRank < approxRank):
    sys.exit()

A = np.random.rand(numRows, numColumns)

# I need to change the singular values of A in order to vary the numerical rank
#   and properly test this method
U,D,V = la.svd(A,0)
for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0

tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
U1,D1,V1 = la.svd(A,0)

# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
print "condition number of input matrix A - ", la.cond(A)

# Note: SSRQCP does not return a permutation matrix because that operation is already embedded into the algorithm
Reflectors,R,P = SSRQRCP(A, approxRank, blockSize, oversamplingParameter)
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:approxRank].T,Q[:,:approxRank])-np.eye(approxRank),2)
print "Residual - ", la.norm(np.dot(Q[:,:approxRank],R)-np.dot(A_copy, np.eye(numColumns)[:,P[:numColumns]]),2)

U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-approxRank):
    D2[numColumns-i-1] = 0

tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
print "Residual of truncated SVD - ", la.norm(A_trunc-A_copy,2)
"""


# Repeated-sample randomized QR with column pivoting

# Kernel
# Version 1 -- Not rank-detecting. User specifies an approximation rank
# Version 2 -- Rank-detecting to some tolerance

def RSRQRCP_setUp1(A,rankEstimate,blockSize,oversamplingParameter):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    sampleRank = blockSize+oversamplingParameter
    R = np.zeros((rankEstimate, numColumns))
    Reflectors = np.zeros((numRows, rankEstimate))
    Perm = np.arange(numColumns,dtype=int)
    return (numRows,numColumns,sampleRank,R,Reflectors,Perm)

def RSRQRCP_loopSetUp1(i,blockSize,rankEstimate,numRows,sampleRank):
        trueBlockSize = min(blockSize, rankEstimate-i)        
        rowStartRange = i
        columnStartRange = i
        rowEndRange = min(i+blockSize, rankEstimate)
        columnEndRange = min(i+blockSize, rankEstimate)
        # Generate the random G.I.I.D matrix using numpy random normal
        # Note that the compression matrix gets smaller with each block iteration
        #   as we recurse into smaller submatrices of A
        Omega = np.random.normal(0.0, 1.0, (sampleRank,numRows-rowStartRange))
        Sample = np.dot(Omega, A[rowStartRange:,columnStartRange:])
        return (trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange,Omega,Sample)

def RSRQRCP_loopSwap1(Perm,A,R,Perm_temp,columnStartRange,numColumns,i):
        # Note: We are also permuting the columns of R that lie in A above the current submatrix
        # Later optimization: I think this is doing too much work if Perm.size << numColumns
        # Still need to update Perm itself, now that I changed it above to Perm_temp
        # Carefully shuffle the contents in Perm form Perm_temp
        Perm[columnStartRange:] = Perm[columnStartRange:][Perm_temp]
        # need to do something here with Permutation  -- Should we only use the trailing ccolumns of A here?
        A[:,i:] = np.dot(A[:,i:],np.eye(numColumns-i)[:,Perm_temp])
        # either store R explicitely in A, or need to permute R itself
        R[:,i:] = np.dot(R[:,i:],np.eye(numColumns-i)[:,Perm_temp])

def RSRQRCP_factorUpdate1(A,Reflectors,R,rankDetect,rowStartRange,columnStartRange,rowEndRange,columnEndRange,trueBlockSize,currIteration,rankDetecting):
    # Factor to find the upper-left R_11 upper-triangular matrix
    Reflectors[rowStartRange:,columnStartRange:columnEndRange],R_temp,Connector = HouseHolderQR_BLAS3_version_1(A[rowStartRange:,columnStartRange:columnEndRange],trueBlockSize)
    if (Connector.shape != (trueBlockSize,trueBlockSize)):
        sys.exit()
    # The copy below is because HQR_BLAS3 routine factors A and stuffs R in A, so the shape of R is the shape of A
    #   We could optimize this later, but this isn't a major thing.
    R[rowStartRange:rowEndRange,columnStartRange:columnEndRange] = R_temp[:trueBlockSize,:]   # R_temp is A, don't forget
    # We can use the data we have already to solve for submatrix R_12,
    #  which is all we need to be able to create an approximation to original matrix A.

    # But first we need to recreate Q.T from the reflectors we have
    #ReflectorsCopy = Reflectors.copy()  # just for safety and debugging
    # Try version_3 later to see if it works
    tempMatrix1 = np.dot(Reflectors[rowStartRange:,columnStartRange:columnEndRange].T,A[rowStartRange:,columnEndRange:])
    tempMatrix2 = np.dot(Connector.T,tempMatrix1)
    tempMatrix3 = np.dot(Reflectors[rowStartRange:rowEndRange,columnStartRange:columnEndRange],tempMatrix2)
    # Watch out. I think we may have a problem with the panel of A being overwritten by HQR_BLAS3
    R[rowStartRange:rowEndRange,columnEndRange:] = A[rowStartRange:rowEndRange,columnEndRange:] + tempMatrix3
    #Q = convertReflectorsToOrthgonal_version_1(Reflectors[rowStartRange:,columnStartRange:columnEndRange])
    #R[rowStartRange:rowEndRange,columnEndRange:] = np.dot(Q[:,0:trueBlockSize].T, A[currIteration:,columnEndRange:])
    
    if (rankDetecting):
        # Check on rank
        if (rankDetect < trueBlockSize):
            rankEstimate = currIteration+rankDetect
            return (True,rankEstimate)
    # Now we need to update the trailing matrix and continue on with next block iteration
    tempMatrix3 = np.dot(Reflectors[rowEndRange:,columnStartRange:columnEndRange],tempMatrix2)
    A[rowEndRange:,columnEndRange:] = A[rowEndRange:,columnEndRange:] + tempMatrix3
    return False,trueBlockSize

def RSRQRCP_version1(A,rankEstimate,blockSize, oversamplingParameter):
    numRows,numColumns,sampleRank,R,Reflectors,Perm = RSRQRCP_setUp1(A,rankEstimate,blockSize,oversamplingParameter)

    # Iterate over blocks, resampling each time
    for i in range(0,rankEstimate,blockSize):
        trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange,Omega,Sample = RSRQRCP_loopSetUp1(i,blockSize,rankEstimate,numRows,sampleRank)
        
        # Perform QR with column-pivoting solely to find the correct pivot order for the next b pivots
        # QRCP can only select blockSize pivot columns
        # Later optimization: I could change the blockSize here to utilize BLAS level 3 more, since right now,
        #  blockSize == maxPivots as arguments        
        Reflectors_temp,R_temp,Perm_temp = HouseHolderQRCP_BLAS3_version_1(Sample, trueBlockSize, trueBlockSize,0)
        
        # Permute the columns of A so that we can run blocked QR on the sampleRank-size panel of A
        RSRQRCP_loopSwap1(Perm,A,R,Perm_temp,columnStartRange,numColumns,i)

        lastIteration,rankEstimateFunc = RSRQRCP_factorUpdate1(A,Reflectors,R,trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange,trueBlockSize,i,False)
    return (Reflectors,R,Perm)

def RSRQRCP_version2(A,blockSize,Tolerance,oversamplingParameter):
    numRows,numColumns,sampleRank,R,Reflectors,Perm = RSRQRCP_setUp1(A,A.shape[1],blockSize,oversamplingParameter)
    rankEstimate = numColumns

    # Iterate over blocks, resampling each time
    for i in range(0,numColumns,blockSize):
        trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange,Omega,Sample = RSRQRCP_loopSetUp1(i,blockSize,numColumns,numRows,sampleRank)
        
        # Perform QR with column-pivoting solely to find the correct pivot order for the next b pivots
        # QRCP can only select blockSize pivot columns
        # Later optimization: I could change the blockSize here to utilize BLAS level 3 more, since right now,
        #  blockSize == maxPivots as arguments
        Reflectors_temp,R_temp,Perm_temp,rankDetect = HouseHolderQRCP_BLAS3_version_4(Sample, trueBlockSize, Tolerance, trueBlockSize,0)
        
        # Wait on checking rank, because there is still some background computation
        #   that we need to do if detectRank > 1
        
        # Permute the columns of A so that we can run blocked QR on the sampleRank-size panel of A
        RSRQRCP_loopSwap1(Perm,A,R,Perm_temp,columnStartRange,numColumns,i)
        lastIteration,rankEstimateFunc = RSRQRCP_factorUpdate1(A,Reflectors,R,rankDetect,rowStartRange,columnStartRange,rowEndRange,columnEndRange,trueBlockSize,i,True)
        if (lastIteration):
            rankEstimate = rankEstimateFunc
            break

    return (Reflectors,R,Perm,rankEstimate)


"""
5. Repeated-sample randomized QR with column pivoting external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
testRank = input("What rank matrix do you want to test with: ")
approxRank = input("Enter approximation rank: ")
oversamplingParameter = input("Enter oversampling parameter : ")

# Error checking
if (testRank < approxRank):
    sys.exit()

A = np.random.rand(numRows, numColumns)

# I need to change the singular values of A in order to vary the numerical rank
#   and properly test this method
U,D,V = la.svd(A,0)
for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0

tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
U1,D1,V1 = la.svd(A,0)

# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
print "condition number of input matrix A - ", la.cond(A)

# Note: SSRQCP does not return a permutation matrix because that operation is already embedded into the algorithm
Reflectors,R,P = RSRQRCP_version1(A, approxRank, blockSize, oversamplingParameter)
#print "Reflectors - ", Reflectors
#print "R - ", R
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:approxRank].T,Q[:,:approxRank])-np.eye(approxRank),2)
print "Residual - ", la.norm(np.dot(Q[:,:approxRank],R)-np.dot(A_copy, np.eye(numColumns)[:,P]),2)

U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-approxRank):
    D2[numColumns-i-1] = 0

tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
print "Residual of truncated SVD - ", la.norm(A_trunc-A_copy,2)
"""


"""
Rank-detecting Repeated-sample randomized QR with column pivoting external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
testRank = input("What rank matrix do you want to test with: ")
oversamplingParameter = input("Enter oversampling parameter : ")

A = np.random.rand(numRows, numColumns)

# I need to change the singular values of A in order to vary the numerical rank
#   and properly test this method
U,D,V = la.svd(A,0)
for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0

tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
U1,D1,V1 = la.svd(A,0)

# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
print "condition number of input matrix A - ", la.cond(A)

# Note: SSRQCP does not return a permutation matrix because that operation is already embedded into the algorithm
Reflectors,R,P,rankEstimate = RSRQRCP_version2(A, blockSize, 1e-13,oversamplingParameter)
print "Rank estimate - ", rankEstimate

#print "Reflectors - ", Reflectors
#print "R - ", R
Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:rankEstimate])

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:rankEstimate].T,Q[:,:rankEstimate])-np.eye(rankEstimate),2)
print "Residual - ", la.norm(np.dot(Q[:,:rankEstimate],R[:rankEstimate,:])-np.dot(A_copy, np.eye(numColumns)[:,P]),2)

U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-testRank):
    D2[numColumns-i-1] = 0

tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
print "Residual of truncated SVD - ", la.norm(A_trunc-A_copy,2)
"""


# Randomized QR with column pivoting

# Version 1 -- not rank detecting. User passes in an approximation rank
# Version 2 -- rank detecting

def RQRCP_setUp1(A,rankEstimate,blockSize,oversamplingParameter):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    sampleRank = blockSize+oversamplingParameter
    R = np.zeros((rankEstimate, numColumns))
    Reflectors = np.zeros((numRows, rankEstimate))
    Perm = np.arange(numColumns,dtype=int)
    return (numRows,numColumns,sampleRank,R,Reflectors,Perm)

def RQRCP_loopSetUp1(currIteration,blockSize,rankEstimate,numRows,sampleRank):
        trueBlockSize = min(blockSize, rankEstimate-currIteration)        
        rowStartRange = currIteration
        columnStartRange = currIteration
        rowEndRange = min(currIteration+blockSize, rankEstimate)
        columnEndRange = min(currIteration+blockSize, rankEstimate)
        # Generate the random G.I.I.D matrix using numpy random normal
        # Note that the compression matrix gets smaller with each block iteration
        #   as we recurse into smaller submatrices of A
        Omega = np.random.normal(0.0, 1.0, (sampleRank,numRows-rowStartRange))
        Sample = np.dot(Omega, A[rowStartRange:,columnStartRange:])
        return (trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange,Omega,Sample)

def RQRCP_loopSwap1(Perm,A,R,Perm_temp,columnStartRange,numColumns,currIteration):
        # Note: We are also permuting the columns of R that lie in A above the current submatrix
        # Later optimization: I think this is doing too much work if Perm.size << numColumns
        # Still need to update Perm itself, now that I changed it above to Perm_temp
        # Carefully shuffle the contents in Perm form Perm_temp
        Perm[columnStartRange:] = Perm[columnStartRange:][Perm_temp]
        # need to do something here with Permutation  -- Should we only use the trailing ccolumns of A here?
        A[:,currIteration:] = np.dot(A[:,currIteration:],np.eye(numColumns-currIteration)[:,Perm_temp])
        # either store R explicitely in A, or need to permute R itself
        R[:,currIteration:] = np.dot(R[:,currIteration:],np.eye(numColumns-currIteration)[:,Perm_temp])

def RQRCP_factorUpdate1(A,Reflectors,R,rankDetect,rowStartRange,columnStartRange,rowEndRange,columnEndRange,trueBlockSize,currIteration,rankDetecting):
    # Factor to find the upper-left R_11 upper-triangular matrix
    Reflectors[rowStartRange:,columnStartRange:columnEndRange],R_temp,Connector = HouseHolderQR_BLAS3_version_1(A[rowStartRange:,columnStartRange:columnEndRange],trueBlockSize)
    if (Connector.shape != (trueBlockSize,trueBlockSize)):
        sys.exit()
    R[rowStartRange:rowEndRange,columnStartRange:columnEndRange] = R_temp[:trueBlockSize,:]   # R_temp is A, don't forget
    # We can use the data we have already to solve for submatrix R_12,
    #  which is all we need to be able to create an approximation to original matrix A.

    # But first we need to recreate Q.T from the reflectors we have
    #ReflectorsCopy = Reflectors.copy()  # just for safety and debugging
    # Try version_3 later to see if it works
    tempMatrix1 = np.dot(Reflectors[rowStartRange:,columnStartRange:columnEndRange].T,A[rowStartRange:,columnEndRange:])
    tempMatrix2 = np.dot(Connector.T,tempMatrix1)
    tempMatrix3 = np.dot(Reflectors[rowStartRange:rowEndRange,columnStartRange:columnEndRange],tempMatrix2)
    # Watch out. I think we may have a problem with the panel of A being overwritten by HQR_BLAS3
    R[rowStartRange:rowEndRange,columnEndRange:] = A[rowStartRange:rowEndRange,columnEndRange:] + tempMatrix3   
    
    #Q = convertReflectorsToOrthgonal_version_1(Reflectors[rowStartRange:,columnStartRange:columnEndRange])
    #R[rowStartRange:rowEndRange,columnEndRange:] = np.dot(Q[:,0:trueBlockSize].T, A[currIteration:,columnEndRange:])
    
    if (rankDetecting):
        # Check on rank
        if (rankDetect < trueBlockSize):
            rankEstimate = currIteration+rankDetect
            return (True,rankEstimate)
    # Now we need to update the trailing matrix and continue on with next block iteration
    tempMatrix3 = np.dot(Reflectors[rowEndRange:,columnStartRange:columnEndRange],tempMatrix2)
    A[rowEndRange:,columnEndRange:] = A[rowEndRange:,columnEndRange:] + tempMatrix3
    #A[rowEndRange:,columnEndRange:] = np.dot(Q[:,trueBlockSize:].T, A[rowStartRange:,columnEndRange:])
    return False,trueBlockSize

def RQRCP_Downsample(currIteration,blockSize,trueBlockSize,sampleRank,rankEstimate,rowStartRange,rowEndRange,columnStartRange,columnEndRange,Sample,R,S_temp):
    # Time to update the Sample, subtracting out the components from the submatrix above
    # Don't update unless we have another iteration. Bad matrix alignment occurs otherwise
    if (currIteration+blockSize < rankEstimate):
        # Update shape of downsampled sample matrix
        Sample = Sample[:,trueBlockSize:]
        Sample[blockSize:,:] = S_temp[blockSize:sampleRank,blockSize:]
        tempMatrix1 = np.dot(la.inv(R[rowStartRange:rowEndRange,columnStartRange:columnEndRange]), R[rowStartRange:rowEndRange,columnEndRange:])
        tempMatrix2 = np.dot(S_temp[:blockSize,:blockSize], tempMatrix1)
        Sample[:blockSize,:] = S_temp[:blockSize,blockSize:] - tempMatrix2
    return Sample

def RQRCP_version1(A,rankEstimate,blockSize,oversamplingParameter):
    numRows,numColumns,sampleRank,R,Reflectors,Perm = RQRCP_setUp1(A,rankEstimate,blockSize,oversamplingParameter)

    # Iterate over blocks, resampling each time
    for i in range(0,rankEstimate,blockSize):
        trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange,Omega,Sample = RQRCP_loopSetUp1(i,blockSize,rankEstimate,numRows,sampleRank)
        
        # Perform QR with column-pivoting solely to find the correct pivot order for the next b pivots
        # QRCP can only select blockSize pivot columns
        # Later optimization: I could change the blockSize here to utilize BLAS level 3 more, since right now,
        #  blockSize == maxPivots as arguments        
        Reflectors_temp,S_temp,Perm_temp = HouseHolderQRCP_BLAS3_version_1(Sample, trueBlockSize, trueBlockSize,oversamplingParameter)
        
        # Permute the columns of A so that we can run blocked QR on the sampleRank-size panel of A
        RQRCP_loopSwap1(Perm,A,R,Perm_temp,columnStartRange,numColumns,i)

        lastIteration,rankEstimateFunc = RQRCP_factorUpdate1(A,Reflectors,R,trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange,trueBlockSize,i,False)
        Sample = RQRCP_Downsample(i,blockSize,trueBlockSize,sampleRank,rankEstimate,rowStartRange,rowEndRange,columnStartRange,columnEndRange,Sample,R,S_temp)
    return (Reflectors,R,Perm)

def RQRCP_version2(A,blockSize,Tolerance,oversamplingParameter):
    numRows,numColumns,sampleRank,R,Reflectors,Perm = RQRCP_setUp1(A,A.shape[1],blockSize,oversamplingParameter)
    rankEstimate = numColumns

    # Iterate over blocks, resampling each time
    for i in range(0,numColumns,blockSize):
        trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange,Omega,Sample = RQRCP_loopSetUp1(i,blockSize,numColumns,numRows,sampleRank)
        
        # Perform QR with column-pivoting solely to find the correct pivot order for the next b pivots
        # QRCP can only select blockSize pivot columns
        # Later optimization: I could change the blockSize here to utilize BLAS level 3 more, since right now,
        #  blockSize == maxPivots as arguments
        Reflectors_temp,S_temp,Perm_temp,rankDetect = HouseHolderQRCP_BLAS3_version_4(Sample, trueBlockSize, Tolerance, trueBlockSize,oversamplingParameter)
        
        # Wait on checking rank, because there is still some background computation
        #   that we need to do if detectRank > 1
        
        # Permute the columns of A so that we can run blocked QR on the sampleRank-size panel of A
        RQRCP_loopSwap1(Perm,A,R,Perm_temp,columnStartRange,numColumns,i)
        lastIteration,rankEstimateFunc = RQRCP_factorUpdate1(A,Reflectors,R,rankDetect,rowStartRange,columnStartRange,rowEndRange,columnEndRange,trueBlockSize,i,True)
        if (lastIteration):
            rankEstimate = rankEstimateFunc
            break
        Sample = RQRCP_Downsample(i,blockSize,trueBlockSize,sampleRank,rankEstimate,rowStartRange,rowEndRange,columnStartRange,columnEndRange,Sample,R,S_temp)
    return (Reflectors,R,Perm,rankEstimate)


"""
5. Randomized QR with column pivoting external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
testRank = input("What rank matrix do you want to test with: ")
approxRank = input("Enter approximation rank: ")
oversamplingParameter = input("Enter oversampling parameter: ")

# Error checking
if (testRank < approxRank):
    sys.exit()

A = np.random.rand(numRows, numColumns)

# I need to change the singular values of A in order to vary the numerical rank
#   and properly test this method
U,D,V = la.svd(A,0)
for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0

tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
U1,D1,V1 = la.svd(A,0)

# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
print "condition number of input matrix A - ", la.cond(A)

# Note: SSRQCP does not return a permutation matrix because that operation is already embedded into the algorithm
Reflectors,R,P = RQRCP_version1(A, approxRank, blockSize,oversamplingParameter)
#print "Reflectors - ", Reflectors
#print "R - ", R
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:approxRank].T,Q[:,:approxRank])-np.eye(approxRank),2)
print "Residual - ", la.norm(np.dot(Q[:,:approxRank],R)-np.dot(A_copy, np.eye(numColumns)[:,P]),2)

U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-approxRank):
    D2[numColumns-i-1] = 0

tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
print "Residual of truncated SVD - ", la.norm(A_trunc-A_copy,2)
"""


"""
Rank-detecting Randomized QR with column pivoting external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
testRank = input("What rank matrix do you want to test with: ")
oversamplingParameter = input("Enter oversampling parameter : ")

A = np.random.rand(numRows, numColumns)

# I need to change the singular values of A in order to vary the numerical rank
#   and properly test this method
U,D,V = la.svd(A,0)
for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0

tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
U1,D1,V1 = la.svd(A,0)

# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
print "condition number of input matrix A - ", la.cond(A)

# Note: SSRQCP does not return a permutation matrix because that operation is already embedded into the algorithm
Reflectors,R,P,rankEstimate = RQRCP_version2(A, blockSize, 1e-13,oversamplingParameter)
print "Rank estimate - ", rankEstimate

#print "Reflectors - ", Reflectors
#print "R - ", R
Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:rankEstimate])

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:rankEstimate].T,Q[:,:rankEstimate])-np.eye(rankEstimate),2)
print "Residual - ", la.norm(np.dot(Q[:,:rankEstimate],R[:rankEstimate,:])-np.dot(A_copy, np.eye(numColumns)[:,P]),2)

U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-testRank):
    D2[numColumns-i-1] = 0

tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
print "Residual of truncated SVD - ", la.norm(A_trunc-A_copy,2)
"""


# Truncated Randomized QR with column pivoting with no trailing matrix update

# Version 1 -- not rank detecting. User passes in an approximation rank
# Version 2 -- rank detecting

def TRQRCP_setUp1(A,rankEstimate,blockSize,oversamplingParameter):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    sampleRank = blockSize+oversamplingParameter
    R = np.zeros((rankEstimate, numColumns))
    Connector = np.zeros((rankEstimate, rankEstimate))
    Reflectors = np.zeros((numRows, rankEstimate))
    Perm = np.arange(numColumns,dtype=int)
    # Generate the random G.I.I.D matrix using numpy random normal
    # Note that the compression matrix gets smaller with each block iteration
    #   as we recurse into smaller submatrices of A
    Omega = np.random.normal(0.0, 1.0, (sampleRank,numRows))
    Sample = np.dot(Omega, A)
    return (numRows,numColumns,sampleRank,R,Reflectors,Perm,Omega,Sample,Connector)

def TRQRCP_loopSetUp1(currIteration,blockSize,rankEstimate,numRows,sampleRank):
        trueBlockSize = min(blockSize, rankEstimate-currIteration)        
        rowStartRange = currIteration
        columnStartRange = currIteration
        rowEndRange = min(currIteration+blockSize, rankEstimate)
        columnEndRange = min(currIteration+blockSize, rankEstimate)
        return (trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange)

def TRQRCP_loopSwap1(Perm,A,R,Perm_temp,columnStartRange,numColumns,currIteration):
        # Note: We are also permuting the columns of R that lie in A above the current submatrix
        # Later optimization: I think this is doing too much work if Perm.size << numColumns
        # Still need to update Perm itself, now that I changed it above to Perm_temp
        # Carefully shuffle the contents in Perm form Perm_temp
        Perm[columnStartRange:] = Perm[columnStartRange:][Perm_temp]
        # need to do something here with Permutation  -- Should we only use the trailing ccolumns of A here?
        A[:,currIteration:] = np.dot(A[:,currIteration:],np.eye(numColumns-currIteration)[:,Perm_temp])
        # either store R explicitely in A, or need to permute R itself
        R[:,currIteration:] = np.dot(R[:,currIteration:],np.eye(numColumns-currIteration)[:,Perm_temp])

def TRQRCP_accumulateUpdatesIntoColumnPanel(A,Reflectors,Connector,rowStartRange,rowEndRange,columnStartRange,columnEndRange):
    # Before factoring, we need to accumulate delayed updates into the panel
    tempMatrix1 = np.dot(Reflectors[:,:columnStartRange].T, A[:,columnStartRange:columnEndRange])
    tempMatrix2 = np.dot(Connector[:rowStartRange,:columnStartRange].T, tempMatrix1)
    tempMatrix3 = np.dot(Reflectors[rowStartRange:,:columnStartRange], tempMatrix2)
    A[rowStartRange:,columnStartRange:columnEndRange] = A[rowStartRange:,columnStartRange:columnEndRange] + tempMatrix3

def TRQRCP_accumulateUpdatesIntoRowPanel(A,Reflectors,R,Connector,rowStartRange,rowEndRange,columnStartRange,columnEndRange):
    # We use all rows of A, not just all rows of sub-matrix A, for this complicated update
    tempMatrix1 = np.dot(Reflectors[:,:columnEndRange].T, A[:,columnEndRange:])
    tempMatrix2 = np.dot(Connector[:rowEndRange,:columnEndRange].T, tempMatrix1)
    tempMatrix3 = np.dot(Reflectors[rowStartRange:rowEndRange,:columnEndRange], tempMatrix2)
    R[rowStartRange:rowEndRange,columnEndRange:] = A[rowStartRange:rowEndRange,columnEndRange:] + tempMatrix3
        
def TRQRCP_buildConnector(Connector,Reflectors,R,rowStartRange,rowEndRange,columnStartRange,columnEndRange,currIteration):
    # Separate this out if we are on first iteration of block
    if (currIteration > 0):
        tempMatrix1 = np.dot(Reflectors[:,columnStartRange:columnEndRange], Connector[rowStartRange:rowEndRange,columnStartRange:columnEndRange])
        tempMatrix2 = np.dot(Reflectors[:,:rowStartRange].T, tempMatrix1)
        Connector[:rowStartRange,columnStartRange:columnEndRange] = np.dot(Connector[:rowStartRange,:columnStartRange], tempMatrix2)

def TRQRCP_factorRowPanel(A,R,Reflectors,Connector,rowStartRange,rowEndRange,columnStartRange,columnEndRange,trueBlockSize):
    # Factor to find the upper-left R_11 upper-triangular matrix
    Reflectors[rowStartRange:,columnStartRange:columnEndRange],R_temp,Connector[rowStartRange:rowEndRange,columnStartRange:columnEndRange] = HouseHolderQR_BLAS3_version_1(A[rowStartRange:,columnStartRange:columnEndRange],trueBlockSize)
    R[rowStartRange:rowEndRange,columnStartRange:columnEndRange] = R_temp[:trueBlockSize,:]   # R_temp is A, don't forget

def TRQRCP_Downsample(currIteration,blockSize,trueBlockSize,sampleRank,rankEstimate,rowStartRange,rowEndRange,columnStartRange,columnEndRange,Sample,R,S_temp):
    # Time to update the Sample, subtracting out the components from the submatrix above
    # Don't update unless we have another iteration. Bad matrix alignment occurs otherwise
    if (currIteration+blockSize < rankEstimate):
        # Update shape of downsampled sample matrix
        Sample = Sample[:,trueBlockSize:]
        Sample[blockSize:,:] = S_temp[blockSize:sampleRank,blockSize:]
        tempMatrix1 = np.dot(la.inv(R[rowStartRange:rowEndRange,columnStartRange:columnEndRange]), R[rowStartRange:rowEndRange,columnEndRange:])
        tempMatrix2 = np.dot(S_temp[:blockSize,:blockSize], tempMatrix1)
        Sample[:blockSize,:] = S_temp[:blockSize,blockSize:] - tempMatrix2
    return Sample

def TRQRCP_version1(A,rankEstimate,blockSize,oversamplingParameter):
    numRows,numColumns,sampleRank,R,Reflectors,Perm,Omega,Sample,Connector = TRQRCP_setUp1(A,rankEstimate,blockSize,oversamplingParameter)

    # Iterate over blocks, resampling each time
    for i in range(0,rankEstimate,blockSize):
        trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange = TRQRCP_loopSetUp1(i,blockSize,numColumns,numRows,sampleRank)

        # Perform QR with column-pivoting solely to find the correct pivot order for the next b pivots
        # QRCP can only select blockSize pivot columns
        Reflectors_temp,S_temp,Perm_temp = HouseHolderQRCP_BLAS3_version_1(Sample, trueBlockSize, trueBlockSize,oversamplingParameter)
        
        # Permute the columns of A so that we can run blocked QR on the sampleRank-size panel of A
        # Note: We are also permuting the columns of R that lie in A above the current submatrix
        # Later optimization: I think this is doing too much work if Perm.size << numColumns
        TRQRCP_loopSwap1(Perm,A,R,Perm_temp,columnStartRange,numColumns,i)
        
        # Before factoring, we need to accumulate delayed updates into the panel
        TRQRCP_accumulateUpdatesIntoColumnPanel(A,Reflectors,Connector,rowStartRange,rowEndRange,columnStartRange,columnEndRange)
        
        # Factor to find the upper-left R_11 upper-triangular matrix
        TRQRCP_factorRowPanel(A,R,Reflectors,Connector,rowStartRange,rowEndRange,columnStartRange,columnEndRange,trueBlockSize)
        
        # We can use the data we have already to solve for submatrix R_12,
        #  which is all we need to be able to create an approximation to original matrix A.
        # Now we need to build up Connector
        TRQRCP_buildConnector(Connector,Reflectors,R,rowStartRange,rowEndRange,columnStartRange,columnEndRange,i)
        
        # Factor the row-panel of R with newly formed updates
        TRQRCP_accumulateUpdatesIntoRowPanel(A,Reflectors,R,Connector,rowStartRange,rowEndRange,columnStartRange,columnEndRange)
        
        # No more need to update the trailing matrix and continue on with next block iteration
        Sample = TRQRCP_Downsample(i,blockSize,trueBlockSize,sampleRank,rankEstimate,rowStartRange,rowEndRange,columnStartRange,columnEndRange,Sample,R,S_temp)
    return (Reflectors,R,Perm)

def TRQRCP_version2(A,blockSize,Tolerance,oversamplingParameter):
    numRows,numColumns,sampleRank,R,Reflectors,Perm,Omega,Sample,Connector = TRQRCP_setUp1(A,A.shape[1],blockSize,oversamplingParameter)
    rankEstimate = numColumns

    # Iterate over blocks, resampling each time
    for i in range(0,numColumns,blockSize):
        trueBlockSize,rowStartRange,columnStartRange,rowEndRange,columnEndRange = TRQRCP_loopSetUp1(i,blockSize,numColumns,numRows,sampleRank)

        # Perform QR with column-pivoting solely to find the correct pivot order for the next b pivots
        # QRCP can only select blockSize pivot columns
        Reflectors_temp,S_temp,Perm_temp,rankDetect = HouseHolderQRCP_BLAS3_version_4(Sample, trueBlockSize,Tolerance,trueBlockSize,oversamplingParameter)
        
        # Permute the columns of A so that we can run blocked QR on the sampleRank-size panel of A
        # Note: We are also permuting the columns of R that lie in A above the current submatrix
        # Later optimization: I think this is doing too much work if Perm.size << numColumns
        TRQRCP_loopSwap1(Perm,A,R,Perm_temp,columnStartRange,numColumns,i)
        
        # Before factoring, we need to accumulate delayed updates into the panel
        TRQRCP_accumulateUpdatesIntoColumnPanel(A,Reflectors,Connector,rowStartRange,rowEndRange,columnStartRange,columnEndRange)
        
        # Factor to find the upper-left R_11 upper-triangular matrix
        TRQRCP_factorRowPanel(A,R,Reflectors,Connector,rowStartRange,rowEndRange,columnStartRange,columnEndRange,trueBlockSize)
        
        # We can use the data we have already to solve for submatrix R_12,
        #  which is all we need to be able to create an approximation to original matrix A.
        # Now we need to build up Connector
        TRQRCP_buildConnector(Connector,Reflectors,R,rowStartRange,rowEndRange,columnStartRange,columnEndRange,i)
        
        # Factor the row-panel of R with newly formed updates
        TRQRCP_accumulateUpdatesIntoRowPanel(A,Reflectors,R,Connector,rowStartRange,rowEndRange,columnStartRange,columnEndRange)
                
        # Check on rank
        if (rankDetect < trueBlockSize):
            rankEstimate = i+rankDetect
            break
        
        # No more need to update the trailing matrix and continue on with next block iteration
        Sample = TRQRCP_Downsample(i,blockSize,trueBlockSize,sampleRank,rankEstimate,rowStartRange,rowEndRange,columnStartRange,columnEndRange,Sample,R,S_temp)
    return (Reflectors,R,Perm,rankEstimate)


"""
6. Truncated Randomized QR with column pivoting (with no trailing matrix update) external interface for user to play around
     with timings/performance and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
testRank = input("What rank matrix do you want to test with: ")
approxRank = input("Enter approximation rank: ")
oversamplingParameter = input("Enter oversampling parameter: ")

# Error checking
if (testRank < approxRank):
    sys.exit()

A = np.random.rand(numRows, numColumns)

# I need to change the singular values of A in order to vary the numerical rank
#   and properly test this method
U,D,V = la.svd(A,0)
for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0

tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
U1,D1,V1 = la.svd(A,0)

# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
A_temp = A.copy()
print "condition number of input matrix A - ", la.cond(A)

# Note: SSRQCP does not return a permutation matrix because that operation is already embedded into the algorithm
Reflectors,R,P = TRQRCP_version1(A, approxRank, blockSize, oversamplingParameter)

# For debugging, lets compare R
#print "Look at this element-wise comparison - ", np.abs(RCopy-R)

#print "Reflectors - ", Reflectors
#print "R - ", R
Q = convertReflectorsToOrthgonal_version_1(Reflectors)

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:approxRank].T,Q[:,:approxRank])-np.eye(approxRank),2)
print "Residual - ", la.norm(np.dot(Q[:,:approxRank],R)-np.dot(A_copy, np.eye(numColumns)[:,P]),2)

#print "check this difference between QR and A - ", np.abs(np.dot(Q[:,:approxRank],R)-np.dot(A_copy, np.eye(numColumns)[:,P]))

U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-approxRank):
    D2[numColumns-i-1] = 0

tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
print "Residual of truncated SVD - ", la.norm(A_trunc-A_copy,2)
"""


"""
Rank-detecting Truncated randomized QR with column pivoting external interface for user to play around with timings/performance
     and numerical aspects of the computed factorization
"""

"""
# Call the function
numRows = input("Enter number of rows: ")
numColumns = input("Enter number of columns: ")
blockSize = input("Enter block size: ")
testRank = input("What rank matrix do you want to test with: ")
oversamplingParameter = input("Enter oversampling parameter : ")

A = np.random.rand(numRows, numColumns)

# I need to change the singular values of A in order to vary the numerical rank
#   and properly test this method
U,D,V = la.svd(A,0)
for i in range(numColumns-testRank):
    D[numColumns-i-1] = 0

tempTrunc1 = np.dot(np.diag(D), V)
A = np.dot(U, tempTrunc1)
U1,D1,V1 = la.svd(A,0)

# Copy A because it is corrupted in function call, yet is needed for residual check
A_copy = A.copy()
A_sanity = A.copy()
print "condition number of input matrix A - ", la.cond(A)

# Note: SSRQCP does not return a permutation matrix because that operation is already embedded into the algorithm
Reflectors,R,P,rankEstimate = TRQRCP_version2(A, blockSize, 1e-13,oversamplingParameter)
print "Rank estimate - ", rankEstimate

#print "Reflectors - ", Reflectors
#print "R - ", R
Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:rankEstimate])

# Test deviation from orthogonality
# Test residual
print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:rankEstimate].T,Q[:,:rankEstimate])-np.eye(rankEstimate),2)
print "Residual - ", la.norm(np.dot(Q[:,:rankEstimate],R[:rankEstimate,:])-np.dot(A_copy, np.eye(numColumns)[:,P]),2)

U2,D2,V2 = la.svd(A_sanity,0)
for i in range(numColumns-testRank):
    D2[numColumns-i-1] = 0

tempTrunc2 = np.dot(np.diag(D2), V2)
A_trunc = np.dot(U2, tempTrunc2)
print "Residual of truncated SVD - ", la.norm(A_trunc-A_copy,2)
"""


# Testing playground

# Test ID inventory
"""
1 - Householder QR, BLAS level 1
2 - Householder QR, BLAS level 2
3 - Householder QR, BLAS level 3
4 - Householder QR with column pivoting, BLAS level 1
5 - Householder QR with column pivoting, BLAS level 2
6 - Householder QR with column pivoting, BLAS level 3
7 - Single-sampled randomized Householder QR with column pivoting
8 - Repeated-sampled randomized Householder QR with column pivoting
9 - Randomized Householder QR with column pivoting
10 - Truncated Randomized QR with column pivoting and no trailing matrix update
"""

moreTests = True
while (moreTests):
    
    # Lets create a playground for non-randomized QRCP first
    # Allow them to detect rank, which is important
    # Call the function
    
    testID = input("Enter the function ID you want to test: (Look at top of cell for directory)")
    if (testID == 1):
        print "You are testing Householder QR, BLAS level 1"
    elif (testID == 2):
        print "You are testing Householder QR, BLAS level 2"
    elif (testID == 3):
        print "You are testing Householder QR, BLAS level 3"
    elif (testID == 4):
        print "You are testing Householder QR with column pivoting, BLAS level 1"
    elif (testID == 5):
        print "You are testing Householder QR with column pivoting, BLAS level 2"
    elif (testID == 6):
        print "You are testing Householder QR with column pivoting, BLAS level 3"
    elif (testID == 7):
        print "You are testing Single-sampled randomized Householder QR with column pivoting"
    elif (testID == 8):
        print "You are testing Repeated-sampled randomized Householder QR with column pivoting"
    elif (testID == 9):
        print "You are testing Randomized Householder QR with column pivoting"
    elif (testID == 10):
        print "You are testing Truncated Randomized Householder QR with column pivoting and no trailing matrix update"
    numRows = input("Enter number of rows: ")
    numColumns = input("Enter number of columns: ")
    # Error check:
    if (numColumns > numRows):
        print """Sorry. Householder QR does not support matrices that are underdetermined. Good news is I am going to implement
                  the truncated SVD that directly follows from the algorithm: Truncated Randomized Householder QR with column pivoting and no trailing matrix update
                  , so I can update this and give you access to that algorithm in the coming weeks."""
        continue
    A = np.random.rand(numRows, numColumns)
    # I need to change the singular values of A in order to vary the numerical rank
    #   and properly test this method
    U,D,V = la.svd(A,0)
    
    if (testID == 1):
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        start1 = time.clock()
        Reflectors,R = HouseHolderQR_BLAS1_version_1(A)
        end1 = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors)
        # Test deviation from orthogonality
        # Test residual
        print "Computation took ", end1-start1, " seconds"
        print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
        print "Residual - ", la.norm(np.dot(Q,R)-A_copy,2)
    elif (testID == 2):
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        start2 = time.clock()
        Reflectors,R = HouseHolderQR_BLAS2_version_1(A)
        end2 = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors)
        # Test deviation from orthogonality
        # Test residual
        print "Computation took ", end2-start2, " seconds"
        print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
        print "Residual - ", la.norm(np.dot(Q,R)-A_copy,2)
    elif (testID == 3):
        blockSize = input("Enter block size: ")
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        start3 = time.clock()
        Reflectors,R,Connector = HouseHolderQR_BLAS3_version_1(A, blockSize)
        end3 = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors)
        # Test deviation from orthogonality
        # Test residual
        print "Computation took ", end3-start3, " seconds"
        print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
        print "Residual - ", la.norm(np.dot(Q,R)-A_copy,2)
    elif (testID == 4):
        testRank = input("What rank matrix do you want to test with: ")
        # Error check
        if (testRank > numColumns):
            print "Requested rank must be less than or equal to numColumns"
            continue
        rankDecision= input("Do you want to detect rank dynamically (1), or specify approximation rank explicitely (0)? : ")
        for i in range(numColumns-testRank):
            D[numColumns-i-1] = 0
        # Re-form A with rank testRank
        tempTrunc1 = np.dot(np.diag(D), V)
        A = np.dot(U, tempTrunc1)
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        # figure out how to deal  with detecting rank
        if (rankDecision == 0):
            approxRank = input("Enter approximation rank if specified above. Otherwise, type anything: ")
            # Error checking
            if ((rankDecision == 0) and (testRank < approxRank)):
                print "Make sure that the matrix rank is greater than or equal to your approximation rank"
                continue
            start41 = time.clock()
            Reflectors,R,P = HouseHolderQRCP_BLAS1_version_1(A, approxRank)
            end41 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors)[:,:approxRank]
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end41-start41, " seconds"
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q,np.dot(R,np.eye(numColumns)[:,P].T))-A_copy,2)
        else:
            Tolerance = input("Enter tolerance to which rank detection should be made: ")
            print "Your tolerance is ", Tolerance, ", shown to make sure you know that you specified properly."
            start44 = time.clock()
            Reflectors,R,P,computedRankEstimate = HouseHolderQRCP_BLAS1_version_4(A, Tolerance)
            end44 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end44-start44, " seconds"
            print "Algorithm estimates rank to be - ", computedRankEstimate
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)
        U2,D2,V2 = la.svd(A_sanity,0)
        for i in range(numColumns-testRank):
            D2[numColumns-i-1] = 0
        tempTrunc2 = np.dot(np.diag(D2), V2)
        A_trunc = np.dot(U2, tempTrunc2)
        print "Residual of truncated SVD of rank-",testRank," matrix - ", la.norm(A_trunc-A_copy,2)
    elif (testID == 5):
        testRank = input("What rank matrix do you want to test with: ")
        # Error check
        if (testRank > numColumns):
            print "Requested rank must be less than or equal to numColumns"
            continue
        rankDecision= input("Do you want to detect rank dynamically (1), or specify approximation rank explicitely (0)? : ")
        for i in range(numColumns-testRank):
            D[numColumns-i-1] = 0
        # Re-form A with rank testRank
        tempTrunc1 = np.dot(np.diag(D), V)
        A = np.dot(U, tempTrunc1)
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        # figure out how to deal  with detecting rank
        if (rankDecision == 0):
            approxRank = input("Enter approximation rank: ")
            # Error checking
            if ((rankDecision == 0) and (testRank < approxRank)):
                print "Make sure that the matrix rank is greater than or equal to your approximation rank"
                continue
            start51 = time.clock()
            Reflectors,R,P = HouseHolderQRCP_BLAS2_version_1(A, approxRank)
            end51 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors)[:,:approxRank]
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end51-start51, " seconds"
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "check matrix shapes - ", Q.shape, " ", R.shape, " ", np.dot(R,np.eye(numColumns)[:,P].T).shape
            print "Residual - ", la.norm(np.dot(Q,np.dot(R[:approxRank,:],np.eye(numColumns)[:,P].T))-A_copy,2)
        else:
            Tolerance = input("Enter tolerance to which rank detection should be made: ")
            print "Your tolerance is ", Tolerance, ", shown to make sure you know that you specified properly."
            start54 = time.clock()
            Reflectors,R,P,computedRankEstimate = HouseHolderQRCP_BLAS2_version_4(A, Tolerance)
            end54 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end54-start54, " seconds"
            print "Algorithm estimates rank to be - ", computedRankEstimate
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)
        U2,D2,V2 = la.svd(A_sanity,0)
        for i in range(numColumns-testRank):
            D2[numColumns-i-1] = 0
        tempTrunc2 = np.dot(np.diag(D2), V2)
        A_trunc = np.dot(U2, tempTrunc2)
        print "Residual of truncated SVD of rank-",testRank," matrix - ", la.norm(A_trunc-A_copy,2)
    elif (testID == 6):
        blockSize = input("Enter block size: ")
        # Error check:
        if (blockSize > numColumns):
            print "You cannot enter a blockSize that is greater than the number of columns. Try again"
            continue
        testRank = input("What rank matrix do you want to test with: ")
        # Error check
        if (testRank > numColumns):
            print "Requested rank must be less than or equal to numColumns"
            continue
        rankDecision= input("Do you want to detect rank dynamically (1), or specify approximation rank explicitely (0)? : ")
        for i in range(numColumns-testRank):
            D[numColumns-i-1] = 0
        # Re-form A with rank testRank
        tempTrunc1 = np.dot(np.diag(D), V)
        A = np.dot(U, tempTrunc1)
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        # figure out how to deal  with detecting rank
        if (rankDecision == 0):
            approxRank = input("Enter approximation rank if specified above. Otherwise, type anything: ")
            # Error checking
            if ((rankDecision == 0) and (testRank < approxRank)):
                print "Make sure that the matrix rank is greater than or equal to your approximation rank"
                continue
            start61 = time.clock()
            Reflectors,R,P = HouseHolderQRCP_BLAS3_version_1(A, blockSize, approxRank,0)
            end61 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors)[:,:approxRank]
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end61-start61, " seconds"
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:approxRank].T,Q[:,:approxRank])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q,np.dot(R[:approxRank,:],np.eye(numColumns)[:,P].T))-A_copy,2)
        else:
            Tolerance = input("Enter tolerance to which rank detection should be made: ")
            print "Your tolerance is ", Tolerance, ", shown to make sure you know that you specified properly."
            start64 = time.clock()
            Reflectors,R,P,computedRankEstimate = HouseHolderQRCP_BLAS3_version_4(A, blockSize, Tolerance, numColumns,0)
            end64 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end64-start64, " seconds"
            print "Algorithm estimates rank to be - ", computedRankEstimate
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)
        U2,D2,V2 = la.svd(A_sanity,0)
        for i in range(numColumns-testRank):
            D2[numColumns-i-1] = 0
        tempTrunc2 = np.dot(np.diag(D2), V2)
        A_trunc = np.dot(U2, tempTrunc2)
        print "Residual of truncated SVD of rank-",testRank," matrix - ", la.norm(A_trunc-A_copy,2)
    elif (testID == 7):
        blockSize = input("Enter block size: ")
        # Error check:
        if (blockSize > numColumns):
            print "You cannot enter a blockSize that is greater than the number of columns. Try again"
            continue
        testRank = input("What rank matrix do you want to test with: ")
        # Error check
        if (testRank > numColumns):
            print "Requested rank must be less than or equal to numColumns"
            continue
        approxRank= input("Specify approximation rank: ")
        oversamplingParameter = input("Specify oversampling parameter: ")
        # Error checking
        if (testRank < approxRank):
            print "Make sure that the matrix rank is greater than or equal to your approximation rank"
            continue
        for i in range(numColumns-testRank):
            D[numColumns-i-1] = 0
        # Re-form A with rank testRank
        tempTrunc1 = np.dot(np.diag(D), V)
        A = np.dot(U, tempTrunc1)
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        # figure out how to deal  with detecting rank
        start7 = time.clock()
        Reflectors,R,P = SSRQRCP(A, approxRank, blockSize,oversamplingParameter)
        end7 = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors)
        # Test deviation from orthogonality
        # Test residual
        print "Computation took ", end7-start7, " seconds"
        print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:approxRank].T,Q[:,:approxRank])-np.eye(approxRank),2)
        print "Residual - ", la.norm(np.dot(Q[:,:approxRank],R)-np.dot(A_copy, np.eye(numColumns)[:,P[:numColumns]]),2)
        U2,D2,V2 = la.svd(A_sanity,0)
        for i in range(numColumns-approxRank):
            D2[numColumns-i-1] = 0
        tempTrunc2 = np.dot(np.diag(D2), V2)
        A_trunc = np.dot(U2, tempTrunc2)
        print "Residual of truncated SVD of rank-",testRank," matrix - ", la.norm(A_trunc-A_copy,2)
    elif (testID == 8):
        blockSize = input("Enter block size: ")
        # Error check:
        if (blockSize > numColumns):
            print "You cannot enter a blockSize that is greater than the number of columns. Try again"
            continue
        testRank = input("What rank matrix do you want to test with: ")
        # Error check
        if (testRank > numColumns):
            print "Requested rank must be less than or equal to numColumns"
            continue
        rankDecision= input("Do you want to detect rank dynamically (1), or specify approximation rank explicitely (0)? : ")
        oversamplingParameter = input("Specify oversampling parameter: ")
        for i in range(numColumns-testRank):
            D[numColumns-i-1] = 0
        # Re-form A with rank testRank
        tempTrunc1 = np.dot(np.diag(D), V)
        A = np.dot(U, tempTrunc1)
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        # figure out how to deal  with detecting rank
        if (rankDecision == 0):
            approxRank = input("Enter approximation rank if specified above. Otherwise, type anything: ")
            # Error checking
            if ((rankDecision == 0) and (testRank < approxRank)):
                print "Make sure that the matrix rank is greater than or equal to your approximation rank"
                continue
            start81 = time.clock()
            Reflectors,R,P = RSRQRCP_version1(A, approxRank, blockSize,oversamplingParameter)
            end81 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors)
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end81-start81, " seconds"
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q[:,:approxRank],np.dot(R,np.eye(numColumns)[:,P].T))-A_copy,2)
        else:
            Tolerance = input("Enter tolerance to which rank detection should be made: ")
            print "Your tolerance is ", Tolerance, ", shown to make sure you know that you specified properly."
            start82 = time.clock()
            Reflectors,R,P,computedRankEstimate = RSRQRCP_version2(A, blockSize, Tolerance,oversamplingParameter)
            end82 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end82-start82, " seconds"
            print "Algorithm estimates rank to be - ", computedRankEstimate
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)
        U2,D2,V2 = la.svd(A_sanity,0)
        for i in range(numColumns-testRank):
            D2[numColumns-i-1] = 0
        tempTrunc2 = np.dot(np.diag(D2), V2)
        A_trunc = np.dot(U2, tempTrunc2)
        print "Residual of truncated SVD of rank-",testRank," matrix - ", la.norm(A_trunc-A_copy,2)
    elif (testID == 9):
        blockSize = input("Enter block size: ")
        # Error check:
        if (blockSize > numColumns):
            print "You cannot enter a blockSize that is greater than the number of columns. Try again"
            continue
        testRank = input("What rank matrix do you want to test with: ")
        # Error check
        if (testRank > numColumns):
            print "Requested rank must be less than or equal to numColumns"
            continue
        rankDecision= input("Do you want to detect rank dynamically (1), or specify approximation rank explicitely (0)? : ")
        oversamplingParameter = input("Specify oversampling parameter: ")
        for i in range(numColumns-testRank):
            D[numColumns-i-1] = 0
        # Re-form A with rank testRank
        tempTrunc1 = np.dot(np.diag(D), V)
        A = np.dot(U, tempTrunc1)
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        # figure out how to deal  with detecting rank
        if (rankDecision == 0):
            approxRank = input("Enter approximation rank if specified above. Otherwise, type anything: ")
            # Error checking
            if ((rankDecision == 0) and (testRank < approxRank)):
                print "Make sure that the matrix rank is greater than or equal to your approximation rank"
                continue
            start91 = time.clock()
            Reflectors,R,P = RQRCP_version1(A, approxRank, blockSize,oversamplingParameter)
            end91 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors)
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end91-start91, " seconds"
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q[:,:approxRank],np.dot(R,np.eye(numColumns)[:,P].T))-A_copy,2)
        else:
            Tolerance = input("Enter tolerance to which rank detection should be made: ")
            print "Your tolerance is ", Tolerance, ", shown to make sure you know that you specified properly."
            start92 = time.clock()
            Reflectors,R,P,computedRankEstimate = RQRCP_version2(A, blockSize, Tolerance,oversamplingParameter)
            end92 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end92-start92, " seconds"
            print "Algorithm estimates rank to be - ", computedRankEstimate
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)
        U2,D2,V2 = la.svd(A_sanity,0)
        for i in range(numColumns-testRank):
            D2[numColumns-i-1] = 0
        tempTrunc2 = np.dot(np.diag(D2), V2)
        A_trunc = np.dot(U2, tempTrunc2)
        print "Residual of truncated SVD of rank-",testRank," matrix - ", la.norm(A_trunc-A_copy,2)
    elif (testID == 10):
        blockSize = input("Enter block size: ")
        # Error check:
        if (blockSize > numColumns):
            print "You cannot enter a blockSize that is greater than the number of columns. Try again"
            continue
        testRank = input("What rank matrix do you want to test with: ")
        # Error check
        if (testRank > numColumns):
            print "Requested rank must be less than or equal to numColumns"
            continue
        rankDecision= input("Do you want to detect rank dynamically (1), or specify approximation rank explicitely (0)? : ")
        oversamplingParameter = input("Specify oversampling parameter: ")
        for i in range(numColumns-testRank):
            D[numColumns-i-1] = 0
        # Re-form A with rank testRank
        tempTrunc1 = np.dot(np.diag(D), V)
        A = np.dot(U, tempTrunc1)
        # Copy A because it is corrupted in function call, yet is needed for residual check
        A_copy = A.copy()
        A_sanity = A.copy()
        print "condition number of input matrix A - ", la.cond(A)
        # figure out how to deal  with detecting rank
        if (rankDecision == 0):
            approxRank = input("Enter approximation rank if specified above. Otherwise, type anything: ")
            # Error checking
            if ((rankDecision == 0) and (testRank < approxRank)):
                print "Make sure that the matrix rank is greater than or equal to your approximation rank"
                continue
            start10_1 = time.clock()
            Reflectors,R,P = TRQRCP_version1(A, approxRank, blockSize,oversamplingParameter)
            end10_1 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors)
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end10_1-start10_1, " seconds"
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q[:,:approxRank],np.dot(R,np.eye(numColumns)[:,P].T))-A_copy,2)
        else:
            Tolerance = input("Enter tolerance to which rank detection should be made: ")
            print "Your tolerance is ", Tolerance, ", shown to make sure you know that you specified properly."
            start10_2 = time.clock()
            Reflectors,R,P,computedRankEstimate = TRQRCP_version2(A, blockSize, Tolerance,oversamplingParameter)
            end10_2 = time.clock()
            Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])
            # Test deviation from orthogonality
            # Test residual
            print "Computation took ", end10_2-start10_2, " seconds"
            print "Algorithm estimates rank to be - ", computedRankEstimate
            print "Deviation from orthogonality of computed Q - ", la.norm(np.dot(Q[:,:numColumns].T,Q[:,:numColumns])-np.eye(numColumns),2)
            print "Residual - ", la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)
        U2,D2,V2 = la.svd(A_sanity,0)
        for i in range(numColumns-testRank):
            D2[numColumns-i-1] = 0
        tempTrunc2 = np.dot(np.diag(D2), V2)
        A_trunc = np.dot(U2, tempTrunc2)
        print "Residual of truncated SVD of rank-",testRank," matrix - ", la.norm(A_trunc-A_copy,2)
    else:
        print "You did not specify a valid test."
    
    moreTests = input("Are you done testing? (No - 1, Yes - 0)")
    print "\n\n\n"


"""
We can also test the 6 rank-detecting algorithms and the numpy SVD side-by-side to compare wallclock time,
  residual of the computed factorization, and detected rank
"""

moreTests = True
while (moreTests):
    
    # Lets create a playground for non-randomized QRCP first
    # Allow them to detect rank, which is important
    # Call the function
    
    numRows = input("Enter number of rows: ")
    numColumns = input("Enter number of columns: ")
    blockSize = input("Enter block size (not relevant for all algorithms): ")
    oversamplingParameter = input("Specify oversampling parameter (not relevant for all algorithms): ")
    # Error check:
    if (blockSize > numColumns):
        print "You cannot enter a blockSize that is greater than the number of columns. Try again"
        continue
    # Error check:
    if (numColumns > numRows):
        print """Sorry. Householder QR does not support matrices that are underdetermined. Good news is I am going to implement
                  the truncated SVD that directly follows from the algorithm: Truncated Randomized Householder QR with column pivoting and no trailing matrix update
                  , so I can update this and give you access to that algorithm in the coming weeks."""
        continue
    A = np.random.rand(numRows, numColumns)
    # I need to change the singular values of A in order to vary the numerical rank
    #   and properly test this method
    U,D,V = la.svd(A,0)
    # D will change, but U and V will not. So we need to save a copy of D
    D_save = D.copy()
    testRankStart = input("Enter the starting rank of the matrix do you want to test with: ")
    # Error check
    if (testRankStart > numColumns):
        print "Requested rank must be less than or equal to numColumns"
        continue
    testRankInterval = input("Enter the interval between ranks that you want to test with: ")
    Tolerance = input("Enter tolerance to which rank detection should be made: ")
    print "Your tolerance is ", Tolerance, ", shown to make sure you know that you specified properly."

    # Create lists to save results in order to create the 3 plots
    rankAxis = []
    qrcpBLAS1_Time = []
    qrcpBLAS1_Rank = []
    qrcpBLAS1_Residual = []
    qrcpBLAS2_Time = []
    qrcpBLAS2_Rank = []
    qrcpBLAS2_Residual = []
    qrcpBLAS3_Time = []
    qrcpBLAS3_Rank = []
    qrcpBLAS3_Residual = []
    ssrqrcp_Time = []
    ssrqrcp_Rank = []
    ssrqrcp_Residual = []
    rsrqrcp_Time = []
    rsrqrcp_Rank = []
    rsrqrcp_Residual = []
    rqrcp_Time = []
    rqrcp_Rank = []
    rqrcp_Residual = []
    trqrcp_Time = []
    trqrcp_Rank = []
    trqrcp_Residual = []
    svd_Time = []
    svd_Rank = []
    svd_Residual = []
    
    while (testRankStart <= numColumns):
        rankAxis.append(testRankStart)
        
        for i in range(testRankStart):
            D[i] = D_save[i]
        for i in range(numColumns-testRankStart):
            D[numColumns-i-1] = 0
        # Re-form A with rank testRank
        tempTrunc1 = np.dot(np.diag(D), V)
        A = np.dot(U, tempTrunc1)
        # Copy A because it is corrupted in function call, yet is needed for residual check 
        A_copy = A.copy()
        A_residual = A.copy()
        print "condition number of matrix A with rank", testRankStart, " is - ", la.cond(A)
        
        # SVD
        startTime = time.clock()
        U2,D2,V2 = la.svd(A,0)
        endTime = time.clock()
        for i in range(numColumns-testRankStart):
            D2[numColumns-i-1] = 0
        tempTrunc2 = np.dot(np.diag(D2), V2)
        A_trunc = np.dot(U2, tempTrunc2)
        svd_Residual.append(la.norm(A_trunc-A,2))
        svd_Rank.append(testRankStart)
        svd_Time.append(endTime-startTime)
        
        # QRCP Blas 1
        A = A_copy.copy()
        startTime = time.clock()
        Reflectors,R,P,computedRankEstimate = HouseHolderQRCP_BLAS1_version_4(A, Tolerance)
        endTime = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])

        qrcpBLAS1_Time.append(endTime-startTime)
        qrcpBLAS1_Rank.append(computedRankEstimate)
        qrcpBLAS1_Residual.append(la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2)) 
        
        # QRCP Blas 2
        A = A_copy.copy()
        startTime = time.clock()
        Reflectors,R,P,computedRankEstimate = HouseHolderQRCP_BLAS2_version_4(A, Tolerance)
        endTime = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])

        qrcpBLAS2_Time.append(endTime-startTime)
        qrcpBLAS2_Rank.append(computedRankEstimate)
        qrcpBLAS2_Residual.append(la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2))
        
        # QRCP Blas 3
        A = A_copy.copy()
        startTime = time.clock()
        Reflectors,R,P,computedRankEstimate = HouseHolderQRCP_BLAS3_version_4(A, blockSize, Tolerance, numColumns,0)
        endTime = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])

        qrcpBLAS3_Time.append(endTime-startTime)
        qrcpBLAS3_Rank.append(computedRankEstimate)
        qrcpBLAS3_Residual.append(la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2))
        
        # Repeated-sample Randomized QRCP
        A = A_copy.copy()
        startTime = time.clock()
        Reflectors,R,P,computedRankEstimate = RSRQRCP_version2(A, blockSize, Tolerance,oversamplingParameter)
        endTime = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])

        rsrqrcp_Time.append(endTime-startTime)
        rsrqrcp_Rank.append(computedRankEstimate)
        rsrqrcp_Residual.append(la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2))
        
        # Randomized QRCP
        A = A_copy.copy()
        startTime = time.clock()
        Reflectors,R,P,computedRankEstimate = RQRCP_version2(A, blockSize, Tolerance,oversamplingParameter)
        endTime = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])

        rqrcp_Time.append(endTime-startTime)
        rqrcp_Rank.append(computedRankEstimate)
        rqrcp_Residual.append(la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2))

        # Truncated Randomized QRCP
        A = A_copy.copy()
        startTime = time.clock()
        Reflectors,R,P,computedRankEstimate = TRQRCP_version2(A, blockSize, Tolerance,oversamplingParameter)
        endTime = time.clock()
        Q = convertReflectorsToOrthgonal_version_1(Reflectors[:,:computedRankEstimate])

        trqrcp_Time.append(endTime-startTime)
        trqrcp_Rank.append(computedRankEstimate)
        trqrcp_Residual.append(la.norm(np.dot(Q[:,:computedRankEstimate],np.dot(R[:computedRankEstimate,:],np.eye(numColumns)[:,P].T))-A_copy,2))
        
        testRankStart = testRankStart + testRankInterval

    # Generate plots
    fig1 = ppt.figure(1)
    sub1 = ppt.subplot(111)
    ppt.xlim([rankAxis[0]-10,rankAxis[-1]+10])
    ppt.title('Wall-clock time vs. Matrix rank')
    ppt.xlabel('Matrix rank')
    ppt.ylabel('Wall-clock time (sec)')
    ppt.plot(rankAxis, qrcpBLAS1_Time, 'o', label='qrcpBLAS1')
    ppt.plot(rankAxis, qrcpBLAS2_Time, 'o', label='qrcpBLAS2')
    ppt.plot(rankAxis, qrcpBLAS3_Time, 'o', label='qrcpBLAS3')
    ppt.plot(rankAxis, rsrqrcp_Time, 'o', label='rsrqrcp')
    ppt.plot(rankAxis, rqrcp_Time, 'o', label='rqrcp')
    ppt.plot(rankAxis, trqrcp_Time, 'o', label='trqrcp')
    ppt.plot(rankAxis, svd_Time, 'o', label='svd')
    chartBox = sub1.get_position()
    sub1.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    sub1.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    ppt.show()
    
    fig2 = ppt.figure(2)
    sub2 = ppt.subplot(111)
    ppt.xlim([rankAxis[0]-10,rankAxis[-1]+10])
    ppt.title('Rank estimate vs. Matrix rank')
    ppt.xlabel('Matrix rank')
    ppt.ylabel('Rank estimate')
    ppt.plot(rankAxis, qrcpBLAS1_Rank, 'o', label='qrcpBLAS1')
    ppt.plot(rankAxis, qrcpBLAS2_Rank, 'o', label='qrcpBLAS2')
    ppt.plot(rankAxis, qrcpBLAS3_Rank, 'o', label='qrcpBLAS3')
    ppt.plot(rankAxis, rsrqrcp_Rank, 'o', label='rsrqrcp')
    ppt.plot(rankAxis, rqrcp_Rank, 'o', label='rqrcp')
    ppt.plot(rankAxis, trqrcp_Rank, 'o', label='trqrcp')
    ppt.plot(rankAxis, svd_Rank, 'o', label='svd')
    chartBox = sub2.get_position()
    sub2.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    sub2.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    ppt.show()
    
    fig3 = ppt.figure(3)
    sub3 = ppt.subplot(111)
    ppt.xlim([rankAxis[0]-10,rankAxis[-1]+10])
    ppt.title('Computed residual vs. Matrix rank')
    ppt.xlabel('Matrix rank')
    ppt.ylabel('Computed residual')
    ppt.plot(rankAxis, qrcpBLAS1_Residual, 'o', label='qrcpBLAS1')
    ppt.plot(rankAxis, qrcpBLAS2_Residual, 'o', label='qrcpBLAS2')
    ppt.plot(rankAxis, qrcpBLAS3_Residual, 'o', label='qrcpBLAS3')
    ppt.plot(rankAxis, rsrqrcp_Residual, 'o', label='rsrqrcp')
    ppt.plot(rankAxis, rqrcp_Residual, 'o', label='rqrcp')
    ppt.plot(rankAxis, trqrcp_Residual, 'o', label='trqrcp')
    ppt.plot(rankAxis, svd_Residual, 'o', label='svd')
    chartBox = sub3.get_position()
    sub3.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    sub3.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    ppt.show()
    
    moreTests = input("Are you done testing? (No - 1, Yes - 0)")
    print "\n\n\n"

# End of code
