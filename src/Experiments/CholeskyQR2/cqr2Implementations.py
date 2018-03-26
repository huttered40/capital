#!/usr/bin/env python

'''
Implementation of different Cholesky-QR2 algorithms
'''

import numpy as np
import numpy.linalg as la
import sys

def CQR(A):
    m = A.shape[0]
    n = A.shape[1]
    B = np.dot(A.T,A)
    print "Condition number of computed B - ", la.cond(B)
    R = la.cholesky(B)
    # Note that B_1 is lower-triangular, so we need to transpose it
    Q = np.dot(A, la.inv(R).T)
    return (Q,R)

def CQR2(A):
    m = A.shape[0]
    n = A.shape[1]

    Q1,R1 = CQR(A)
    Q,R2 = CQR(Q1)
    R = np.dot(R2.T, R1.T)    
    return (Q,R)

def CQR2_Alt(A,b):
    m = A.shape[0]
    n = A.shape[1]
    
    Q_1 = np.zeros((b,m/b,n))
    R_1 = np.zeros((b,n,n))
    R_1_sum = np.zeros((n,n))
    for i in range(b):
        Q_1[i,:,:],R_1[i,:,:] = la.qr(A[(i*(m/b)):((i+1)*(m/b)),:])
        R_1_sum = R_1_sum + np.dot(R_1[i,:,:].T, R_1[i,:,:])
    B_1 = la.cholesky(R_1_sum)
    # Note that B_1 is lower-triangular, so we need to transpose it
    
    R_2 = np.zeros((b,n,n))
    R_2_sum = np.zeros((n,n))
    for i in range(b):
        R_2[i,:,:] = np.dot(R_1[i,:,:], la.inv(B_1).T)
        R_2_sum = R_2_sum + np.dot(R_2[i,:,:].T, R_2[i,:,:])
    C_1 = la.cholesky(R_2_sum)
    
    R = np.dot(C_1.T, B_1.T)
    Q = np.zeros((b,m/b,n))
    for i in range(b):
        Q[i,:,:] = np.dot(Q_1[i,:,:], R_1[i,:,:])
        Q[i,:,:] = np.dot(Q[i,:,:], la.inv(R))
    return (Q,R)

# 2D experimentation with CholeskyQR2 as panel factorization

#def qr2D(A,blockSize,numProcessors):
#    numRows = A.shape[0]
#    numColumns = A.shape[1]
#    Q = np.zeros((A.shape))
#    R = np.zeros((numColumns,numColumns))
#    for i in range(0,numColumns,blockSize):
#        trueBlockSize = min(blockSize,numColumns-i)
#        endColumn = min(i+blockSize,numColumns)
#        # panel factorization
#        Q[:,i:endColumn],R[i:endColumn,i:endColumn] = CQR2(A[:,i:endColumn],numProcessors)
#        # trailing matrix update and finish row-panel of R at same time
#        R[i:endColumn,endColumn:] = np.dot(Q[:,i:endColumn].T,A[:,endColumn:])
#        A[:,endColumn:] = A[:,endColumn:] - np.dot(Q[:,i:endColumn],R[i:endColumn,endColumn:])
#    return (Q,R)

# For debugging purposes, lets use BLAS-2 MGS2 for trailing matrix update
def qr2D(A,blockSize,numProcessors):
    numRows = A.shape[0]
    numColumns = A.shape[1]
    Q = np.zeros((A.shape))
    R = np.zeros((numColumns,numColumns))
    for i in range(0,numColumns,blockSize):
        trueBlockSize = min(blockSize,numColumns-i)
        endColumn = min(i+blockSize,numColumns)
        # panel factorization
        Q[:,i:endColumn],R[i:endColumn,i:endColumn] = CQR2(A[:,i:endColumn],numProcessors)
        # trailing matrix update and finish row-panel of R at same time
        # This is wasteful to update both R and A, but this is just for debugging
        
        # lets use blocked updates, not BLAS-2 updates
        futureEndColumn = min(endColumn+blockSize,numColumns)
        R[:endColumn,endColumn:futureEndColumn] = np.dot(Q[:,:endColumn].T,A[:,endColumn:futureEndColumn])
        A[:,endColumn:futureEndColumn] = A[:,endColumn:futureEndColumn] - np.dot(Q[:,:endColumn],R[:endColumn,endColumn:futureEndColumn])
        # Again
        R[:endColumn,endColumn:futureEndColumn] = np.dot(Q[:,:endColumn].T,A[:,endColumn:futureEndColumn])
        A[:,endColumn:futureEndColumn] = A[:,endColumn:futureEndColumn] - np.dot(Q[:,:endColumn],R[:endColumn,endColumn:futureEndColumn])
  
        # Lets use MGS2 BLAS-2 level trailing matrix update
#        for j in range(endColumn,min(endColumn+blockSize,numColumns)):
            # Maybe we don't need to update R, we can just get R after we find Q via R = Q^T*A
                # NO! A will be corrupted if we do that, unless we copy A, which is really just a waste
            # R comes directly from Q_prev^* * a_next (those inner-products)
            #R[i:endColumn,endColumn:] = np.dot(Q[:,i:endColumn].T,A[:,endColumn:])
            #A[:,endColumn:] = A[:,endColumn:] - np.dot(Q[:,i:endColumn],R[i:endColumn,endColumn:])
#            R[:endColumn,j] = np.dot(Q[:,:endColumn].T,A[:,j])
#            A[:,j] = A[:,j] - np.dot(Q[:,:endColumn],R[:endColumn,j])
            
            # Do it again to model MGS2
#            R[:endColumn,j] = np.dot(Q[:,:endColumn].T,A[:,j])
#            A[:,j] = A[:,j] - np.dot(Q[:,:endColumn],R[:endColumn,j])
    return (Q,R)


"""
Run numerical tests on alternate CholeskyQR(2), CholeskyQR(2), and la.QR
Arguments: 1 -> CholeskyQR
           2 -> CholeskyQR2
           3 -> Alternate CholeskyQR
           4 -> Alternate CholeskyQR2
           5 -> la.QR
           6 -> 2D QR with CholeskyQR2 panel factorization
"""

def printResults(A, orthogonalityCheckMatrix1, orthogonalityCheckMatrix2, residualCheckMatrix, numColumns, numRows):
    print "Deviation from orthogonality (Q.T*Q-I) - ",la.norm(orthogonalityCheckMatrix1 - np.eye(numColumns),2) / np.sqrt(numColumns)
    print "Deviation from orthogonality (Q*Q.T-I)- ",la.norm(orthogonalityCheckMatrix2 - np.eye(numRows),2) / np.sqrt(numColumns)
    print "Residual - ",la.norm(residualCheckMatrix - A, 2) / la.norm(A,2)

def testQRalg():
    arg = input("Enter method to test: CQR[1], CQR2[2], Alternate CQR2[3]: ")
    numRows = input("Enter number of rows: ")
    numColumns = input("Enter number of columns: ")
    blockSize = input("Enter block size: ")
    conditionNumberStart = input("Enter condition number start: ")
    conditionNumberEnd = input("Enter condition number end: ")
    IntervalSize = input("Enter interval size: ")

    A = np.asarray(np.random.rand(numRows, numColumns), dtype=np.float64)

    # Now lets modify the condition number by expanding out its SVD and modifying the singular values
    U,D,V = la.svd(A,0)

    # Try tests for all different condition numbers in the range
    interval = 2**(np.arange(conditionNumberStart, conditionNumberEnd, IntervalSize))

    for z in interval:
        # Change the singular values, which will change the condition number in the Euclidean norm
        D = np.linspace(1,z,D.shape[0])
        D = D[::-1]
        A = np.dot(U,np.diag(D))
        A = np.dot(A,V.T)

        print "Condition number of current test is ", la.cond(A)  # why not use cond?
#        Z = np.dot(A,la.inv(A))
#        print("Condition number of A, A_inverse, and Z - ", la.cond(A), la.cond(la.inv(A)), la.cond(Z))
#        print("Error in I-AA^-1", la.norm(np.eye(A.shape[0])-np.dot(A,la.inv(A)),2))
    
        if (arg == 1):
            Q,R = CQR(A)
            orthogonalityCheck1 = np.dot(Q.T, Q)
            orthogonalityCheck2 = np.dot(Q, Q.T)
            residualCheck = np.dot(Q, R)
            print "Condition number of computed Q - ", la.cond(Q)
            print "Condition number of computed R - ", la.cond(R)
            print "Condition number of computed R_inv - ", la.cond(la.inv(R))
            U1,D1,V1 = la.svd(Q,1)
            print "Singular values of Q - ", D1
            printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows)
        elif (arg == 2):
            Q,R = CQR2(A)
            orthogonalityCheck1 = np.dot(Q.T, Q)
            orthogonalityCheck2 = np.dot(Q, Q.T)
            residualCheck = np.dot(Q, R)
            print "Condition number of computed Q - ", la.cond(Q)
            U2,D2,V2 = la.svd(Q,1)
            print "Singular values of Q - ", D2
            print "Condition number of computed R - ", la.cond(R)
            print "Condition number of computed R_inv - ", la.cond(la.inv(R))
            printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows)
        elif (arg == 3):
            Q_i,R = CQR2_Alt(A, blockSize)
            orthogonalityCheck1 = np.zeros((numColumns,numColumns))
            orthogonalityCheck2 = np.zeros((numRows,numRows))
            residualCheck = np.zeros((numRows, numColumns))

            for i in range(blockSize):
                orthogonalityCheck1 += np.dot(Q_i[i,:,:].T, Q_i[i,:,:])
#               orthogonalityCheck2 += np.dot(Q_i[i,:,:], Q_i[i,:,:].T)
                residualCheck[i*(numRows/blockSize) : (i+1)*(numRows/blockSize),:] = np.dot(Q_i[i,:,:], R)
            printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows)
        elif (arg == 5):
            Q,R = la.qr(A)
            print "Condition number of computed Q - ", la.cond(Q)
            U5,D5,V5 = la.svd(Q,1)
            print "Singular values - ", D5
            print "Condition number of computed R - ", la.cond(R)
            print "Condition number of computed R_inv - ", la.cond(la.inv(R))
            orthogonalityCheck1 = np.dot(Q.T, Q)
            orthogonalityCheck2 = np.dot(Q, Q.T)
            residualCheck = np.dot(Q, R)
            printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows)
        elif (arg == 6):
            Q,R = qr2D(A,blockSize,numProcessors)
            orthogonalityCheck1 = np.dot(Q.T, Q)
            orthogonalityCheck2 = np.dot(Q, Q.T)
            residualCheck = np.dot(Q, R)
            print "Condition number of computed Q - ", la.cond(Q)
            U2,D2,V2 = la.svd(Q,1)
            print "Singular values - ", D
            print "Condition number of computed R - ", la.cond(R)
            print "Condition number of computed R_inv - ", la.cond(la.inv(R))
            printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows)
            
        print "\n"



testQRalg()
