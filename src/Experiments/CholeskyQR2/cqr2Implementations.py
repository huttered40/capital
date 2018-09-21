#!/usr/bin/env python

'''
Implementation of different Cholesky-QR2 algorithms
'''

import numpy as np
import numpy.linalg as la
import sys
import matplotlib
import matplotlib.pyplot as plt

def CQR(A):
    print "Tell me condition number of A : ", la.cond(A)
    m = A.shape[0]
    n = A.shape[1]
    B = np.dot(A.T,A)
    print "B - ",B
    print "Condition number of computed B - ", la.cond(B)
    try:
        R = la.cholesky(B)
    except np.linalg.LinAlgError as err:
        return (0,0,0)
    R = la.cholesky(B)
    print "R - ",R
    # Note that B_1 is lower-triangular, so we need to transpose it
    Q = np.dot(A, la.inv(R).T)
    return (Q,R,1)

def ShiftedCQR(A):
    print "Tell me condition number of A : ", la.cond(A)
    m = A.shape[0]
    n = A.shape[1]
    B = np.dot(A.T,A)
    eps = 10e-16
    tr = np.trace(B)
    shiftNumerator = (n+2)*eps*tr
    shiftDenominator = 1 - (n+1)*(n+3)*eps
    shift = shiftNumerator*1. / (shiftDenominator*1.)
    print "Shift - ", shift
    print "Condition number of computed B - ", la.cond(B)
    print "Condition number of shifted B - ", la.cond(B + shift*np.eye(n))
    U1,D1,V1 = la.svd(B,1)
    U2,D2,V2 = la.svd(B + shift*np.eye(n),1)
    print "Singular values of B before shift - ", D1
    print "Singular values of B after shift - ", D2
    try:
        R = la.cholesky(B + shift*np.eye(n))
    except np.linalg.LinAlgError as err:
        return (0,0,0)
    R = la.cholesky(B + shift*np.eye(n))
    # Note that B_1 is lower-triangular, so we need to transpose it
    Q = np.dot(A, la.inv(R).T)
    return (Q,R,1)

def CQR2(A):
    m = A.shape[0]
    n = A.shape[1]

    Q1,R1,flag = CQR(A)
    if flag == 0:
        return (Q1,R1,flag)
    Q,R2,flag = CQR(Q1)
    if flag == 0:
        return (Q,R2,flag)
    R = np.dot(R2.T, R1.T)    
    return (Q,R,1)

def ShiftedCQR3(A):
    m = A.shape[0]
    n = A.shape[1]

    Q1,R1,flag = ShiftedCQR(A)
    if (flag == 0):
        return (Q1,R1,flag)
    Q2,R2,flag = ShiftedCQR(Q1)
    if (flag == 0):
        return (Q2,R2,flag)
    Q,R3,flag = ShiftedCQR(Q2)
    if (flag == 0):
        return (Q,R3,flag)
    R = np.dot(R3.T,np.dot(R2.T, R1.T))
    return (Q,R,1)

# This routine is in development. I need to figure out a way to extract householder vectors from the computed Q factor from CholeskyQR2
def CQR2panel(A,blockSize):
    m = A.shape[0]
    n = A.shape[1]
    Q = np.zeros((m,n))
    R = np.zeros((n,n))

    if (n%blockSize != 0):
        exit()
    for i in range(0,n,blockSize):
        Q_panel,R_panel,flag = CQR2(A[i:,i:i+blockSize])
        if (flag == 0):
          return (Q,R,flag)
        Q[:,i:i+blockSize] = Q_panel
        R[i:i+blockSize,i:i+blockSize] = R_panel

        # Now update trailing matrix with Q_panel. This is a step I am unsure of
        A[:,(i+blockSize):] = A[:,(i+blockSize):] - np.dot(Q_panel,np.dot(Q_panel.T,A[:,(i+blockSize):]))
        R[i:(i+blockSize),(i+blockSize)] = A[i:(i+blockSize),(i+blockSize)]

    return (Q,R,1)

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

#def qr2D(A,blockSize,numPro):
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

"""
Note: As of May 2018, this shit is wrong!
Run numerical tests on alternate CholeskyQR(2), CholeskyQR(2), and la.QR
Arguments: 1 -> CholeskyQR
           2 -> CholeskyQR2
           3 -> Alternate CholeskyQR
           4 -> Alternate CholeskyQR2
           5 -> la.QR
           6 -> 2D QR with CholeskyQR2 panel factorization
"""

def printResults(A, orthogonalityCheckMatrix1, orthogonalityCheckMatrix2, residualCheckMatrix, numColumns, numRows, devOrthList, residualList):
    devOrth = la.norm(orthogonalityCheckMatrix1 - np.eye(numColumns),2) / np.sqrt(numColumns)
    residual = la.norm(residualCheckMatrix - A, 2) / la.norm(A,2)
    print "Deviation from orthogonality (Q.T*Q-I) - ", devOrth
    print "Residual - ", residual
    devOrthList.append(devOrth)
    residualList.append(residual)
    print "new residual value - ", len(residualList), residualList[-1]
    if (len(residualList) == 240):
        print residualList
    return (devOrthList,residualList)

def testQRalg(conditionNumberStart,conditionNumberEnd,IntervalSize):
    #arg = input("Enter method to test: CQR[1], CQR2[2], Alternate CQR2[3], Shifted CQR3[4], Householder QR[6]: ")
    numRows = input("Enter number of rows: ")
    numColumns = input("Enter number of columns: ")
    blockSize = input("Enter block size for panel-based CQR2: ")

    condList=[]
    residualList=[]
    devOrthList=[]
    # Iterate over Householder QR, CholeskyQR, CholeskyQR2, and Shifted CholeskyQR3
    for i in range(1,5):
        A = np.asarray(np.random.rand(numRows, numColumns), dtype=np.float64)

        # Now lets modify the condition number by expanding out its SVD and modifying the singular values
        U,D,V = la.svd(A,0)

        # Try tests for all different condition numbers in the range
        interval = 2**(np.arange(conditionNumberStart, conditionNumberEnd, IntervalSize))
        jump = interval.shape[0]
        for z in interval:
            # Change the singular values, which will change the condition number in the Euclidean norm
            D = np.linspace(1,z,D.shape[0])
            D = D[::-1]
            A = np.dot(U,np.diag(D))
            A = np.dot(A,V.T)

            print "Condition number of current test is ", la.cond(A)  # why not use cond?
            condList.append(la.cond(A))
    
            if (i == 2):
                Q,R,flag = CQR(A)
                if flag == 1:
                    orthogonalityCheck1 = np.dot(Q.T, Q)
                    orthogonalityCheck2 = np.dot(Q, Q.T)
                    residualCheck = np.dot(Q, R)
                    print "Condition number of computed Q - ", la.cond(Q)
                    print "Condition number of computed R - ", la.cond(R)
                    print "Condition number of computed R_inv - ", la.cond(la.inv(R))
                    U1,D1,V1 = la.svd(Q,1)
                    print "Singular values of Q - ", D1
                    devOrthList,residualList = printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows, devOrthList, residualList)
                else:
                    devOrthList.append(1)
                    residualList.append(1)
            elif (i == 3):
                Q,R,flag = CQR2(A)
                if flag == 1:
                    orthogonalityCheck1 = np.dot(Q.T, Q)
                    orthogonalityCheck2 = np.dot(Q, Q.T)
                    residualCheck = np.dot(Q, R)
                    print "Condition number of computed Q - ", la.cond(Q)
                    U2,D2,V2 = la.svd(Q,1)
                    print "Singular values of Q - ", D2
                    print "Condition number of computed R - ", la.cond(R)
                    print "Condition number of computed R_inv - ", la.cond(la.inv(R))
                    devOrthList,residualList = printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows, devOrthList, residualList)
                else:
                    devOrthList.append(1)
                    residualList.append(1)
            elif (i == 5):
                Q,R,flag = CQR2panel(A,blockSize)
                if flag == 1:
                    orthogonalityCheck1 = np.dot(Q.T, Q)
                    orthogonalityCheck2 = np.dot(Q, Q.T)
                    residualCheck = np.dot(Q, R)
                    print "Condition number of computed Q - ", la.cond(Q)
                    U2,D2,V2 = la.svd(Q,1)
                    print "Singular values of Q - ", D2
                    print "Condition number of computed R - ", la.cond(R)
                    print "Condition number of computed R_inv - ", la.cond(la.inv(R))
                    devOrthList,residualList = printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows, devOrthList, residualList)
                else:
                    devOrthList.append(1)
                    residualList.append(1)
            #elif (i == 5):
                #Q_i,R,pass = CQR2_Alt(A, blockSize)
                #orthogonalityCheck1 = np.zeros((numColumns,numColumns))
                #orthogonalityCheck2 = np.zeros((numRows,numRows))
                #residualCheck = np.zeros((numRows, numColumns))

                #for i in range(blockSize):
                #    orthogonalityCheck1 += np.dot(Q_i[i,:,:].T, Q_i[i,:,:])
#               #    orthogonalityCheck2 += np.dot(Q_i[i,:,:], Q_i[i,:,:].T)
                #    residualCheck[i*(numRows/blockSize) : (i+1)*(numRows/blockSize),:] = np.dot(Q_i[i,:,:], R)
                #printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows)
            elif (i == 4):
                Q,R,flag = ShiftedCQR3(A)
                if flag == 1:
                    orthogonalityCheck1 = np.dot(Q.T, Q)
                    orthogonalityCheck2 = np.dot(Q, Q.T)
                    residualCheck = np.dot(Q, R)
                    print "Condition number of computed Q - ", la.cond(Q)
                    U2,D2,V2 = la.svd(Q,1)
                    print "Singular values of Q - ", D2
                    print "Condition number of computed R - ", la.cond(R)
                    print "Condition number of computed R_inv - ", la.cond(la.inv(R))
                    devOrthList,residualList = printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows, devOrthList, residualList)
                else:
                    devOrthList.append(1)
                    residualList.append(1)
            elif (i == 1):
                Q,R = la.qr(A)
                print "Condition number of computed Q - ", la.cond(Q)
                U5,D5,V5 = la.svd(Q,1)
                print "Singular values - ", D5
                print "Condition number of computed R - ", la.cond(R)
                print "Condition number of computed R_inv - ", la.cond(la.inv(R))
                orthogonalityCheck1 = np.dot(Q.T, Q)
                orthogonalityCheck2 = np.dot(Q, Q.T)
                residualCheck = np.dot(Q, R)
                printResults(A, orthogonalityCheck1, orthogonalityCheck2, residualCheck, numColumns, numRows, devOrthList, residualList)
            print "\n"
    return (residualList,devOrthList,condList,jump)

conditionNumberStart = input("Enter condition number start: ")
conditionNumberEnd = input("Enter condition number end: ")
IntervalSize = input("Enter interval size: ")
residual,devOrth,condList,jump = testQRalg(conditionNumberStart,conditionNumberEnd,IntervalSize)

fig,ax = plt.subplots()
ax.loglog(condList[0:jump],residual[0:jump], label='HQR')
ax.loglog(condList[jump:2*jump],residual[jump:2*jump], label='CQR')
ax.loglog(condList[2*jump:3*jump],residual[2*jump:3*jump], label='CQR2')
ax.loglog(condList[3*jump:4*jump],residual[3*jump:4*jump], label='Shifted CQR3')
ax.loglog(condList[4*jump:5*jump],residual[4*jump:5*jump], label='Panel-based CQR2')
ax.legend(loc='upper left')
plt.title('Residual, 100 x 10 matrix')
plt.xlabel('Condition number')
plt.ylabel('Residual')
fig.savefig("CholeskyQRresidual")
plt.show()

fig1,ax1 = plt.subplots()
ax1.loglog(condList[0:jump],devOrth[0:jump], label='HQR')
ax1.loglog(condList[jump:2*jump],devOrth[jump:2*jump], label='CQR')
ax1.loglog(condList[2*jump:3*jump],devOrth[2*jump:3*jump], label='CQR2')
ax1.loglog(condList[3*jump:4*jump],devOrth[3*jump:4*jump], label='Shifted CQR3')
ax1.loglog(condList[4*jump:5*jump],devOrth[4*jump:5*jump], label='Panel-based CQR2')
ax1.legend(loc='upper left')
plt.title('Deviation from orthogonality, 100 x 10 matrix')
plt.xlabel('Condition number')
plt.ylabel('Deviation from orthogonality')
fig1.savefig("CholeskyQRdevOrth")
plt.show()
