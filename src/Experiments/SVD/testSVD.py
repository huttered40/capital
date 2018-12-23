import numpy as np
import numpy.linalg as la
import math as m
import matplotlib.pyplot as pt

import sys
sys.path.append('./SVDmethods/')

from RSI import RandomizedSubspaceIteration
from RBL import RandomizedKrylovMethod

isDense = input("Low rank dense (1) or low rank sparse (0)")

if (isDense):
	# Create matrix with exponentially decaying singular values
	numRows = input("Enter number of rows: ")
	numColumns = input("Enter number of columns: ")
	A = np.asarray(np.random.rand(numRows, numColumns),dtype=np.float64)

	# Now lets modify the condition number by expanding out its SVD and modifying the singular values
	U,D,V = la.svd(A,full_matrices=0)

	# Try tests for all different condition numbers in the range
	D = np.logspace(1,50,base=2.,num=numColumns)
	D = D[::-1]
	#D = np.exp(np.arange(numColumns,0,-1))
	#for i in range(numColumns):
	#    D[i] = (numColumns-i-1)**(numColumns-i-1)

	A = U.dot(np.diag(D).dot(V.T))
else:
	A = scio.mmread("fs_760_2/fs_760_2")
	print "Matrix shape - ", A.shape
	numRows = A.shape[0]
	numColumns = A.shape[1]

u,s,v = la.svd(A.todense())
pt.semilogy(s, label='singular values')
pt.legend()
pt.show()




epsilon = input("Enter epsilon threshhold: ")
Ranks = []
while (1):
    rank = input("Enter rank approximation to test (0 to end): ")
    if (rank == 0):
        break
    Ranks.append(rank)
    
SingularValuesToTest = []
while (1):
    sv = input("Enter singular value to test (-1 to end): ")
    if (sv == -1):
        break
    SingularValuesToTest.append(sv)

BlockSizes = []
while (1):
    bs = input("Enter block size (for Krylov) to test (0 to end): ")
    test = 1
    if (bs == 0):
        break
    if ((numColumns % bs) != 0):
        print "Bad block size. Try again"
        continue
    for i in Ranks:
        if ((i % bs) != 0):
            print "Bad block size. Try again"
            test = -1
            break
    if (test == -1):
        continue
            
    BlockSizes.append(bs)

for i in Ranks:
    # Lists for Randomized Subspace iteration
    FrobResidsListRSI = []
    SpectralResidsListRSI = []
    NumMatVecsListRSI = []
    FrobSVListRSI = []
    SpectralSVListRSI = []

    # Lists for Randomized block-lanczos method
    FrobResidsListRBL = []
    SpectralResidsListRBL = []
    NumMatVecsListRBLRes = []      # One for finding low-rank approximation residuals
    NumMatVecsListRBLSV = []      # One for finding singular values
    FrobSVListRBL = []
    SpectralSVListRBL = []
    
    for j in BlockSizes:

        frobResidualsRSI,SpectralResidualsRSI,NumMatVecsRSI,FrobSV,SpectralSV = RandomizedSubspaceIteration(A,epsilon,i,SingularValuesToTest)
        # Add to lists
        FrobResidsListRSI.append(frobResidualsRSI)
        SpectralResidsListRSI.append(SpectralResidualsRSI)
        NumMatVecsListRSI.append(NumMatVecsRSI)
        FrobSVListRSI.append(FrobSV)
        SpectralSVListRSI.append(SpectralSV)

        frobResidualsRBL,SpectralResidualsRBL,NumMatVecsRBLRes,blah1,blah2 = RandomizedKrylovMethod(A,epsilon,i,j,SingularValuesToTest,i)
        blah1,blah2,NumMatVecsRBLSV,FrobSV,SpectralSV = RandomizedKrylovMethod(A,epsilon,i,j,SingularValuesToTest,A.shape[1])
        # Add to lists
        FrobResidsListRBL.append(frobResidualsRBL)
        SpectralResidsListRBL.append(SpectralResidualsRBL)
        NumMatVecsListRBLRes.append(NumMatVecsRBLRes)
        NumMatVecsListRBLSV.append(NumMatVecsRBLSV)
        FrobSVListRBL.append(FrobSV)
        SpectralSVListRBL.append(SpectralSV)
    
    # Plot the low-rank approximation residual norm convergence
    print len(NumMatVecsListRSI[0]),len(SpectralResidsListRSI[0])
    print "check RSI - ",NumMatVecsListRSI[0],SpectralResidsListRSI[0]
    pt.semilogy(NumMatVecsListRSI[0],SpectralResidsListRSI[0], label='RSI')
    pt.legend()

    y=0
    for z in BlockSizes:
        #print len(NumMatVecsListRBL[i]),len(SpectralResidsListRBL[i])
        print "check RBL - ",NumMatVecsListRBLRes[y],SpectralResidsListRBL[y]
        pt.semilogy(NumMatVecsListRBLRes[y],SpectralResidsListRBL[y], label='RBL block size %d'%(z))
        pt.legend()
        y = y+1
    
    # show the spectral norm results
    pt.xlabel("Number of matvecs")
    pt.ylabel("Spectral norm residual")
    pt.title("Quality of Low-rank (%d) approximation vs. number of Mat vecs"%(i))
    pt.show()
    
    # Plot the low-rank approximation Frobenius norm convergence
    pt.semilogy(NumMatVecsListRSI[0],FrobResidsListRSI[0], label='RSI')
    pt.legend()

    y=0
    for z in BlockSizes:
        pt.semilogy(NumMatVecsListRBLRes[y],FrobResidsListRBL[y], label='RBL block size %d'%(z))
        pt.legend()
        y = y+1
    
    # show the spectral norm results
    pt.xlabel("Number of matvecs")
    pt.ylabel("Frobenius norm residual")
    pt.title("Quality of Low-rank (%d) approximation vs. number of Mat vecs"%(i))
    pt.show()
    
    # Plot the 1st singular value convergence via Spectral norm
    k = 0
    for j in SingularValuesToTest:

        pt.semilogy(NumMatVecsListRSI[0],SpectralSVListRSI[0][k], label='RSI')
        pt.legend()

        y=0
        for z in BlockSizes:
            pt.semilogy(NumMatVecsListRBLSV[y],SpectralSVListRBL[y][k], label='RBL block size %d'%(z))
            pt.legend()
            y = y+1
    
        # show the spectral norm results
        pt.xlabel("Number of matvecs")
        pt.ylabel("Spectral norm residual")
        pt.title("Quality of %d'th singular value (rank %d approximation) vs. number of Mat vecs"%(j,i))
        pt.show()
    
        k = k+1
    
    # Plot the 1st singular value convergence via Spectral norm
    k = 0
    for j in SingularValuesToTest:

        pt.semilogy(NumMatVecsListRSI[0],FrobSVListRSI[0][k], label='RSI')
        pt.legend()

        y=0
        for z in BlockSizes:
            pt.semilogy(NumMatVecsListRBLSV[y],FrobSVListRBL[y][k], label='RBL block size %d'%(z))
            pt.legend()
            y = y+1
        
        # show the spectral norm results
        pt.xlabel("Number of matvecs")
        pt.ylabel("Frobenius norm residual")
        pt.title("Quality of %d'th singular value (rank %d approximation) vs. number of Mat vecs"%(j,i))
        pt.show()
    
        k = k+1
