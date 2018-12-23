# Randomized Subspace iteration for dense matrices

import numpy as np
import numpy.linalg as la
import math as m

# Assumes that the rank 'k' is known apriori
def RandomizedSubspaceIteration(A,epsilon,k,SingularValuesToTest,Decisions):

    """
	Decisions[0] :	0 -> Starting iterate is non-orthogonal random matrix
		       	1 -> Starting iterate is orthogonal random matrix
		       	2 -> Starting iterate is A multiplied by random matrix
		       	3 -> Starting iterate is the orthogonalization of A multiplied by random matrix
	
	Decisions[1] :	0 -> Multiply by A^T*A to find the right singular vectors
			1 -> Multiply by A*A^T to find the left singular vectors

	Decisions[2] : 	0 -> Do not perform SVD to correct singular vector approximants
			1 -> Perform SVD to correct singular vector approximants

	Decisions[3] : 	0 -> Use the maximum amount of iterations as prescribed by tolerance epsilon
			!=0 -> Used this number of iterations
    """

    # Create lists to fill out
    FrobResids = []
    SpectralResids = []
    FrobSV = []
    SpectralSV = []
    NumMatVecs = []
    NumFlopsMM = []
    NumFlopsQR = []
    NumColumnsOrthogonalized = []
    for j in range(len(SingularValuesToTest)):
        SpectralSV.append([])
        FrobSV.append([])

    m = A.shape[0]
    n = A.shape[1]

    if (Decisions[3] != 0):
	q = Decisions[3]
    else:
	q = int(m.ceil(np.log(n)/epsilon))

    """
	Note: the only costs we don't count are:
		1) forming the random matrix
		2) orthogonalizing the random matrix
    """

    # Create starting iterate 'K' from a random matrix
    if (Decisions[0] == 0):
	RandMat = np.random.normal(size=(m,k))
	K = RandMat
	NumMatVecs[0].append(0)
	NumColumnsOrthogonalized.append(0)
	NumFlopsQR.append(0)
	NumFlopsMM.append(0)
    else if (Decisions[0] == 1):
	RandMat = np.random.normal(size=(m,k))
	RandMatOrth,RandmatR = la.qr(RandMat)
	K = RandMatOrth
	NumMatVecs[0].append(0)
	NumColumnsOrthogonalized.append(k)
	NumFlopsQR.append(2*m*k*k - 5./3*k**3)		# Householder estimate
	NumFlopsMM.append(0)
    else if (Decisions[0] == 2):
	RandMat = np.random.normal(size=(n,k))
	K = A.dot(RandMat)
	NumMatVecs.append(k)
	NumColumnsOrthogonalized.append(0)
	NumFlopsQR.append(0)
	NumFlopsMM.append(2*m*n*k)
    else if (Decisions[0] == 3):
	RandMat = np.random.normal(size=(n,k))
	Temp = A.dot(RandMat)
	RandMatOrth,RandmatR = la.qr(Temp)
	K = RandMatOrth
	NumMatVecs.append(k)
	NumColumnsOrthogonalized.append(k)
	NumFlopsQR.append(2*m*k*k - 5./3*k**3)          # Householder estimate
	NumFlopsMM.append(2*m*n*k)

    # 'G' is iterate matrix
    if (Decisions[1] == 0):
	G = A.T.dot(A)
    else if (Decisions[1] == 1):
	G = A.dot(A.T)

    for i in range(q):
	# When i==0, we are ready to test with 'K'
        if (i>0):
		K = G.dot(K)
        	NumMatVecs.append(K.shape[1] + NumMatVecs[len(NumMatVecs)-1])
		NumFlopsMM.append(2*G.shape[0]*G.shape[1]*K.shape[1])
        	K,R = la.qr(K)
		NumColumnsOrthogonalized.append(K.shape[1])
		NumFlopsQR.append(2*K.shape[0]*K.shape[1]*K.shape[1] - 5./3*K.shape[1]**3)          # Householder estimate

	if (Decisions[2] == 0):
		Z=K
	else if (Decisions[2] == 1):
		.. abort .. No reason to try this
		M = K.T.dot(A.dot(A.T.dot(K)))
		U_k,s,v = la.svd(M,full_matrices=False)
		Z = K.dot(U_k)

	if (Decisions[1] == 1):
		.. Is this right notion of a low-rank approximation when we have left singular vectors?
		LowRankApprox = Z.dot(Z.T.dot(A))    # used dense, not sparse version?????
	else if (Decisions[1] == 0):
		..
		LowRandApprox = A.dot(Z.T.dot(Z))

        # get low-rank approximation residual
        SpectralResids.append(la.norm(A-LowRankApprox,'fro')/la.norm(A,'fro'))
        FrobResids.append(la.norm(A-LowRankApprox,2)/la.norm(A,2))

        # Just to compare the singular values
        S = la.svd(LowRankApprox,full_matrices=False,compute_uv=False)
        for y in range(len(SingularValuesToTest)):
            LowRankApprox_SV = S[SingularValuesToTest[y]]
            A_SV = s_a[SingularValuesToTest[y]]
            SpectralSV[y].append(np.abs(A_SV-LowRankApprox_SV)/np.abs(A_SV))
            FrobSV[y].append(np.abs(A_SV-LowRankApprox_SV)/np.abs(A_SV))
            
    return FrobResids,SpectralResids,NumMatVecs,NumColumnsOrthogonalized,FrobSV,SpectralSV
