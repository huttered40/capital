/*
	Author: Edward Hutter
*/

#ifndef ALG_H_
#define ALG_H_

// Algorithmic design choices yield type parameters
class NoAcceleration;		// Used primarily for validation routines
class OffloadEachGemm;	// Offload each GEMM to a GPU via Cuda
class OffloadImmediate;	// Immediately offload matrix data to GPU and keep it resident there

// Local includes
#include "./../util/shared.h"
#include "./../timer/CTFtimer.h"
#include "./../blas/blasEngine.h"
#include "./../lapack/lapackEngine.h"
#include "./../matrix/Matrix.h"
#include "./../matrix/MatrixSerializer.h"
#include "./../alg/topology.h"
#include "./../util/util.h"

#endif // ALGORITHMS_H_
