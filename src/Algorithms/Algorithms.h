/*
	Author: Edward Hutter
*/

#ifndef ALGORITHMS_H_
#define ALGORITHMS_H_

// Algorithmic design choices yield type parameters
class NoAcceleration;		// Used primarily for validation routines
class OffloadEachGemm;	// Offload each GEMM to a GPU via Cuda
class OffloadImmediate;	// Immediately offload matrix data to GPU and keep it resident there

// Local includes
#include "./../Util/shared.h"
#include "./../Timer/CTFtimer.h"
#include "./../BLAS/blasEngine.h"
#include "./../LAPACK/lapackEngine.h"
#include "./../Matrix/Matrix.h"
#include "./../Matrix/MatrixSerializer.h"
#include "./../Util/util.h"

#endif // ALGORITHMS_H_
