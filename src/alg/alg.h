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
#include "./../blas/engine.h"
#include "./../lapack/engine.h"
#include "./../matrix/matrix.h"
#include "./../matrix/serialize.h"
#include "./../util/topology.h"
#include "./../util/util.h"

#endif // ALGORITHMS_H_
