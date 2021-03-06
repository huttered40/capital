/* Author: Edward Hutter */

#ifndef SHARED
#define SHARED

// System includes
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <complex>
#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <tuple>
#include <cmath>
#include <string>
#include <assert.h>
#include <complex>

#include <mpi.h>
#include "mkl.h"

#ifdef CRITTER
#include "critter.h"
#else
//#ifdef PORTER
//#include <cblas.h>
//#include "/home/hutter2/hutter2/external/BLAS/OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"
//#endif
#define CRITTER_START(ARG)
#define CRITTER_STOP(ARG)
#endif

template<typename ScalarType>
class mpi_type;

template<>
class mpi_type<float>{
public:
  constexpr static size_t type = MPI_FLOAT;
};
template<>
class mpi_type<double>{
public:
  constexpr static size_t type = MPI_DOUBLE;
};


#endif /*SHARED*/
