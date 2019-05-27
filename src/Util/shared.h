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
#include <algorithm>
#include <utility>
#include <tuple>
#include <cmath>
#include <string>

#ifdef CRITTER
#ifdef PORTER
#include "../../../ExternalLibraries/CRITTER/critter/critter.h"
#endif /*PORTER*/
#ifdef STAMPEDE2
#include "../../../critter/critter.h"
#endif /*STAMPEDE2*/
#ifdef BLUEWATERS
#include "../../../critter/critter.h"
#endif /*BLUEWATERS*/
#endif /*CRITTER*/

#ifndef CRITTER
#include <mpi.h>
#endif /*CRITTER*/

#ifdef FLOAT_TYPE
  #define DATATYPE float
  #define MPI_DATATYPE MPI_FLOAT
#endif
#ifdef DOUBLE_TYPE
  #define DATATYPE double
  #define MPI_DATATYPE MPI_DOUBLE
#endif
#ifdef COMPLEX_FLOAT_TYPE
  #define DATATYPE std::complex<float>
  #define MPI_DATATYPE MPI_C_FLOAT_COMPLEX
#endif
#ifdef COMPLEX_DOUBLE_TYPE
  #define DATATYPE std::complex<double>
  #define MPI_DATATYPE MPI_C_DOUBLE_COMPLEX
#endif
#ifdef INT_TYPE
  #define INTTYPE int
#endif
#ifdef INT64_T_TYPE
  #define INTTYPE int64_t
#endif

#endif /*SHARED*/
