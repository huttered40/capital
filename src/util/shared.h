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

#ifdef PORTER
#include "../../../ExternalLibraries/critter/src/critter.h"
#endif /*PORTER*/
#ifdef STAMPEDE2
#include "../../../critter/src/critter.h"
#endif /*STAMPEDE2*/
#ifdef BLUEWATERS
#include "../../../critter/src/critter.h"
#endif /*BLUEWATERS*/

// Note: no need to include mpi header file when using critter
// #include <mpi.h>

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
