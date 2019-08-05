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
#include "../../../ExternalLibraries/critter/src/critter.h"
#endif /*PORTER*/
#ifdef STAMPEDE2
#include "../../../critter/critter.h"
#endif /*STAMPEDE2*/
#ifdef BLUEWATERS
#include "../../../critter/critter.h"
#endif /*BLUEWATERS*/
#endif /*CRITTER*/

// Note: no need to include mpi header file when using critter
// #include <mpi.h>

#endif /*SHARED*/
