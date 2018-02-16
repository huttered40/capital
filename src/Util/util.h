/* Author: Edward Hutter */

#ifndef UTIL_H_
#define UTIL_H_

#include <vector>
#include <tuple>

#include "shared.h"
#include "../Matrix/Matrix.h"
#include "../AlgebraicAlgorithms/MatrixMultiplication/MM3D/MM3D.h"

template<typename T, typename U>
class util
{
public:
  util() = delete;
  ~util();
  util(const util& rhs) = delete;
  util(util&& rhs) = delete;
  util& operator=(const util& rhs) = delete;
  util& operator=(util&& rhs) = delete;

  static std::vector<T> blockedToCyclic(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, int pGridDimensionSize);

  template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
  static std::vector<T> getReferenceMatrix(
              Matrix<T,U,StructureArg,Distribution>& myMatrix,
							U key,
							std::tuple<MPI_Comm, int, int, int, int> commInfo
						  );

  template< template<typename,typename,template<typename,typename,int> class> class StructureArg,template<typename,typename,int> class Distribution>
  static void transposeSwap(
					Matrix<T,U,StructureArg,Distribution>& mat,
				  int myRank,
					int transposeRank,
					MPI_Comm commWorld
					);

  static std::tuple<MPI_Comm, int, int, int, int> getCommunicatorSlice(MPI_Comm commWorld);

  template< template<typename,typename,template<typename,typename,int> class> class StructureArg1,
    template<typename,typename,template<typename,typename,int> class> class StructureArg2,
    template<typename,typename,template<typename,typename,int> class> class StructureArg3,
    template<typename,typename,int> class Distribution>
  static void validateResidualParallel(
                        Matrix<T,U,StructureArg1,Distribution>& matrixA,
                        Matrix<T,U,StructureArg2,Distribution>& matrixB,
                        Matrix<T,U,StructureArg3,Distribution>& matrixC,
                        char dir,
                        MPI_Comm commWorld,
                        MPI_Comm columnAltComm = MPI_COMM_WORLD
                      );

  template< template<typename,typename,template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>
  static void validateOrthogonalityParallel(
                        Matrix<T,U,StructureArg,Distribution>& matrixQ,
                        MPI_Comm commWorld,
                        MPI_Comm columnAltComm = MPI_COMM_WORLD
                      );
  static U getNextPowerOf2(U localShift);
};

#include "util.hpp"
#endif /*UTIL_H_*/
