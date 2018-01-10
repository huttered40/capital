/* Author: Edward Hutter */

#ifndef CHOLESKYQR2_H_
#define CHOLESKYQR2_H_

// System includes
#include <iostream>
#include <mpi.h>

// Local includes
#include "../../../AlgebraicContainers/Matrix/Matrix.h"
#include "./../../../AlgebraicBLAS/blasEngine.h"
#include "./../../MatrixMultiplication/MM3D/MM3D.h"
#include "./../../CholeskyFactorization/CFR3D/CFR3D.h"

// Need template parameters for all 3 matrices (A,Q,R), as well as some other things, right?
template<typename T,typename U,template<typename,typename> class blasEngine>
class CholeskyQR2
{
public:

  CholeskyQR2() = delete;
  ~CholeskyQR2() = delete;
  CholeskyQR2(const CholeskyQR2& rhs) = delete;
  CholeskyQR2(CholeskyQR2&& rhs) = delete;
  CholeskyQR2& operator=(const CholeskyQR2& rhs) = delete;
  CholeskyQR2& operator=(CholeskyQR2&& rhs) = delete;

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void Factor1D(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U globalDimensionM, U globalDimensionN, MPI_Comm commWorld);

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void Factor3D(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U globalDimensionM, U globalDimensionN, MPI_Comm commWorld);

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void FactorTunable(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U globalDimensionM, U globalDimensionN, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld);

private:
  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void Factor1D_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U localDimensionM, U localDimensionN, MPI_Comm commWorld);

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void Factor3D_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U localDimensionM, U localDimensionN, MPI_Comm commWorld);

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void FactorTunable_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U localDimensionM, U localDimensionN, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld,
      std::tuple<MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm> tunableCommunicators);
  
  static void BroadcastPanels(
				std::vector<T>& data,
				U size,
				bool isRoot,
				int pGridCoordZ,
				MPI_Comm panelComm
			      );
};

#include "CholeskyQR2.hpp"

#endif /* CHOLESKYQR2_H_ */
