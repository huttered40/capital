/* Author: Edward Hutter */

#ifndef TRSM3D_H_
#define TRSM3D_H_

// System includes
#include <iostream>
#include <complex>
#include <mpi.h>
#include "/home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"

// Local includes
#include "./../../../AlgebraicContainers/Matrix/Matrix.h"
#include "./../../../AlgebraicContainers/Matrix/MatrixSerializer.h"
#include "./../../../AlgebraicBLAS/blasEngine.h"
#include "./../../MatrixMultiplication/MM3D/MM3D.h"

// Lets use partial template specialization
// So only declare the fully templated class
// Why not just use square? Because later on, I may want to experiment with LowerTriangular Structure.
// Also note, we do not need an extra template parameter for L-inverse. Presumably if the user wants L to be LowerTriangular, then he wants L-inverse
//   to be LowerTriangular as well

template<typename T, typename U, template<typename, typename> class blasEngine>
class TRSM3D
{
public:
  // Prevent instantiation of this class
  TRSM3D() = delete;
  TRSM3D(const TRSM3D& rhs) = delete;
  TRSM3D(TRSM3D&& rhs) = delete;
  TRSM3D& operator=(const TRSM3D& rhs) = delete;
  TRSM3D& operator=(TRSM3D&& rhs) = delete;

  template<template<typename,typename,int> class Distribution>
  static void Solve(
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
                      char dir,
                      int tune,
                      MPI_Comm commWorld,
                      int MM_id = 0
                    );


private:
  template<template<typename,typename,int> class Distribution>
  static void iSolveLower(
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
                       U localDimension,
                       U trueLocalDimenion,
                       U bcDimension,
                       U globalDimension,
                       U trueGlobalDimension,
                       U matAstartX,
                       U matAendX,
                       U matAstartY,
                       U matAendY,
                       U matLstartX,
                       U matLendX,
                       U matLstartY,
                       U matLendY,
                       U matLIstartX,
                       U matLIendX,
                       U matLIstartY,
                       U matLIendY,
                       U tranposePartner,
                       int MM_id,
                       MPI_Comm commWorld
                     );

  template<template<typename,typename,int> class Distribution>
  static void iSolveUpper(
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixRI,
                       U localDimension,
                       U trueLocalDimension,
                       U bcDimension,
                       U globalDimension,
                       U trueGlobalDimension,
                       U matAstartX,
                       U matAendX,
                       U matAstartY,
                       U matAendY,
                       U matRstartX,
                       U matRendX,
                       U matRstartY,
                       U matRendY,
                       U matRIstartX,
                       U matRIendX,
                       U matRIstartY,
                       U matRIendY,
                       U transposePartner,
                       int MM_id,
                       MPI_Comm commWorld
                     );

  template<template<typename,typename,template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>
  static void transposeSwap(
				Matrix<T,U,StructureArg,Distribution>& mat,
				int myRank,
				int transposeRank,
				MPI_Comm commWorld
			   );
};

#include "TRSM3D.hpp"

#endif /* TRSM3D_H_ */
