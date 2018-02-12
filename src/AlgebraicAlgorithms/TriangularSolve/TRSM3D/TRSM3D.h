/* Author: Edward Hutter */

#ifndef TRSM3D_H_
#define TRSM3D_H_

// System includes
#include <iostream>
#include <complex>
#include <mpi.h>
#include "/home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"

// Local includes
#include "./../../../Matrix/Matrix.h"
#include "./../../../Matrix/MatrixSerializer.h"
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
/*
  template<template<typename,typename,int> class Distribution>
  static void Solve(
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixB,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixC,
                      char dir,
                      int tune,
                      MPI_Comm commWorld,
                      int MM_id = 0
                    );
*/

  template<
    template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution
  >
  static void iSolveLowerLeft(
                       Matrix<T,U,StructureArg,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
                       Matrix<T,U,StructureArg,Distribution>& matrixB,
                       U matAstartX,
                       U matAendX,
                       U matAstartY,
                       U matAendY,
                       U matLstartX,
                       U matLendX,
                       U matLstartY,
                       U matLendY,
                       U matBstartX,
                       U matBendX,
                       U matBstartY,
                       U matBendY,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& srcPackage,
                       MPI_Comm commWorld,
                       int MM_id = 0,
                       int TR_id = 1
                     );

  template<
    template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution
  >
  static void iSolveUpperLeft(
                       Matrix<T,U,StructureArg,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixU,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixUI,
                       Matrix<T,U,StructureArg,Distribution>& matrixB,
                       U matAstartX,
                       U matAendX,
                       U matAstartY,
                       U matAendY,
                       U matUstartX,
                       U matUendX,
                       U matUstartY,
                       U matUendY,
                       U matBstartX,
                       U matBendX,
                       U matBstartY,
                       U matBendY,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& srcPackage,
                       MPI_Comm commWorld,
                       int MM_id = 0,
                       int TR_id = 1
                     );
  
  template<
    template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution
  >
  static void iSolveLowerRight(
                       Matrix<T,U,StructureArg,Distribution>& matrixL,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                       Matrix<T,U,StructureArg,Distribution>& matrixB,
                       U matLstartX,
                       U matLendX,
                       U matLstartY,
                       U matLendY,
                       U matAstartX,
                       U matAendX,
                       U matAstartY,
                       U matAendY,
                       U matBstartX,
                       U matBendX,
                       U matBstartY,
                       U matBendY,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& srcPackage,
                       MPI_Comm commWorld,
                       int MM_id = 0,
                       int TR_id = 1         // allows for benchmarking to see which version is faster 
                     );

  template<
    template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution
  >
  static void iSolveUpperRight(
                       Matrix<T,U,StructureArg,Distribution>& matrixU,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixUI,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                       Matrix<T,U,StructureArg,Distribution>& matrixB,
                       U matUstartX,
                       U matUendX,
                       U matUstartY,
                       U matUendY,
                       U matAstartX,
                       U matAendX,
                       U matAstartY,
                       U matAendY,
                       U matBstartX,
                       U matBendX,
                       U matBstartY,
                       U matBendY,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& srcPackage,
                       MPI_Comm commWorld,
                       int MM_id = 0,
                       int TR_id = 1
                     );

private:
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
