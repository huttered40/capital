/* Author: Edward Hutter */

#ifndef CFR3D_H_
#define CFR3D_H_

// System includes
#include <iostream>
#include <complex>
#include <mpi.h>
#include "/home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/LAPACKE/include/lapacke.h"

// Local includes
#include "./../../../../../AlgebraicContainers/Matrix/Matrix.h"
#include "./../../../../../AlgebraicContainers/Matrix/MatrixSerializer.h"
#include "./../../../../../AlgebraicBLAS/blasEngine.h"
#include "./../../MatrixMultiplication/SquareMM3D/SquareMM3D.h"

// Lets use partial template specialization
// So only declare the fully templated class
// Why not just use square? Because later on, I may want to experiment with LowerTriangular Structure.
// Also note, we do not need an extra template parameter for L-inverse. Presumably if the user wants L to be LowerTriangular, then he wants L-inverse
//   to be LowerTriangular as well

template<typename T, typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,
  template<typename,typename,template<typename,typename,int> class> class StructureL,
  template<typename, typename> class blasEngine>
class CFR3D;

// Partial specialization of CFR3D algorithm class
template<typename T, typename U, template<typename, typename> class blasEngine>
class CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare, blasEngine>
{
public:
  // Prevent instantiation of this class
  CFR3D() = delete;
  CFR3D(const CFR3D& rhs) = delete;
  CFR3D(CFR3D&& rhs) = delete;
  CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare,blasEngine>& operator=(const CFR3D& rhs) = delete;
  CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare,blasEngine>& operator=(CFR3D&& rhs) = delete;

  template<template<typename,typename,int> class Distribution>
  static void Factor(
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
                      U dimension,
                      char dir,
                      MPI_Comm commWorld
                    );


private:
  template<template<typename,typename,int> class Distribution>
  static void rFactorLower(
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
                       U dimension,
                       U bcDimension,
                       U globalDimension,
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
                       MPI_Comm commWorld
                     );

  template<template<typename,typename,int> class Distribution>
  static void rFactorUpper(
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixRI,
                       U dimension,
                       U bcDimension,
                       U globalDimension,
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
                       MPI_Comm commWorld
                     );

  template<template<typename,typename,template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>
  static void transposeSwap(
				Matrix<T,U,StructureArg,Distribution>& mat,
				int myRank,
				int transposeRank
			   );

  template<template<typename,typename,int> class Distribution>
  static std::vector<T> blockedToCyclicTransformation(
							Matrix<T,U,MatrixStructureSquare,Distribution>& matA,
							U localDimension,
							U globalDimension,
							U bcDimension,
							U matAstartX,
							U matAendX,
							U matAstartY,
							U matAendY,
							int pGridDimensionSize,
							MPI_Comm slice2Dcomm
						     );

  static void cyclicToLocalTransformation(
						std::vector<T>& storeT,
						std::vector<T>& storeTI,
						U localDimension,
						U globalDimension,
						U bcDimension,
						int pGridDimensionSize,
						int rankSlice,
						char dir
					 );

};

#include "CFR3D.hpp"

#endif /* CFR3D_H_ */
