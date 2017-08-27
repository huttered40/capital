/* Author: Edward Hutter */

#ifndef CFR3D_H_
#define CFR3D_H_

// System includes
#include <iostream>
#include <mpi.h>

// Local includes
#include "./../../../../../AlgebraicStructures/Matrix/Matrix.h"
#include "./../../../../../AlgebraicStructures/Matrix/MatrixSerializer.h"
#include "./../../../../../AlgebraicBLAS/blasEngine.h"

// Lets use partial template specialization
// So only declare the fully templated class
template<typename T, typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,
  template<typename,typename,template<typename,typename,int> class> class StructureL>
class CFR3D;

// Partial specialization of CFR3D algorithm class
template<typename T, typename U>
class CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare>
{
public:
  // Prevent instantiation of this class
  CFR3D() = delete;
  CFR3D(const CFR3D& rhs) = delete;
  CFR3D(CFR3D&& rhs) = delete;
  CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare>& operator=(const CFR3D& rhs) = delete;
  CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare>& operator=(CFR3D&& rhs) = delete;

  template<template<typename,typename,int> class Distribution>
  static void Factor(
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
                      U dimension,
                      MPI_Comm commWorld
                    );

private:
  template<template<typename,typename,int> class Distribution>
  static void rFactor(
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

};

#include "CFR3D.hpp"

#endif /* CFR3D_H_ */
