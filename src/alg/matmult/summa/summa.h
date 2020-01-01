/* Author: Edward Hutter */

#ifndef MATMULT__SUMMA_H_
#define MATMULT__SUMMA_H_

#include "./../../alg.h"

namespace matmult{
/*
  We can implement square MM for now, but soon, we will need triangular MM
    and triangular matrices, as well as Square-Triangular Multiplication and Triangular-Square Multiplication
  Also, we need to figure out what to do with Rectangular.
*/

class summa{
public:
  // Format: matrixA is M x K
  //         matrixB is K x N
  //         matrixC is M x N

  // New design: user will specify via an argument to the overloaded Multiply() method what underlying BLAS routine he wants called.
  //             I think this is a reasonable assumption to make and will allow me to optimize each routine.

  template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
  static void invoke(MatrixAType& A, MatrixBType& B, MatrixCType& C, CommType&& CommInfo,
                     blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage);

  template<typename ScalarType, typename DimensionType, typename ArgType, typename CommType>
  static ScalarType* invoke(ScalarType* A, ScalarType* B, ScalarType* C, DimensionType localNumRowsA, DimensionType localNumColumnsA, DimensionType localNumColumnsB,
                            DimensionType globalNumRowsA, DimensionType globalNumColumnsA, DimensionType globalNumColumnsB, ArgType&& args, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixBType, typename CommType>
  static void invoke(MatrixAType& A, MatrixBType& B, CommType&& CommInfo,
                     blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage);

  template<typename MatrixAType, typename MatrixCType, typename CommType>
  static void invoke(MatrixAType& A, MatrixCType& C, CommType&& CommInfo,
                     blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage);

  template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
  static void invoke(MatrixAType& A, MatrixBType& B, MatrixCType& C, typename MatrixAType::DimensionType AcutXstart,
                     typename MatrixAType::DimensionType AcutXend, typename MatrixAType::DimensionType AcutYstart,
                     typename MatrixAType::DimensionType AcutYend, typename MatrixBType::DimensionType BcutZstart,
                     typename MatrixBType::DimensionType BcutZend, typename MatrixBType::DimensionType BcutXstart,
                     typename MatrixBType::DimensionType BcutXend, typename MatrixCType::DimensionType CcutZstart,
                     typename MatrixCType::DimensionType CcutZend, typename MatrixCType::DimensionType CcutYstart,
                     typename MatrixCType::DimensionType CcutYend, CommType&& CommInfo,
                     blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB, bool cutC);

  template<typename MatrixAType, typename MatrixBType, typename CommType>
  static void invoke(MatrixAType& A, MatrixBType& B, typename MatrixAType::DimensionType AcutXstart,
                     typename MatrixAType::DimensionType AcutXend, typename MatrixAType::DimensionType AcutYstart,
                     typename MatrixAType::DimensionType AcutYend, typename MatrixBType::DimensionType BcutZstart,
                     typename MatrixBType::DimensionType BcutZend, typename MatrixBType::DimensionType BcutXstart,
                     typename MatrixBType::DimensionType BcutXend, CommType&& CommInfo,
                     blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB);

  template<typename MatrixAType, typename MatrixCType, typename CommType>
  static void invoke(MatrixAType& A, MatrixCType& C, typename MatrixAType::DimensionType AcutXstart,
                     typename MatrixAType::DimensionType AcutXend, typename MatrixAType::DimensionType AcutYstart,
                     typename MatrixAType::DimensionType AcutYend, typename MatrixCType::DimensionType CcutZstart,
                     typename MatrixCType::DimensionType CcutZend, typename MatrixCType::DimensionType CcutXstart,
                     typename MatrixCType::DimensionType CcutXend, CommType&& CommInfo,
                     blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage, bool cutA = true, bool cutC = true);

private:

  template<typename MatrixAType, typename MatrixBType, typename CommType>
  static void distribute(MatrixAType& A, MatrixBType& B, CommType&& CommInfo);

  template<typename MatrixType, typename CommType>
  static void collect(MatrixType& matrix, CommType&& CommInfo);

  template<typename MatrixType>
  static MatrixType extract(MatrixType& srcMatrix, typename MatrixType::DimensionType matrixArgColumnStart, typename MatrixType::DimensionType matrixArgColumnEnd,
                            typename MatrixType::DimensionType matrixArgRowStart, typename MatrixType::DimensionType matrixArgRowEnd, int64_t sliceDim, bool getSub);
};
}

#include "summa.hpp"

#endif /* MATMULT__SUMMA_H_ */
