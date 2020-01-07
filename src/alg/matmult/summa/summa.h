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
  static void invoke(MatrixAType& A, MatrixBType& B, MatrixCType& C, CommType&& CommInfo, blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage);

  template<typename MatrixAType, typename MatrixBType, typename CommType>
  static void invoke(MatrixAType& A, MatrixBType& B, CommType&& CommInfo, blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage);

  template<typename MatrixSrcType, typename MatrixDestType, typename CommType>
  static void invoke(MatrixSrcType& A, MatrixDestType& C, CommType&& CommInfo, blas::ArgPack_syrk<typename MatrixSrcType::ScalarType>& srcPackage);

  template<typename MatrixSrcType, typename MatrixDestType, typename CommType>
  static void invoke(MatrixSrcType& A, MatrixSrcType& B, MatrixDestType& C, CommType&& CommInfo, blas::ArgPack_syrk<typename MatrixSrcType::ScalarType>& srcPackage);

private:

  template<typename MatrixAType, typename MatrixBType, typename CommType>
  static void distribute(MatrixAType& A, MatrixBType& B, CommType&& CommInfo);

  template<typename MatrixType, typename CommType>
  static void collect(MatrixType& matrix, CommType&& CommInfo);

  template<typename MatrixSrcType, typename MatrixDestType, typename CommType>
  static void syrk_internal(MatrixSrcType& A, MatrixSrcType& B, MatrixDestType& C, CommType&& CommInfo, blas::ArgPack_syrk<typename MatrixSrcType::ScalarType>& srcPackage);
};
}

#include "summa.hpp"

#endif /* MATMULT__SUMMA_H_ */
