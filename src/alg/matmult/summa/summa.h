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

  template<typename MatrixBType, typename CommType>
  static void invoke(typename MatrixBType::ScalarType* matrixA, MatrixBType& matrixB, typename MatrixBType::ScalarType* matrixC,
                     typename MatrixBType::DimensionType matrixAnumColumns, typename MatrixBType::DimensionType matrixAnumRows,
                     typename MatrixBType::DimensionType matrixBnumColumns, typename MatrixBType::DimensionType matrixBnumRows,
                     typename MatrixBType::DimensionType matrixCnumColumns, typename MatrixBType::DimensionType matrixCnumRows,
                     CommType&& CommInfo, const blas::ArgPack_gemm<typename MatrixBType::ScalarType>& srcPackage);

  template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
  static void invoke(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, CommType&& CommInfo,
                     const blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage, int64_t methodKey = 0);

  template<typename MatrixAType, typename MatrixBType, typename CommType>
  static void invoke(MatrixAType& matrixA, MatrixBType& matrixB, CommType&& CommInfo,
                     const blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage, int64_t methodKey = 0);

  template<typename MatrixAType, typename CommType>
  static void invoke(MatrixAType& matrixA, typename MatrixAType::ScalarType* matrixB, typename MatrixAType::DimensionType matrixAnumColumns,
                     typename MatrixAType::DimensionType matrixAnumRows, typename MatrixAType::DimensionType matrixBnumColumns,
                     typename MatrixAType::DimensionType matrixBnumRows, CommType&& CommInfo,
                     const blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage);

  template<typename MatrixAType, typename MatrixCType, typename CommType>
  static void invoke(MatrixAType& matrixA, MatrixCType& matrixC, CommType&& CommInfo,
                     const blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage, int64_t methodKey = 0);

  template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
  static void invoke(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, typename MatrixAType::DimensionType matrixAcutXstart,
                     typename MatrixAType::DimensionType matrixAcutXend, typename MatrixAType::DimensionType matrixAcutYstart,
                     typename MatrixAType::DimensionType matrixAcutYend, typename MatrixBType::DimensionType matrixBcutZstart,
                     typename MatrixBType::DimensionType matrixBcutZend, typename MatrixBType::DimensionType matrixBcutXstart,
                     typename MatrixBType::DimensionType matrixBcutXend, typename MatrixCType::DimensionType matrixCcutZstart,
                     typename MatrixCType::DimensionType matrixCcutZend, typename MatrixCType::DimensionType matrixCcutYstart,
                     typename MatrixCType::DimensionType matrixCcutYend, CommType&& CommInfo,
                     const blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB, bool cutC, int64_t methodKey = 0);

  template<typename MatrixAType, typename MatrixBType, typename CommType>
  static void invoke(MatrixAType& matrixA, MatrixBType& matrixB, typename MatrixAType::DimensionType matrixAcutXstart,
                     typename MatrixAType::DimensionType matrixAcutXend, typename MatrixAType::DimensionType matrixAcutYstart,
                     typename MatrixAType::DimensionType matrixAcutYend, typename MatrixBType::DimensionType matrixBcutZstart,
                     typename MatrixBType::DimensionType matrixBcutZend, typename MatrixBType::DimensionType matrixBcutXstart,
                     typename MatrixBType::DimensionType matrixBcutXend, CommType&& CommInfo,
                     const blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB, int64_t methodKey = 0);

  template<typename MatrixAType, typename MatrixCType, typename CommType>
  static void invoke(MatrixAType& matrixA, MatrixCType& matrixC, typename MatrixAType::DimensionType matrixAcutXstart,
                     typename MatrixAType::DimensionType matrixAcutXend, typename MatrixAType::DimensionType matrixAcutYstart,
                     typename MatrixAType::DimensionType matrixAcutYend, typename MatrixCType::DimensionType matrixCcutZstart,
                     typename MatrixCType::DimensionType matrixCcutZend, typename MatrixCType::DimensionType matrixCcutXstart,
                     typename MatrixCType::DimensionType matrixCcutXend, CommType&& CommInfo,
                     const blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage, bool cutA = true, bool cutC = true, int64_t methodKey = 0);

private:

  template<typename MatrixAType, typename MatrixBType, typename CommType>
  static void distribute_bcast(MatrixAType& matrixA, MatrixBType& matrixB, CommType&& CommInfo, typename MatrixAType::ScalarType*& matrixAEnginePtr,
                      typename MatrixBType::ScalarType*& matrixBEnginePtr, std::vector<typename MatrixAType::ScalarType>& matrixAEngineVector,
                      std::vector<typename MatrixBType::ScalarType>& matrixBEngineVector, std::vector<typename MatrixAType::ScalarType>& foreignA,
                      std::vector<typename MatrixBType::ScalarType>& foreignB, bool& serializeKeyA, bool& serializeKeyB);

  template<typename MatrixAType, typename MatrixBType, typename CommType>
  static void distribute_allgather(MatrixAType& matrixA, MatrixBType& matrixB, CommType&& CommInfo,
                      std::vector<typename MatrixAType::ScalarType>& matrixAEngineVector, std::vector<typename MatrixBType::ScalarType>& matrixBEngineVector,
                      bool& serializeKeyA, bool& serializeKeyB);

  template<typename MatrixType, typename CommType>
  static void collect(typename MatrixType::ScalarType* matrixEnginePtr, MatrixType& matrix, CommType&& CommInfo, int64_t dir = 0);

  template<typename MatrixSrcType, typename MatrixDestType>
  static void getEnginePtr(MatrixSrcType& matrixArg, MatrixDestType& matrixDest, std::vector<typename MatrixSrcType::ScalarType>& data, bool isRoot);

  template<typename MatrixType>
  static MatrixType getSubMatrix(MatrixType& srcMatrix, typename MatrixType::DimensionType matrixArgColumnStart, typename MatrixType::DimensionType matrixArgColumnEnd,
                                 typename MatrixType::DimensionType matrixArgRowStart, typename MatrixType::DimensionType matrixArgRowEnd, int64_t sliceDim, bool getSub);

};
}

#include "summa.hpp"

#endif /* MATMULT__SUMMA_H_ */
