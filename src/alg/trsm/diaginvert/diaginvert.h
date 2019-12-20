/* Author: Edward Hutter */

#ifndef TRSM3D_H_
#define TRSM3D_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"

namespace trsm{

class diaginvert{
public:
  template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
  static void invoke(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, CommType&& CommInfo,
                     char UpLo, char Dir, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                     blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage);

private:
  /*
  template<typename MatrixAType, typename MatrixTriType, typename CommType>
  static void iSolveLowerLeft(MatrixAType& matrixA, MatrixTriType& matrixL, MatrixTriType& matrixLI, CommType&& CommInfo,
                              std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                              blas::ArgPack_gemm<typename MatrixTriType::ScalarType>& gemmPackage);
  */

  template<typename MatrixAType, typename MatrixUType, typename MatrixUIType, typename CommType>
  static void iSolveUpperLeft(MatrixAType& matrixA, MatrixUType& matrixU, MatrixUIType& matrixUI, CommType&& CommInfo,
                              std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                              blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage);
  
  template<typename MatrixLType, typename MatrixLIType, typename MatrixAType, typename CommType>
  static void iSolveLowerRight(MatrixLType& matrixL, MatrixLIType& matrixLI, MatrixAType& matrixA, CommType&& CommInfo,
                               std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                               blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage);
  /*
  template<typename MatrixTriType, typename MatrixAType, typename CommType>
  static void iSolveUpperRight(MatrixTriType& matrixU, MatrixTriType& matrixUI, MatrixAType& matrixA, CommType&& CommInfo,
                               std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                               blas::ArgPack_gemm<typename MatrixTriType::ScalarType>& gemmPackage);
  */
};
}

#include "diaginvert.hpp"

#endif /* TRSM__DIAGINVERT_H_ */
