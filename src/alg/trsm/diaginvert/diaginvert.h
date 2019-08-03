/* Author: Edward Hutter */

#ifndef TRSM3D_H_
#define TRSM3D_H_

#include "./../../alg.h"
#include "./../../matmult/summa3d/summa3d.h"

namespace trsm{

// Lets use partial template specialization
// So only declare the fully templated class
// Why not just use square? Because later on, I may want to experiment with LowerTriangular Structure.
// Also note, we do not need an extra template parameter for L-inverse. Presumably if the user wants L to be LowerTriangular, then he wants L-inverse
//   to be LowerTriangular as well

class diaginvert{
public:
  template<typename MatrixAType, typename MatrixTriType, typename CommType>
  static void iSolveLowerLeft(MatrixAType& matrixA, MatrixTriType& matrixL, MatrixTriType& matrixLI, std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                               blasEngineArgumentPackage_gemm<typename MatrixTriType::ScalarType>& gemmPackage, blasEngineArgumentPackage_trmm<typename MatrixTriType::ScalarType>& trmmPackage,
                               CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixTriType, typename CommType>
  static void iSolveUpperLeft(MatrixAType& matrixA, MatrixTriType& matrixU, MatrixTriType& matrixUI, std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                               blasEngineArgumentPackage_gemm<typename MatrixTriType::ScalarType>& gemmPackage, blasEngineArgumentPackage_trmm<typename MatrixTriType::ScalarType>& trmmPackage,
                               CommType&& CommInfo);
  
  template<typename MatrixTriType, typename MatrixAType, typename CommType>
  static void iSolveLowerRight(MatrixTriType& matrixL, MatrixTriType& matrixLI, MatrixAType& matrixA, std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                               blasEngineArgumentPackage_gemm<typename MatrixTriType::ScalarType>& gemmPackage, blasEngineArgumentPackage_trmm<typename MatrixTriType::ScalarType>& trmmPackage,
                               CommType&& CommInfo);

  template<typename MatrixTriType, typename MatrixAType, typename CommType>
  static void iSolveUpperRight(MatrixTriType& matrixU, MatrixTriType& matrixUI, MatrixAType& matrixA, std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                               blasEngineArgumentPackage_gemm<typename MatrixTriType::ScalarType>& gemmPackage, blasEngineArgumentPackage_trmm<typename MatrixTriType::ScalarType>& trmmPackage,
                               CommType&& CommInfo);
};
}

#include "diaginvert.hpp"

#endif /* TRSM__DIAGINVERT_H_ */
