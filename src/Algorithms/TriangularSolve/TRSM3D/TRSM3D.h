/* Author: Edward Hutter */

#ifndef TRSM3D_H_
#define TRSM3D_H_

#include "./../../Algorithms.h"
#include "./../../MatrixMultiplication/MM3D/MM3D.h"

// Lets use partial template specialization
// So only declare the fully templated class
// Why not just use square? Because later on, I may want to experiment with LowerTriangular Structure.
// Also note, we do not need an extra template parameter for L-inverse. Presumably if the user wants L to be LowerTriangular, then he wants L-inverse
//   to be LowerTriangular as well

class TRSM3D{
public:
  template<typename MatrixAType, typename MatrixTriType>
  static void iSolveLowerLeft(MatrixAType& matrixA, MatrixTriType& matrixL, MatrixTriType& matrixLI, std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                               blasEngineArgumentPackage_gemm<typename MatrixTriType::ScalarType>& gemmPackage, blasEngineArgumentPackage_trmm<typename MatrixTriType::ScalarType>& trmmPackage,
                               MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D);

  template<typename MatrixAType, typename MatrixTriType>
  static void iSolveUpperLeft(MatrixAType& matrixA, MatrixTriType& matrixU, MatrixTriType& matrixUI, std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                               blasEngineArgumentPackage_gemm<typename MatrixTriType::ScalarType>& gemmPackage, blasEngineArgumentPackage_trmm<typename MatrixTriType::ScalarType>& trmmPackage,
                               MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D);
  
  template<typename MatrixTriType, typename MatrixAType>
  static void iSolveLowerRight(MatrixTriType& matrixL, MatrixTriType& matrixLI, MatrixAType& matrixA, std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                               blasEngineArgumentPackage_gemm<typename MatrixTriType::ScalarType>& gemmPackage, blasEngineArgumentPackage_trmm<typename MatrixTriType::ScalarType>& trmmPackage,
                               MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D);

  template<typename MatrixTriType, typename MatrixAType>
  static void iSolveUpperRight(MatrixTriType& matrixU, MatrixTriType& matrixUI, MatrixAType& matrixA, std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                               blasEngineArgumentPackage_gemm<typename MatrixTriType::ScalarType>& gemmPackage, blasEngineArgumentPackage_trmm<typename MatrixTriType::ScalarType>& trmmPackage,
                               MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D);
};

#include "TRSM3D.hpp"

#endif /* TRSM3D_H_ */
