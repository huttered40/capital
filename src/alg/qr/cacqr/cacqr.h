/* Author: Edward Hutter */

#ifndef QR__CACQR_H_
#define QR__CACQR_H_

#include "./../../alg.h"
#include "./../../MatrixMultiplication/MM3D/MM3D.h"
#include "./../../TriangularSolve/TRSM3D/TRSM3D.h"
#include "./../../CholeskyFactorization/CFR3D/CFR3D.h"

namespace qr{

class cacqr{
public:
  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& commInfo,
                     size_t inverseCutOffMultiplier = 0, size_t baseCaseMultiplier = 0, size_t panelDimensionMultiplier = 0);

protected:
  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke_1d(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke_3d(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& commInfo,
                        size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier);

  template<typename T, typename U> 
  static void broadcast_panels(std::vector<T>& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panelComm);
};

class cacqr2 : public cacqr{
public:
  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& commInfo,
                     size_t inverseCutOffMultiplier = 0, size_t baseCaseMultiplier = 0, size_t panelDimensionMultiplier = 0);

protected:
  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke_1d(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke_3d(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo,
                        size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier);
};
}

#include "cacqr.hpp"

#endif /* QR__CACQR_H_ */
