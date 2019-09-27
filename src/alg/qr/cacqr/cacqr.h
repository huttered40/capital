/* Author: Edward Hutter */

#ifndef QR__CACQR_H_
#define QR__CACQR_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../../trsm/diaginvert/diaginvert.h"
#include "./../../cholesky/cholinv/cholinv.h"
#include "./../policies/cacqr/policy.h"
#include "./../../cholesky/policies/cholinv/policy.h"

namespace qr{

template<class SerializeSymmetricPolicy = policy::cacqr::SerializeSymmetricToTriangle,
         class CholInvPolicy = cholesky::cholinv<cholesky::policy::cholinv::TrmmUpdate>,
         class TrsmPolicy = trsm::diaginvert
         >
class cacqr{
public:
  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo,
                     size_t inverseCutOffMultiplier = 0, size_t baseCaseMultiplier = 0, size_t panelDimensionMultiplier = 0);

protected:
  // Special overload to avoid recreating MPI communicator topologies
  template<typename MatrixAType, typename MatrixRType, typename RectCommType, typename SquareCommType>
  static void invoke(MatrixAType& MatrixA, MatrixRType& MatrixR, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo,
                     size_t inverseCutOffMultiplier = 0, size_t baseCaseMultiplier = 0, size_t panelDimensionMultiplier = 0);

  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke_1d(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke_3d(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo,
                        size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier);

  template<typename T, typename U> 
  static void broadcast_panels(std::vector<T>& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panelComm);
};

template<class SerializeSymmetricPolicy = policy::cacqr::SerializeSymmetricToTriangle,
         class CholInvPolicy = cholesky::cholinv<cholesky::policy::cholinv::TrmmUpdate>,
         class TrsmPolicy = trsm::diaginvert
	 >
class cacqr2 : public cacqr<SerializeSymmetricPolicy>{
public:
  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo,
                     size_t inverseCutOffMultiplier = 0, size_t baseCaseMultiplier = 0, size_t panelDimensionMultiplier = 0);

protected:
  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke_1d(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixRType, typename CommType>
  static void invoke_3d(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo,
                        size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier);
};
}

#include "cacqr.hpp"

#endif /* QR__CACQR_H_ */
