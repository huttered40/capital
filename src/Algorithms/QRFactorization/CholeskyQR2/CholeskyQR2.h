/* Author: Edward Hutter */

#ifndef CHOLESKYQR2_H_
#define CHOLESKYQR2_H_

#include "./../../Algorithms.h"
#include "./../../MatrixMultiplication/MM3D/MM3D.h"
#include "./../../TriangularSolve/TRSM3D/TRSM3D.h"
#include "./../../CholeskyFactorization/CFR3D/CFR3D.h"

// Need template parameters for all 3 matrices (A,Q,R), as well as some other things, right?
class CholeskyQR2{
public:
  template<typename MatrixAType, typename MatrixRType>
  static void Factor1D(MatrixAType& matrixA, MatrixRType& matrixR, MPI_Comm commWorld);

  template<typename MatrixAType, typename MatrixRType>
  static void Factor3D(MatrixAType& matrixA, MatrixRType& matrixR, MPI_Comm commWorld,
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D, size_t inverseCutOffMultiplier = 0, size_t baseCaseMultiplier = 0, size_t panelDimensionMultiplier = 0);

  template<typename MatrixAType, typename MatrixRType>
  static void FactorTunable(MatrixAType& matrixA, MatrixRType& matrixR, size_t gridDimensionD, size_t gridDimensionC,
      MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm>& commInfoTunable, size_t inverseCutOffMultiplier = 0, size_t baseCaseMultiplier = 0, size_t panelDimensionMultiplier = 0);

private:
  template<typename MatrixAType, typename MatrixRType>
  static void Factor1D_cqr(MatrixAType& matrixA, MatrixRType& matrixR, MPI_Comm commWorld);

  template<typename MatrixAType, typename MatrixRType>
  static void Factor3D_cqr(MatrixAType& matrixA, MatrixRType& matrixR,
      MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D,
      size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier);

  template<typename MatrixAType, typename MatrixRType>
  static void FactorTunable_cqr(MatrixAType& matrixA, MatrixRType& matrixR,
      size_t gridDimensionD, size_t gridDimensionC, MPI_Comm commWorld, std::tuple<MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm>& tunableCommunicators,
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D, size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier);

  template<typename T, typename U> 
  static void BroadcastPanels(std::vector<T>& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panelComm);
};

#include "CholeskyQR2.hpp"

#endif /* CHOLESKYQR2_H_ */
