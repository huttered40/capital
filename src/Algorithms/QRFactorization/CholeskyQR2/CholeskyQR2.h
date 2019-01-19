/* Author: Edward Hutter */

#ifndef CHOLESKYQR2_H_
#define CHOLESKYQR2_H_

// System includes
#include <iostream>

#ifdef CRITTER
#ifdef PORTER
#include "../../../../../ExternalLibraries/CRITTER/critter/critter.h"
#endif /*PORTER*/
#ifdef STAMPEDE2
#include "../../../../../critter/critter.h"
#endif /*STAMPEDE2*/
#ifdef BLUEWATERS
#include "../../../../../critter/critter.h"
#endif /*BLUEWATERS*/
#endif /*CRITTER*/

#ifndef CRITTER
#include <mpi.h>
#endif /*CRITTER*/

// Local includes
#include "./../../../Util/shared.h"
#include "./../../../Timer/CTFtimer.h"
#include "../../../Matrix/Matrix.h"
#include "./../../../AlgebraicBLAS/blasEngine.h"
#include "./../../MatrixMultiplication/MM3D/MM3D.h"
#include "./../../CholeskyFactorization/CFR3D/CFR3D.h"
#include "./../../TriangularSolve/TRSM3D/TRSM3D.h"
#include "./../../../Util/util.h"

// Need template parameters for all 3 matrices (A,Q,R), as well as some other things, right?
template<typename T,typename U,template<typename,typename> class blasEngine>
class CholeskyQR2
{
public:

  CholeskyQR2() = delete;
  ~CholeskyQR2() = delete;
  CholeskyQR2(const CholeskyQR2& rhs) = delete;
  CholeskyQR2(CholeskyQR2&& rhs) = delete;
  CholeskyQR2& operator=(const CholeskyQR2& rhs) = delete;
  CholeskyQR2& operator=(CholeskyQR2&& rhs) = delete;

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void Factor1D(
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld);

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void Factor3D(
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld,
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D, int inverseCutOffMultiplier = 0, int baseCaseMultiplier = 0, int panelDimensionMultiplier = 0);

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void FactorTunable(
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld, 
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm>& commInfoTunable, int inverseCutOffMultiplier = 0, int baseCaseMultiplier = 0,
      int panelDimensionMultiplier = 0);

private:
  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void Factor1D_cqr(
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld);

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void Factor3D_cqr(
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
      int inverseCutOffMultiplier, int baseCaseMultiplier, int panelDimensionMultiplier);

  template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
  static void FactorTunable_cqr(
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld,
      std::tuple<MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm>& tunableCommunicators,
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D, int inverseCutOffMultiplier, int baseCaseMultiplier, int panelDimensionMultiplier);
  
  static void BroadcastPanels(
				std::vector<T>& data,
				U size,
				bool isRoot,
				int pGridCoordZ,
				MPI_Comm panelComm
			      );
};

#include "CholeskyQR2.hpp"

#endif /* CHOLESKYQR2_H_ */