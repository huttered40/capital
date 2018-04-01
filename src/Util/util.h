/* Author: Edward Hutter */

#ifndef UTIL_H_
#define UTIL_H_

#include <fstream>
#include <vector>
#include <tuple>

#include "shared.h"
#include "./../Timer/Timer.h"
#include "../Matrix/Matrix.h"
#include "../Matrix/MatrixSerializer.h"

template<typename T, typename U>
class util
{
public:
  util() = delete;
  ~util();
  util(const util& rhs) = delete;
  util(util&& rhs) = delete;
  util& operator=(const util& rhs) = delete;
  util& operator=(util&& rhs) = delete;

  static std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int> build3DTopology(
                    MPI_Comm commWorld);
  static std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm> buildTunableTopology(
      MPI_Comm commWorld, int pGridDimensionD, int pGridDimensionC);

  static void destroy3DTopology(std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D);
  static void destroyTunableTopology(std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm>& commInfoTunable);

  static std::vector<T> blockedToCyclic(
    std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, int pGridDimensionSize);

  static std::vector<T> blockedToCyclicSpecial(
  std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, int pGridDimensionSize, char dir);

  template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
  static std::vector<T> getReferenceMatrix(
              Matrix<T,U,StructureArg,Distribution>& myMatrix,
							U key,
							std::tuple<MPI_Comm, int, int, int, int> commInfo
						  );

  template< template<typename,typename,template<typename,typename,int> class> class StructureArg,template<typename,typename,int> class Distribution>
  static void transposeSwap(
					Matrix<T,U,StructureArg,Distribution>& mat,
				  int myRank,
					int transposeRank,
					MPI_Comm commWorld
					);

  static std::tuple<MPI_Comm, int, int, int, int> getCommunicatorSlice(
    MPI_Comm commWorld);

  static U getNextPowerOf2(U localShift);

  template< template<typename,typename,template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>
  static void removeTriangle(Matrix<T,U,StructureArg,Distribution>& matrix, int pGridCoordX, int pGridCoordY, int pGridDimensionSize, char dir);
  
  static void processAveragesFromFile(FILE* fptrAvg, std::string& fileStrTotal, int numFuncs, int numIterations, int rank);
};

#include "util.hpp"
#endif /*UTIL_H_*/
