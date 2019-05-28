/* Author: Edward Hutter */

#ifndef UTIL_H_
#define UTIL_H_

class util{
public:
  static std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t> build3DTopology(MPI_Comm commWorld);
  static std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm> buildTunableTopology(MPI_Comm commWorld, size_t pGridDimensionD, size_t pGridDimensionC);

  static void destroy3DTopology(std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D);
  static void destroyTunableTopology(std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm>& commInfoTunable);

  template<typename T, typename U>
  static std::vector<T> blockedToCyclic(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, size_t pGridDimensionSize);

  template<typename T, typename U>
  static std::vector<T> blockedToCyclicSpecial(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, size_t pGridDimensionSize, char dir);

  template<typename MatrixType>
  static std::vector<typename MatrixType::ScalarType>
         getReferenceMatrix(MatrixType& myMatrix, size_t key, std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> commInfo);

  template<typename MatrixType>
  static void transposeSwap(MatrixType& mat, size_t myRank, size_t transposeRank, MPI_Comm commWorld);

  static std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> getCommunicatorSlice(MPI_Comm commWorld);

  template<typename U>
  static U getNextPowerOf2(U localShift);

  template<typename MatrixType>
  static void removeTriangle(MatrixType& matrix, size_t pGridCoordX, size_t pGridCoordY, size_t pGridDimensionSize, char dir);
  
  static void processAveragesFromFile(std::ofstream& fptrAvg, std::string& fileStrTotal, size_t numFuncs, size_t numIterations, size_t rank);

  template<typename T>
  static void InitialGEMM();
};

#include "util.hpp"
#endif /*UTIL_H_*/
