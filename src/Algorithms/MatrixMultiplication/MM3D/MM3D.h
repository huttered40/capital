/* Author: Edward Hutter */

#ifndef MM3D_H_
#define MM3D_H_

#include "./../../Algorithms.h"

/*
  We can implement square MM for now, but soon, we will need triangular MM
    and triangular matrices, as well as Square-Triangular Multiplication and Triangular-Square Multiplication
  Also, we need to figure out what to do with Rectangular.
*/

class MM3D{
public:
  // Format: matrixA is M x K
  //         matrixB is K x N
  //         matrixC is M x N

  // New design: user will specify via an argument to the overloaded Multiply() method what underlying BLAS routine he wants called.
  //             I think this is a reasonable assumption to make and will allow me to optimize each routine.

  template<typename MatrixBType>
  static void Multiply(typename MatrixBType::ScalarType* matrixA, MatrixBType& matrixB, typename MatrixBType::ScalarType* matrixC,
                       typename MatrixBType::DimensionType matrixAnumColumns, typename MatrixBType::DimensionType matrixAnumRows,
                       typename MatrixBType::DimensionType matrixBnumColumns, typename MatrixBType::DimensionType matrixBnumRows,
                       typename MatrixBType::DimensionType matrixCnumColumns, typename MatrixBType::DimensionType matrixCnumRows, MPI_Comm commWorld,
                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D, const blasEngineArgumentPackage_gemm<typename MatrixBType::ScalarType>& srcPackage);

  template<typename MatrixAType, typename MatrixBType, typename MatrixCType>
  static void Multiply(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D,
                       const blasEngineArgumentPackage_gemm<typename MatrixAType::ScalarType>& srcPackage, size_t methodKey = 0);

  template<typename MatrixAType, typename MatrixBType>
  static void Multiply(MatrixAType& matrixA, MatrixBType& matrixB, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D,
                       const blasEngineArgumentPackage_trmm<typename MatrixAType::ScalarType>& srcPackage, size_t methodKey = 0);

  template<typename MatrixAType>
  static void Multiply(MatrixAType& matrixA, typename MatrixAType::ScalarType* matrixB, typename MatrixAType::DimensionType matrixAnumColumns,
                       typename MatrixAType::DimensionType matrixAnumRows, typename MatrixAType::DimensionType matrixBnumColumns,
                       typename MatrixAType::DimensionType matrixBnumRows, MPI_Comm commWorld,
                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D, const blasEngineArgumentPackage_trmm<typename MatrixAType::ScalarType>& srcPackage);

  template<typename MatrixAType, typename MatrixCType>
  static void Multiply(MatrixAType& matrixA, MatrixCType& matrixC, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D,
                       const blasEngineArgumentPackage_syrk<typename MatrixAType::ScalarType>& srcPackage, size_t methodKey = 0);

  template<typename MatrixAType, typename MatrixBType, typename MatrixCType>
  static void Multiply(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, typename MatrixAType::DimensionType matrixAcutXstart,
                       typename MatrixAType::DimensionType matrixAcutXend, typename MatrixAType::DimensionType matrixAcutYstart,
                       typename MatrixAType::DimensionType matrixAcutYend, typename MatrixBType::DimensionType matrixBcutZstart,
                       typename MatrixBType::DimensionType matrixBcutZend, typename MatrixBType::DimensionType matrixBcutXstart,
                       typename MatrixBType::DimensionType matrixBcutXend, typename MatrixCType::DimensionType matrixCcutZstart,
                       typename MatrixCType::DimensionType matrixCcutZend, typename MatrixCType::DimensionType matrixCcutYstart,
                       typename MatrixCType::DimensionType matrixCcutYend, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D,
                        const blasEngineArgumentPackage_gemm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB, bool cutC, size_t methodKey = 0);

  template<typename MatrixAType, typename MatrixBType>
  static void Multiply(MatrixAType& matrixA, MatrixBType& matrixB, typename MatrixAType::DimensionType matrixAcutXstart,
                       typename MatrixAType::DimensionType matrixAcutXend, typename MatrixAType::DimensionType matrixAcutYstart,
                       typename MatrixAType::DimensionType matrixAcutYend, typename MatrixBType::DimensionType matrixBcutZstart,
                       typename MatrixBType::DimensionType matrixBcutZend, typename MatrixBType::DimensionType matrixBcutXstart,
                       typename MatrixBType::DimensionType matrixBcutXend, MPI_Comm commWorld,
                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D,
                       const blasEngineArgumentPackage_trmm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB, size_t methodKey = 0);

  template<typename MatrixAType, typename MatrixCType>
  static void Multiply(MatrixAType& matrixA, MatrixCType& matrixC, typename MatrixAType::DimensionType matrixAcutXstart,
                       typename MatrixAType::DimensionType matrixAcutXend, typename MatrixAType::DimensionType matrixAcutYstart,
                       typename MatrixAType::DimensionType matrixAcutYend, typename MatrixCType::DimensionType matrixCcutZstart,
                       typename MatrixCType::DimensionType matrixCcutZend, typename MatrixCType::DimensionType matrixCcutXstart,
                       typename MatrixCType::DimensionType matrixCcutXend, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D,
                       const blasEngineArgumentPackage_syrk<typename MatrixAType::ScalarType>& srcPackage, bool cutA = true, bool cutC = true, size_t methodKey = 0);

private:

  template<typename MatrixAType, typename MatrixBType, typename tupleStructure>
  static void _start1(MatrixAType& matrixA, MatrixBType& matrixB, tupleStructure& commInfo3D, typename MatrixAType::ScalarType*& matrixAEnginePtr,
                      typename MatrixBType::ScalarType*& matrixBEnginePtr, std::vector<typename MatrixAType::ScalarType>& matrixAEngineVector,
                      std::vector<typename MatrixBType::ScalarType>& matrixBEngineVector, std::vector<typename MatrixAType::ScalarType>& foreignA,
                      std::vector<typename MatrixBType::ScalarType>& foreignB, bool& serializeKeyA, bool& serializeKeyB);

  template<typename MatrixAType, typename MatrixBType, typename tupleStructure>
  static void _start2(MatrixAType& matrixA, MatrixBType& matrixB, tupleStructure& commInfo3D,
                      std::vector<typename MatrixAType::ScalarType>& matrixAEngineVector, std::vector<typename MatrixBType::ScalarType>& matrixBEngineVector,
                      bool& serializeKeyA, bool& serializeKeyB);

  template<typename MatrixType, typename tupleStructure>
  static void _end1(typename MatrixType::ScalarType* matrixEnginePtr, MatrixType& matrix, tupleStructure& commInfo3D, size_t dir = 0);

  template<typename T, typename U>
  static void BroadcastPanels(std::vector<T>& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panelComm);

  template<typename T, typename U>
  static void BroadcastPanels(T*& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panelComm);

  template<typename MatrixSrcType, typename MatrixDestType>
  static void getEnginePtr(MatrixSrcType& matrixArg, MatrixDestType& matrixDest, std::vector<typename MatrixSrcType::ScalarType>& data, bool isRoot);

  template<typename MatrixType>
  static MatrixType getSubMatrix(MatrixType& srcMatrix, typename MatrixType::DimensionType matrixArgColumnStart, typename MatrixType::DimensionType matrixArgColumnEnd,
                                 typename MatrixType::DimensionType matrixArgRowStart, typename MatrixType::DimensionType matrixArgRowEnd, size_t pGridDimensionSize, bool getSub);

};

#include "MM3D.hpp"

#endif /* MM3D_H_ */
