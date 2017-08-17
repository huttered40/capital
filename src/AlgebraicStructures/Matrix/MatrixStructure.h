/* Author: Edward Hutter */

#ifndef MATRIX_STRUCTURE_H_
#define MATRIX_STRUCTURE_H_

// System includes
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>


// Local includes
#include "MatrixDistributer.h"

// Create the many Structure classes that I want to experiment with.
//   See MatrixDistributer.h for discussions on the format of this code.

// These class policies implement the Structure Policy, which I could
//   define at a later time.

// For now, I will assume that the data structure utilized in this Policy
//   will be a vector<T*>. We can change and generalize this later as needed.

template<typename T, typename U, template<typename,typename,int> class Distributer>
class MatrixStructureSquare
{
public:
  // Prevent all compiler-generated constructors/destructors
  MatrixStructureSquare() = delete;
  MatrixStructureSquare(const MatrixStructureSquare& rhs) = delete;
  MatrixStructureSquare(MatrixStructureSquare&& rhs) = delete;
  MatrixStructureSquare& operator=(const MatrixStructureSquare& rhs) = delete;
  MatrixStructureSquare& operator=(MatrixStructureSquare&& rhs) = delete;
  ~MatrixStructureSquare() = delete;

  static void Allocate();
  static void Construct();
  static void Assemble(std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  static void Deallocate();
  static void Destroy();
  static void Dissamble(std::vector<T*>& matrix);
  static void Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY);
  static void DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimY);
  static void DistributeSymmetric(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimY, bool diagonallyDominant);
  static void Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
};


template<typename T, typename U, template<typename,typename,int> class Distributer>
class MatrixStructureRectangle
{
public:
  // Prevent all compiler-generated constructors/destructors
  MatrixStructureRectangle() = delete;
  MatrixStructureRectangle(const MatrixStructureRectangle& rhs) = delete;
  MatrixStructureRectangle(MatrixStructureRectangle&& rhs) = delete;
  MatrixStructureRectangle& operator=(const MatrixStructureRectangle& rhs) = delete;
  MatrixStructureRectangle& operator=(MatrixStructureRectangle&& rhs) = delete;
  ~MatrixStructureRectangle() = delete;

  static void Allocate();
  static void Construct();
  static void Assemble(std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  static void Deallocate();
  static void Destroy();
  static void Dissamble(std::vector<T*>& matrix);
  static void Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY);
  static void DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimY);
  static void Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
};

template<typename T, typename U, template<typename,typename,int> class Distributer>
class MatrixStructureUpperTriangular
{
public:
  // Prevent all compiler-generated constructors/destructors
  MatrixStructureUpperTriangular() = delete;
  MatrixStructureUpperTriangular(const MatrixStructureUpperTriangular& rhs) = delete;
  MatrixStructureUpperTriangular(MatrixStructureUpperTriangular&& rhs) = delete;
  MatrixStructureUpperTriangular& operator=(const MatrixStructureUpperTriangular& rhs) = delete;
  MatrixStructureUpperTriangular& operator=(MatrixStructureUpperTriangular&& rhs) = delete;
  ~MatrixStructureUpperTriangular() = delete;

  static void Allocate();
  static void Construct();
  static void Assemble(std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  static void Deallocate();
  static void Destroy();
  static void Dissamble(std::vector<T*>& matrix);
  static void Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY);
  static void DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimY);
  static void Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
};

template<typename T, typename U, template<typename,typename,int> class Distributer>
class MatrixStructureLowerTriangular
{
public:
  // Prevent all compiler-generated constructors/destructors
  MatrixStructureLowerTriangular() = delete;
  MatrixStructureLowerTriangular(const MatrixStructureLowerTriangular& rhs) = delete;
  MatrixStructureLowerTriangular(MatrixStructureLowerTriangular&& rhs) = delete;
  MatrixStructureLowerTriangular& operator=(const MatrixStructureLowerTriangular& rhs) = delete;
  MatrixStructureLowerTriangular& operator=(MatrixStructureLowerTriangular&& rhs) = delete;
  ~MatrixStructureLowerTriangular() = delete;

  static void Allocate();
  static void Construct();
  static void Assemble(std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  static void Deallocate();
  static void Destroy();
  static void Dissamble(std::vector<T*>& matrix);
  static void Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY);
  static void DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimY);
  static void Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
};

#include "MatrixStructure.hpp"

#endif /* MATRIX_STRUCTURE_H_ */
