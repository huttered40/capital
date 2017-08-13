/* Author: Edward Hutter */

#ifndef MATRIX_ALLOCATOR_H_
#define MATRIX_ALLOCATOR_H_

// System includes
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>


// Local includes

// Create the many Allocator classes that I want to experiment with.
//   See MatrixDistributer.h for discussions on the format of this code.

// These class policies implement the Allocator Policy, which I could
//   define at a later time.

// For now, I will assume that the data structure utilized in this Policy
//   will be a vector<T*>. We can change and generalize this later as needed.

template<typename T, typename U>
class MatrixAllocatorSquare
{
public:
  // Prevent all compiler-generated constructors/destructors
  MatrixAllocatorSquare() = delete;
  MatrixAllocatorSquare(const MatrixAllocatorSquare& rhs) = delete;
  MatrixAllocatorSquare(MatrixAllocatorSquare&& rhs) = delete;
  MatrixAllocatorSquare& operator=(const MatrixAllocatorSquare& rhs) = delete;
  MatrixAllocatorSquare& operator=(MatrixAllocatorSquare&& rhs) = delete;
  ~MatrixAllocatorSquare() = delete;

  static void Allocate();
  static void Construct();
  static void Assemble(std::vector<T*>& matrix, U dimensionX, U dimensionY);
  static void Deallocate();
  static void Destroy();
  static void Dissamble(std::vector<T*>& matrix);
  static void Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY);
  static void Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
};


template<typename T, typename U>
class MatrixAllocatorRectangle
{
public:
  // Prevent all compiler-generated constructors/destructors
  MatrixAllocatorRectangle() = delete;
  MatrixAllocatorRectangle(const MatrixAllocatorRectangle& rhs) = delete;
  MatrixAllocatorRectangle(MatrixAllocatorRectangle&& rhs) = delete;
  MatrixAllocatorRectangle& operator=(const MatrixAllocatorRectangle& rhs) = delete;
  MatrixAllocatorRectangle& operator=(MatrixAllocatorRectangle&& rhs) = delete;
  ~MatrixAllocatorRectangle() = delete;

  static void Allocate();
  static void Construct();
  static void Assemble(std::vector<T*>& matrix, U dimensionX, U dimensionY);
  static void Deallocate();
  static void Destroy();
  static void Dissamble(std::vector<T*>& matrix);
  static void Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY);
  static void Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
};

template<typename T, typename U>
class MatrixAllocatorUpperTriangular
{
public:
  // Prevent all compiler-generated constructors/destructors
  MatrixAllocatorUpperTriangular() = delete;
  MatrixAllocatorUpperTriangular(const MatrixAllocatorUpperTriangular& rhs) = delete;
  MatrixAllocatorUpperTriangular(MatrixAllocatorUpperTriangular&& rhs) = delete;
  MatrixAllocatorUpperTriangular& operator=(const MatrixAllocatorUpperTriangular& rhs) = delete;
  MatrixAllocatorUpperTriangular& operator=(MatrixAllocatorUpperTriangular&& rhs) = delete;
  ~MatrixAllocatorUpperTriangular() = delete;

  static void Allocate();
  static void Construct();
  static void Assemble(std::vector<T*>& matrix, U dimensionX, U dimensionY);
  static void Deallocate();
  static void Destroy();
  static void Dissamble(std::vector<T*>& matrix);
  static void Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY);
  static void Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
};

template<typename T, typename U>
class MatrixAllocatorLowerTriangular
{
public:
  // Prevent all compiler-generated constructors/destructors
  MatrixAllocatorLowerTriangular() = delete;
  MatrixAllocatorLowerTriangular(const MatrixAllocatorLowerTriangular& rhs) = delete;
  MatrixAllocatorLowerTriangular(MatrixAllocatorLowerTriangular&& rhs) = delete;
  MatrixAllocatorLowerTriangular& operator=(const MatrixAllocatorLowerTriangular& rhs) = delete;
  MatrixAllocatorLowerTriangular& operator=(MatrixAllocatorLowerTriangular&& rhs) = delete;
  ~MatrixAllocatorLowerTriangular() = delete;

  static void Allocate();
  static void Construct();
  static void Assemble(std::vector<T*>& matrix, U dimensionX, U dimensionY);
  static void Deallocate();
  static void Destroy();
  static void Dissamble(std::vector<T*>& matrix);
  static void Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY);
  static void Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
};

#include "MatrixAllocator.hpp"

#endif /* MATRIX_ALLOCATOR_H_ */
