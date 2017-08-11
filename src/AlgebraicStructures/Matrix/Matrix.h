/* Author: Edward Hutter */

#ifndef MATRIX_H_
#define MATRIX_H_

/* system includes */
#include <vector>

template<typename T, typename U>
class Matrix
{
public:
  explicit Matrix() = delete;
  explicit Matrix(U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY);
  explicit Matrix(const Matrix& rhs);
  explicit Matrix(Matrix&& rhs);
  Matrix& operator=(const Matrix& rhs);
  Matrix& operator=(Matrix&& rhs);
  ~Matrix();

  //I want to have a serialize method that will give me say an upper triangular
  //  we can do this without creating another full object or copying by just iterating over the pointer offsets and modifying them.
  //  We should overload the method so that there can be 2 ways: copying into a new matrix and moving into a new matrix.
  void serialize(const Matrix& rhs);
  void serialize(Matrix&& rhs);

  // Later on: Listen to the red book and design a distribution policy and implement policy classes that implement specific distribution strategies
  //           so that the user of the Matrix class can choose the kind of distribution it wants.
  void distributeCyclic();

  // Just a local print. For a distributed print, we must implement another class that takes the Matrix and operates on it.
  //   That is not something that this class policy needs to worry about.
  void print();

  // create my own allocator class?

private:

  void copy(const Matrix& rhs);
  void mover(Matrix&& rhs);		// will need to use std::forward<T> with this I think.

  std::vector<T*> _matrix;
  U _dimensionX;
  U _dimensionY;
  // This Matrix is most likely a sub-matrix of a global Matrix partitioned among many processors.
  U _globalDimensionX;
  U _globalDimensionY;
};

#include "Matrix.hpp"

#endif /* MATRIX_H_ */
