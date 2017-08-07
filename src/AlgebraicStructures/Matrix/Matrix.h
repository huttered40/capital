/* Author: Edward Hutter */

#ifndef MATRIX_H_
#define MATRIX_H_

/* system includes */
#include <vector>

template<typename T, typename U>
class Matrix
{
public:
  Matrix();
  Matrix(U dimensionX, U dimensionY);
  Matrix(const Matrix& rhs);
  Matrix(Matrix&& rhs);
  Matrix& operator=(const Matrix& rhs);
  Matrix& operator=(Matrix&& rhs);
  ~Matrix();

  .. I want to have a serialize method that will give me say an upper triangular
  .. we can do this without creating another full object or copying by just iterating over the pointer offsets and modifying them.

  // create my own allocator class?

private:

  void copy(const Matrix& rhs);
  void mover(Matrix&& rhs);		// will need to use std::forward<T> with this I think.

  std::vector<T*> _matrix;
  U _dimensionX;
  U _dimensionY;
};

#include "Matrix.hpp"

#endif /* MATRIX_H_ */
