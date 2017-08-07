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
  Matrix(const Matrix& rhs);
  Matrix(Matrix&& rhs);
  Matrix& operator=(const Matrix& rhs);
  Matrix& operator=(Matrix&& rhs);
  ~Matrix();

  // create my own allocator class?

private:
  std::vector<T> _data;
  std::vector<T*> _matrix;
  U _dimensionX;
  U _dimensionY;
};

#include "Matrix.hpp"

#endif /* MATRIX_H_ */
