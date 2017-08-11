/* Author: Edward Hutter */

#ifndef MATRIX_H_
#define MATRIX_H_

// System includes
#include <vector>
#include <iostream>

// Local includes -- the policy classes
#include "MatrixDistributer.h"
#include "MatrixAllocator.h"
// Local includes -- non-policy classes
#include "MatrixSerializer.h"

/*
  Divide the Matrix class into policies.
  As of right now, I can identify 2 policies -> creation and distribution
  As explicitey stated in Modern C++ Design, each policiy should be treated as
  its own set of classes, with each different policiy choice called a policy class
  and given its own class.

  I do not want to overload the class template with more than 6 parameters. But I do want
  to give the user of the Matrix class options as to how to build and distribute its data.
  As of right now, the best design choice I can come up with would be to add another template
  parameter to the Matrix class, and write a family of classes for each policy, where each policy class
  can be simple. Each can have just a single static method and take as arguments a reference to a Matrix.
  I choose static methods so that we don't need a class instance in order to use the class, as the class will
  not have any member variables either (only static would work anyways with a single static method). In addition,
  the policy class will be treated as a friend class so that we can make the static method a protected member and prevent
  the user from directly using the class.
*/
template<typename T, typename U, class Allocator, class Distributer>
class Matrix
{
public:

  // Mark these two classes as friends so that we can use their protected members.
  friend Allocator;
  friend Distributer;
  friend MatrixSerializer;

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
  void Serialize(const Matrix& matrix);
  void Serialize(Matrix&& matrix);

  // Host method, will call Allocator<T,U,?>::Distribute(...) method
  void Distribute();

  // Just a local print. For a distributed print, we must implement another class that takes the Matrix and operates on it.
  //   That is not something that this class policy needs to worry about.
  void print();

  // create my own allocator class using the template parameter? Yes, I want to do this in the future.

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
