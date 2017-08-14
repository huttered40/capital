/* Author: Edward Hutter */

#ifndef MATRIX_SERIALIZER_H_
#define MATRIX_SERIALIZER_H_

// System includes
#include <iostream>
#include <vector>

// Local includes

// See MatrixDistributer.h for discussions on the format of this code.


/*
Notes: I may want to look into the new pass-by-value is cheaper than fill by reference
         due to move semantics and move constructor being called automatically
         since the return value of a function is an rvalue unless its a lvalue reference.

       Also, I may want to look into just changing the base matrix into whatever
         I am serializing into. This would be only if I needed it.

       For now, lets focus on the case where I pass the matrix data structure by reference
         and fill it up.

*/


// Fully templated class is only declared
template<typename T, typename U, int Z>
class MatrixSerializer;

// Partially specialized template classes
template<typename T, typename U>
class MatrixSerializer<T,U,0>
{
public:
  MatrixSerializer() = delete;
  MatrixSerializer(const MatrixSerializer& rhs) = delete;
  MatrixSerializer(MatrixSerializer&& rhs) = delete;
  MatrixSerializer<T,U,0> operator=(const MatrixSerializer& rhs) = delete;
  MatrixSerializer<T,U,0> operator=(MatrixSerializer&& rhs) = delete;
  ~MatrixSerializer() = delete;

  static void Serialize(std::vector<T*>& matrix);
  static void Serialize(std::vector<T*>&& matrix);

};

#include "MatrixSerializer.hpp"

#endif /* MATRIX_SERIALIZER_H_ */
