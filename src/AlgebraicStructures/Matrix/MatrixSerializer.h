/* Author: Edward Hutter */

#ifndef MATRIX_SERIALIZER_H_
#define MATRIX_SERIALIZER_H_

// System includes
#include <iostream>
#include <vector>

// Local includes

// See MatrixDistributer.h for discussions on the format of this code.

template<typename T, typename U, typename Z>
class MatrixSerializer;

template<typename T, typename U>
class MatrixSerializer<T,U,std::vector<T*>>
{
public:
  MatrixSerializer() = delete;
  MatrixSerializer(const MatrixSerializer& rhs) = delete;
  MatrixSerializer(MatrixSerializer&& rhs) = delete;
  MatrixSerializer<T,U,std::vector<T*>> operator=(const MatrixSerializer& rhs) = delete;
  MatrixSerializer<T,U,std::vector<T*>> operator=(MatrixSerializer&& rhs) = delete;
  ~MatrixSerializer() = delete;

protected:
  static void SerializeUpperTriangular(std::vector<T*>& matrix);
  static void SerializeLowerTriangular(std::vector<T*>& matrix);
  static void SerializeUpperTriangular(std::vector<T*>&& matrix);
  static void SerializeLowerTriangular(std::vector<T*>&& matrix);

};

#include "MatrixSerializer.hpp"

#endif /* MATRIX_SERIALIZER_H_ */
