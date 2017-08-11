/* Author: Edward Hutter */

#ifndef MATRIX_DISTRIBUTER_H_
#define MATRIX_DISTRIBUTER_H_

// System includes
#include <iostream>
#include <vector>

// Local includes -> Note that this class should need to know nothing about the Matrix class.

// In order to use a partial specialization of a fully templated class, we must first
//   define a partial specialized class, and in order to do that, we must declare/define a
//   templated 'host' class. Note that we can however define a fully specialized class member
//   method, but that is not what we want here.
template<typename T, typename U, typename Z>
class MatrixDistributerCyclic;

// Partially specialized class that will allow us to use the partially specialized method.
template<typename T, typename U>
class MatrixDistributerCyclic<T,U,std::vector<T*>>
{
public:
  // Prevent anyone from instantiating this class with any default constructor.
  // It is used solely to separate the policy of Matrix distribution
  //   from the Matrix class.
  MatrixDistributerCyclic() = delete;
  MatrixDistributerCyclic(const MatrixDistributerCyclic& rhs) = delete;
  MatrixDistributerCyclic(MatrixDistributerCyclic&& rhs) = delete;
  MatrixDistributerCyclic<T,U,std::vector<T*>> operator=(const MatrixDistributerCyclic& rhs) = delete;
  MatrixDistributerCyclic<T,U,std::vector<T*>> operator=(MatrixDistributerCyclic&& rhs) = delete;
  ~MatrixDistributerCyclic() = delete;

protected:
  // Reason why we must provide a templated parameter here is because
  //   we don't know exactly what type the matrix argument is.
  // This actually might cause problems down the road, because the Distribute()
  //   method will be written a certain way with a certain data structure for matrix
  //   in mind. So once we start modifying the data structures, we might want to rethink
  //   how we either implement Distribute or how to template this class/method.
  // Wait, I just thought of a potential solution: since the Matrix class is the only class
  //   that has access to this method, what we can do, is we can put a separate template purely on this
  //   method, and perform a template specialization so as to implement only specific data structures.
  // No! This will not work because templated class with more than a single template parameter can not have
  //   templated member methods.
  // So, I think the design choice for now will be add the template parameter to the class itself (Z), and
  //   then specialize just that template parameter and implement the cyclic distribution routine just for
  //   that data structure Z.
  static void Distribute(std::vector<T*>& matrix, U dimension, U dimensionY);
};

// We must provide the template class definitions below. Read up on why exactly we need to do this.
#include "MatrixDistributer.hpp"

#endif /* MATRIX_DISTRIBUTER_H_ */
