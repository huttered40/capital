/* Author: Edward Hutter */

#ifndef MATRIX_DISTRIBUTER_H_
#define MATRIX_DISTRIBUTER_H_

// System includes
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>

// Local includes -> Note that this class should need to know nothing about the Matrix class.

// In order to use a partial specialization of a fully templated class, we must first
//   define a partial specialized class, and in order to do that, we must declare/define a
//   templated 'host' class. Note that we can however define a fully specialized class member
//   method, but that is not what we want here.
/*
template<typename T, typename U, typename Z>
class MatrixDistributerCyclic;
*/

// Partially specialized class that will allow us to use the partially specialized method.
// Newsflash! Not partially specialized anymore. I just got rid of the 3rd template parameter. std::vector<T*> is here to stay!
template<typename T, typename U>
class MatrixDistributerCyclic
{
public:
  // Prevent anyone from instantiating this class with any default constructor.
  // It is used solely to separate the policy of Matrix distribution
  //   from the Matrix class.
  MatrixDistributerCyclic() = delete;
  MatrixDistributerCyclic(const MatrixDistributerCyclic& rhs) = delete;
  MatrixDistributerCyclic(MatrixDistributerCyclic&& rhs) = delete;
  MatrixDistributerCyclic<T,U> operator=(const MatrixDistributerCyclic& rhs) = delete;
  MatrixDistributerCyclic<T,U> operator=(MatrixDistributerCyclic&& rhs) = delete;
  ~MatrixDistributerCyclic() = delete;

//protected:
  static void Distribute(std::vector<T*>& matrix, U dimensionX, U dimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimY);
};

// We must provide the template class definitions below. Read up on why exactly we need to do this.
#include "MatrixDistributer.hpp"

#endif /* MATRIX_DISTRIBUTER_H_ */
