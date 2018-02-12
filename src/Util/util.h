/* Author: Edward Hutter */

#ifndef UTIL_H_
#define UTIL_H_

#include <vector>

#include "../Matrix/Matrix.h"

template<typename T, typename U>
class util
{
public:
  util() = delete;
  ~util();
  util(const util& rhs) = delete;
  util(util&& rhs) = delete;
  util& operator=(const util& rhs) = delete;
  util& operator=(util&& rhs) = delete;

  static std::vector<T> blockedToCyclic(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, int pGridDimensionSize);

  template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
  static std::vector<T> getReferenceMatrix(
              Matrix<T,U,StructureArg,Distribution>& myMatrix,
							U key,
							std::tuple<MPI_Comm, int, int, int, int> commInfo
						  );
};

#include "util.hpp"
#endif /*UTIL_H_*/
