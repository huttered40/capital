/* Author: Edward Hutter */

#ifndef UTIL_H_
#define UTIL_H_

#include <vector>

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
};

#include "util.hpp"
#endif /*UTIL_H_*/
