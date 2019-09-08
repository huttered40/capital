/* Author: Edward Hutter */

#ifndef MATRIX_STRUCTURE_H_
#define MATRIX_STRUCTURE_H_

// These class policies implement the Structure Policy

class square{
public:
  template<typename T, typename U>
  static void _Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename T>
  static void _Dissamble(std::vector<T*>& matrix);
  template<typename T, typename U>
  static void _Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename U>
  static inline U _getNumElems(U rangeX, U rangeY) { return rangeX*rangeY; }
};


class rect{
public:
  template<typename T, typename U>
  static void _Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename T>
  static void _Dissamble(std::vector<T*>& matrix);
  template<typename T, typename U>
  static void _Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename U>
  static inline U _getNumElems(U rangeX, U rangeY) { return rangeX*rangeY; }
};

class uppertri{
public:
  template<typename T, typename U>
  static void _Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename T>
  static void _Dissamble(std::vector<T*>& matrix);
  template<typename T, typename U>
  static void _Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename U>
  static inline U _getNumElems(U rangeX, U rangeY) { return ((rangeX*(rangeX+1))>>1); }
};

class lowertri{
public:
  template<typename T, typename U>
  static void _Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename T>
  static void _Dissamble(std::vector<T*>& matrix);
  template<typename T, typename U>
  static void _Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename U>
  static inline U _getNumElems(U rangeX, U rangeY) { return ((rangeX*(rangeX+1))>>1); }
};

#include "structure.hpp"

#endif /* MATRIX_STRUCTURE_H_ */