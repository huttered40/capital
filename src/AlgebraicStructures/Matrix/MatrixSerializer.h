/* Author: Edward Hutter */

#ifndef MATRIX_SERIALIZER_H_
#define MATRIX_SERIALIZER_H_

// System includes
#include <vector>
#include <iostream>
#include <cstring>

// Local includes
#include "MatrixStructure.h"

/*
  Note: Serialize is an engine that can take any Structure combo
  Example: Source is a (cyclically distributed) Upper-triangular matrix and Dest must be a Square (cyclically distributed) matrix
  Future: Need to deal with changing between distributions. Maybe add this to the Distributer Policy.
*/

// Fully templated class is declared, not defined
template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class StructureSource,
  template<typename,typename,template<typename,typename,int> class> class StructureDest>
class Serializer;

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureSquare, MatrixStructureSquare>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureSquare,MatrixStructureUpperTriangular>& operator=(Serializer&& rhs) = delete;

  static void Serialize(T* src, T*& dest, U dimensionX, U dimensionY);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend);
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureSquare,MatrixStructureUpperTriangular>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureSquare,MatrixStructureUpperTriangular>& operator=(Serializer&& rhs) = delete;

  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, bool fillZeros);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros);
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>& operator=(Serializer&& rhs) = delete;

  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, bool fillZeros);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros);
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureUpperTriangular,MatrixStructureSquare>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureUpperTriangular,MatrixStructureSquare>& operator=(Serializer&& rhs) = delete;

  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros);
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureUpperTriangular,MatrixStructureUpperTriangular>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureUpperTriangular,MatrixStructureUpperTriangular>& operator=(Serializer&& rhs) = delete;

  static void Serialize(T* src, T*& dest, U dimensionX, U dimensionY);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend);
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureLowerTriangular,MatrixStructureSquare>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureLowerTriangular,MatrixStructureSquare>& operator=(Serializer&& rhs) = delete;

  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros);
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureLowerTriangular,MatrixStructureLowerTriangular>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureLowerTriangular,MatrixStructureLowerTriangular>& operator=(Serializer&& rhs) = delete;

  static void Serialize(T* src, T*& dest, U dimensionX, U dimensionY);
  static void Serialize(const T* src, T*& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend);
};

#include "MatrixSerializer.hpp"

#endif /* MATRIX_SERIALIZE_H_ */
