/* Author: Edward Hutter */

#ifndef MATRIX_SERIALIZER_H_
#define MATRIX_SERIALIZER_H_

// System includes
#include <vector>
#include <iostream>
#include <cstring>

// Local includes
#include "./../Util/shared.h"
#include "./../Timer/Timer.h"
#include "Matrix.h"
#include "MatrixStructure.h"		// Should be included from within Matrix.h, but whatever because I used a ifndef

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

  // Need to provide an extra template parameter so that this class knows what Matrix template type it is dealing with.
  //   Otherwise, Distributer is just a tag with no meaning. With the overloaded templated template class method, Distributer will be able
  //   to stand for a template class that takes 3 template parameters as shown above.
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare, Distributer>& src,Matrix<T,U,MatrixStructureSquare,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);


private:
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureSquare, MatrixStructureRectangle>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureSquare,MatrixStructureRectangle>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureSquare,MatrixStructureRectangle>& operator=(Serializer&& rhs) = delete;

  // Need to provide an extra template parameter so that this class knows what Matrix template type it is dealing with.
  //   Otherwise, Distributer is just a tag with no meaning. With the overloaded templated template class method, Distributer will be able
  //   to stand for a template class that takes 3 template parameters as shown above.
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureRectangle,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare, Distributer>& src,Matrix<T,U,MatrixStructureRectangle,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);


private:
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

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest);

/* I am removing this method. It just doesn't make sense. We cannot allow matrices with UT structure to be square. Its no cheaper than square to square anyway
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest,
    bool fillZeros, bool dir = false);
*/

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);

/*
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir = false);
*/


private:
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

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest);

/*
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest,
    bool fillZeros, bool dir = false);
*/

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);

/*
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src, Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir = false);
*/


private:
};


// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureRectangle, MatrixStructureSquare>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureRectangle,MatrixStructureSquare>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureRectangle,MatrixStructureSquare>& operator=(Serializer&& rhs) = delete;

  // Need to provide an extra template parameter so that this class knows what Matrix template type it is dealing with.
  //   Otherwise, Distributer is just a tag with no meaning. With the overloaded templated template class method, Distributer will be able
  //   to stand for a template class that takes 3 template parameters as shown above.
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureRectangle,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureRectangle, Distributer>& src,Matrix<T,U,MatrixStructureSquare,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);

private:
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureRectangle, MatrixStructureRectangle>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureRectangle,MatrixStructureRectangle>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureRectangle,MatrixStructureRectangle>& operator=(Serializer&& rhs) = delete;

  // Need to provide an extra template parameter so that this class knows what Matrix template type it is dealing with.
  //   Otherwise, Distributer is just a tag with no meaning. With the overloaded templated template class method, Distributer will be able
  //   to stand for a template class that takes 3 template parameters as shown above.
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureRectangle,Distributer>& src, Matrix<T,U,MatrixStructureRectangle,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureRectangle, Distributer>& big,Matrix<T,U,MatrixStructureRectangle,Distributer>& small,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);

private:
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureRectangle, MatrixStructureUpperTriangular>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureRectangle,MatrixStructureUpperTriangular>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureRectangle,MatrixStructureUpperTriangular>& operator=(Serializer&& rhs) = delete;

  // Need to provide an extra template parameter so that this class knows what Matrix template type it is dealing with.
  //   Otherwise, Distributer is just a tag with no meaning. With the overloaded templated template class method, Distributer will be able
  //   to stand for a template class that takes 3 template parameters as shown above.
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureRectangle,Distributer>& src, Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureRectangle, Distributer>& src,Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);

private:
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureRectangle, MatrixStructureLowerTriangular>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureRectangle,MatrixStructureLowerTriangular>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureRectangle,MatrixStructureLowerTriangular>& operator=(Serializer&& rhs) = delete;

  // Need to provide an extra template parameter so that this class knows what Matrix template type it is dealing with.
  //   Otherwise, Distributer is just a tag with no meaning. With the overloaded templated template class method, Distributer will be able
  //   to stand for a template class that takes 3 template parameters as shown above.
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureRectangle,Distributer>& src, Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureRectangle, Distributer>& src,Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);

private:
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

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);

/*
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir = false);
*/


private:
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureRectangle>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureUpperTriangular,MatrixStructureRectangle>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureUpperTriangular,MatrixStructureRectangle>& operator=(Serializer&& rhs) = delete;

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src, Matrix<T,U,MatrixStructureRectangle,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src, Matrix<T,U,MatrixStructureRectangle,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);

/*
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir = false);
*/


private:
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

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src, Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src, Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);


private:
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

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);
/*
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir = false);
*/


private:
};

// Use partial specialization to define certain combinations
template<typename T, typename U>
class Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureRectangle>
{
public:
  // Prevent this class from being instantiated.
  Serializer() = delete;
  Serializer(const Serializer& rhs) = delete;
  Serializer(Serializer&& rhs) = delete;
  Serializer<T,U,MatrixStructureLowerTriangular,MatrixStructureRectangle>& operator=(const Serializer& rhs) = delete;
  Serializer<T,U,MatrixStructureLowerTriangular,MatrixStructureRectangle>& operator=(Serializer&& rhs) = delete;

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src, Matrix<T,U,MatrixStructureRectangle,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src, Matrix<T,U,MatrixStructureRectangle,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);

/*
  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src, Matrix<T,U,MatrixStructureSquare,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir = false);
*/


private:
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

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src, Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest);

  template<template<typename, typename,int> class Distributer>
  static void Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src, Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest,
    U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir = false);


private:
};

#include "MatrixSerializer.hpp"

#endif /* MATRIX_SERIALIZE_H_ */
