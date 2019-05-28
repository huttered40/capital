/* Author: Edward Hutter */

#ifndef MATRIX_SERIALIZER_H_
#define MATRIX_SERIALIZER_H_

/*
  Note: Serialize is an engine that can take any Structure combo
  Example: Source is a (cyclically distributed) Upper-triangular matrix and Dest must be a Square (cyclically distributed) matrix
  TODO: Future: Do we need to deal with changing between distributions? Maybe add this to the Distributer Policy?
*/

// Fully templated class is declared, not defined
template<typename Structure1, typename Structure2>
class Serializer;

template<>
class Serializer<Square,Square>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<Square,Rectangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<Square,UpperTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<Square,LowerTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<Rectangular,Square>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<Rectangular,Rectangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<Rectangular,UpperTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<Rectangular,LowerTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<UpperTriangular,Square>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<UpperTriangular,Rectangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<UpperTriangular,UpperTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<LowerTriangular,Square>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<LowerTriangular,Rectangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class Serializer<LowerTriangular,LowerTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void Serialize(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

#include "MatrixSerializer.hpp"

#endif /* MATRIX_SERIALIZE_H_ */
