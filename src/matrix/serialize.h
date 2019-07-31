/* Author: Edward Hutter */

#ifndef MATRIX_SERIALIZE_H_
#define MATRIX_SERIALIZE_H_

/*
  Note: Serialize is an engine that can take any Structure combo
  Example: Source is a (cyclically distributed) Upper-triangular matrix and Dest must be a Square (cyclically distributed) matrix
  TODO: Future: Do we need to deal with changing between distributions? Maybe add this to the Distributer Policy?
*/

// Fully templated class is declared, not defined
template<typename Structure1, typename Structure2>
class serialize;

template<>
class serialize<Square,Square>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<Square,Rectangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<Square,UpperTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<Square,LowerTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<Rectangular,Square>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<Rectangular,Rectangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<Rectangular,UpperTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<Rectangular,LowerTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<UpperTriangular,Square>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<UpperTriangular,Rectangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<UpperTriangular,UpperTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<LowerTriangular,Square>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<LowerTriangular,Rectangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<LowerTriangular,LowerTriangular>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

#include "serialize.hpp"

#endif /* MATRIX_SERIALIZE_H_ */
