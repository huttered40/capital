/* Author: Edward Hutter */

#ifndef MATRIX_SERIALIZE_H_
#define MATRIX_SERIALIZE_H_

/*
  Note: Serialize is an engine that can take any Structure combo
  Example: Source is a (cyclically distributed) Upper-triangular matrix and Dest must be a square (cyclically distributed) matrix
  TODO: Future: Do we need to deal with changing between distributions? Maybe add this to the Distributer Policy?
*/

// Fully templated class is declared, not defined
template<typename Structure1, typename Structure2>
class serialize;

template<>
class serialize<square,square>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<square,rect>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<square,uppertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<square,lowertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<rect,square>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<rect,rect>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<rect,uppertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<rect,lowertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<uppertri,square>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<uppertri,rect>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<uppertri,uppertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<lowertri,square>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<lowertri,rect>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

template<>
class serialize<lowertri,lowertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(SrcType& src, DestType& dest);

  template<typename BigType, typename SmallType>
  static void invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir = false);
};

#include "serialize.hpp"

#endif /* MATRIX_SERIALIZE_H_ */
