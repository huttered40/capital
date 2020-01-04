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
class serialize<rect,rect>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                     typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer=0, size_t dest_buffer=0);
};

template<>
class serialize<rect,uppertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                     typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer=0, size_t dest_buffer=0);
};

template<>
class serialize<rect,lowertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                     typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer=0, size_t dest_buffer=0);
};

template<>
class serialize<uppertri,rect>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                     typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer=0, size_t dest_buffer=0);
};

template<>
class serialize<uppertri,uppertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                     typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer=0, size_t dest_buffer=0);
};

template<>
class serialize<lowertri,rect>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                     typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer=0, size_t dest_buffer=0);
};

template<>
class serialize<lowertri,lowertri>{
public:
  template<typename SrcType, typename DestType>
  static void invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                     typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer=0, size_t dest_buffer=0);
};

#include "serialize.hpp"

#endif /* MATRIX_SERIALIZE_H_ */
