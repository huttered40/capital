#ifndef QR__POLICY__CACQR
#define QR__POLICY__CACQR

namespace qr{
namespace policy{
namespace cacqr{

// ***********************************************************************************************************************************************************************
class NoSerialize{
public:
  using structure = rect;

  template<typename MatrixType, typename BufferType, typename CommType>
  static void gram(MatrixType& Matrix, BufferType& buffer, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType;
    U localDimensionN = Matrix.num_columns_local();
    MPI_Allreduce(MPI_IN_PLACE, Matrix.data(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, CommInfo.world);
    return;
  }

  template<typename MatrixType, typename BufferType>
  static MatrixType& invoke(MatrixType& Matrix, BufferType& buffer){
    return Matrix;
  }

  template<typename MatrixType, typename BufferType>
  static void complete(MatrixType& Matrix, BufferType& buffer){}
};

class Serialize{
public:
  using structure = uppertri;

  template<typename MatrixType, typename BufferType, typename CommType>
  static void gram(MatrixType& Matrix, BufferType& buffer, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using Offload = typename MatrixType::OffloadType;
    U globalDimensionN = Matrix.num_columns_global();
    serialize<rect,structure>::invoke(Matrix, buffer);
    MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.num_elems(), mpi_type<T>::type, MPI_SUM, CommInfo.world);
    serialize<structure,rect>::invoke(buffer,Matrix);
    return;
  }

  template<typename MatrixType, typename BufferType>
  static BufferType& invoke(MatrixType& Matrix, BufferType& buffer){
    serialize<rect,structure>::invoke(Matrix, buffer);
    return buffer;
  }

  template<typename MatrixType, typename BufferType>
  static void complete(MatrixType& Matrix, BufferType& buffer){
    serialize<structure,rect>::invoke(buffer,Matrix);
  }
};
// ***********************************************************************************************************************************************************************

// ***********************************************************************************************************************************************************************
class SaveIntermediates{
public:
  template<typename TableType, typename KeyType, typename... ValueTypes>
  static void init(TableType& table, KeyType&& key, ValueTypes&&... values){
    if (table.find(key) == table.end()){
      table.emplace(std::piecewise_construct,std::forward_as_tuple(std::forward<KeyType>(key)),std::forward_as_tuple(std::forward<ValueTypes>(values)...));
    }
  }

  template<typename TableType, typename KeyType>
  static inline typename TableType::mapped_type& invoke(TableType& table, KeyType&& key){
    return table[std::forward<KeyType>(key)];
  }

  template<typename MatrixType>
  static void flush(MatrixType& matrix){}
};

class FlushIntermediates{
public:
  template<typename TableType, typename KeyType, typename... ValueTypes>
  static void init(TableType& table, KeyType&& key, ValueTypes&&... values){
    if (table.find(key) == table.end()){
      table.emplace(std::piecewise_construct,std::forward_as_tuple(std::forward<KeyType>(key)),std::forward_as_tuple(std::forward<ValueTypes>(values)...,true));
    }
  }

  template<typename TableType, typename KeyType>
  static inline typename TableType::mapped_type& invoke(TableType& table, KeyType&& key){
    table[std::forward<KeyType>(key)].fill();
    return table[std::forward<KeyType>(key)];
  }

  template<typename MatrixType>
  static void flush(MatrixType& matrix){
    matrix.destroy();
  }
};
// ***********************************************************************************************************************************************************************

};
};
};
#endif // QR__POLICY__CACQR
