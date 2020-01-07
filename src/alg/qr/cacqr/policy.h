#ifndef QR__POLICY__CACQR
#define QR__POLICY__CACQR

namespace qr{
namespace policy{
namespace cacqr{

// ***********************************************************************************************************************************************************************
class NoSerialize{
protected:
  using structure = rect;

  template<typename MatrixType, typename BufferType>
  static MatrixType& buffer(MatrixType& Matrix, BufferType& buffer){
    return Matrix; 
  }

  template<typename MatrixType, typename BufferType, typename CommType>
  static void compute_gram(MatrixType& Matrix, BufferType& buffer, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    auto localDimensionN = Matrix.num_columns_local();
    MPI_Allreduce(MPI_IN_PLACE, Matrix.data(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, CommInfo.world);
    return;
  }

  template<typename MatrixType, typename BufferType>
  static void save_R_1d(MatrixType& Matrix, BufferType& buffer){
    auto num_rows = Matrix.num_rows_local(); auto num_columns = Matrix.num_columns_local();
    serialize<uppertri,uppertri>::invoke(Matrix,buffer,0,num_columns,0,num_rows,0,num_columns,0,num_rows);
  }

  template<typename MatrixType, typename BufferType>
  static typename BufferType::ScalarType* retrieve_intermediate_R_1d(MatrixType& Matrix, BufferType& buffer){
    return buffer.data();
  }

  template<typename MatrixType, typename BufferType>
  static typename MatrixType::ScalarType* retrieve_final_R_1d(MatrixType& Matrix, BufferType& buffer){
    return Matrix.data();
  }

  template<typename MatrixType, typename BufferType1, typename BufferType2>
  static void save_R_3d(MatrixType& Matrix, BufferType1& buffer1, BufferType2& buffer2){
    auto num_rows = Matrix.num_rows_local(); auto num_columns = Matrix.num_columns_local();
    serialize<uppertri,uppertri>::invoke(Matrix,buffer2,0,num_columns,0,num_rows,0,num_columns,0,num_rows);
  }

  template<typename MatrixType, typename BufferType>
  static BufferType& retrieve_intermediate_R_3d(MatrixType& Matrix, BufferType& buffer){
    return buffer;
  }

  template<typename MatrixType, typename BufferType>
  static MatrixType& retrieve_final_R_3d(MatrixType& Matrix, BufferType& buffer){
    return Matrix;
  }

  template<typename MatrixType, typename BufferType>
  static void complete_1d(MatrixType& Matrix, BufferType& buffer){}

  template<typename MatrixType, typename BufferType>
  static void transfer_start(MatrixType& Matrix, BufferType& buffer){}

  template<typename MatrixType>
  static void transfer_end(MatrixType& Matrix){}
};

class Serialize{
protected:
  using structure = uppertri;

  template<typename MatrixType, typename BufferType>
  static BufferType& buffer(MatrixType& Matrix, BufferType& buffer){
    return buffer; 
  }

  template<typename MatrixType, typename BufferType, typename CommType>
  static void compute_gram(MatrixType& Matrix, BufferType& buffer, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    auto num_rows = Matrix.num_rows_local(); auto num_columns = Matrix.num_columns_local();
    serialize<uppertri,uppertri>::invoke(buffer,Matrix,0,num_columns,0,num_rows,0,num_columns,0,num_rows);
    MPI_Allreduce(MPI_IN_PLACE, Matrix.data(), Matrix.num_elems(), mpi_type<T>::type, MPI_SUM, CommInfo.world);
    serialize<uppertri,uppertri>::invoke(Matrix,buffer,0,num_columns,0,num_rows,0,num_columns,0,num_rows);
    return;
  }

  template<typename MatrixType, typename BufferType>
  static void save_R_1d(MatrixType& Matrix, BufferType& buffer){
    auto num_rows = Matrix.num_rows_local(); auto num_columns = Matrix.num_columns_local();
    serialize<uppertri,uppertri>::invoke(buffer,Matrix,0,num_columns,0,num_rows,0,num_columns,0,num_rows,0,2);
  }

  template<typename MatrixType, typename BufferType>
  static typename MatrixType::ScalarType* retrieve_intermediate_R_1d(MatrixType& Matrix, BufferType& buffer){
    return Matrix.pad();
  }

  template<typename MatrixType, typename BufferType>
  static typename MatrixType::ScalarType* retrieve_final_R_1d(MatrixType& Matrix, BufferType& buffer){
    return buffer.data();
  }

  template<typename MatrixType, typename BufferType1, typename BufferType2>
  static void save_R_3d(MatrixType& Matrix, BufferType1& buffer1, BufferType2& buffer2){
    auto num_rows = Matrix.num_rows_local(); auto num_columns = Matrix.num_columns_local();
    serialize<uppertri,uppertri>::invoke(Matrix,buffer1,0,num_columns,0,num_rows,0,num_columns,0,num_rows,0,0);
  }

  template<typename MatrixType, typename BufferType>
  static MatrixType& retrieve_intermediate_R_3d(MatrixType& Matrix, BufferType& buffer){
    return Matrix;
  }

  template<typename MatrixType, typename BufferType>
  static BufferType& retrieve_final_R_3d(MatrixType& Matrix, BufferType& buffer){
    return buffer;
  }

  template<typename MatrixType, typename BufferType>
  static void complete_1d(MatrixType& Matrix, BufferType& buffer){
    auto num_rows = Matrix.num_rows_local(); auto num_columns = Matrix.num_columns_local();
    serialize<uppertri,uppertri>::invoke(buffer,Matrix,0,num_columns,0,num_rows,0,num_columns,0,num_rows);
  }

  template<typename MatrixType, typename BufferType>
  static void transfer_start(MatrixType& Matrix, BufferType& buffer){
    Matrix.swap();
    auto num_rows = Matrix.num_rows_local(); auto num_columns = Matrix.num_columns_local();
    serialize<uppertri,uppertri>::invoke(buffer,Matrix,0,num_columns,0,num_rows,0,num_columns,0,num_rows,0,0);
  }

  template<typename MatrixType>
  static void transfer_end(MatrixType& Matrix){
    Matrix.swap();
  }
};
// ***********************************************************************************************************************************************************************

// ***********************************************************************************************************************************************************************
class SaveIntermediates{
protected:
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
protected:
  template<typename TableType, typename KeyType, typename... ValueTypes>
  static void init(TableType& table, KeyType&& key, ValueTypes&&... values){
    if (table.find(key) == table.end()){
      table.emplace(std::piecewise_construct,std::forward_as_tuple(std::forward<KeyType>(key)),std::forward_as_tuple(std::forward<ValueTypes>(values)...,true));
    }
  }

  template<typename TableType, typename KeyType>
  static inline typename TableType::mapped_type& invoke(TableType& table, KeyType&& key){
    table[std::forward<KeyType>(key)]._fill_();
    return table[std::forward<KeyType>(key)];
  }

  template<typename MatrixType>
  static void flush(MatrixType& matrix){
    matrix._destroy_();
  }
};
// ***********************************************************************************************************************************************************************

};
};
};
#endif // QR__POLICY__CACQR
