#ifndef CHOLESKY__POLICY__CHOLINV
#define CHOLESKY__POLICY__CHOLINV

namespace cholesky{
namespace policy{
namespace cholinv{

// Policy classes for the policy describing whether or not to serialize from symmetric Gram matrix
//   to triangular matrix before AllReduction.

class Serialize;
class NoSerialize;

class SaveIntermediates;
class FlushIntermediates;

class NoOverlap;
class OverlapComp;
class OverlapComm;

// ***********************************************************************************************************************************************************************
/*
template<class PolicyClass>
class OverlapGatherPolicyClass{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& Matrix, std::vector<typename MatrixType::ScalarType>& blocked, typename MatrixType::ScalarType* cyclic, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    using U = typename MatrixType::DimensionType;
    U localDimension = Matrix.num_columns_local();
    MPI_Allgather(Matrix.data(), Matrix.num_elems(), mpi_type<T>::type, &blocked[0], Matrix.num_elems(), mpi_type<T>::type, CommInfo.slice);
    util::block_to_cyclic(blocked, cyclic, localDimension, localDimension, CommInfo.d, 'U');
    return;
  }
};

template<>
class OverlapGatherPolicyClass<OverlapGather>{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& Matrix, std::vector<typename MatrixType::ScalarType>& blocked, typename MatrixType::ScalarType* cyclic, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    using U = typename MatrixType::DimensionType;
    using Distribution = typename MatrixType::DistributionType;
    using Offload = typename MatrixType::OffloadType;
    U localDimension = Matrix.num_columns_local();
    // initiate distribution of allgather into chunks of local columns, multiples of localDimension
    std::vector<MPI_Request> req(CommInfo.num_chunks);
    std::vector<MPI_Status> stat(CommInfo.num_chunks);
    U offset = localDimension*(localDimension%CommInfo.num_chunks);
    U progress=0;
    for (size_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Iallgather(Matrix.data()+progress, idx==(CommInfo.num_chunks-1) ? localDimension*(localDimension/CommInfo.num_chunks+offset) : localDimension*(localDimension/CommInfo.num_chunks),
                     mpi_type<T>::type, &blocked[progress], idx==(CommInfo.num_chunks-1) ? localDimension*(localDimension/CommInfo.num_chunks+offset) : localDimension*(localDimension/CommInfo.num_chunks),
                     mpi_type<T>::type, CommInfo.slice, &req[idx]);
      progress += localDimension * (localDimension/CommInfo.num_chunks);
    }
    // initiate distribution along columns and complete distribution across rows
    progress=0;
    for (size_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Wait(&req[idx],&stat[idx]);
      util::block_to_cyclic(&blocked[progress], &cyclic[progress], localDimension,
                            idx==(CommInfo.num_chunks-1) ? (localDimension+offset)/CommInfo.num_chunks : localDimension/CommInfo.num_chunks, CommInfo.d);
      progress += (localDimension * (localDimension/CommInfo.num_chunks))*CommInfo.d*CommInfo.d;
    }
    return;
  }
};
*/
// ***********************************************************************************************************************************************************************

// ***********************************************************************************************************************************************************************
class Serialize{
public:
  using structure = uppertri;

  template<typename TriMatrixType, typename SquareMatrixType, typename CommType>
  static void invoke(TriMatrixType& matrix, std::vector<typename TriMatrixType::ScalarType>& blocked, SquareMatrixType& cyclic, CommType&& CommInfo){
    using T = typename TriMatrixType::ScalarType;
    using U = typename TriMatrixType::DimensionType;
    U localDimension = matrix.num_columns_local();
    MPI_Allgather(matrix.data(), matrix.num_elems(), mpi_type<T>::type, &blocked[0], matrix.num_elems(), mpi_type<T>::type, CommInfo.slice);
    util::block_to_cyclic(blocked, cyclic.data(), localDimension, localDimension, CommInfo.d, 'U');
    return;
  }
};

class NoSerialize{
public:
  using structure = square;	// might need 'rect'

  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& matrix, std::vector<typename MatrixType::ScalarType>& blocked, MatrixType& cyclic, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    using U = typename MatrixType::DimensionType;
    U localDimension = matrix.num_columns_local();
    MPI_Allgather(matrix.data(), matrix.num_elems(), mpi_type<T>::type, &blocked[0], matrix.num_elems(), mpi_type<T>::type, CommInfo.slice);
    util::block_to_cyclic(&blocked[0], cyclic.data(), localDimension, localDimension, CommInfo.d);
    return;
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

// ***********************************************************************************************************************************************************************
class NoOverlap{
public:
  static void invoke_stage1(){
  }
  static void invoke_stage2(){
  }
  static void invoke_stage3(){
  }
  static void invoke_stage4(){
  }
};
class OverlapComp{
};
class OverlapComm{
};

};
};
};
#endif // CHOLESKY__POLICY__CHOLINV
