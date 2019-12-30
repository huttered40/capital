#ifndef CHOLESKY__POLICY__CHOLINV
#define CHOLESKY__POLICY__CHOLINV

namespace cholesky{
namespace policy{
namespace cholinv{

// Policy classes for the policy describing whether or not to serialize from symmetric Gram matrix
//   to triangular matrix before AllReduction.
class GemmUpdate;
class ReduceFlopUpdate;

class SerializeAvoidComm;
class NoSerializeAvoidComm;

class NoIntermediateOverlap;
class IntermediateOverlapComp;
class IntermediateOverlapComm;

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
template<class PolicyClass>
class SerializePolicyClass{
public:
  using structure = uppertri;

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
class SerializePolicyClass<NoSerializeAvoidComm>{
public:
  using structure = square;	// might need 'rect'

  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& Matrix, std::vector<typename MatrixType::ScalarType>& blocked, typename MatrixType::ScalarType* cyclic, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    using U = typename MatrixType::DimensionType;
    U localDimension = Matrix.num_columns_local();
    MPI_Allgather(Matrix.data(), Matrix.num_elems(), mpi_type<T>::type, &blocked[0], Matrix.num_elems(), mpi_type<T>::type, CommInfo.slice);
    util::block_to_cyclic(&blocked[0], cyclic, localDimension, localDimension, CommInfo.d);
    return;
  }
};
// ***********************************************************************************************************************************************************************

};
};
};
#endif // CHOLESKY__POLICY__CHOLINV
