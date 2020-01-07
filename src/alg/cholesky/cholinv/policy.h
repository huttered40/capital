#ifndef CHOLESKY__POLICY__CHOLINV
#define CHOLESKY__POLICY__CHOLINV

namespace cholesky{
namespace policy{
namespace cholinv{

// Policy classes for the policy describing whether or not to serialize from symmetric Gram matrix
//   to triangular matrix before AllReduction.

// ***********************************************************************************************************************************************************************
/*
template<class PolicyClass>
class OverlapGatherPolicyClass{
protected:
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
protected:
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
protected:
  using structure = uppertri;

  template<typename TriMatrixType, typename SquareMatrixType, typename CommType>
  static void invoke(TriMatrixType& matrix, std::vector<typename TriMatrixType::ScalarType>& blocked, SquareMatrixType& cyclic, CommType&& CommInfo){
    using T = typename TriMatrixType::ScalarType;
    auto localDimension = matrix.num_columns_local();
    MPI_Allgather(matrix.data(), matrix.num_elems(), mpi_type<T>::type, &blocked[0], matrix.num_elems(), mpi_type<T>::type, CommInfo.slice);
    util::block_to_cyclic(blocked, cyclic.data(), localDimension, localDimension, CommInfo.d, 'U');
    return;
  }
};

class NoSerialize{
protected:
  using structure = rect;

  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& matrix, std::vector<typename MatrixType::ScalarType>& blocked, MatrixType& cyclic, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    auto localDimension = matrix.num_columns_local();
    MPI_Allgather(matrix.data(), matrix.num_elems(), mpi_type<T>::type, &blocked[0], matrix.num_elems(), mpi_type<T>::type, CommInfo.slice);
    util::block_to_cyclic(&blocked[0], cyclic.data(), localDimension, localDimension, CommInfo.d);
    return;
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

// ***********************************************************************************************************************************************************************
class NoPipeline{
protected:
  template<typename TableType1, typename TableType2, typename TableType3, typename ArgType, typename CommType>
  static void initiate(TableType1& t1, TableType2& t2, TableType3& t3, ArgType& args, CommType&& CommInfo){
    using T = typename ArgType::ScalarType; using ArgTypeRR = typename std::remove_reference<ArgType>::type;
    auto split1 = (args.localDimension>>args.split); split1 = util::get_next_power2(split1); auto split2 = args.localDimension-split1;
    blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, -1., 1.);
    serialize<uppertri,uppertri>::invoke(args.R, t1, args.AstartX+split1, args.AendX, args.AstartY+split1, args.AendY,0,split2,0,split2);
    matmult::summa::invoke(t2, t3, t1, std::forward<CommType>(CommInfo), syrkArgs);
    serialize<uppertri,uppertri>::invoke(t1, args.R, 0,split2,0,split2,args.AstartX+split1, args.AendX, args.AstartY+split1, args.AendY);
  }

  template<typename ArgType, typename CommType>
  static void update_panel(ArgType& args, CommType&& CommInfo){
    // needs to be some recognition that the top panel does not need updating
  }
};
class Pipeline{
protected:
  template<typename TableType1, typename TableType2, typename TableType3, typename ArgType, typename CommType>
  static void initiate(TableType1& t1, TableType2& t2, TableType3& t3, ArgType& args, CommType&& CommInfo){
  }

  template<typename ArgType, typename CommType>
  static void update_panel(ArgType& args, CommType&& CommInfo){
    // needs to be some recognition that the top panel does not need updating
  }
};

};
};
};
#endif // CHOLESKY__POLICY__CHOLINV
