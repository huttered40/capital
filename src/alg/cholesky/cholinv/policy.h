#ifndef CHOLESKY__POLICY__CHOLINV
#define CHOLESKY__POLICY__CHOLINV

namespace cholesky{
namespace policy{
namespace cholinv{

// ***********************************************************************************************************************************************************************
class Serialize{
protected:
  using structure = uppertri;
};

class NoSerialize{
protected:
  using structure = rect;
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
class ReplicateCommComp{
protected:
  template<typename ArgType, typename CommType>
  static void initiate(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgType::ScalarType;
    auto aggregDim = (args.AendX-args.AstartX)*CommInfo.d; auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    auto localDimension = args.base_case_table[index_pair].num_columns_local();
    serialize<uppertri,uppertri>::invoke(args.R, args.base_case_table[index_pair], args.AstartX, args.AendX, args.AstartY, args.AendY,0,args.AendX-args.AstartX,0,args.AendY-args.AstartY);
    MPI_Allgather(args.base_case_table[index_pair].data(), args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, &args.base_case_blocked_table[index_pair][0],
                  args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, CommInfo.slice);
    if (std::is_same<typename ArgTypeRR::SP,Serialize>::value){
      util::block_to_cyclic_triangle(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(),
                                     args.base_case_blocked_table[index_pair].size(), localDimension, localDimension, CommInfo.d);
    } else{
      util::block_to_cyclic_rect(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(), localDimension, localDimension, CommInfo.d);
    }
  }

  template<typename ArgType, typename CommType>
  static void compute(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgType::ScalarType;
    auto aggregDim = (args.AendX-args.AstartX)*CommInfo.d; auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    auto span = (args.AendX!=args.trueLocalDimension ? aggregDim :aggregDim-(args.trueLocalDimension*CommInfo.d-args.trueGlobalDimension));
    lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
    lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
    lapack::engine::_potrf(args.base_case_cyclic_table[index_pair].data(),span,aggregDim,potrfArgs);
    std::memcpy(args.base_case_cyclic_table[index_pair].scratch(),args.base_case_cyclic_table[index_pair].data(),sizeof(T)*args.base_case_cyclic_table[index_pair].num_elems());
    lapack::engine::_trtri(args.base_case_cyclic_table[index_pair].scratch(),span,aggregDim,trtriArgs);
  }

  template<typename ArgType, typename CommType>
  static void complete(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgType::ScalarType;
    int rankSlice; MPI_Comm_rank(CommInfo.slice, &rankSlice);
    auto aggregDim = (args.AendX-args.AstartX)*CommInfo.d; auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    util::cyclic_to_local(args.base_case_cyclic_table[index_pair].data(),args.base_case_cyclic_table[index_pair].scratch(), args.localDimension, args.globalDimension, aggregDim, CommInfo.d,rankSlice);
    serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.R, 0,args.AendX-args.AstartX,0,args.AendY-args.AstartY,args.AstartY, args.AendY, args.AstartY, args.AendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.Rinv,0,args.AendX-args.AstartX,0,args.AendY-args.AstartY,args.TIstartX, args.TIendX, args.TIstartY, args.TIendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
  }
};

class ReplicateComp{
protected:
  template<typename ArgType, typename CommType>
  static void initiate(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    auto aggregDim = (args.AendX-args.AstartX)*CommInfo.d; auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    auto localDimension = args.base_case_table[index_pair].num_columns_local();
    if (CommInfo.z==0){
      serialize<uppertri,uppertri>::invoke(args.R, args.base_case_table[index_pair], args.AstartX, args.AendX, args.AstartY, args.AendY,0,args.AendX-args.AstartX,0,args.AendY-args.AstartY);
      MPI_Allgather(args.base_case_table[index_pair].data(), args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, &args.base_case_blocked_table[index_pair][0],
                    args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, CommInfo.slice);
      if (std::is_same<typename ArgTypeRR::SP,Serialize>::value){
        util::block_to_cyclic_triangle(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(),
                                       args.base_case_blocked_table[index_pair].size(), localDimension, localDimension, CommInfo.d);
      } else{
        util::block_to_cyclic_rect(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(), localDimension, localDimension, CommInfo.d);
      }
    }
  }

  template<typename ArgType, typename CommType>
  static void compute(ArgType&& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    if (CommInfo.z==0){
      auto aggregDim = (args.AendX-args.AstartX)*CommInfo.d; auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
      auto span = (args.AendX!=args.trueLocalDimension ? aggregDim :aggregDim-(args.trueLocalDimension*CommInfo.d-args.trueGlobalDimension));
      lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
      lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
      lapack::engine::_potrf(args.base_case_cyclic_table[index_pair].data(),span,aggregDim,potrfArgs);
      std::memcpy(args.base_case_cyclic_table[index_pair].scratch(),args.base_case_cyclic_table[index_pair].data(),sizeof(T)*args.base_case_cyclic_table[index_pair].num_elems());
      lapack::engine::_trtri(args.base_case_cyclic_table[index_pair].scratch(),span,aggregDim,trtriArgs);
    }
  }

  template<typename ArgType, typename CommType>
  static void complete(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    int rankSlice; MPI_Comm_rank(CommInfo.slice, &rankSlice);
    auto aggregDim = (args.AendX-args.AstartX)*CommInfo.d; auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    MPI_Bcast(args.base_case_cyclic_table[index_pair].data(),aggregDim*aggregDim,mpi_type<T>::type,0,CommInfo.depth);
    MPI_Bcast(args.base_case_cyclic_table[index_pair].scratch(),aggregDim*aggregDim,mpi_type<T>::type,0,CommInfo.depth);
    util::cyclic_to_local(args.base_case_cyclic_table[index_pair].data(),args.base_case_cyclic_table[index_pair].scratch(), args.localDimension, args.globalDimension, aggregDim, CommInfo.d,rankSlice);
    serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.R, 0,args.AendX-args.AstartX,0,args.AendY-args.AstartY,args.AstartY, args.AendY, args.AstartY, args.AendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.Rinv,0,args.AendX-args.AstartX,0,args.AendY-args.AstartY,args.TIstartX, args.TIendX, args.TIstartY, args.TIendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
  }
};

class NoReplication{
protected:
/*
  template<typename ArgType, typename CommType>
  static void initiate(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    auto aggregDim = (args.AendX-args.AstartX)*CommInfo.d; auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    auto localDimension = args.base_case_table[index_pair].num_columns_local();
    if (CommInfo.z==0){
      serialize<uppertri,uppertri>::invoke(args.R, args.base_case_table[index_pair], args.AstartX, args.AendX, args.AstartY, args.AendY,0,args.AendX-args.AstartX,0,args.AendY-args.AstartY);
      MPI_Gather(args.base_case_table[index_pair].data(), args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, &args.base_case_blocked_table[index_pair][0],
                    args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, 0, CommInfo.slice);
      if (CommInfo.x==0 && CommInfo.y==0){
        if (std::is_same<typename ArgTypeRR::SP,Serialize>::value){
          util::block_to_cyclic_triangle(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(),
                                         args.base_case_blocked_table[index_pair].size(), localDimension, localDimension, CommInfo.d);
        } else{
          util::block_to_cyclic_rect(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(), localDimension, localDimension, CommInfo.d);
        }
      }
    }
  }

  template<typename ArgType, typename CommType>
  static void compute(ArgType&& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    if (CommInfo.x==0 && CommInfo.y==0 && CommInfo.z==0){
      auto aggregDim = (args.AendX-args.AstartX)*CommInfo.d; auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
      auto span = (args.AendX!=args.trueLocalDimension ? aggregDim :aggregDim-(args.trueLocalDimension*CommInfo.d-args.trueGlobalDimension));
      lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
      lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
      lapack::engine::_potrf(args.base_case_cyclic_table[index_pair].data(),span,aggregDim,potrfArgs);
      std::memcpy(args.base_case_cyclic_table[index_pair].scratch(),args.base_case_cyclic_table[index_pair].data(),sizeof(T)*args.base_case_cyclic_table[index_pair].num_elems());
      util::cyclic_to_block(
    }
    if (CommInfo.z==0){
      MPI_Iscatter(args.base_case_cyclic_table[index_pair].data(),&args.req);
    }
    if (CommInfo.x==0 && CommInfo.y==0 && CommInfo.z==0){
      lapack::engine::_trtri(args.base_case_cyclic_table[index_pair].scratch(),span,aggregDim,trtriArgs);
    }
  }

  template<typename ArgType, typename CommType>
  static void complete(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    int rankSlice; MPI_Comm_rank(CommInfo.slice, &rankSlice); MPI_Status st;
    auto aggregDim = (args.AendX-args.AstartX)*CommInfo.d; auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    MPI_Wait(&args.req, &st);
    MPI_Ibcast(args.base_case_cyclic_table[index_pair].data(),aggregDim*aggregDim,mpi_type<T>::type,0,CommInfo.depth,&args.req);
    MPI_Scatter(..);
    MPI_Wait();
    MPI_Bcast(args.base_case_cyclic_table[index_pair].scratch(),aggregDim*aggregDim,mpi_type<T>::type,0,CommInfo.depth);
    
    ..serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.R, 0,args.AendX-args.AstartX,0,args.AendY-args.AstartY,args.AstartY, args.AendY, args.AstartY, args.AendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.Rinv,0,args.AendX-args.AstartX,0,args.AendY-args.AstartY,args.TIstartX, args.TIendX, args.TIstartY, args.TIendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
  }
*/
};

};
};
};
#endif // CHOLESKY__POLICY__CHOLINV
