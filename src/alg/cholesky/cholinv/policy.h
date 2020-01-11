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

  template<typename ArgType, typename CommType>
  static void create_buffers(bool bc_strategy_id, ArgType& args, CommType&& CommInfo){
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    init(args.base_case_table, index_pair, nullptr,index_pair.first,index_pair.second,CommInfo.d,CommInfo.d);
    auto num_elems = args.base_case_table[index_pair].num_elems()*CommInfo.d*CommInfo.d;
    if (bc_strategy_id==0){
      init(args.base_case_cyclic_table, index_pair, nullptr,aggregDim,aggregDim,CommInfo.d,CommInfo.d);
      init(args.base_case_blocked_table,index_pair, num_elems);
    }
    else if (bc_strategy_id==1){
      init(args.base_case_cyclic_table, index_pair, nullptr,aggregDim,aggregDim,CommInfo.d,CommInfo.d);
      if (CommInfo.z==0){
        init(args.base_case_blocked_table,index_pair, num_elems);
      }
    }
    else if (bc_strategy_id>=2){
      if (CommInfo.x==0 && CommInfo.y==0 && CommInfo.z==0){
        init(args.base_case_cyclic_table, index_pair, nullptr,aggregDim,aggregDim,CommInfo.d,CommInfo.d);
        init(args.base_case_blocked_table,index_pair, num_elems);
      }
    }
  }

  template<typename ArgType, typename CommType>
  static void init_buffers(bool bc_strategy_id, ArgType& args, CommType&& CommInfo){
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    auto& m1 = invoke(args.base_case_table,index_pair);
    if (bc_strategy_id==0){
      auto& m2 = invoke(args.base_case_cyclic_table,index_pair); auto& m3 = invoke(args.base_case_cyclic_table,index_pair);
    }
    else if (bc_strategy_id==1){
      auto& m3 = invoke(args.base_case_cyclic_table,index_pair);
      if (CommInfo.z==0){
        auto& m2 = invoke(args.base_case_cyclic_table,index_pair);
      }
    }
    else if (bc_strategy_id>=2){
      if (CommInfo.x==0 && CommInfo.y==0 && CommInfo.z==0){
        auto& m2 = invoke(args.base_case_cyclic_table,index_pair); auto& m3 = invoke(args.base_case_cyclic_table,index_pair);
      }
    }
  }

  template<typename ArgType, typename CommType>
  static void remove_buffers(bool bc_strategy_id, ArgType& args, CommType&& CommInfo){}
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

  template<typename ArgType, typename CommType>
  static void create_buffers(bool bc_strategy_id, ArgType& args, CommType&& CommInfo){
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    init(args.base_case_table, index_pair, nullptr,index_pair.first,index_pair.second,CommInfo.d,CommInfo.d);
    auto num_elems = args.base_case_table[index_pair].num_elems()*CommInfo.d*CommInfo.d;
    if (bc_strategy_id==0){
      init(args.base_case_cyclic_table, index_pair, nullptr,aggregDim,aggregDim,CommInfo.d,CommInfo.d);
      init(args.base_case_blocked_table,index_pair, num_elems);
    }
    else if (bc_strategy_id==1){
      init(args.base_case_cyclic_table, index_pair, nullptr,aggregDim,aggregDim,CommInfo.d,CommInfo.d);
      if (CommInfo.z==0){
        init(args.base_case_blocked_table,index_pair, num_elems);
      }
    }
    else if (bc_strategy_id==2){
      if (CommInfo.x==0 && CommInfo.y==0 && CommInfo.z==0){
        init(args.base_case_cyclic_table, index_pair, nullptr,aggregDim,aggregDim,CommInfo.d,CommInfo.d);
        init(args.base_case_blocked_table,index_pair, num_elems);
      }
    }
  }

  template<typename ArgType, typename CommType>
  static void init_buffers(bool bc_strategy_id, ArgType& args, CommType&& CommInfo){
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    auto& m1 = invoke(args.base_case_table,index_pair);
    if (bc_strategy_id==0){
      auto& m2 = invoke(args.base_case_cyclic_table,index_pair); auto& m3 = invoke(args.base_case_cyclic_table,index_pair);
    }
    else if (bc_strategy_id==1){
      auto& m3 = invoke(args.base_case_cyclic_table,index_pair);
      if (CommInfo.z==0){
        auto& m2 = invoke(args.base_case_cyclic_table,index_pair);
      }
    }
    else if (bc_strategy_id==2){
      if (CommInfo.x==0 && CommInfo.y==0 && CommInfo.z==0){
        auto& m2 = invoke(args.base_case_cyclic_table,index_pair); auto& m3 = invoke(args.base_case_cyclic_table,index_pair);
      }
    }
  }

  template<typename ArgType, typename CommType>
  static void remove_buffers(bool bc_strategy_id, ArgType& args, CommType&& CommInfo){
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    flush(args.base_case_table[index_pair]);
    if (bc_strategy_id<=1){
      flush(args.base_case_cyclic_table[index_pair]);
    }
  }
};
// ***********************************************************************************************************************************************************************

// ***********************************************************************************************************************************************************************
class ReplicateCommComp{
protected:
  static size_t get_id(){return 0;}

  template<typename ArgType, typename CommType>
  static void initiate(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgType::ScalarType;
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    auto localDimension = args.base_case_table[index_pair].num_columns_local();
    serialize<uppertri,uppertri>::invoke(args.R, args.base_case_table[index_pair], args.AstartX, args.AendX, args.AstartY, args.AendY,0,index_pair.first,0,index_pair.second);
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
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
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
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    util::cyclic_to_local(args.base_case_cyclic_table[index_pair].data(),args.base_case_cyclic_table[index_pair].scratch(), args.localDimension, aggregDim, CommInfo.d,rankSlice);
    serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.R, 0,index_pair.first,0,index_pair.second,args.AstartY, args.AendY, args.AstartY, args.AendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.Rinv,0,index_pair.first,0,index_pair.second,args.TIstartX, args.TIendX, args.TIstartY, args.TIendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
  }
};

class ReplicateComp{
protected:
  static size_t get_id(){return 1;}

  template<typename ArgType, typename CommType>
  static void initiate(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    auto localDimension = args.base_case_table[index_pair].num_columns_local();
    if (CommInfo.z==0){
      serialize<uppertri,uppertri>::invoke(args.R, args.base_case_table[index_pair], args.AstartX, args.AendX, args.AstartY, args.AendY,0,index_pair.first,0,index_pair.second);
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
      auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
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
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    MPI_Bcast(args.base_case_cyclic_table[index_pair].data(),aggregDim*aggregDim,mpi_type<T>::type,0,CommInfo.depth);
    MPI_Bcast(args.base_case_cyclic_table[index_pair].scratch(),aggregDim*aggregDim,mpi_type<T>::type,0,CommInfo.depth);
    util::cyclic_to_local(args.base_case_cyclic_table[index_pair].data(),args.base_case_cyclic_table[index_pair].scratch(), args.localDimension, aggregDim, CommInfo.d,rankSlice);
    serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.R, 0,index_pair.first,0,index_pair.second,args.AstartY, args.AendY, args.AstartY, args.AendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<uppertri,uppertri>::invoke(args.base_case_cyclic_table[index_pair], args.Rinv,0,index_pair.first,0,index_pair.second,args.TIstartX, args.TIendX, args.TIstartY, args.TIendY);
    args.base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
  }
};

class NoReplication{
protected:
  static size_t get_id(){return 2;}

  template<typename ArgType, typename CommType>
  static void initiate(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    auto localDimension = args.base_case_table[index_pair].num_columns_local();
    if (CommInfo.z==0){
      serialize<uppertri,uppertri>::invoke(args.R, args.base_case_table[index_pair], args.AstartX, args.AendX, args.AstartY, args.AendY,0,index_pair.first,0,index_pair.second);
      if (CommInfo.x==0 && CommInfo.y==0){
        MPI_Gather(args.base_case_table[index_pair].data(), args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, &args.base_case_blocked_table[index_pair][0],
                   args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, 0, CommInfo.slice);
        if (std::is_same<typename ArgTypeRR::SP,Serialize>::value){
          util::block_to_cyclic_triangle(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(),
                                         args.base_case_blocked_table[index_pair].size(), localDimension, localDimension, CommInfo.d);
        } else{
          util::block_to_cyclic_rect(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(), localDimension, localDimension, CommInfo.d);
        }
      }
      else{
        MPI_Gather(args.base_case_table[index_pair].data(), args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, nullptr, 0, mpi_type<T>::type, 0, CommInfo.slice);
      }
    }
  }

  template<typename ArgType, typename CommType>
  static void compute(ArgType&& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    auto localDimension = args.base_case_table[index_pair].num_columns_local();
    auto span = (args.AendX!=args.trueLocalDimension ? aggregDim :aggregDim-(args.trueLocalDimension*CommInfo.d-args.trueGlobalDimension));
    lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
    lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
    if (CommInfo.z==0){
      if (CommInfo.x==0 && CommInfo.y==0){
        lapack::engine::_potrf(args.base_case_cyclic_table[index_pair].data(),span,aggregDim,potrfArgs);
        std::memcpy(args.base_case_cyclic_table[index_pair].scratch(),args.base_case_cyclic_table[index_pair].data(),sizeof(T)*args.base_case_cyclic_table[index_pair].num_elems());
        if (std::is_same<typename ArgTypeRR::SP,Serialize>::value){
          util::cyclic_to_block_triangle(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(),
                                         args.base_case_blocked_table[index_pair].size(), localDimension, localDimension, CommInfo.d);
        } else{
          util::cyclic_to_block_rect(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(), localDimension, localDimension, CommInfo.d);
        }
        MPI_Scatter(&args.base_case_blocked_table[index_pair][0],args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,args.base_case_table[index_pair].data(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.slice);
      }
      else{
        MPI_Scatter(nullptr,0,mpi_type<T>::type,args.base_case_table[index_pair].data(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.slice);
      }
      if (CommInfo.x==0 && CommInfo.y==0){
        lapack::engine::_trtri(args.base_case_cyclic_table[index_pair].scratch(),span,aggregDim,trtriArgs);
        if (std::is_same<typename ArgTypeRR::SP,Serialize>::value){
          util::cyclic_to_block_triangle(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].scratch(),
                                         args.base_case_blocked_table[index_pair].size(), localDimension, localDimension, CommInfo.d);
        } else{
          util::cyclic_to_block_rect(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].scratch(), localDimension, localDimension, CommInfo.d);
        }
        MPI_Scatter(&args.base_case_blocked_table[index_pair][0],args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,args.base_case_table[index_pair].scratch(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.slice);
      }
      else{
        MPI_Scatter(nullptr,0,mpi_type<T>::type,args.base_case_table[index_pair].scratch(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.slice);
      }
    }
  }

  template<typename ArgType, typename CommType>
  static void complete(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
    MPI_Bcast(args.base_case_table[index_pair].data(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.depth);
    MPI_Bcast(args.base_case_table[index_pair].scratch(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.depth);
    serialize<uppertri,uppertri>::invoke(args.base_case_table[index_pair], args.R, 0,index_pair.first,0,index_pair.second,args.AstartY, args.AendY, args.AstartY, args.AendY);
    args.base_case_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<uppertri,uppertri>::invoke(args.base_case_table[index_pair], args.Rinv,0,index_pair.first,0,index_pair.second,args.TIstartX, args.TIendX, args.TIstartY, args.TIendY);
    args.base_case_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
  }
};

class NoReplicationOverlap{
protected:
  static size_t get_id(){return 3;}

  template<typename ArgType, typename CommType>
  static void initiate(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    auto localDimension = args.base_case_table[index_pair].num_columns_local();
    if (CommInfo.z==0){
      serialize<uppertri,uppertri>::invoke(args.R, args.base_case_table[index_pair], args.AstartX, args.AendX, args.AstartY, args.AendY,0,index_pair.first,0,index_pair.second);
      if (CommInfo.x==0 && CommInfo.y==0){
        MPI_Gather(args.base_case_table[index_pair].data(), args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, &args.base_case_blocked_table[index_pair][0],
                   args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, 0, CommInfo.slice);
        if (std::is_same<typename ArgTypeRR::SP,Serialize>::value){
          util::block_to_cyclic_triangle(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(),
                                         args.base_case_blocked_table[index_pair].size(), localDimension, localDimension, CommInfo.d);
        } else{
          util::block_to_cyclic_rect(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(), localDimension, localDimension, CommInfo.d);
        }
      }
      else{
        MPI_Gather(args.base_case_table[index_pair].data(), args.base_case_table[index_pair].num_elems(), mpi_type<T>::type, nullptr, 0, mpi_type<T>::type, 0, CommInfo.slice);
      }
    }
  }

  template<typename ArgType, typename CommType>
  static void compute(ArgType&& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); auto aggregDim = index_pair.first*CommInfo.d;
    auto localDimension = args.base_case_table[index_pair].num_columns_local(); MPI_Status st;
    auto span = (args.AendX!=args.trueLocalDimension ? aggregDim :aggregDim-(args.trueLocalDimension*CommInfo.d-args.trueGlobalDimension));
    lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
    lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
    if (CommInfo.z==0){
      if (CommInfo.x==0 && CommInfo.y==0){
        lapack::engine::_potrf(args.base_case_cyclic_table[index_pair].data(),span,aggregDim,potrfArgs);
        std::memcpy(args.base_case_cyclic_table[index_pair].scratch(),args.base_case_cyclic_table[index_pair].data(),sizeof(T)*args.base_case_cyclic_table[index_pair].num_elems());
        if (std::is_same<typename ArgTypeRR::SP,Serialize>::value){
          util::cyclic_to_block_triangle(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(),
                                         args.base_case_blocked_table[index_pair].size(), localDimension, localDimension, CommInfo.d);
        } else{
          util::cyclic_to_block_rect(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].data(), localDimension, localDimension, CommInfo.d);
        }
        MPI_Iscatter(&args.base_case_blocked_table[index_pair][0],args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,args.base_case_table[index_pair].data(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.slice, &args.req);
      }
      else{
        MPI_Iscatter(nullptr,0,mpi_type<T>::type,args.base_case_table[index_pair].data(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.slice,&args.req);
      }
      if (CommInfo.x==0 && CommInfo.y==0){
        lapack::engine::_trtri(args.base_case_cyclic_table[index_pair].scratch(),span,aggregDim,trtriArgs);
        MPI_Wait(&args.req,&st);
        if (std::is_same<typename ArgTypeRR::SP,Serialize>::value){
          util::cyclic_to_block_triangle(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].scratch(),
                                         args.base_case_blocked_table[index_pair].size(), localDimension, localDimension, CommInfo.d);
        } else{
          util::cyclic_to_block_rect(&args.base_case_blocked_table[index_pair][0], args.base_case_cyclic_table[index_pair].scratch(), localDimension, localDimension, CommInfo.d);
        }
        MPI_Iscatter(&args.base_case_blocked_table[index_pair][0],args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,args.base_case_table[index_pair].scratch(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.slice,&args.req);
      }
      else{
        MPI_Wait(&args.req,&st);
        MPI_Iscatter(nullptr,0,mpi_type<T>::type,args.base_case_table[index_pair].scratch(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.slice,&args.req);
      }
    }
    MPI_Bcast(args.base_case_table[index_pair].data(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.depth);
  }

  template<typename ArgType, typename CommType>
  static void complete(ArgType& args, CommType&& CommInfo){
    using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
    auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY); MPI_Status st;
    if (CommInfo.z==0){ MPI_Wait(&args.req,&st); }
    MPI_Bcast(args.base_case_table[index_pair].scratch(),args.base_case_table[index_pair].num_elems(),mpi_type<T>::type,0,CommInfo.depth);
    serialize<uppertri,uppertri>::invoke(args.base_case_table[index_pair], args.R, 0,index_pair.first,0,index_pair.second,args.AstartY, args.AendY, args.AstartY, args.AendY);
    args.base_case_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<uppertri,uppertri>::invoke(args.base_case_table[index_pair], args.Rinv,0,index_pair.first,0,index_pair.second,args.TIstartX, args.TIendX, args.TIstartY, args.TIendY);
    args.base_case_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
  }
};

};
};
};
#endif // CHOLESKY__POLICY__CHOLINV
