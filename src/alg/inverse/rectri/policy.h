#ifndef INVERSE__POLICY__RECTRI
#define INVERSE__POLICY__RECTRI

namespace inverse{
namespace policy{
namespace rectri{

// ***********************************************************************************************************************************************************************
class Serialize{
protected:
  using structure = lowertri;
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
};
};
};
#endif // INVERSE__POLICY__RECTRI
