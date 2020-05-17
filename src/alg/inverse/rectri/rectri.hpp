/* Author: Edward Hutter */

namespace inverse{
template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void rectri<SerializePolicy,IntermediatesPolicy>::invoke(const MatrixType& A, ArgType& args, CommType&& CommInfo){
  assert(args.dir == 'L');
  auto localDimension = A.num_rows_local(); auto globalDimension = A.num_rows_global();
//  args.L._register_(A.num_columns_global(),A.num_rows_global(),CommInfo.d,CommInfo.d);
//  args.Linv._register_(A.num_columns_global(),A.num_rows_global(),CommInfo.d,CommInfo.d);
  args.L_block_table[0] = decltype(args.L)(A.num_columns_global(),A.num_rows_global(),CommInfo.d,CommInfo.d);
  args.Linv_block_table[0] = decltype(args.L)(A.num_columns_global(),A.num_rows_global(),CommInfo.d,CommInfo.d);
  serialize<lowertri,lowertri>::invoke(A,args.L_block_table[0],0,localDimension,0,localDimension,0,localDimension,0,localDimension);
  args.num_levels=0;
  simulate(args, std::forward<CommType>(CommInfo));
  invert(args, std::forward<CommType>(CommInfo));
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> rectri<SerializePolicy,IntermediatesPolicy>::construct_Linv(ArgType& args, CommType&& CommInfo){
  auto localDimension = args.Linv.num_rows_local();
  matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> ret(args.Linv.num_columns_global(),args.Linv.num_rows_global(),CommInfo.c, CommInfo.c);
  serialize<typename SerializePolicy::structure,rect>::invoke(args.Linv, ret,0,localDimension,0,localDimension,0,localDimension,0,localDimension);
  return ret;
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
void rectri<SerializePolicy,IntermediatesPolicy>::simulate(ArgType& args, CommType&& CommInfo){
  args.num_levels++;
  if (CommInfo.size==1){
    return;
  }
  args.process_grids.push_back(CommInfo);
  MPI_Comm swap_comm, recurse_comm;
  int swap_color = 1*(CommInfo.y%(CommInfo.c/2)) + (CommInfo.c/2)*(CommInfo.x%(CommInfo.c/2)) + (CommInfo.d*CommInfo.d/4)*CommInfo.z;
  int recurse_color = 1*(CommInfo.y/((CommInfo.c/2))) + 2*(CommInfo.x/((CommInfo.c/2))) + 4*(CommInfo.z/((CommInfo.c/2)));
  MPI_Comm_split(CommInfo.world,swap_color,CommInfo.rank,&swap_comm);		// key might be wrong
  MPI_Comm_split(CommInfo.world,recurse_color,CommInfo.rank,&recurse_comm);	// key might be wrong
  args.swap_communicators.push_back(swap_comm);
  args.L_panel_table[args.num_levels-1] = decltype(args.L)(args.L_block_table[args.num_levels-1].num_columns_global()/8,args.L_block_table[args.num_levels-1].num_rows_global(),CommInfo.d/2,CommInfo.d/2);
  args.Linv_panel_table[args.num_levels-1] = decltype(args.L)(args.L_block_table[args.num_levels-1].num_columns_global()/8,args.L_block_table[args.num_levels-1].num_rows_global(),CommInfo.d/2,CommInfo.d/2);
  MPI_Alltoall(&args.L_block_table[args.num_levels-1].data()[(recurse_color<4) ? 0 : args.L_block_table[args.num_levels-1].num_elems()/2], args.L_block_table[args.num_levels-1].num_elems()/8, MPI_DOUBLE,
               &args.L_panel_table[args.num_levels-1].scratch()[0], args.L_block_table[args.num_levels-1].num_elems()/8, MPI_DOUBLE, swap_comm);
  args.L_block_table[args.num_levels] = decltype(args.L)(args.L_block_table[args.num_levels-1].num_columns_global()/8,args.L_block_table[args.num_levels-1].num_rows_global()/8,CommInfo.d/2,CommInfo.d/2);
  args.Linv_block_table[args.num_levels] = decltype(args.L)(args.L_block_table[args.num_levels-1].num_columns_global()/8,args.L_block_table[args.num_levels-1].num_rows_global()/8,CommInfo.d/2,CommInfo.d/2);
  int64_t blocked_offset = args.L_block_table[args.num_levels-1].num_elems()/8;
  std::array<int,4> counters; counters.fill(0.0); std::array<int,2> offsets; offsets[0]=0; offsets[1]=2*blocked_offset;
  int num_rows_local = args.L_panel_table[args.num_levels-1].num_rows_local(); int num_columns_local = args.L_panel_table[args.num_levels-1].num_columns_local();
  for (int i=0; i<num_columns_local; i++){
    for (int j=0; j<num_rows_local; j++){
      args.L_panel_table[args.num_levels-1].data()[i*num_rows_local+j] = args.L_panel_table[args.num_levels-1].scratch()[offsets[i%2]+(j%2)*blocked_offset+counters[(j%2)+2*(i%2)]]; counters[(j%2)+2*(i%2)]++;
    }
  }
  serialize<rect,rect>::invoke(args.L_panel_table[args.num_levels-1],args.L_block_table[args.num_levels],0,args.L_panel_table[args.num_levels-1].num_columns_local(),0,args.L_panel_table[args.num_levels-1].num_columns_local(),
                               0,args.L_panel_table[args.num_levels-1].num_columns_local(),0,args.L_panel_table[args.num_levels-1].num_columns_local());
  simulate(args,topo::square(recurse_comm,CommInfo.c/2,CommInfo.num_chunks,true));
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
void rectri<SerializePolicy,IntermediatesPolicy>::invert(ArgType& args, CommType&& CommInfo){
  using ArgTypeRR = typename std::remove_reference<ArgType>::type; using ScalarType = typename ArgTypeRR::ScalarType;
  auto num_columns_local = args.L_block_table[args.num_levels-1].num_columns_local();
  lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackLower, lapack::Diag::AlapackNonUnit);
  std::memcpy(args.Linv_block_table[args.num_levels-1].data(),args.L_block_table[args.num_levels-1].data(),num_columns_local*num_columns_local*sizeof(ScalarType));
  lapack::engine::_trtri(args.Linv_block_table[args.num_levels-1].data(),args.L_block_table[args.num_levels-1].num_columns_local(),args.L_block_table[args.num_levels-1].num_columns_local(),trtriArgs);
  //TODO: Then need to reshuffle it?
/*
  int group_size=1;
  for (int i=0; i<args.num_levels; i++){
    int z = args.process_grids[num_levels-2-i].rank%8;		.. definitely not sure if correct
    int partner;
    if (args.process_grids[num_levels-2-i].rank%2==0){
      args.process_grids[num_levels-2-i].rank+group_size;
    } else{
      args.process_grids[num_levels-2-i].rank-group_size;
    }
    for (int j=0; j<7; j++){
      if ((z%2==0) && (z<(7-j))){
        MPI_Recv(..);
      }
      if ((z%2==1) && (z<(8-j))){
        MPI_Send(..);
      }
      if ((z%2==0) && (z>0) && (z<(8-j))){
        MPI_Recv(..);
      }
      if ((z%2==1) && (z<(7-j))){
        MPI_Send(..);
      }
      matmult::summa::invoke(..);
    }
    .. we need to store the newly formed inverse into args.Linv
    Alltoall(...);
    group_size*=8;
  }
*/
}

}
