/* Author: Edward Hutter */

#include <fstream>
#include <iomanip>

#include "../../../src/alg/cholesky/cholinv/cholinv.h"
#include "../../../test/cholesky/validate.h"
#include "../../util.h"

using namespace std;

int _rank_,_size_,_provided_,_sample_constraint_mode_,_reset_mode_;
size_t num_iter,compare,configuration_id;
double total_reference_time;
double total_time;
double overhead_bin;
std::ofstream cp_stream_times, cp_stream_costs, cross_stream_times, cross_stream_costs;

template<typename alg_type>
class launch{
public:
  template<typename MatrixType, typename CommType>
  static void invoke_decomposition_compare(MatrixType& M, CommType& topo, std::vector<int>& resets, char dir, bool complete_inv, int split,
                                           int bcMultiplier, int configuration_id, double& overhead_bin){
    using T = double; using U = int64_t; using MatrixType = matrix<T,U,rect>; using namespace cholesky;
    double overhead_timer = MPI_Wtime();
    typename alg_type::info<T,U> pack(complete_inv,split,bcMultiplier,dir);
    M.distribute_symmetric(topo.x, topo.y, topo.d, topo.d, _rank_/topo.c,true);
    critter::set_mode(0);
    alg_type::factor(M,pack,topo);// Avoid allocation times
    double reference_time = MPI_Wtime();
    alg_type::factor(M,pack,topo);// Avoid allocation times
    reference_time = MPI_Wtime() - reference_time;
    PMPI_Allreduce(MPI_IN_PLACE,&reference_time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    total_reference_time += reference_time;
    critter::set_mode();
    critter::set_mechanism(0);
    critter::set_debug(1);
    M.distribute_symmetric(topo.x, topo.y, topo.d, topo.d, _rank_/topo.c,true);
    critter::start();
    alg_type::factor(M,pack,topo);
    critter::stop();
    critter::record(configuration_id,1,0);
    std::vector<double> decomp_cp_info(critter::get_critical_path_costs()); critter::get_critical_path_costs(&decomp_cp_info[0]);
    std::vector<double> decomp_pp_info(critter::get_max_per_process_costs()); critter::get_max_per_process_costs(&decomp_pp_info[0]);
    std::vector<double> decomp_vol_info(critter::get_volumetric_costs()); critter::get_volumetric_costs(&decomp_vol_info[0]);
    write_cross_info(cross_stream_times,cross_stream_costs,compare,configuration_id,decomp_cp_info,decomp_pp_info,decomp_vol_info);
    critter::set_debug(0);
    critter::set_mechanism(1);
    M.distribute_symmetric(topo.x, topo.y, topo.d, topo.d, _rank_/topo.c,true);
    overhead_bin += (MPI_Wtime() - overhead_timer);
    if (_sample_constraint_mode_==3){
      critter::set_mechanism(2);
      critter::start();
      alg_type::factor(M,pack,topo);
      critter::stop();
      critter::set_mechanism(1);
    }
    for (size_t i=0; i<num_iter; i++){
      overhead_timer = MPI_Wtime();
      M.distribute_symmetric(topo.x, topo.y, topo.d, topo.d, _rank_/topo.c,true);
      overhead_bin += (MPI_Wtime() - overhead_timer);
      critter::start();
      alg_type::factor(M,pack,topo);
      critter::stop();
      overhead_timer = MPI_Wtime();
      critter::record(configuration_id,1,0);
      critter::record(configuration_id,2);
      std::vector<double> disc_cp_info(critter::get_critical_path_costs()); critter::get_critical_path_costs(&disc_cp_info[0]);
      std::vector<double> disc_pp_info(critter::get_max_per_process_costs()); critter::get_max_per_process_costs(&disc_pp_info[0]);
      std::vector<double> disc_vol_info(critter::get_volumetric_costs()); critter::get_volumetric_costs(&disc_vol_info[0]);
      write_cp_info(cp_stream_times,cp_stream_costs,compare,configuration_id,reference_time, decomp_cp_info,disc_cp_info);
      overhead_bin += (MPI_Wtime() - overhead_timer);
    }
    overhead_timer = MPI_Wtime();
    critter::record(configuration_id,3,0);
    if (_reset_mode_==0) critter::clear();
    else critter::clear(0,resets.size(), &resets[0]);
    overhead_bin += (MPI_Wtime() - overhead_timer);
  }
  template<typename MatrixType, typename CommType>
  static void invoke_discretization_compare(MatrixType& M, CommType& topo, std::vector<int>& resets, char dir, bool complete_inv, int split,
                                            int bcMultiplier, int configuration_id, double& overhead_bin){
    using T = double; using U = int64_t; using MatrixType = matrix<T,U,rect>; using namespace cholesky;
    double overhead_timer = MPI_Wtime();
    typename alg_type::info<T,U> pack(complete_inv,split,bcMultiplier,dir);
    M.distribute_symmetric(topo.x, topo.y, topo.d, topo.d, _rank_/topo.c,true);
    critter::set_mode(0);
    alg_type::factor(M,pack,topo);// Avoid allocation times
    double reference_time = MPI_Wtime();
    alg_type::factor(M,pack,topo);// Avoid allocation times
    reference_time = MPI_Wtime() - reference_time;
    PMPI_Allreduce(MPI_IN_PLACE,&reference_time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    total_reference_time += reference_time;
    critter::set_mode();
    critter::set_debug(1);
    M.distribute_symmetric(topo.x, topo.y, topo.d, topo.d, _rank_/topo.c,true);
    critter::start();
    alg_type::factor(M,pack,topo);
    critter::stop();
    std::vector<double> decomp_cp_info(critter::get_critical_path_costs()); critter::get_critical_path_costs(&decomp_cp_info[0]);
    std::vector<double> decomp_pp_info(critter::get_max_per_process_costs()); critter::get_max_per_process_costs(&decomp_pp_info[0]);
    std::vector<double> decomp_vol_info(critter::get_volumetric_costs()); critter::get_volumetric_costs(&decomp_vol_info[0]);
    critter::set_debug(0);
    M.distribute_symmetric(topo.x, topo.y, topo.d, topo.d, _rank_/topo.c,true);
    overhead_bin += (MPI_Wtime() - overhead_timer);
    if (_sample_constraint_mode_==3){
      critter::set_mechanism(2);
      critter::start();
      alg_type::factor(M,pack,topo);
      critter::stop();
      critter::set_mechanism(1);
    }
    for (size_t i=0; i<num_iter; i++){
      overhead_timer = MPI_Wtime();
      M.distribute_symmetric(topo.x, topo.y, topo.d, topo.d, _rank_/topo.c,true);
      overhead_bin += (MPI_Wtime() - overhead_timer);
      critter::start();
      alg_type::factor(M,pack,topo);
      critter::stop();
      overhead_timer = MPI_Wtime();
      critter::record(configuration_id,1,0);
      std::vector<double> disc_cp_info(critter::get_critical_path_costs()); critter::get_critical_path_costs(&disc_cp_info[0]);
      std::vector<double> disc_pp_info(critter::get_max_per_process_costs()); critter::get_max_per_process_costs(&disc_pp_info[0]);
      std::vector<double> disc_vol_info(critter::get_volumetric_costs()); critter::get_volumetric_costs(&disc_vol_info[0]);
      write_cp_info(cp_stream_times,cp_stream_costs,compare,configuration_id,reference_time, decomp_cp_info,disc_cp_info);
      overhead_bin += (MPI_Wtime() - overhead_timer);
    }
    overhead_timer = MPI_Wtime();
    critter::record(configuration_id,3,0);
    if (_reset_mode_==0) critter::clear();
    else critter::clear(0,resets.size(), &resets[0]);
    overhead_bin += (MPI_Wtime() - overhead_timer);
  }
};

int main(int argc, char** argv){
  using T = double; using U = int64_t; using MatrixType = matrix<T,U,rect>; using namespace cholesky;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &_provided_);
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &_size_);

  // Read in all relevant tuning arguments
  char dir            = 'U';
  int num_rows        = atoi(argv[1]);// number of rows in global matrix
  int rep_div         = atoi(argv[2]);// cuts the depth of cubic process grid (only trivial support of value '1' is supported)
  bool complete_inv   = atoi(argv[3]);// decides whether to complete inverse in cholinv
  int split           = atoi(argv[4]);// split factor in cholinv
  int bcMultiplier    = atoi(argv[5]);// base case depth factor in cholinv
  size_t layout       = atoi(argv[6]);// arranges sub-communicator layout
  size_t num_chunks   = atoi(argv[7]);// splits up communication in summa into nonblocking chunks
  num_iter            = atoi(argv[8]);// number of simulations of the algorithm for performance testing
  compare             = atoi(argv[9]);// compare with decomposition or discretization mechanism
  assert(compare==0 || compare==1);
  size_t space_dim = 5;
  if (argc > 10){ space_dim = atoi(argv[10]); }
  using cholesky_type0 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::NoReplication>;
  using cholesky_type1 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicateCommComp>;
  using cholesky_type2 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicateComp>;
  size_t process_cube_dim = std::nearbyint(std::ceil(pow(_size_,1./3.)));
  size_t rep_factor = process_cube_dim/rep_div;

  // Set two tuning environment variables
  _sample_constraint_mode_=0;
  if (std::getenv("CRITTER_AUTOTUNING_SAMPLE_CONSTRAINT_MODE") != NULL){
    _sample_constraint_mode_ = atoi(std::getenv("CRITTER_AUTOTUNING_SAMPLE_CONSTRAINT_MODE"));
  }
  _reset_mode_=0;
  if (std::getenv("RESET_MODE") != NULL){
    _reset_mode_ = atoi(std::getenv("RESET_MODE"));
  }
  std::string stream_name_cp_times = "";
  std::string stream_name_cp_costs = "";
  std::string stream_name_cross_times = "";
  std::string stream_name_cross_costs = "";
  if (std::getenv("CRITTER_VIZ_FILE") != NULL){
    stream_name_cp_times += std::getenv("CRITTER_VIZ_FILE");
    stream_name_cp_costs += std::getenv("CRITTER_VIZ_FILE");
    stream_name_cross_times += std::getenv("CRITTER_VIZ_FILE");
    stream_name_cross_costs += std::getenv("CRITTER_VIZ_FILE");
  }
  stream_name_cp_times += "_reset-" + std::to_string(_reset_mode_) + "_compare-" + std::to_string(compare) + "_cp_times.txt";
  stream_name_cp_costs += "_reset-" + std::to_string(_reset_mode_) + "_compare-" + std::to_string(compare) + "_cp_costs.txt";
  stream_name_cross_times += "_reset-" + std::to_string(_reset_mode_) + "_compare-" + std::to_string(compare) + "_cross_times.txt";
  stream_name_cross_costs += "_reset-" + std::to_string(_reset_mode_) + "_compare-" + std::to_string(compare) + "_cross_costs.txt";
  if (_rank_==0){
    cp_stream_times.open(stream_name_cp_times.c_str(),std::ofstream::app);
    cp_stream_costs.open(stream_name_cp_costs.c_str(),std::ofstream::app);
    cross_stream_times.open(stream_name_cross_times.c_str(),std::ofstream::app);
    cross_stream_costs.open(stream_name_cross_costs.c_str(),std::ofstream::app);
    write_cp_times_header(cp_stream_times,compare);
    write_cp_costs_header(cp_stream_costs,compare);
    write_cross_times_header(cross_stream_times,compare);
    write_cross_costs_header(cross_stream_costs,compare);
  }

  // Set all kernel distributions that should be reset with each distinct configuration
  std::vector<int> reset_routines(5);
  reset_routines[0] = 201;	// LAPACK_POTRF
  reset_routines[1] = 202;	// LAPACK_TRTRI
  reset_routines[2] = 5;	// MPI_Allgather
  reset_routines[3] = 300;	// CAPITAL_blktocyc
  reset_routines[4] = 301;	// CAPITAL_cyctoblk

  { 
    // Generate communicator structure and matrix structure
    auto SquareTopo = topo::square(MPI_COMM_WORLD,rep_factor,layout,num_chunks);
    MatrixType A(num_rows,num_rows, SquareTopo.d, SquareTopo.d);

    configuration_id=0;
    total_reference_time=0;
    overhead_bin = 0;
    total_time = MPI_Wtime();
    PMPI_Barrier(MPI_COMM_WORLD);
    critter::start();
    double total_time = MPI_Wtime();
    for (auto k=0; k<space_dim; k++){
      if (compare) launch<cholesky_type0>::invoke_decomposition_compare(A,SquareTopo,reset_routines,dir,complete_inv,split,bcMultiplier+k,configuration_id,overhead_bin);
      else         launch<cholesky_type0>::invoke_discretization_compare(A,SquareTopo,reset_routines,dir,complete_inv,split,bcMultiplier+k,configuration_id,overhead_bin);
      configuration_id++;
    }
    for (auto k=0; k<space_dim; k++){
      if (compare) launch<cholesky_type1>::invoke_decomposition_compare(A,SquareTopo,reset_routines,dir,complete_inv,split,bcMultiplier+k,configuration_id,overhead_bin);
      else         launch<cholesky_type1>::invoke_discretization_compare(A,SquareTopo,reset_routines,dir,complete_inv,split,bcMultiplier+k,configuration_id,overhead_bin);
      configuration_id++;
    }
    for (auto k=0; k<space_dim; k++){
      if (compare) launch<cholesky_type2>::invoke_decomposition_compare(A,SquareTopo,reset_routines,dir,complete_inv,split,bcMultiplier+k,configuration_id,overhead_bin);
      else         launch<cholesky_type2>::invoke_discretization_compare(A,SquareTopo,reset_routines,dir,complete_inv,split,bcMultiplier+k,configuration_id,overhead_bin);
      configuration_id++;
    }
    total_time = MPI_Wtime() - total_time;
    critter::stop();
    critter::record(-1,0,overhead_bin);
    if (_rank_==0) std::cout << "Total reference time - " << total_reference_time << std::endl;
    if (_rank_==0) std::cout << "Total time - " << total_time << std::endl;
  }
  if (_rank_==0){
    cp_stream_times.close();
    cp_stream_costs.close();
    cross_stream_times.close();
    cross_stream_costs.close();
  }
  MPI_Finalize();
  return 0;
}
