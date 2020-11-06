/* Author: Edward Hutter */

#include "../../../src/alg/cholesky/cholinv/cholinv.h"
#include "../../../test/cholesky/validate.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t; using MatrixType = matrix<T,U,rect>; using namespace cholesky;

  int rank,size,provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

  char dir          = 'U';
  U num_rows        = atoi(argv[1]);// number of rows in global matrix
  U rep_div         = atoi(argv[2]);// cuts the depth of cubic process grid (only trivial support of value '1' is supported)
  bool complete_inv = atoi(argv[3]);// decides whether to complete inverse in cholinv
  U split           = atoi(argv[4]);// split factor in cholinv
  U bcMultiplier    = atoi(argv[5]);// base case depth factor in cholinv
  size_t layout     = atoi(argv[6]);// arranges sub-communicator layout
  size_t num_chunks = atoi(argv[7]);// splits up communication in summa into nonblocking chunks
  size_t num_iter   = atoi(argv[8]);// number of simulations of the algorithm for performance testing

  int sample_constraint_mode=0;
  if (std::getenv("CRITTER_AUTOTUNING_SAMPLE_CONSTRAINT_MODE") != NULL){
    sample_constraint_mode = atoi(std::getenv("CRITTER_AUTOTUNING_SAMPLE_CONSTRAINT_MODE"));
  }

  using cholesky_type0 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::NoReplication>;
  using cholesky_type1 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicateCommComp>;
  using cholesky_type2 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicateComp>;
  size_t process_cube_dim = std::nearbyint(std::ceil(pow(size,1./3.)));
  size_t rep_factor = process_cube_dim/rep_div;
  { 
    auto SquareTopo = topo::square(MPI_COMM_WORLD,rep_factor,layout,num_chunks);
    MatrixType A(num_rows,num_rows, SquareTopo.d, SquareTopo.d);
    A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    // Generate algorithmic structure via instantiating packs

    size_t space_dim = 15;
    // Stage 1: tune the parameterization space
    double overhead_bin = 0;
    PMPI_Barrier(MPI_COMM_WORLD);
    critter::start();
    volatile double st3 = MPI_Wtime();
    for (auto k=0; k<space_dim; k++){
      if (k/5==0){
        cholesky_type0::info<T,U> pack(complete_inv,split,bcMultiplier+k%5,dir);
        double overhead_timer = MPI_Wtime();
        A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
        critter::set_mode(0);
        cholesky_type0::factor(A,pack,SquareTopo);// Avoid allocation times
        critter::set_mode();
        critter::set_mechanism(0);
        critter::set_debug(1);
        critter::start();
        cholesky_type0::factor(A,pack,SquareTopo);
        critter::stop();
	critter::record(k,1,0);
        critter::set_debug(0);
        critter::set_mechanism(1);
        if (sample_constraint_mode==3){
          critter::set_mechanism(2);
          critter::start();
          cholesky_type0::factor(A,pack,SquareTopo);
          critter::stop();
          critter::set_mechanism(1);
        }
        overhead_bin += (MPI_Wtime() - overhead_timer);
        for (size_t i=0; i<num_iter; i++){
          critter::start();
          cholesky_type0::factor(A,pack,SquareTopo);
          critter::stop();
          overhead_timer = MPI_Wtime();
	  critter::record(k,1,0);
          critter::record(k,2);
          overhead_bin += (MPI_Wtime() - overhead_timer);
          //if (rank==0) std::cout << "in stage 1 - " << k << "\n";
        }
      }
      else if (k/5==1){
        cholesky_type1::info<T,U> pack(complete_inv,split,bcMultiplier+k%5,dir);
        double overhead_timer = MPI_Wtime();
        A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
        critter::set_mode(0);
        cholesky_type1::factor(A,pack,SquareTopo);// Avoid allocation times
        critter::set_mode();
        critter::set_mechanism(0);
        critter::set_debug(1);
        critter::start();
        cholesky_type1::factor(A,pack,SquareTopo);
        critter::stop();
	critter::record(k,1,0);
        critter::set_debug(0);
        critter::set_mechanism(1);
        if (sample_constraint_mode==3){
          critter::set_mechanism(2);
          critter::start();
          cholesky_type1::factor(A,pack,SquareTopo);
          critter::stop();
          critter::set_mechanism(1);
        }
        overhead_bin += (MPI_Wtime() - overhead_timer);
        for (size_t i=0; i<num_iter; i++){
          critter::start();
          cholesky_type1::factor(A,pack,SquareTopo);
          critter::stop();
          overhead_timer = MPI_Wtime();
	  critter::record(k,1,0);
          critter::record(k,2);
          overhead_bin += (MPI_Wtime() - overhead_timer);
          //if (rank==0) std::cout << "in stage 1 - " << k << "\n";
        }
      }
      else if (k/5==2){
        cholesky_type2::info<T,U> pack(complete_inv,split,bcMultiplier+k%5,dir);
        double overhead_timer = MPI_Wtime();
        A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
        critter::set_mode(0);
        cholesky_type2::factor(A,pack,SquareTopo);// Avoid allocation times
        critter::set_mode();
        critter::set_mechanism(0);
        critter::set_debug(1);
        critter::start();
        cholesky_type2::factor(A,pack,SquareTopo);
        critter::stop();
	critter::record(k,1,0);
        critter::set_debug(0);
        critter::set_mechanism(1);
        if (sample_constraint_mode==3){
          critter::set_mechanism(2);
          critter::set_debug(1);
          critter::start();
          cholesky_type2::factor(A,pack,SquareTopo);
          critter::stop();
          critter::set_debug(0);
          critter::set_mechanism(1);
        }
        overhead_bin += (MPI_Wtime() - overhead_timer);
        for (size_t i=0; i<num_iter; i++){
          critter::start();
          cholesky_type2::factor(A,pack,SquareTopo);
          critter::stop();
          overhead_timer = MPI_Wtime();
	  critter::record(k,1,0);
          critter::record(k,2);
          overhead_bin += (MPI_Wtime() - overhead_timer);
          //if (rank==0) std::cout << "in stage 1 - " << k << "\n";
        }
      }
      double overhead_timer = MPI_Wtime();
      //critter::record(k,2);
      critter::clear();
      overhead_bin += (MPI_Wtime() - overhead_timer);
      //if (rank==0) std::cout << "progress stage 1 - " << k << std::endl;
    }
    st3 = MPI_Wtime() - st3;
    critter::stop();
    critter::record(-1,0,overhead_bin);
    if (rank==0) std::cout << "wallclock time of stage 1 - " << st3 << std::endl;
  }
  MPI_Finalize();
  return 0;
}
