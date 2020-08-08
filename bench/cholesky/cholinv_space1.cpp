/* Author: Edward Hutter */

#include "../../src/alg/cholesky/cholinv/cholinv.h"
#include "../../test/cholesky/validate.h"

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

  using cholesky_type0 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::NoReplication>;
  using cholesky_type1 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicateCommComp>;
  using cholesky_type2 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicateComp>;
  size_t process_cube_dim = std::nearbyint(std::ceil(pow(size,1./3.)));
  size_t rep_factor = process_cube_dim/rep_div; double time_global;
  T residual_error_local,residual_error_global; auto mpi_dtype = mpi_type<T>::type;
  { 
    auto SquareTopo = topo::square(MPI_COMM_WORLD,rep_factor,layout,num_chunks);
    MatrixType A(num_rows,num_rows, SquareTopo.d, SquareTopo.d);
    A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    // Generate algorithmic structure via instantiating packs

    // First attain the "true" execution times for each variant

    for (size_t i=0; i<num_iter; i++){
      MPI_Barrier(MPI_COMM_WORLD);
      volatile double st = MPI_Wtime();
      for (auto j=0; j<3; j++){
        for (auto k=0; k<15; k++){
          if (k==0){
            cholesky_type0::info<T,U> pack0_0(complete_inv,split,bcMultiplier,dir);
            critter::start(false,true);
            cholesky_type0::factor(A, pack0_0, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==1){
            cholesky_type0::info<T,U> pack1_0(complete_inv,split,bcMultiplier+1,dir);
            critter::start(false,true);
            cholesky_type0::factor(A, pack1_0, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==2){
            cholesky_type0::info<T,U> pack2_0(complete_inv,split,bcMultiplier+2,dir);
            critter::start(false,true);
            cholesky_type0::factor(A, pack2_0, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==3){
            cholesky_type0::info<T,U> pack3_0(complete_inv,split,bcMultiplier+3,dir);
            critter::start(false,true);
            cholesky_type0::factor(A, pack3_0, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==4){
            cholesky_type0::info<T,U> pack4_0(complete_inv,split,bcMultiplier+4,dir);
            critter::start(false,true);
            cholesky_type0::factor(A, pack4_0, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==5){
            cholesky_type1::info<T,U> pack0_1(complete_inv,split,bcMultiplier,dir);
            critter::start(false,true);
            cholesky_type1::factor(A, pack0_1, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==6){
            cholesky_type1::info<T,U> pack1_1(complete_inv,split,bcMultiplier+1,dir);
            critter::start(false,true);
            cholesky_type1::factor(A, pack1_1, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==7){
            cholesky_type1::info<T,U> pack2_1(complete_inv,split,bcMultiplier+2,dir);
            critter::start(false,true);
            cholesky_type1::factor(A, pack2_1, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==8){
            cholesky_type1::info<T,U> pack3_1(complete_inv,split,bcMultiplier+3,dir);
            critter::start(false,true);
            cholesky_type1::factor(A, pack3_1, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==9){
            cholesky_type1::info<T,U> pack4_1(complete_inv,split,bcMultiplier+4,dir);
            critter::start(false,true);
            cholesky_type1::factor(A, pack4_1, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==10){
            cholesky_type2::info<T,U> pack0_2(complete_inv,split,bcMultiplier,dir);
            critter::start(false,true);
            cholesky_type2::factor(A, pack0_2, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==11){
            cholesky_type2::info<T,U> pack1_2(complete_inv,split,bcMultiplier+1,dir);
            critter::start(false,true);
            cholesky_type2::factor(A, pack1_2, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==12){
            cholesky_type2::info<T,U> pack2_2(complete_inv,split,bcMultiplier+2,dir);
            critter::start(false,true);
            cholesky_type2::factor(A, pack2_2, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==13){
            cholesky_type2::info<T,U> pack3_2(complete_inv,split,bcMultiplier+3,dir);
            critter::start(false,true);
            cholesky_type2::factor(A, pack3_2, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
          else if (k==14){
            cholesky_type2::info<T,U> pack4_2(complete_inv,split,bcMultiplier+4,dir);
            critter::start(false,true);
            cholesky_type2::factor(A, pack4_2, SquareTopo);
            critter::stop(false,true,false);
            if (rank==0) std::cout << "progress stage 1 " << i << " " << j << " " << k << std::endl;
          }
        }
      }
      st = MPI_Wtime() - st;
      if (rank==0) std::cout << "wallclock time of stage 1 - " << st << std::endl;
    }


    // Next tune the parameterization space

    for (size_t i=0; i<num_iter; i++){
      MPI_Barrier(MPI_COMM_WORLD);
#ifdef CRITTER
      critter::start(true,false);
#endif
      volatile double st = MPI_Wtime();
      for (auto j=0; j<5; j++){
        for (auto k=0; k<15; k++){
          if (k==0){
            cholesky_type0::info<T,U> pack0_0(complete_inv,split,bcMultiplier,dir);
            cholesky_type0::factor(A, pack0_0, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==1){
            cholesky_type0::info<T,U> pack1_0(complete_inv,split,bcMultiplier+1,dir);
            cholesky_type0::factor(A, pack1_0, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==2){
            cholesky_type0::info<T,U> pack2_0(complete_inv,split,bcMultiplier+2,dir);
            cholesky_type0::factor(A, pack2_0, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==3){
            cholesky_type0::info<T,U> pack3_0(complete_inv,split,bcMultiplier+3,dir);
            cholesky_type0::factor(A, pack3_0, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==4){
            cholesky_type0::info<T,U> pack4_0(complete_inv,split,bcMultiplier+4,dir);
            cholesky_type0::factor(A, pack4_0, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==5){
            cholesky_type1::info<T,U> pack0_1(complete_inv,split,bcMultiplier,dir);
            cholesky_type1::factor(A, pack0_1, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==6){
            cholesky_type1::info<T,U> pack1_1(complete_inv,split,bcMultiplier+1,dir);
            cholesky_type1::factor(A, pack1_1, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==7){
            cholesky_type1::info<T,U> pack2_1(complete_inv,split,bcMultiplier+2,dir);
            cholesky_type1::factor(A, pack2_1, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==8){
            cholesky_type1::info<T,U> pack3_1(complete_inv,split,bcMultiplier+3,dir);
            cholesky_type1::factor(A, pack3_1, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==9){
            cholesky_type1::info<T,U> pack4_1(complete_inv,split,bcMultiplier+4,dir);
            cholesky_type1::factor(A, pack4_1, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==10){
            cholesky_type2::info<T,U> pack0_2(complete_inv,split,bcMultiplier,dir);
            cholesky_type2::factor(A, pack0_2, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==11){
            cholesky_type2::info<T,U> pack1_2(complete_inv,split,bcMultiplier+1,dir);
            cholesky_type2::factor(A, pack1_2, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==12){
            cholesky_type2::info<T,U> pack2_2(complete_inv,split,bcMultiplier+2,dir);
            cholesky_type2::factor(A, pack2_2, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==13){
            cholesky_type2::info<T,U> pack3_2(complete_inv,split,bcMultiplier+3,dir);
            cholesky_type2::factor(A, pack3_2, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==14){
            cholesky_type2::info<T,U> pack4_2(complete_inv,split,bcMultiplier+4,dir);
            cholesky_type2::factor(A, pack4_2, SquareTopo);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank==0) std::cout << "progress stage 2 - " << i << " " << j << " " << k << std::endl;
          }
        }
      }
      st = MPI_Wtime() - st;
#ifdef CRITTER
      critter::stop(true,false);
#endif
      if (rank==0) std::cout << "wallclock time for stage 2 (autotuning) - " << st << std::endl;
/*
      cholesky_type::factor(A, pack, SquareTopo);
      residual_error_local = cholesky::validate<cholesky_type>::residual(A, pack, SquareTopo);
      MPI_Reduce(&residual_error_local, &residual_error_global, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
      if (rank==0){ std::cout << residual_error_global << std::endl; }
*/
    }

    // Stage 3: evaluate the estimated execution times using the autotuned parameterization space

    for (size_t i=0; i<num_iter; i++){
      MPI_Barrier(MPI_COMM_WORLD);
      volatile double st = MPI_Wtime();
      for (auto j=0; j<3; j++){
        for (auto k=0; k<15; k++){
          if (k==0){
            cholesky_type0::info<T,U> pack0_0(complete_inv,split,bcMultiplier,dir);
            critter::start(true,false);
            cholesky_type0::factor(A, pack0_0, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==1){
            cholesky_type0::info<T,U> pack1_0(complete_inv,split,bcMultiplier+1,dir);
            critter::start(true,false);
            cholesky_type0::factor(A, pack1_0, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==2){
            cholesky_type0::info<T,U> pack2_0(complete_inv,split,bcMultiplier+2,dir);
            critter::start(true,false);
            cholesky_type0::factor(A, pack2_0, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==3){
            cholesky_type0::info<T,U> pack3_0(complete_inv,split,bcMultiplier+3,dir);
            critter::start(true,false);
            cholesky_type0::factor(A, pack3_0, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==4){
            cholesky_type0::info<T,U> pack4_0(complete_inv,split,bcMultiplier+4,dir);
            critter::start(true,false);
            cholesky_type0::factor(A, pack4_0, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==5){
            cholesky_type1::info<T,U> pack0_1(complete_inv,split,bcMultiplier,dir);
            critter::start(true,false);
            cholesky_type1::factor(A, pack0_1, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==6){
            cholesky_type1::info<T,U> pack1_1(complete_inv,split,bcMultiplier+1,dir);
            critter::start(true,false);
            cholesky_type1::factor(A, pack1_1, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==7){
            cholesky_type1::info<T,U> pack2_1(complete_inv,split,bcMultiplier+2,dir);
            critter::start(true,false);
            cholesky_type1::factor(A, pack2_1, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==8){
            cholesky_type1::info<T,U> pack3_1(complete_inv,split,bcMultiplier+3,dir);
            critter::start(true,false);
            cholesky_type1::factor(A, pack3_1, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==9){
            cholesky_type1::info<T,U> pack4_1(complete_inv,split,bcMultiplier+4,dir);
            critter::start(true,false);
            cholesky_type1::factor(A, pack4_1, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==10){
            cholesky_type2::info<T,U> pack0_2(complete_inv,split,bcMultiplier,dir);
            critter::start(true,false);
            cholesky_type2::factor(A, pack0_2, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==11){
            cholesky_type2::info<T,U> pack1_2(complete_inv,split,bcMultiplier+1,dir);
            critter::start(true,false);
            cholesky_type2::factor(A, pack1_2, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==12){
            cholesky_type2::info<T,U> pack2_2(complete_inv,split,bcMultiplier+2,dir);
            critter::start(true,false);
            cholesky_type2::factor(A, pack2_2, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==13){
            cholesky_type2::info<T,U> pack3_2(complete_inv,split,bcMultiplier+3,dir);
            critter::start(true,false);
            cholesky_type2::factor(A, pack3_2, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
          else if (k==14){
            cholesky_type2::info<T,U> pack4_2(complete_inv,split,bcMultiplier+4,dir);
            critter::start(true,false);
            cholesky_type2::factor(A, pack4_2, SquareTopo);
            critter::stop(true,false,false);
            if (rank==0) std::cout << "progress stage 3 - " << i << " " << j << " " << k << std::endl;
          }
        }
      }
      st = MPI_Wtime() - st;
      if (rank==0) std::cout << "wallclock time of stage 1 - " << st << std::endl;
    }


  }
  MPI_Finalize();
  return 0;
}
