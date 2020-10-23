/* Author: Edward Hutter */

#include "../../../src/alg/qr/cacqr/cacqr.h"
#include "../../../test/qr/validate.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t; using MatrixType = matrix<T,U,rect>;

  int rank,size,provided; MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

  size_t variant    = atoi(argv[1]);// 1 - cacqr, 2 - cacqr2
  U num_rows        = atoi(argv[2]);// number of rows in global matrix
  U num_columns     = atoi(argv[3]);// number of columns in global matrix
  U rep_factor      = atoi(argv[4]);// decides depth of process grid and replication factor of matrix
  bool complete_inv = atoi(argv[5]);// decides whether to complete inverse in cholinv
  U split           = atoi(argv[6]);// split factor in cholinv
  U bcMultiplier    = atoi(argv[7]);// base case depth factor in cholinv
  size_t layout     = atoi(argv[8]);// arranges sub-communicator layout
  size_t num_chunks = atoi(argv[9]);// splits up communication in summa into nonblocking chunks
  size_t num_iter   = atoi(argv[10]);// number of simulations of the algorithm for performance testing

  bool schedule_kernels=true;
  if (std::getenv("CRITTER_AUTOTUNING_TEST") != NULL){
    int _tuning_test_ = atoi(std::getenv("CRITTER_AUTOTUNING_TEST"));
    if (_tuning_test_ == 0) schedule_kernels=false;
  } else assert(0);
  int sample_constraint_mode=0;
  if (std::getenv("CRITTER_AUTOTUNING_SAMPLE_CONSTRAINT_MODE") != NULL){
    sample_constraint_mode = atoi(std::getenv("CRITTER_AUTOTUNING_SAMPLE_CONSTRAINT_MODE"));
  }

  using cholesky_type = typename cholesky::cholinv<cholesky::policy::cholinv::NoSerialize,cholesky::policy::cholinv::SaveIntermediates,cholesky::policy::cholinv::NoReplication>;
  using qr_type = qr::cacqr<qr::policy::cacqr::NoSerialize,qr::policy::cacqr::SaveIntermediates>;
  {
    auto RectTopo1 = topo::rect(MPI_COMM_WORLD,rep_factor*1,layout,num_chunks);
    MatrixType A1(num_columns,num_rows,RectTopo1.c,RectTopo1.d);
    A1.distribute_random(RectTopo1.x, RectTopo1.y, RectTopo1.c, RectTopo1.d, rank/RectTopo1.c);
    auto RectTopo2 = topo::rect(MPI_COMM_WORLD,rep_factor*2,layout,num_chunks);
    MatrixType A2(num_columns,num_rows,RectTopo2.c,RectTopo2.d);
    A2.distribute_random(RectTopo2.x, RectTopo2.y, RectTopo2.c, RectTopo2.d, rank/RectTopo2.c);
    auto RectTopo3 = topo::rect(MPI_COMM_WORLD,rep_factor*4,layout,num_chunks);
    MatrixType A3(num_columns,num_rows,RectTopo3.c,RectTopo3.d);
    A3.distribute_random(RectTopo3.x, RectTopo3.y, RectTopo3.c, RectTopo3.d, rank/RectTopo3.c);
    // Generate algorithmic structure via instantiating packs

    size_t space_dim = 15;

    // Attain the (profiled) execution times for each variant
    PMPI_Barrier(MPI_COMM_WORLD);
    volatile double st1 = MPI_Wtime();
    for (auto k=0; k<space_dim; k++){
      if (k/5==0){
        cholesky_type::info<T,U> ci_pack(complete_inv,split,bcMultiplier+k%5,'U');
        qr_type::info<T,U,decltype(ci_pack)::alg_type> pack(variant,ci_pack);
        qr_type::factor(A1, pack, RectTopo1);
        for (size_t i=0; i<num_iter; i++){
          critter::start();
          qr_type::factor(A1, pack, RectTopo1);
          critter::stop();
          critter::record(k);
          critter::clear();
        }
      }
      else if (k/5==1){
        cholesky_type::info<T,U> ci_pack(complete_inv,split,bcMultiplier+k%5,'U');
        qr_type::info<T,U,decltype(ci_pack)::alg_type> pack(variant,ci_pack);
        qr_type::factor(A2, pack, RectTopo2);
        for (size_t i=0; i<num_iter; i++){
          critter::start();
          qr_type::factor(A2, pack, RectTopo2);
          critter::stop();
          critter::record(k);
          critter::clear();
        }
      }
      else if (k/5==2){
        cholesky_type::info<T,U> ci_pack(complete_inv,split,bcMultiplier+k%5,'U');
        qr_type::info<T,U,decltype(ci_pack)::alg_type> pack(variant,ci_pack);
        qr_type::factor(A3, pack, RectTopo3);
        for (size_t i=0; i<num_iter; i++){
          critter::start();
          qr_type::factor(A3, pack, RectTopo3);
          critter::stop();
          critter::record(k);
          critter::clear();
        }
      }
      if (rank==0) std::cout << "progress stage 1 - " << k << std::endl;
    }
    st1 = MPI_Wtime() - st1;
    if (rank==0) std::cout << "wallclock time of stage 1 - " << st1 << std::endl;
  }
  MPI_Finalize();
  return 0;
}
