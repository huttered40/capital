/* Author: Edward Hutter */

#include <iomanip>

#include "../../../src/alg/cholesky/cholinv/cholinv.h"
#include "../../../test/cholesky/validate.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t; using MatrixType = matrix<T,U,rect>; using namespace cholesky;

  int rank,size,provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
  PMPI_Barrier(MPI_COMM_WORLD);

  char dir          = 'U';
  U num_rows        = atoi(argv[1]);// number of rows in global matrix
  U rep_div         = atoi(argv[2]);// cuts the depth of cubic process grid (only trivial support of value '1' is supported)
  bool complete_inv = atoi(argv[3]);// decides whether to complete inverse in cholinv
  U split           = atoi(argv[4]);// split factor in cholinv
  U bcMultiplier    = atoi(argv[5]);// base case depth factor in cholinv
  size_t layout     = atoi(argv[6]);// arranges sub-communicator layout
  size_t num_chunks = atoi(argv[7]);// splits up communication in summa into nonblocking chunks
  size_t num_iter   = atoi(argv[8]);// number of simulations of the algorithm for performance testing

  std::string stream_name;
  std::ofstream stream,stream_max,stream_vol;
  if (std::getenv("CRITTER_VIZ_FILE") != NULL){
    stream_name = std::getenv("CRITTER_VIZ_FILE");
  }
  auto stream_name_max = stream_name+"_max.txt";
  auto stream_name_vol = stream_name+"_vol.txt";
  stream_name += ".txt";
  if (rank==0){
    stream.open(stream_name.c_str(),std::ofstream::app);
    stream_max.open(stream_name_max.c_str(),std::ofstream::app);
    stream_vol.open(stream_name_vol.c_str(),std::ofstream::app);
  }
  size_t width = 18;

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

    // Initial simulations to warm cache, etc.
    cholesky_type0::info<T,U> pack_init(complete_inv,split,bcMultiplier,dir);
    cholesky_type0::factor(A,pack_init,SquareTopo);

    size_t space_dim = 15;
    vector<double> save_data(num_iter*space_dim*(1+39));

    // Stage 1: autotune each schedule variant individually

    MPI_Barrier(MPI_COMM_WORLD);
    volatile double st4 = MPI_Wtime();
    for (auto k=0; k<space_dim; k++){
      if (k/5==0){
        cholesky_type0::info<T,U> pack(complete_inv,split,bcMultiplier+k%5,dir);
        for (size_t i=0; i<num_iter; i++){
#ifdef CRITTER
          critter::start(true,true,false,true);
#endif
          volatile double _st = MPI_Wtime();
          cholesky_type0::factor(A,pack,SquareTopo);
          MPI_Barrier(MPI_COMM_WORLD);
          volatile double _st_ = MPI_Wtime();
#ifdef CRITTER
          critter::stop();
          if (i<(num_iter-1)) critter::record(&save_data[1*num_iter*space_dim+39*(k*num_iter+i)],false,true);
          else                critter::record(&save_data[1*num_iter*space_dim+39*(k*num_iter+i)],true,true);
#endif
          save_data[k*num_iter+i] = _st_-_st;
        }
      }
      else if (k/5==1){
        cholesky_type1::info<T,U> pack(complete_inv,split,bcMultiplier+k%5,dir);
        for (size_t i=0; i<num_iter; i++){
#ifdef CRITTER
          critter::start(true,true,false,true);
#endif
          volatile double _st = MPI_Wtime();
          cholesky_type1::factor(A,pack,SquareTopo);
          MPI_Barrier(MPI_COMM_WORLD);
          volatile double _st_ = MPI_Wtime();
#ifdef CRITTER
          critter::stop();
          if (i<(num_iter-1)) critter::record(&save_data[1*num_iter*space_dim+39*(k*num_iter+i)],false,true);
          else                critter::record(&save_data[1*num_iter*space_dim+39*(k*num_iter+i)],true,true);
#endif
          save_data[k*num_iter+i] = _st_-_st;
        }
      }
      else if (k/5==2){
        cholesky_type2::info<T,U> pack(complete_inv,split,bcMultiplier+k%5,dir);
        for (size_t i=0; i<num_iter; i++){
#ifdef CRITTER
          critter::start(true,true,false,true);
#endif
          volatile double _st = MPI_Wtime();
          cholesky_type2::factor(A,pack,SquareTopo);
          MPI_Barrier(MPI_COMM_WORLD);
          volatile double _st_ = MPI_Wtime();
#ifdef CRITTER
          critter::stop();
          if (i<(num_iter-1)) critter::record(&save_data[1*num_iter*space_dim+39*(k*num_iter+i)],false,true);
          else                critter::record(&save_data[1*num_iter*space_dim+39*(k*num_iter+i)],true,true);
#endif
          save_data[k*num_iter+i] = _st_-_st;
        }
      }
#ifdef CRITTER
      critter::clear();
#endif
      if (rank==0) stream << "progress stage 1 - " << k << std::endl;
    }
    st4 = MPI_Wtime() - st4;
    if (rank==0) stream << "wallclock time of stage 1 - " << st4 << std::endl;

    // Print out autotuning data
    if (rank==0){
      stream_max << std::left << std::setw(width) << "ID";
      stream_max << std::left << std::setw(width) << "AutoOverhead";
      stream_max << std::left << std::setw(width) << "EstET";
      stream_max << std::left << std::setw(width) << "EstCompTime";
      stream_max << std::left << std::setw(width) << "EstCompKTime";
      stream_max << std::left << std::setw(width) << "EstCommKTime";
      stream_max << std::left << std::setw(width) << "SchedComm";
      stream_max << std::left << std::setw(width) << "SchedLocalComm";
      stream_max << std::left << std::setw(width) << "SkipComm";
      stream_max << std::left << std::setw(width) << "SchedBytes";
      stream_max << std::left << std::setw(width) << "SchedLocalBytes";
      stream_max << std::left << std::setw(width) << "SkipBytes";
      stream_max << std::left << std::setw(width) << "SchedCommTime";
      stream_max << std::left << std::setw(width) << "SchedLocalCommTime";
      stream_max << std::left << std::setw(width) << "SchedComp";
      stream_max << std::left << std::setw(width) << "SchedLocalComp";
      stream_max << std::left << std::setw(width) << "SkipComp";
      stream_max << std::left << std::setw(width) << "SchedFlops";
      stream_max << std::left << std::setw(width) << "SchedLocalFlops";
      stream_max << std::left << std::setw(width) << "SkipFlops";
      stream_max << std::left << std::setw(width) << "SchedCompTime";
      stream_max << std::left << std::setw(width) << "SchedLocalCompTime";
      stream_max << std::left << std::setw(width) << "CompOverhead";
      stream_max << std::left << std::setw(width) << "CommOverhead1";
      stream_max << std::left << std::setw(width) << "CommOverhead2";
      stream_max << std::endl;

      stream_vol << std::left << std::setw(width) << "ID";
      stream_vol << std::left << std::setw(width) << "AutoOverhead";
      stream_vol << std::left << std::setw(width) << "EstET";
      stream_vol << std::left << std::setw(width) << "EstCompTime";
      stream_vol << std::left << std::setw(width) << "EstCompKTime";
      stream_vol << std::left << std::setw(width) << "EstCommKTime";
      stream_vol << std::left << std::setw(width) << "SchedComm";
      stream_vol << std::left << std::setw(width) << "SchedLocalComm";
      stream_vol << std::left << std::setw(width) << "SkipComm";
      stream_vol << std::left << std::setw(width) << "SchedBytes";
      stream_vol << std::left << std::setw(width) << "SchedLocalBytes";
      stream_vol << std::left << std::setw(width) << "SkipBytes";
      stream_vol << std::left << std::setw(width) << "SchedCommTime";
      stream_vol << std::left << std::setw(width) << "SchedLocalCommTime";
      stream_vol << std::left << std::setw(width) << "SchedComp";
      stream_vol << std::left << std::setw(width) << "SchedLocalComp";
      stream_vol << std::left << std::setw(width) << "SkipComp";
      stream_vol << std::left << std::setw(width) << "SchedFlops";
      stream_vol << std::left << std::setw(width) << "SchedLocalFlops";
      stream_vol << std::left << std::setw(width) << "SkipFlops";
      stream_vol << std::left << std::setw(width) << "SchedCompTime";
      stream_vol << std::left << std::setw(width) << "SchedLocalCompTime";
      stream_vol << std::left << std::setw(width) << "CompOverhead";
      stream_vol << std::left << std::setw(width) << "CommOverhead1";
      stream_vol << std::left << std::setw(width) << "CommOverhead2";
      stream_vol << std::endl;

      for (size_t k=0; k<space_dim; k++){
        for (size_t i=0; i<num_iter; i++){
          stream_max << std::left << std::setw(width) << k;
          stream_max << std::left << std::setw(width) << save_data[0*space_dim*num_iter+k*num_iter+i];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+0];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+1];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+2];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+3];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+4];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+5];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+6];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+7];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+8];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+9];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+10];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+11];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+12];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+13];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+14];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+15];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+16];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+17];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+18];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+19];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+36];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+37];
          stream_max << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+38];
          stream_max << std::endl;

          stream_vol << std::left << std::setw(width) << k;
          stream_vol << std::left << std::setw(width) << save_data[0*space_dim*num_iter+k*num_iter+i];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+0];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+1];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+2];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+3];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+20];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+21];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+22];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+23];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+24];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+25];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+26];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+27];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+28];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+29];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+30];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+31];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+32];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+33];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+34];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+35];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+36];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+37];
          stream_vol << std::left << std::setw(width) << save_data[1*space_dim*num_iter+39*(k*num_iter+i)+38];
          stream_vol << std::endl;
        }
      }
    }
  }
  if (rank==0){
    stream.close();
    stream_max.close();
    stream_vol.close();
  }
  MPI_Finalize();
  return 0;
}
