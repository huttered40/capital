/* Author: Edward Hutter */

#include "../../alg/matmult/summa/summa.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t;
  using MatrixTypeR = matrix<T,U,rect>;
  using MatrixTypeLT = matrix<T,U,lowertri>;
  using MatrixTypeUT = matrix<T,U,uppertri>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  U globalMatrixSizeM  = atoi(argv[1]);
  U globalMatrixSizeN  = atoi(argv[2]);
  U globalMatrixSizeK  = atoi(argv[3]);
  U pGridDimensionC    = atoi(argv[4]);
  size_t layout        = atoi(argv[5]);// arranges sub-communicator layout
  size_t num_chunks    = atoi(argv[6]);
  size_t numIterations = atoi(argv[7]);
  size_t factor        = atoi(argv[8]);// factor by which to multiply the critter stats internally
  size_t id            = atoi(argv[9]);// 0 for critter-only, 1 for critter+production, 2 for critter+production+numerical

  auto mpi_dtype = mpi_type<T>::type;
  U pGridCubeDim = std::nearbyint(std::ceil(pow(size,1./3.)));
  pGridDimensionC = pGridCubeDim/pGridDimensionC;
  {
    auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC,layout,num_chunks);
    MatrixTypeR matA(globalMatrixSizeK,globalMatrixSizeM,SquareTopo.d,SquareTopo.d);
    MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeK,SquareTopo.d,SquareTopo.d);
    MatrixTypeR matC(globalMatrixSizeN,globalMatrixSizeM,SquareTopo.d,SquareTopo.d);
    blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, 1., 0.);
    matA.distribute_random(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c);
    matB.distribute_random(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c*(-1));
    matC.distribute_random(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c*(-1));

    // Loop for getting a good range of results.
    for (size_t i=0; i<numIterations; i++){
      MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
#ifdef CRITTER
      if (id != 3) critter::start(id);
#endif
      matmult::summa::invoke(matA, matB, matC, SquareTopo, blasArgs);
#ifdef CRITTER
      if (id != 3) critter::stop(id,factor);
#endif
    }
  }
  MPI_Finalize();
  return 0;
}
