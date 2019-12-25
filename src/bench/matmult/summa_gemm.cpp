/* Author: Edward Hutter */

#include "../../alg/matmult/summa/summa.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeR = matrix<double,int64_t,rect,cyclic>;
  using MatrixTypeLT = matrix<double,int64_t,lowertri,cyclic>;
  using MatrixTypeUT = matrix<double,int64_t,uppertri,cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int64_t globalMatrixSizeM = atoi(argv[1]);
  int64_t globalMatrixSizeN = atoi(argv[2]);
  int64_t globalMatrixSizeK = atoi(argv[3]);
  int64_t pGridDimensionC   = atoi(argv[4]);
  int64_t num_chunks        = atoi(argv[5]);
  int64_t numIterations     = atoi(argv[6]);

  int64_t pGridCubeDim = std::nearbyint(std::ceil(pow(size,1./3.)));
  pGridDimensionC = pGridCubeDim/pGridDimensionC;
  {
  auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC,num_chunks);
  MatrixTypeR matA(globalMatrixSizeK,globalMatrixSizeM,SquareTopo.d,SquareTopo.d);
  MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeK,SquareTopo.d,SquareTopo.d);
  MatrixTypeR matC(globalMatrixSizeN,globalMatrixSizeM,SquareTopo.d,SquareTopo.d);
  blas::ArgPack_gemm<double> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, 1., 0.);
  double iterTimeGlobal;

  // Loop for getting a good range of results.
  for (size_t i=0; i<numIterations; i++){
    // Note: I think these calls below are still ok given the new topology mapping on Blue Waters/Stampede2
    matA.distribute_random(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c);
    matB.distribute_random(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c*(-1));
    matC.distribute_random(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c*(-1));
    MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
    critter::start();
    matmult::summa::invoke(matA, matB, matC, SquareTopo, blasArgs);
    critter::stop();

    matA.distribute_random(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c);
    matB.distribute_random(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c*(-1));
    matC.distribute_random(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c*(-1));
    double startTime=MPI_Wtime();
    matmult::summa::invoke(matA, matB, matC, SquareTopo, blasArgs);
    double iterTimeLocal=MPI_Wtime()-startTime;

    MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //matmult::validate<summa3d>::validateLocal(matA,matB,matC,MPI_COMM_WORLD,blasArgs);
  }
  }

  MPI_Finalize();
  return 0;
}
