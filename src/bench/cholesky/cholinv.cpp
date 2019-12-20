/* Author: Edward Hutter */

#include "../../alg/cholesky/cholinv/cholinv.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeA = matrix<double,int64_t,square,cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<double>();

  char dir = 'U';
  int64_t globalMatrixSize = atoi(argv[1]);
  int64_t pGridDimensionC = atoi(argv[2]);
  int64_t inverseCutOffMultiplier = atoi(argv[3]); // multiplies baseCase dimension by sucessive 2
  int64_t num_chunks        = atoi(argv[4]);
  int64_t numIterations = atoi(argv[5]);

  size_t pGridCubeDim = std::nearbyint(std::ceil(pow(size,1./3.)));
  pGridDimensionC = pGridCubeDim/pGridDimensionC;
  if (rank==0){ std::cout << "pGridDimensionC - " << pGridDimensionC << std::endl; }

  for (size_t i=0; i<numIterations; i++){
    // Create new topology each outer-iteration so the instance goes out of scope before MPI_Finalize
    auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC,num_chunks);
    // Reset matrixA
    MatrixTypeA matA(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
    MatrixTypeA matT(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
    double iterTimeGlobal,iterErrorGlobal;
    matA.DistributeSymmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    if (rank==0){ std::cout << "SquareTopo values - " << SquareTopo.c << " " << SquareTopo.d << " " << SquareTopo.x << " " << SquareTopo.y << std::endl; }
    if (rank==0){ std::cout << "matA dims - " << matA.getNumRowsLocal() << " " << matA.getNumColumnsLocal() << " " << matA.getNumRowsGlobal() << " " << matA.getNumColumnsLocal() << std::endl; }
    MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
    critter::start();
    cholesky::cholinv::invoke(matA, matT, SquareTopo, inverseCutOffMultiplier, dir);
    critter::stop();

    matA.DistributeSymmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    double startTime=MPI_Wtime();
    cholesky::cholinv::invoke(matA, matT, SquareTopo, inverseCutOffMultiplier, dir);
    double iterTimeLocal=MPI_Wtime() - startTime;

    MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MatrixTypeA saveA = matA;
    saveA.DistributeSymmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    double iterErrorLocal = cholesky::validate<cholesky::cholinv>::invoke(saveA, matA, dir, SquareTopo);
    MPI_Reduce(&iterErrorLocal, &iterErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank==0){
      std::cout << iterTimeGlobal << " " << iterErrorGlobal << std::endl;
    }
  }
  MPI_Finalize();
  return 0;
}
