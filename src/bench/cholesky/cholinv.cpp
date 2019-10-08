/* Author: Edward Hutter */

#include "../../alg/cholesky/cholinv/cholinv.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeA = matrix<double,size_t,square,cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<double>();

  char dir = 'U';
  size_t globalMatrixSize = atoi(argv[1]);
  size_t pGridDimensionC = atoi(argv[2]);
  size_t inverseCutOffMultiplier = atoi(argv[3]); // multiplies baseCase dimension by sucessive 2
  size_t numIterations = atoi(argv[4]);

  size_t pGridCubeDim = std::nearbyint(std::ceil(pow(size,1./3.)));
  pGridDimensionC = pGridCubeDim/pGridDimensionC;

  for (size_t i=0; i<numIterations; i++){
    // Create new topology each outer-iteration so the instance goes out of scope before MPI_Finalize
    auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC);
    // Reset matrixA
    MatrixTypeA matA(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
    MatrixTypeA matT(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
    double iterTimeGlobal,iterErrorGlobal;
    matA.DistributeSymmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
    critter::start();
    cholesky::cholinv::invoke(matA, matT, SquareTopo, inverseCutOffMultiplier, dir);
    critter::stop();

    double startTime=MPI_Wtime();
    cholesky::cholinv::invoke(matA, matT, SquareTopo, inverseCutOffMultiplier, dir);
    double iterTimeLocal=MPI_Wtime() - startTime;

    MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MatrixTypeA saveA = matA;
    saveA.DistributeSymmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    double iterErrorLocal = cholesky::validate<cholesky::cholinv>::invoke(saveA, matA, dir, SquareTopo);
    MPI_Reduce(&iterErrorLocal, &iterErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> Outputs(2);
    Outputs[0] = iterTimeGlobal; Outputs[1] = iterErrorGlobal;
    critter::print(Outputs.size(), &Outputs[0]);
  }
  MPI_Finalize();
  return 0;
}
