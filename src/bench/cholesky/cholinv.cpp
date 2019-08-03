/* Author: Edward Hutter */

#include "../../alg/cholesky/cholinv/cholinv.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeA = Matrix<double,int64_t,Square,Cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<double>();

  size_t pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  size_t helper = pGridDimensionSize;
  helper *= helper;
  size_t pCoordY = rank/helper;
  size_t pCoordX = (rank%helper)/pGridDimensionSize;

  char dir = 'U';
  int64_t globalMatrixSize = atoi(argv[1]);
  int64_t pGridDimensionC = atoi(argv[2]);
  size_t blockSizeMultiplier = atoi(argv[3]);
  size_t inverseCutOffMultiplier = atoi(argv[4]); // multiplies baseCase dimension by sucessive 2
  size_t panelDimensionMultiplier = atoi(argv[5]);
  size_t numIterations = atoi(argv[6]);
  std::string fileStr1 = argv[7];	// Critter
  std::string fileStr2 = argv[8];	// Performance/Residual/DevOrth

  std::vector<size_t> Inputs{matA.getNumRowsGlobal(),pGridDimensionC,blockSizeMultiplier,inverseCutOffMultiplier,panelDimensionMultiplier};
  std::vector<const char*> InputNames{"n","c","bcm","icm","pdm"};

  for (size_t test=0; test<2; test++){
    switch(test){
      case 0:
        critter::init(1,fileStr1);
      case 1:
        critter::init(0,fileStr2);
    }

    for (size_t i=0; i<numIterations; i++){
      // Reset matrixA
      MatrixTypeA matA(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
      MatrixTypeA matT(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
      double iterTimeGlobal,iterErrorGlobal;
      matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
      MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
      std::vector<size_t> Inputs{matA.getNumRowsGlobal(),pGridDimensionSize,blockSizeMultiplier,inverseCutOffMultiplier,panelDimensionMultiplier};
      std::vector<const char*> InputNames{"n","c","bcm","icm","pdm"};
      Critter::reset();
      double startTime=MPI_Wtime();
      cholesky::cholinv::invoke(matA, matT, inverseCutOffMultiplier, blockSizeMultiplier, panelDimensionMultiplier, dir, Square(MPI_COMM_WORLD,pGridDimensionSize));
      double iterTimeLocal=MPI_Wtime() - startTime;
      MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      Critter::print("Cholesky", size, Inputs.size(), &Inputs[0], &InputNames[0]);

      MatrixAType saveA = matA;
      // Note: I think this call below is still ok given the new topology mapping on Blue Waters/Stampede2
      saveA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
      double iterErrorLocal;
      iterErrorLocal = cholesky::validate::invoke<cholinv>(saveA, matA, dir, Square(MPI_COMM_WORLD,pGridDimensionSize));
      MPI_Reduce(&iterErrorLocal, &iterErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    critter::finalize();
  }
  MPI_Finalize();
  return 0;
}
