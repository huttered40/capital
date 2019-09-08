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
  size_t blockSizeMultiplier = atoi(argv[3]);
  size_t inverseCutOffMultiplier = atoi(argv[4]); // multiplies baseCase dimension by sucessive 2
  size_t panelDimensionMultiplier = atoi(argv[5]);
  size_t numIterations = atoi(argv[6]);
  size_t ppn=atoi(argv[7]);
  size_t tpr=atoi(argv[8]);
  std::string fileStr1 = argv[9];	// Critter
  std::string fileStr2 = argv[10];	// Performance/Residual/DevOrth

  std::vector<size_t> Inputs{globalMatrixSize,pGridDimensionC,blockSizeMultiplier,inverseCutOffMultiplier,panelDimensionMultiplier,numIterations,ppn,tpr};
  std::vector<const char*> InputNames{"n","c","bcm","icm","pdm","numiter","ppn","tpr"};

  size_t pGridCubeDim = std::nearbyint(std::ceil(pow(size,1./3.)));
  pGridDimensionC = pGridCubeDim/pGridDimensionC;
  for (size_t test=0; test<2; test++){
    // Create new topology each outer-iteration so the instance goes out of scope before MPI_Finalize
    auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC);

    switch(test){
      case 0:
        critter::init(1,fileStr1);
	break;
      case 1:
        critter::init(0,fileStr2);
	break;
    }

    for (size_t i=0; i<numIterations; i++){
      // Reset matrixA
      MatrixTypeA matA(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
      MatrixTypeA matT(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
      double iterTimeGlobal,iterErrorGlobal;
      matA.DistributeSymmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
      MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
      critter::reset();
      double startTime=MPI_Wtime();
      cholesky::cholinv::invoke(matA, matT, SquareTopo, inverseCutOffMultiplier, blockSizeMultiplier, panelDimensionMultiplier, dir);
      double iterTimeLocal=MPI_Wtime() - startTime;

      switch(test){
        case 0:{
          critter::print(i==0, "Cholesky", size, Inputs.size(), &Inputs[0], &InputNames[0]);
	  break;
	}
        case 1:{
          MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          MatrixTypeA saveA = matA;
          saveA.DistributeSymmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
          double iterErrorLocal = cholesky::validate<cholesky::cholinv>::invoke(saveA, matA, dir, SquareTopo);
          MPI_Reduce(&iterErrorLocal, &iterErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          std::vector<double> Outputs(2);
	  Outputs[0] = iterTimeGlobal; Outputs[1] = iterErrorGlobal;
          critter::print(i==0, "Cholesky", size, Inputs.size(), &Inputs[0], &InputNames[0], Outputs.size(), &Outputs[0]);
	  break;
	}
      }
    }
    critter::finalize();
  }
  MPI_Finalize();
  return 0;
}