/* Author: Edward Hutter */

#include "../../alg/qr/cacqr/cacqr.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeS = matrix<double,int64_t,square,cyclic>;
  using MatrixTypeR = matrix<double,int64_t,rect,cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<double>();

  int64_t globalMatrixDimensionM = atoi(argv[1]);
  int64_t globalMatrixDimensionN = atoi(argv[2]);
  size_t baseCaseMultiplier = atoi(argv[3]);
  size_t inverseCutOffMultiplier = atoi(argv[4]);
  size_t panelDimensionMultiplier = atoi(argv[5]);
  size_t dimensionC = atoi(argv[6]);
  size_t numIterations=atoi(argv[7]);
  std::string fileStr1 = argv[8];	// Critter
  std::string fileStr2 = argv[9];	// Performance/Residual/DevOrth

  size_t dimensionD = size / (dimensionC*dimensionC);
  size_t sliceSize = dimensionD*dimensionC;
  size_t helper = dimensionC*dimensionC;
  size_t pCoordY = rank/helper;
  size_t pCoordX = (rank%helper)/dimensionC;
  std::vector<size_t> Inputs{globalMatrixDimensionM,globalMatrixDimensionN,dimensionC,baseCaseMultiplier,inverseCutOffMultiplier,panelDimensionMultiplier};
  std::vector<const char*> InputNames{"m","n","c","bcm","icm","pdm"};

  for (test=0; test<2; test++){
    switch(test){
      case 0:
        critter::init(1,fileStr1);
      case 1:
        critter::init(0,fileStr2);
    }

    for (size_t i=0; i<numIterations; i++){
      // reset the matrix before timer starts
      // Note: matA and matR are rectangular, but the pieces owned by the individual processors may be square (so also rectangular)
      MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM, dimensionC, dimensionD);
      MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN, dimensionC, dimensionC);
      matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, rank/dimensionC);
      double iterTimeGlobal = 0;
      double residualErrorGlobal,orthogonalityErrorGlobal;
      MPI_Barrier(MPI_COMM_WORLD);	// make sure each process starts together
      critter::reset();
      volatile double startTime=MPI_Wtime();
      qr::cacqr2::invoke(matA, matR, topo::rect(MPI_COMM_WORLD,dimensionC), inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
      double iterTimeLocal = MPI_Wtime() - startTime;

      switch(test){
        case 0:{
          critter::print("QR", size, Inputs.size(), &Inputs[0], &InputNames[0]);
	  break;
	}
        case 1:{
          MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          MatrixTypeR saveA = matA;
          saveA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, rank/dimensionC);
          auto error = qr::validate<cacqr>::invoke(saveA, matA, matR, topo::rect(MPI_COMM_WORLD,dimensionC));
          MPI_Reduce(&error.first, &residualErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          MPI_Reduce(&error.second, &orthogonalityErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          std::vector<double> Outputs(3);
	  Outputs[0] = iterTimeGlobal; Outputs[1] = residualErrorGlobal; Outputs[2] = orthogonalityErrorGlobal;
          critter::print("QR", size, Inputs.size(), &Inputs[0], &InputNames[0],Outputs.size(),&Outputs[0]);
	  break;
	}
      }
    }
    critter::finalize();
  }
  MPI_Finalize();
  return 0;
}
