/* Author: Edward Hutter */

#include "../../alg/qr/cacqr/cacqr.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeS = matrix<double,size_t,square,cyclic>;
  using MatrixTypeR = matrix<double,size_t,rect,cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<double>();

  size_t globalMatrixDimensionM = atoi(argv[1]);
  size_t globalMatrixDimensionN = atoi(argv[2]);
  size_t baseCaseMultiplier = atoi(argv[3]);
  size_t inverseCutOffMultiplier = atoi(argv[4]);
  size_t panelDimensionMultiplier = atoi(argv[5]);
  size_t dimensionC = atoi(argv[6]);
  size_t numIterations=atoi(argv[7]);
  std::string fileStr1 = argv[8];	// Critter
  std::string fileStr2 = argv[9];	// Performance/Residual/DevOrth

  std::vector<size_t> Inputs{globalMatrixDimensionM,globalMatrixDimensionN,dimensionC,baseCaseMultiplier,inverseCutOffMultiplier,panelDimensionMultiplier};
  std::vector<const char*> InputNames{"m","n","c","bcm","icm","pdm"};

  for (auto test=0; test<2; test++){
    // Create new topology each outer-iteration so the instance goes out of scope before MPI_Finalize
    auto RectTopo = topo::rect(MPI_COMM_WORLD,dimensionC);

    switch(test){
      case 0:
        critter::init(1,fileStr1);
	break;
      case 1:
        critter::init(0,fileStr2);
	break;
    }

    for (size_t i=0; i<numIterations; i++){
      // reset the matrix before timer starts
      // Note: matA and matR are rectangular, but the pieces owned by the individual processors may be square (so also rectangular)
      MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM, RectTopo.c, RectTopo.d);
      MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN, RectTopo.c, RectTopo.c);
      matA.DistributeRandom(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
      double iterTimeGlobal = 0;
      double residualErrorGlobal,orthogonalityErrorGlobal;
      MPI_Barrier(MPI_COMM_WORLD);	// make sure each process starts together
      critter::reset();
      volatile double startTime=MPI_Wtime();
      qr::cacqr2::invoke(matA, matR, RectTopo, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
      double iterTimeLocal = MPI_Wtime() - startTime;

      switch(test){
        case 0:{
          critter::print("QR", size, Inputs.size(), &Inputs[0], &InputNames[0]);
	  break;
	}
        case 1:{
          MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          MatrixTypeR saveA = matA;
          saveA.DistributeRandom(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
          auto error = qr::validate<qr::cacqr>::invoke(saveA, matA, matR, RectTopo);
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
