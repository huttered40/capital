/* Author: Edward Hutter */

template<typename T, typename U>
template< template<typename,typename,template<typename,typename,int> class> class StructureArg1,
  template<typename,typename,template<typename,typename,int> class> class StructureArg2,
  template<typename,typename,template<typename,typename,int> class> class StructureArg3,
  template<typename,typename,int> class Distribution>
void validator<T,U>::validateResidualParallel(
                        Matrix<T,U,StructureArg1,Distribution>& matrixA,
                        Matrix<T,U,StructureArg2,Distribution>& matrixB,
                        Matrix<T,U,StructureArg3,Distribution>& matrixC,
                        char dir,
                        MPI_Comm commWorld,
                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                        MPI_Comm columnAltComm
                      )
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  auto commInfo = util<T,U>::getCommunicatorSlice(
    commWorld);

  MPI_Comm sliceComm = std::get<0>(commInfo);
  U pGridCoordX = std::get<1>(commInfo);
  U pGridCoordY = std::get<2>(commInfo);
  U pGridCoordZ = std::get<3>(commInfo);
  U pGridDimensionSize = std::get<4>(commInfo);
  int helper = pGridDimensionSize;
  helper *= helper;

  if (dir == 'L')
  {
    blasEngineArgumentPackage_gemm<T> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasTrans, 1., -1.);
    MM3D<T,U,cblasEngine>::Multiply(
      matrixA, matrixB, matrixC, commWorld, commInfo3D, blasArgs);
  }
  else if (dir == 'U')
  {
    blasEngineArgumentPackage_gemm<T> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasTrans, blasEngineTranspose::AblasNoTrans, 1., -1.);
    MM3D<T,U,cblasEngine>::Multiply(
      matrixA, matrixB, matrixC, commWorld, commInfo3D, blasArgs);
  }
  else if (dir == 'F')
  {
    blasEngineArgumentPackage_gemm<T> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasNoTrans, 1., -1.);
    MM3D<T,U,cblasEngine>::Multiply(
      matrixA, matrixB, matrixC, commWorld, commInfo3D, blasArgs);
  }
  else if (dir == 'I')
  {
    blasEngineArgumentPackage_gemm<T> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);
    MM3D<T,U,cblasEngine>::Multiply(
      matrixA, matrixB, matrixC, commWorld, commInfo3D, blasArgs);
    if (columnAltComm != MPI_COMM_WORLD)
    {
      MPI_Allreduce(MPI_IN_PLACE, matrixC.getRawData(), matrixC.getNumElems(), MPI_DOUBLE,
        MPI_SUM, columnAltComm);
    }
  }
  else
  {
    abort();
  }

  // Now just calculate residual
  T error = 0;
  U localNumRows = matrixC.getNumRowsLocal();
  U localNumColumns = matrixC.getNumColumnsLocal();
  U globalX = pGridCoordX;
  U globalY = pGridCoordY;
  for (U i=0; i<localNumColumns; i++)
  {
    globalY = pGridCoordY;    // reset
    for (int j=0; j<localNumRows; j++)
    {
      T val = 0;
      if ((dir == 'F') || ((dir == 'L') && (globalY >= globalX)) || ((dir == 'U') && (globalY <= globalX)))
      {
        val = matrixC.getRawData()[i*localNumRows+j];
      }
      else if (dir == 'I')
      {
        if (globalX == globalY)
        {
          val = std::abs(1. - matrixC.getRawData()[i*localNumRows+j]);
        }
        else
        {
          val = matrixC.getRawData()[i*localNumRows+j];
        }
        //if (matrixC.getRawData()[i*localNumRows+j] > .5) {std::cout << "CHECK THIS at global " << globalX << " " << globalY <<  std::endl;}
      }
      val *= val;
      //if (rank == 0) std::cout << val << " " << i << " " << j << std::endl;
      error += std::abs(val);
      globalY += pGridDimensionSize;
    }
    globalX += pGridDimensionSize;
  }
  error = std::sqrt(error);
  //std::cout << "localError = " << error << std::endl;
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (rank == 0) {std::cout << "Residual error = " << error << std::endl;}
  MPI_Comm_free(&sliceComm);
}

template<typename T, typename U>
template< template<typename,typename,template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>
void validator<T,U>::validateOrthogonalityParallel(
                        Matrix<T,U,StructureArg,Distribution>& matrixQ,
                        MPI_Comm commWorld,
                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                        MPI_Comm columnAltComm
                      )
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  auto commInfo = util<T,U>::getCommunicatorSlice(
    commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  U pGridCoordX = std::get<1>(commInfo);
  U pGridCoordY = std::get<2>(commInfo);
  U pGridCoordZ = std::get<3>(commInfo);
  U pGridDimensionSize = std::get<4>(commInfo);
  int helper = pGridDimensionSize;
  helper *= helper;
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;

  Matrix<T,U,StructureArg,Distribution> matrixQtrans = matrixQ;
  util<T,U>::transposeSwap(
    matrixQtrans, rank, transposePartner, commWorld);
  U localNumRows = matrixQtrans.getNumColumnsLocal();
  U localNumColumns = matrixQ.getNumColumnsLocal();
  U globalNumRows = matrixQtrans.getNumColumnsGlobal();
  U globalNumColumns = matrixQ.getNumColumnsGlobal();
  U numElems = localNumRows*localNumColumns;
  Matrix<T,U,StructureArg,Distribution> matrixI(std::vector<T>(numElems,0), localNumColumns, localNumRows, globalNumColumns, globalNumRows, true);
  validateResidualParallel(
    matrixQtrans,matrixQ,matrixI,'I',commWorld, commInfo3D, columnAltComm);
  MPI_Comm_free(&sliceComm);
}
