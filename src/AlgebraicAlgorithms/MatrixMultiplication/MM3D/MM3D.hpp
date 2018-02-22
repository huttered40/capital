/* Author: Edward Hutter */


static std::tuple<MPI_Comm,
                  MPI_Comm,
                  MPI_Comm,
                  MPI_Comm,
		              int,
                  int,
                  int>
                  setUpCommunicators(
#ifdef TIMER
                    pTimer& timer,
#endif
                    MPI_Comm commWorld,
                    int depthManipulation = 0
                  )
{
  int rank,size;
#ifdef TIMER
  size_t index5 = timer.setStartTime("MPI_Comm_rank");
#endif
  MPI_Comm_rank(commWorld, &rank);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_rank", index5);
  size_t index6 = timer.setStartTime("MPI_Comm_size");
#endif
  MPI_Comm_size(commWorld, &size);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_size", index6);
#endif

  int pGridDimensionSize = std::nearbyint(std::ceil(pow(size,1./3.)));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = rank%pGridDimensionSize;
  int pGridCoordY = (rank%helper)/pGridDimensionSize;
  int pGridCoordZ = rank/helper;
  pGridCoordZ += depthManipulation;

  MPI_Comm rowComm, columnComm, sliceComm, depthComm;

  // First, split the 3D Cube processor grid communicator into groups based on what 2D slice they are located on.
  // Then, subdivide further into row groups and column groups
#ifdef TIMER
  size_t index1 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(commWorld, pGridCoordY+pGridDimensionSize*pGridCoordX, rank, &depthComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index1);
  size_t index2 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(commWorld, pGridCoordZ, rank, &sliceComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index2);
  size_t index3 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(sliceComm, pGridCoordY, pGridCoordX, &rowComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index3);
  size_t index4 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(sliceComm, pGridCoordX, pGridCoordY, &columnComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index4);
#endif

  return std::make_tuple(rowComm, columnComm, sliceComm, depthComm, pGridCoordX, pGridCoordY, pGridCoordZ);
}

template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
void MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
                                        pTimer& timer,
#endif
                                   	    T* matrixA,
                                        T* matrixB,
                                        T* matrixC,
                                        U matrixAnumColumns,
                                        U matrixAnumRows,
                                        U matrixBnumColumns,
                                        U matrixBnumRows,
                                        U matrixCnumColumns,
                                        U matrixCnumRows,
                                        MPI_Comm commWorld,
                                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                                        const blasEngineArgumentPackage_gemm<T>& srcPackage,
			                                  int depthManipulation
                                   )
{
  // Note: this is a temporary method that simplifies optimizations by bypassing the Matrix interface
  //       Later on, I can make this prettier and merge with the Matrix-explicit method below.
  //       Also, I only allow method1, not Allgather-based method2

  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  T* foreignA;
  T* foreignB;
  U localDimensionM = (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? matrixAnumRows : matrixAnumColumns);
  U localDimensionN = (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? matrixBnumColumns : matrixBnumRows);
  U localDimensionK = (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? matrixAnumColumns : matrixAnumRows);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);

  U sizeA = matrixAnumRows*matrixAnumColumns;
  U sizeB = matrixBnumRows*matrixBnumColumns;
  U sizeC = matrixCnumRows*matrixCnumColumns;
  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

#ifdef TIMER
  size_t index2 = timer.setStartTime("MM3D::BroadcastPanels");
#endif
  BroadcastPanels(
#ifdef TIMER
    timer,
#endif
    (isRootRow ? matrixA : foreignA), sizeA, isRootRow, pGridCoordZ, rowComm);
  BroadcastPanels(
#ifdef TIMER
    timer,
#endif
    (isRootColumn ? matrixB : foreignB), sizeB, isRootColumn, pGridCoordZ, columnComm);
#ifdef TIMER
  timer.setEndTime("MM3D::BroadcastPanels", index2);
#endif

  matrixAEnginePtr = (isRootRow ? matrixA : foreignA);
  matrixBEnginePtr = (isRootColumn ? matrixB : foreignB);

  // Massive bug fix. Need to use a separate array if beta != 0

  T* matrixCforEnginePtr = matrixC;
  if (srcPackage.beta == 0)
  {
#ifdef TIMER
    size_t index3 = timer.setStartTime("gemm");
#endif
    blasEngine<T,U>::_gemm(matrixAEnginePtr, matrixBEnginePtr, matrixCforEnginePtr, localDimensionM, localDimensionN, localDimensionK,
      (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? localDimensionM : localDimensionK),
      (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? localDimensionK : localDimensionN),
      localDimensionM, srcPackage);
#ifdef TIMER
    timer.setEndTime("gemm", index3);
    size_t index4 = timer.setStartTime("MPI_Allreduce");
#endif
    MPI_Allreduce(MPI_IN_PLACE,matrixCforEnginePtr, sizeC, MPI_DOUBLE, MPI_SUM, depthComm);
#ifdef TIMER
    timer.setEndTime("MPI_Allreduce", index4);
#endif
  }
  else
  {
    // This cancels out any affect beta could have. Beta is just not compatable with MM3D and must be handled separately
     std::vector<T> holdProduct(sizeC,0);
#ifdef TIMER
    size_t index3 = timer.setStartTime("gemm");
#endif
     blasEngine<T,U>::_gemm(matrixAEnginePtr, matrixBEnginePtr, &holdProduct[0], localDimensionM, localDimensionN, localDimensionK,
       (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? localDimensionM : localDimensionK),
       (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? localDimensionK : localDimensionN),
       localDimensionM, srcPackage); 
#ifdef TIMER
    timer.setEndTime("blasEngine::gemm", index3);
    size_t index4 = timer.setStartTime("MPI_Allreduce");
#endif
    MPI_Allreduce(MPI_IN_PLACE, &holdProduct[0], sizeC, MPI_DOUBLE, MPI_SUM, depthComm);
#ifdef TIMER
    timer.setEndTime("MPI_Allreduce", index4);
#endif
    for (U i=0; i<sizeC; i++)
    {
      matrixC[i] = srcPackage.beta*matrixC[i] + holdProduct[i];
    }
  }
  if (!isRootRow) delete[] foreignA;
  if (!isRootColumn) delete[] foreignB;
}


// This algorithm with underlying gemm BLAS routine will allow any Matrix Structure.
//   Of course we will serialize into Square Structure if not in Square Structure already in order to be compatible
//   with BLAS-3 routines.

template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<
	  template<typename,typename, template<typename,typename,int> class> class StructureA,
  	  template<typename,typename, template<typename,typename,int> class> class StructureB,
  	  template<typename,typename, template<typename,typename,int> class> class StructureC,
  	  template<typename,typename,int> class Distribution
	>
void MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
                                        pTimer& timer,
#endif
                                   	    Matrix<T,U,StructureA,Distribution>& matrixA,
                                        Matrix<T,U,StructureB,Distribution>& matrixB,
                                        Matrix<T,U,StructureC,Distribution>& matrixC,
                                        MPI_Comm commWorld,
                                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                                        const blasEngineArgumentPackage_gemm<T>& srcPackage,
                  			                int methodKey, // I chose an integer instead of another template parameter
			                                  int depthManipulation
                                   )
{
  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient

  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  std::vector<T> matrixAEngineVector;
  std::vector<T> matrixBEngineVector;
  std::vector<T> foreignA;
  std::vector<T> foreignB;
  bool serializeKeyA = false;
  bool serializeKeyB = false;
  U localDimensionM = (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? matrixA.getNumRowsLocal() : matrixA.getNumColumnsLocal());
  U localDimensionN = (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? matrixB.getNumColumnsLocal() : matrixB.getNumRowsLocal());
  U localDimensionK = (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? matrixA.getNumColumnsLocal() : matrixA.getNumRowsLocal());

  if (methodKey == 0)
  {
#ifdef TIMER
    size_t index2 = timer.setStartTime("MM3D::_start1");
#endif
    _start1(
#ifdef TIMER
      timer,
#endif
      matrixA,matrixB,commInfo3D,matrixAEnginePtr,matrixBEnginePtr,
      matrixAEngineVector,matrixBEngineVector,foreignA,foreignB,serializeKeyA,serializeKeyB);
#ifdef TIMER
    timer.setEndTime("MM3D::_start1", index2);
#endif
  }
  else if (methodKey == 1)
  {
    serializeKeyA = true;
    serializeKeyB = true;
#ifdef TIMER
    size_t index2 = timer.setStartTime("MM3D::_start2");
#endif
    _start2(
#ifdef TIMER
      timer,
#endif
      matrixA,matrixB,commInfo3D,
      matrixAEngineVector,matrixBEngineVector,serializeKeyA,serializeKeyB);
#ifdef TIMER
    timer.setEndTime("MM3D::_start2", index2);
#endif
  }

  // Assume, for now, that matrixC has Rectangular Structure. In the future, we can always do the same procedure as above, and add a Serialize after the AllReduce

  // Massive bug fix. Need to use a separate array if beta != 0

  T* matrixCforEnginePtr = matrixC.getRawData();
  if (srcPackage.beta == 0)
  {
#ifdef TIMER
    size_t index3 = timer.setStartTime("gemm");
#endif
    blasEngine<T,U>::_gemm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
      matrixCforEnginePtr, localDimensionM, localDimensionN, localDimensionK,
      (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? localDimensionM : localDimensionK),
      (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? localDimensionK : localDimensionN),
      localDimensionM, srcPackage);
#ifdef TIMER
    timer.setEndTime("gemm", index3);
    size_t index4 = timer.setStartTime("MM3D::_end1");
#endif
    _end1(
#ifdef TIMER
      timer,
#endif
      matrixCforEnginePtr,matrixC,commInfo3D);
#ifdef TIMER
    timer.setEndTime("MM3D::_end1", index4);
#endif
   }
   else
   {
     // This cancels out any affect beta could have. Beta is just not compatable with MM3D and must be handled separately
     std::vector<T> holdProduct(matrixC.getNumElems(),0);
#ifdef TIMER
    size_t index3 = timer.setStartTime("gemm");
#endif
     blasEngine<T,U>::_gemm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
       &holdProduct[0], localDimensionM, localDimensionN, localDimensionK,
       (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? localDimensionM : localDimensionK),
       (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? localDimensionK : localDimensionN),
       localDimensionM, srcPackage); 
#ifdef TIMER
    timer.setEndTime("gemm", index3);
    size_t index4 = timer.setStartTime("MM3D::_end1");
#endif
    _end1(
#ifdef TIMER
      timer,
#endif
      &holdProduct[0],matrixC,commInfo3D,1);
#ifdef TIMER
    timer.setEndTime("MM3D::_end1", index4);
#endif
    for (U i=0; i<holdProduct.size(); i++)
    {
      matrixC.getRawData()[i] = srcPackage.beta*matrixC.getRawData()[i] + holdProduct[i];
    }
  }
}

template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<
		template<typename,typename, template<typename,typename,int> class> class StructureA,
  		template<typename,typename, template<typename,typename,int> class> class StructureB,
  		template<typename,typename,int> class Distribution
	>
void MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
                                        pTimer& timer,
#endif
                                   	    Matrix<T,U,StructureA,Distribution>& matrixA,
                                        Matrix<T,U,StructureB,Distribution>& matrixB,
                                        MPI_Comm commWorld,
                                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                                        const blasEngineArgumentPackage_trmm<T>& srcPackage,
					                              int methodKey,						// I chose an integer instead of another template parameter
			                                  int depthManipulation
                                   )
{
  // Need to add timers here. Abort
  abort();

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient

  // Need to do the end_1 fix, same as above. Fix if there is sufficient reason to use TRSM instead of GEMM after testing on Cletus
  assert(0);

  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  std::vector<T> matrixAEngineVector;
  std::vector<T> matrixBEngineVector;
  std::vector<T> foreignA;
  std::vector<T> foreignB;
  bool serializeKeyA = false;
  bool serializeKeyB = false;
  U localDimensionM = matrixB.getNumRowsLocal();
  U localDimensionN = matrixB.getNumColumnsLocal();

  // soon, we will need a methodKey for the different MM algs
  if (srcPackage.side == blasEngineSide::AblasLeft)
  {
    if (methodKey == 0)
    {
      _start1(
#ifdef TIMER
        timer,
#endif
        matrixA, matrixB, commInfo3D, matrixAEnginePtr, matrixBEnginePtr, matrixAEngineVector, matrixBEngineVector, foreignA, foreignB,
        serializeKeyA, serializeKeyB);
    }
    else if (methodKey == 1)
    {
      serializeKeyA = true;
      serializeKeyB = true;
      _start2(
#ifdef TIMER
        timer,
#endif
        matrixA, matrixB, commInfo3D, matrixAEngineVector, matrixBEngineVector,
        serializeKeyA, serializeKeyB);
    }
    blasEngine<T,U>::_trmm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
      localDimensionM, localDimensionN, localDimensionM, (srcPackage.order == blasEngineOrder::AblasColumnMajor ? localDimensionM : localDimensionN),
      srcPackage);
  }
  else
  {
    if (methodKey == 0)
    {
      _start1(
#ifdef TIMER
        timer,
#endif
        matrixB, matrixA, commInfo3D, matrixBEnginePtr, matrixAEnginePtr, matrixBEngineVector, matrixAEngineVector, foreignB, foreignA, serializeKeyB, serializeKeyA);
    }
    else if (methodKey == 1)
    {
      serializeKeyA = true;
      serializeKeyB = true;
      _start2(
#ifdef TIMER
        timer,
#endif
        matrixB, matrixA, commInfo3D, matrixBEngineVector, matrixAEngineVector, serializeKeyB, serializeKeyA);
    }
    blasEngine<T,U>::_trmm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
      localDimensionM, localDimensionN, localDimensionN, (srcPackage.order == blasEngineOrder::AblasColumnMajor ? localDimensionM : localDimensionN),
      srcPackage);
  }
  // We will follow the standard here: matrixA is always the triangular matrix. matrixB is always the rectangular matrix
  _end1(
#ifdef TIMER
    timer,
#endif
    (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),matrixB,commInfo3D);
}

template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<
		template<typename,typename, template<typename,typename,int> class> class StructureA,
  		template<typename,typename, template<typename,typename,int> class> class StructureB,
  		template<typename,typename,int> class Distribution
	>
void MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
                                        pTimer& timer,
#endif
                                   	    Matrix<T,U,StructureA,Distribution>& matrixA,
                                        Matrix<T,U,StructureB,Distribution>& matrixB,
                                        MPI_Comm commWorld,
                                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                                        const blasEngineArgumentPackage_syrk<T>& srcPackage
                                   )
{
/*
  // Not correct right now. Will fix later
  MPI_Abort(commWorld, -1);

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm transComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);

  std::vector<T>& dataA = matrixA.getVectorData(); 
  std::vector<T> dataAtrans = dataA;			// need to make a copy here I think
  U sizeA = matrixA.getNumElems();
  std::vector<T> foreignA;
  std::vector<T> foreignAtrans;
  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootTrans = ((pGridCoordY == pGridCoordZ) ? true : false);

  BroadcastPanels((isRootRow ? dataA : foreignA), sizeA, isRootRow, pGridCoordZ, rowComm);
  BroadcastPanels((isRootTrans ? dataAtrans : foreignAtrans), sizeA, isRootTrans, pGridCoordZ, transComm);

  // Right now, foreignA and/or foreignAtrans might be empty if this processor is the rowRoot or the transRoot
  Matrix<T,U,MatrixStructureRectangle,Distribution> helperA(std::vector<T>(), localDimensionN, localDimensionN, localDimensionN, localDimensionN);
  T* matrixAforEnginePtr = getEnginePtr(matrixA, helperA, (isRootRow ? dataA : foreignA), isRootRow);

  // We assume that matrixB is Square for now. No reason to believe otherwise

  std::vector<T>& matrixBforEngine = matrixB.getVectorData();
  U numElems = matrixB.getNumElems();				// We assume that the user initialized matrixC correctly, even for TRMM

  blasEngine<T,U>::_syrk(matrixAforEnginePtr, &matrixBforEngine[0], (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? localDimensionN : localDimensionK),
    (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? localDimensionN : localDimensionK),
    (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? localDimensionK : localDimensionN),
    (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? localDimensionK : localDimensionN),
    srcPackage);

  // in a syrk, we will end up with a symmetric matrix, so we should serialize into packed buffer first to avoid half the communication!
  MPI_Allreduce(MPI_IN_PLACE, &matrixBforEngine[0], numElems, MPI_DOUBLE, MPI_SUM, depthComm);

*/
}

template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<
		template<typename,typename, template<typename,typename,int> class> class StructureA,
  		template<typename,typename, template<typename,typename,int> class> class StructureB,
  		template<typename,typename, template<typename,typename,int> class> class StructureC,
  		template<typename,typename,int> class Distribution
	>
void MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
                pTimer& timer,
#endif
                Matrix<T,U,StructureA,Distribution>& matrixA,
                Matrix<T,U,StructureB,Distribution>& matrixB,
				        Matrix<T,U,StructureC,Distribution>& matrixC,
				        U matrixAcutXstart,
				        U matrixAcutXend,
				        U matrixAcutYstart,
				        U matrixAcutYend,
				        U matrixBcutZstart,
				        U matrixBcutZend,
				        U matrixBcutXstart,
				        U matrixBcutXend,
				        U matrixCcutZstart,
				        U matrixCcutZend,
				        U matrixCcutYstart,
				        U matrixCcutYend,
				        MPI_Comm commWorld,
                std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
				        const blasEngineArgumentPackage_gemm<T>& srcPackage,
				        bool cutA,
				        bool cutB,
				        bool cutC,
                int methodKey, // I chose an integer instead of another template parameter
			          int depthManipulation
                                   )
{
  // We will set up 3 matrices and call the method above.

  U rangeA_x = matrixAcutXend-matrixAcutXstart;
  U rangeA_y = matrixAcutYend-matrixAcutYstart;
  U rangeB_z = matrixBcutZend-matrixBcutZstart;
  U rangeB_x = matrixBcutXend-matrixBcutXstart;
  U rangeC_z = matrixCcutZend - matrixCcutZstart;
  U rangeC_y = matrixCcutYend - matrixCcutYstart; 

  U sizeA = matrixA.getNumElems(rangeA_x, rangeA_y);
  U sizeB = matrixB.getNumElems(rangeB_z, rangeB_x);
  U sizeC = matrixC.getNumElems(rangeC_y, rangeC_z);

  int size;
  int pGridDimensionSize;

#ifdef TIMER
  size_t index1 = timer.setStartTime("MPI_Comm_size");
#endif
  MPI_Comm_size(std::get<0>(commInfo3D), &pGridDimensionSize);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_size", index1);
#endif

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
#ifdef TIMER
  size_t index2 = timer.setStartTime("MM3D::getSubMatrix");
#endif
  Matrix<T,U,StructureA,Distribution> matA = getSubMatrix(
#ifdef TIMER
    timer,
#endif
    matrixA, matrixAcutXstart, matrixAcutXend, matrixAcutYstart, matrixAcutYend, pGridDimensionSize, cutA);
#ifdef TIMER
  timer.setEndTime("MM3D::getSubMatrix",index2);
  size_t index3 = timer.setStartTime("MM3D::getSubMatrix");
#endif
  Matrix<T,U,StructureB,Distribution> matB = getSubMatrix(
#ifdef TIMER
    timer,
#endif
    matrixB, matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, pGridDimensionSize, cutB);
#ifdef TIMER
  timer.setEndTime("MM3D::getSubMatrix",index3);
  size_t index4 = timer.setStartTime("MM3D::getSubMatrix");
#endif
  Matrix<T,U,StructureC,Distribution> matC = getSubMatrix(
#ifdef TIMER
    timer,
#endif
    matrixC, matrixCcutZstart, matrixCcutZend, matrixCcutYstart, matrixCcutYend, pGridDimensionSize, cutC);
#ifdef TIMER
  timer.setEndTime("MM3D::getSubMatrix",index4);
#endif

#ifdef TIMER
  size_t index5 = timer.setStartTime("MM3D::Multiply");
#endif
  Multiply(
#ifdef TIMER
    timer,
#endif
    (cutA ? matA : matrixA), (cutB ? matB : matrixB), (cutC ? matC : matrixC), commWorld, commInfo3D, srcPackage, methodKey, depthManipulation);
#ifdef TIMER
  timer.setEndTime("MM3D::Multiply", index5);
#endif

  // reverse serialize, to put the solved piece of matrixC into where it should go.
  if (cutC)
  {
#ifdef TIMER
    size_t index5 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,StructureC,StructureC>::Serialize(matrixC, matC,
      matrixCcutZstart, matrixCcutZend, matrixCcutYstart, matrixCcutYend, true);
#ifdef TIMER
    timer.setEndTime("Serializer", index5);
#endif
  }
}


template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<
		template<typename,typename, template<typename,typename,int> class> class StructureA,
  		template<typename,typename, template<typename,typename,int> class> class StructureB,
  		template<typename,typename,int> class Distribution
	 >
void MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
              pTimer& timer,
#endif
				      Matrix<T,U,StructureA,Distribution>& matrixA,
				      Matrix<T,U,StructureB,Distribution>& matrixB,
				      U matrixAcutXstart,
				      U matrixAcutXend,
				      U matrixAcutYstart,
				      U matrixAcutYend,
				      U matrixBcutZstart,
				      U matrixBcutZend,
				      U matrixBcutXstart,
				      U matrixBcutXend,
				      MPI_Comm commWorld,
              std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
				      const blasEngineArgumentPackage_trmm<T>& srcPackage,
				      bool cutA,
				      bool cutB,
              int methodKey, // I chose an integer instead of another template parameter
			        int depthManipulation
                                    )
{
  // Need to add timers here. Abort.
  abort();

  // We will set up 3 matrices and call the method above.

  U rangeA_x = matrixAcutXend-matrixAcutXstart;
  U rangeA_y = matrixAcutYend-matrixAcutYstart;
  U rangeB_x = matrixBcutXend-matrixBcutXstart;
  U rangeB_z = matrixBcutZend-matrixBcutZstart;

  int pGridDimensionSize;
  MPI_Comm_size(std::get<0>(commInfo3D), &pGridDimensionSize);

  U sizeA = matrixA.getNumElems(rangeA_x, rangeA_y);
  U sizeB = matrixB.getNumElems(rangeB_z, rangeB_x);

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  Matrix<T,U,StructureA,Distribution> matA = getSubMatrix(
#ifdef TIMER
    timer,
#endif
    matrixA, matrixAcutXstart, matrixAcutXend, matrixAcutYstart, matrixAcutYend, pGridDimensionSize, cutA);
  Matrix<T,U,StructureB,Distribution> matB = getSubMatrix(
#ifdef TIMER
    timer,
#endif
    matrixB, matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, pGridDimensionSize, cutB);
  Multiply(
#ifdef TIMER
    timer,
#endif
    matA, matB, commWorld, srcPackage, methodKey, depthManipulation);

  // reverse serialize, to put the solved piece of matrixC into where it should go. Only if we need to
  if (cutB)
  {
    Serializer<T,U,StructureB,StructureB>::Serialize(matrixB, matB,
      matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, true);
  }
}


template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<
		template<typename,typename, template<typename,typename,int> class> class StructureA,
  		template<typename,typename, template<typename,typename,int> class> class StructureB,
  		template<typename,typename,int> class Distribution
	 >
void MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
              pTimer& timer,
#endif
				      Matrix<T,U,StructureA,Distribution>& matrixA,
				      Matrix<T,U,StructureB,Distribution>& matrixB,
				      U matrixAcutXstart,
				      U matrixAcutXend,
				      U matrixAcutYstart,
				      U matrixAcutYend,
				      U matrixBcutZstart,
				      U matrixBcutZend,
				      U matrixBcutXstart,
				      U matrixBcutXend,
				      MPI_Comm commWorld,
              std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
				      const blasEngineArgumentPackage_syrk<T>& srcPackage,
				      bool cutA,
				      bool cutB
				    )
{
/*
  // Not correct right now. Will fix later
  MPI_Abort(commWorld, -1);
  // We will set up 3 matrices and call the method above.

  U rangeA_x = matrixAcutXend-matrixAcutXstart;
  U rangeA_y = matrixAcutYend-matrixAcutYstart;
  U rangeB_z = matrixBcutZend-matrixBcutZstart;
  U rangeB_x = matrixBcutXend-matrixBcutXstart;
  U globalDiffA = matrixA.getNumRowsGlobal() / matrixA.getNumRowsLocal();		// picked rows arbitrarily
  U globalDiffB = matrixB.getNumRowsGlobal() / matrixB.getNumRowsLocal();		// picked rows arbitrarily

  U sizeA = matrixA.getNumElems(rangeA_x, rangeA_y);
  U sizeB = matrixB.getNumElems(rangeB_z, rangeB_x);

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  Matrix<T,U,StructureA,Distribution> subMatrixA(std::vector<T>(), rangeA_x, rangeA_y, rangeA_x*globalDiffA, rangeA_y*globalDiffA);
  Matrix<T,U,StructureB,Distribution> subMatrixB(std::vector<T>(), rangeB_z, rangeB_x, rangeB_z*globalDiffB, rangeB_x*globalDiffB);
  Matrix<T,U,StructureA,Distribution>& matA = getSubMatrix(matrixA, subMatrixA, matrixAcutXstart, matrixAcutXend, matrixAcutYstart, matrixAcutYend, globalDiffA, cutA);
  Matrix<T,U,StructureB,Distribution>& matB = getSubMatrix(matrixB, subMatrixB, matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, globalDiffB, cutB);

  Multiply(matA, matB, rangeA_x, rangeA_y, rangeB_z, commWorld, srcPackage);

  // reverse serialize, to put the solved piece of matrixC into where it should go. Only if we need to
  if (cutB)
  {
    Serializer<T,U,StructureB,StructureB>::Serialize(matrixB, matB,
      matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, true);
  }
*/
}


template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<template<typename,typename,int> class Distribution,
  template<typename,typename, template<typename,typename,int> class> class StructureArg1,
  template<typename,typename, template<typename,typename,int> class> class StructureArg2,
  typename tupleStructure>
void MM3D<T,U,blasEngine>::_start1(
#ifdef TIMER
          pTimer& timer,
#endif
					Matrix<T,U,StructureArg1,Distribution>& matrixA,
					Matrix<T,U,StructureArg2,Distribution>& matrixB,
					tupleStructure& commInfo3D,
					T*& matrixAEnginePtr,
					T*& matrixBEnginePtr,
					std::vector<T>& matrixAEngineVector,
					std::vector<T>& matrixBEngineVector,
					std::vector<T>& foreignA,
					std::vector<T>& foreignB,
					bool& serializeKeyA,
					bool& serializeKeyB
				  )
{
  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);

  U localDimensionM = matrixA.getNumRowsLocal();
  U localDimensionN = matrixB.getNumColumnsLocal();
  U localDimensionK = matrixA.getNumColumnsLocal();
  std::vector<T>& dataA = matrixA.getVectorData(); 
  std::vector<T>& dataB = matrixB.getVectorData();
  U sizeA = matrixA.getNumElems();
  U sizeB = matrixB.getNumElems();
  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

#ifdef TIMER
  size_t index1 = timer.setStartTime("BroadcastPanels");
#endif
  BroadcastPanels(
#ifdef TIMER
    timer,
#endif
    (isRootRow ? dataA : foreignA), sizeA, isRootRow, pGridCoordZ, rowComm);
#ifdef TIMER
  timer.setEndTime("BroadcastPanels", index1);
  size_t index2 = timer.setStartTime("BroadcastPanels");
#endif
  BroadcastPanels(
#ifdef TIMER
    timer,
#endif
    (isRootColumn ? dataB : foreignB), sizeB, isRootColumn, pGridCoordZ, columnComm);
#ifdef TIMER
  timer.setEndTime("BroadcastPanels", index2);
#endif

  matrixAEnginePtr = (isRootRow ? &dataA[0] : &foreignA[0]);
  matrixBEnginePtr = (isRootColumn ? &dataB[0] : &foreignB[0]);
  if ((!std::is_same<StructureArg1<T,U,Distribution>,MatrixStructureRectangle<T,U,Distribution>>::value)
    && (!std::is_same<StructureArg1<T,U,Distribution>,MatrixStructureSquare<T,U,Distribution>>::value))		// compile time if statement. Branch prediction should be correct.
  {
    serializeKeyA = true;
    Matrix<T,U,MatrixStructureRectangle,Distribution> helperA(std::vector<T>(), localDimensionK, localDimensionM, localDimensionK, localDimensionM);
    getEnginePtr(
#ifdef TIMER
      timer,
#endif
      matrixA, helperA, (isRootRow ? dataA : foreignA), isRootRow);
    matrixAEngineVector = std::move(helperA.getVectorData());
  }
  if ((!std::is_same<StructureArg2<T,U,Distribution>,MatrixStructureRectangle<T,U,Distribution>>::value)
    && (!std::is_same<StructureArg2<T,U,Distribution>,MatrixStructureSquare<T,U,Distribution>>::value))		// compile time if statement. Branch prediction should be correct.
  {
    serializeKeyB = true;
    Matrix<T,U,MatrixStructureRectangle,Distribution> helperB(std::vector<T>(), localDimensionN, localDimensionK, localDimensionN, localDimensionK);
    getEnginePtr(
#ifdef TIMER
      timer,
#endif
      matrixB, helperB, (isRootColumn ? dataB : foreignB), isRootColumn);
    matrixBEngineVector = std::move(helperB.getVectorData());
  }
}


template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<template<typename,typename, template<typename,typename,int> class> class StructureArg,template<typename,typename,int> class Distribution,
  typename tupleStructure>
void MM3D<T,U,blasEngine>::_end1(
#ifdef TIMER
          pTimer& timer,
#endif
					T* matrixEnginePtr,
					Matrix<T,U,StructureArg,Distribution>& matrix,
					tupleStructure& commInfo3D,
          int dir
				)
{
  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);

  U numElems = matrix.getNumElems();

  // Prevents buffer aliasing, which MPI does not like.
  if ((dir) || (matrixEnginePtr == matrix.getRawData()))
  {
#ifdef TIMER
    size_t index1 = timer.setStartTime("MPI_Allreduce");
#endif
    MPI_Allreduce(MPI_IN_PLACE,matrixEnginePtr, numElems, MPI_DOUBLE, MPI_SUM, depthComm);
#ifdef TIMER
    timer.setEndTime("MPI_Allreduce", index1);
#endif
  }
  else
  {
#ifdef TIMER
    size_t index1 = timer.setStartTime("MPI_Allreduce");
#endif
    MPI_Allreduce(matrixEnginePtr, matrix.getRawData(), numElems, MPI_DOUBLE, MPI_SUM, depthComm);
#ifdef TIMER
    timer.setEndTime("MPI_Allreduce", index1);
#endif
  }

}

template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<template<typename,typename,int> class Distribution,
  template<typename,typename, template<typename,typename,int> class> class StructureArg1,
  template<typename,typename, template<typename,typename,int> class> class StructureArg2,
  typename tupleStructure>
void MM3D<T,U,blasEngine>::_start2(
#ifdef TIMER
          pTimer& timer,
#endif
					Matrix<T,U,StructureArg1,Distribution>& matrixA,
					Matrix<T,U,StructureArg2,Distribution>& matrixB,
					tupleStructure& commInfo3D,
					std::vector<T>& matrixAEngineVector,
					std::vector<T>& matrixBEngineVector,
					bool& serializeKeyA,
					bool& serializeKeyB
				  )
{
  // Need to add timers here. Abort.
  abort();

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);
  int rowCommSize,columnCommSize,depthCommSize;
  MPI_Comm_size(rowComm, &rowCommSize);
  MPI_Comm_size(columnComm, &columnCommSize);
  MPI_Comm_size(depthComm, &depthCommSize);

/* Debugging notes
  How do I deal with serialization? Like, what if matrixA or B is a UT or LT? I do not want to communicate more than the packed data
    in that case obviously, but I also need to make sure that things are still in order, which I think is harder now than it was in _start1( method )
*/

  U localDimensionM = matrixA.getNumRowsLocal();
  U localDimensionN = matrixB.getNumColumnsLocal();
  U localDimensionK = matrixA.getNumColumnsLocal();
  std::vector<T>& dataA = matrixA.getVectorData(); 
  std::vector<T>& dataB = matrixB.getVectorData();
  U sizeA = matrixA.getNumElems();
  U sizeB = matrixB.getNumElems();

  // Allgathering matrixA is no problem because we store matrices column-wise
  // Note: using rowCommSize here instead of rowCommSize shouldn't matter for a 3D grid with uniform 2D slices
  U localNumRowsA = matrixA.getNumRowsLocal();
  U localNumColumnsA = matrixA.getNumColumnsLocal();
  U modA = localNumColumnsA%rowCommSize;
  U divA = localNumColumnsA/rowCommSize;
  U gatherSizeA = (modA == 0 ? sizeA : (divA+1)*rowCommSize*localNumRowsA);
  std::vector<T> collectMatrixA(gatherSizeA);
  U shift = (pGridCoordZ + pGridCoordX) % rowCommSize;
  U dataAOffset = localNumRowsA*divA*shift;
  dataAOffset += std::min(shift,modA)*localNumRowsA;
  // matrixAEngineVector can stay with sizeA elements, because when we move data into it, we will get rid of the zeros.
  matrixAEngineVector.resize(sizeA);			// will need to change upon Serialize changes
  U messageSizeA = gatherSizeA/rowCommSize;

  // Some processors will need to serialize
  if (modA && (shift >= modA))
  {
    std::vector<T> partitionMatrixA(messageSizeA,0);
    memcpy(&partitionMatrixA[0], &dataA[dataAOffset], (messageSizeA - localNumRowsA)*sizeof(T));  // truncation should be fine here. Rest is zeros
    MPI_Allgather(&partitionMatrixA[0], messageSizeA, MPI_DOUBLE, &collectMatrixA[0], messageSizeA, MPI_DOUBLE, rowComm);
  }
  else
  {
    MPI_Allgather(&dataA[dataAOffset], messageSizeA, MPI_DOUBLE, &collectMatrixA[0], messageSizeA, MPI_DOUBLE, rowComm);
  }


  // If pGridCoordZ != 0, then we need to re-shuffle the data. AllGather did not put into optimal order.
  if (pGridCoordZ == 0)
  {
    if (gatherSizeA == sizeA)
    {
      matrixAEngineVector = std::move(collectMatrixA);
    }
    else
    {
      // first serialize into collectMatrixA itself by removing excess zeros
      // then move into matrixAEngineVector
      // Later optimizaion: avoid copying unless writeIndex < readInde
      U readIndex = 0;
      U writeIndex = 0;
      for (U i=0; i<rowCommSize; i++)
      {
        U writeSize = (((modA == 0) || (i < modA)) ? messageSizeA : divA*localNumRowsA);
        memcpy(&collectMatrixA[writeIndex], &collectMatrixA[readIndex], writeSize*sizeof(T));
        writeIndex += writeSize;
        readIndex += messageSizeA;
      }
      collectMatrixA.resize(sizeA);
      matrixAEngineVector = std::move(collectMatrixA);
    }
  }
  else
  {
    matrixAEngineVector.resize(sizeA);
    U shuffleAoffset = messageSizeA*((rowCommSize - pGridCoordZ)%rowCommSize);
    U stepA = 0;
    for (U i=0; i<rowCommSize; i++)
    {
      // Don't really need the 2nd if statement condition like the one above. Actually, neither do
      U writeSize = (((i % rowCommSize) < modA) ? messageSizeA : divA*localNumRowsA);
      memcpy(&matrixAEngineVector[stepA], &collectMatrixA[shuffleAoffset], writeSize*sizeof(T));
      stepA += writeSize;
      shuffleAoffset += messageSizeA;
      shuffleAoffset %= gatherSizeA;
    }
  }

  
  // Now we Allgather partitions of matrix B
  U localNumRowsB = matrixB.getNumRowsLocal();
  U localNumColumnsB = matrixB.getNumColumnsLocal();
  U modB = localNumRowsB%columnCommSize;
  U divB = localNumRowsB/columnCommSize;
  U blockLengthB = (modB == 0 ? divB : divB +1);
  shift = (pGridCoordZ + pGridCoordY) % columnCommSize;
  U dataBOffset = divB*shift;
  dataBOffset += std::min(shift, modB);       // WATCH: could be wrong
  U gatherSizeB = blockLengthB*columnCommSize*localNumColumnsB;
  U messageSizeB = gatherSizeB/columnCommSize;
  std::vector<T> collectMatrixB(gatherSizeB);			// will need to change upon Serialize changes
  std::vector<T> partitionMatrixB(messageSizeB,0);			// Important to fill with zeros first
  // Special serialize. Can't use my MatrixSerializer here.
  U writeSize = (((modB == 0)) || (shift < modB) ? blockLengthB : blockLengthB-1);
  for (U i=0; i<localNumColumnsB; i++)
  {
    memcpy(&partitionMatrixB[i*blockLengthB], &matrixB.getRawData()[dataBOffset + i*localNumRowsB], writeSize*sizeof(T));
  }
  MPI_Allgather(&partitionMatrixB[0], partitionMatrixB.size(), MPI_DOUBLE, &collectMatrixB[0], partitionMatrixB.size(), MPI_DOUBLE, columnComm);
/*
  // Allgathering matrixB is a problem for AMPI when using derived datatypes
  MPI_Datatype matrixBcolumnData;
  MPI_Type_vector(localNumColumnsB,blockLengthB,localNumRowsB,MPI_DOUBLE,&matrixBcolumnData);
  MPI_Type_commit(&matrixBcolumnData);
  U messageSizeB = sizeB/columnCommSize;
  MPI_Allgather(&dataB[dataBOffset], 1, matrixBcolumnData, &collectMatrixB[0], messageSizeB, MPI_DOUBLE, columnComm);
*/
  // Then need to re-shuffle the data in collectMatrixB because of the format Allgather puts the received data in 
  // Note: there is a particular order to it beyond the AllGather order. Depends what z coordinate we are on (that determines the shift)

  if ((rowCommSize == 1) && (columnCommSize == 1) && (depthCommSize == 1))
  {
    matrixBEngineVector = std::move(collectMatrixB);
  }
  else
  {
    matrixBEngineVector.resize(sizeB);			// will need to change upon Serialize changes
    // Open question: Is this the most cache-efficient way to reshuffle the data?
    for (U i=0; i<localNumColumnsB; i++)
    {
      // We always start in the same offset in the gatherBuffer
      U shuffleBoffset = messageSizeB*((columnCommSize - pGridCoordZ)%columnCommSize);
      U saveStepB = i*localNumRowsB;
      for (U j=0; j<columnCommSize; j++)
      {
        U writeSize = (((modB == 0) || ((j % columnCommSize) < modB)) ? blockLengthB : blockLengthB-1);
        U saveOffsetB = shuffleBoffset + i*blockLengthB;
/*
        if (saveStepB+writeSize > matrixBEngineVector.size())
        {
          std::cout << "saveStepB - " << saveStepB << ", writeSize - " << writeSize << ", matrixBEngineVector size - " << matrixBEngineVector.size() << ", j - " << j << ", columnCommSize - " << columnCommSize << ", i - " << i << ", localNumColumnsB - " << localNumColumnsB << ", localNumRowsB - " << localNumRowsB << ", blockSize - " << blockLengthB << ", sizeB - " << sizeB << ", size of collectMatrixB - " << collectMatrixB.size() << ", matrixB things - " << matrixB.getNumRowsLocal() << " " << matrixB.getNumColumnsLocal() << " " << matrixB.getNumElems() << std::endl;
          assert(saveStepB+writeSize <= matrixBEngineVector.size());
        }
*/
        memcpy(&matrixBEngineVector[saveStepB], &collectMatrixB[saveOffsetB], writeSize*sizeof(T));
        saveStepB += writeSize;
        shuffleBoffset += messageSizeB;
        shuffleBoffset %= gatherSizeB;
      }
    }
  }

/*
  if ((!std::is_same<StructureArg1<T,U,Distribution>,MatrixStructureRectangle<T,U,Distribution>>::value)
    && (!std::is_same<StructureArg1<T,U,Distribution>,MatrixStructureSquare<T,U,Distribution>>::value))		// compile time if statement. Branch prediction should be correct.
  {
    Matrix<T,U,MatrixStructureRectangle,Distribution> helperA(std::vector<T>(), localDimensionK, localDimensionM, localDimensionK, localDimensionM);
    getEnginePtr(matrixA, helperA, (isRootRow ? dataA : foreignA), isRootRow);
    matrixAEngineVector = std::move(helperA.getVectorData());
  }
  if ((!std::is_same<StructureArg2<T,U,Distribution>,MatrixStructureRectangle<T,U,Distribution>>::value)
    && (!std::is_same<StructureArg2<T,U,Distribution>,MatrixStructureSquare<T,U,Distribution>>::value))		// compile time if statement. Branch prediction should be correct.
  {
    Matrix<T,U,MatrixStructureRectangle,Distribution> helperB(std::vector<T>(), localDimensionN, localDimensionK, localDimensionN, localDimensionK);
    getEnginePtr(matrixB, helperB, (isRootColumn ? dataB : foreignB), isRootColumn);
    matrixBEngineVector = std::move(helperB.getVectorData());
  }
*/
}



template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
void MM3D<T,U,blasEngine>::BroadcastPanels(
#ifdef TIMER
            pTimer& timer,
#endif
						std::vector<T>& data,
						U size,
						bool isRoot,
						int pGridCoordZ,
						MPI_Comm panelComm
					   )
{
  if (isRoot)
  {
#ifdef TIMER
    size_t index1 = timer.setStartTime("MPI_Bcast");
#endif
    MPI_Bcast(&data[0], size, MPI_DOUBLE, pGridCoordZ, panelComm);
#ifdef TIMER
    timer.setEndTime("MPI_Bcast", index1);
#endif
  }
  else
  {
    data.resize(size);
#ifdef TIMER
    size_t index1 = timer.setStartTime("MPI_Bcast");
#endif
    MPI_Bcast(&data[0], size, MPI_DOUBLE, pGridCoordZ, panelComm);
#ifdef TIMER
    timer.setEndTime("MPI_Bcast", index1);
#endif
  }
}

template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
void MM3D<T,U,blasEngine>::BroadcastPanels(
#ifdef TIMER
            pTimer& timer,
#endif
						T*& data,
						U size,
						bool isRoot,
						int pGridCoordZ,
						MPI_Comm panelComm
					   )
{
  if (isRoot)
  {
#ifdef TIMER
    size_t index1 = timer.setStartTime("MPI_Bcast");
#endif
    MPI_Bcast(data, size, MPI_DOUBLE, pGridCoordZ, panelComm);
#ifdef TIMER
    timer.setEndTime("MPI_Bcast", index1);
#endif
  }
  else
  {
    data = new double[size];
#ifdef TIMER
    size_t index1 = timer.setStartTime("MPI_Bcast");
#endif
    MPI_Bcast(data, size, MPI_DOUBLE, pGridCoordZ, panelComm);
#ifdef TIMER
    timer.setEndTime("MPI_Bcast", index1);
#endif
  }
}


template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
void MM3D<T,U,blasEngine>::getEnginePtr(
#ifdef TIMER
          pTimer& timer,
#endif
					Matrix<T,U,StructureArg, Distribution>& matrixArg,
					Matrix<T,U,MatrixStructureRectangle, Distribution>& matrixDest,
					std::vector<T>& data,
					bool isRoot
				     )
{
  // Need to separate the below out into its own function that will not get instantied into object code
  //   unless it passes the test above. This avoids template-enduced template compiler errors
  if (!isRoot)
  {
    Matrix<T,U,StructureArg,Distribution> matrixToSerialize(std::move(data), matrixArg.getNumColumnsLocal(),
      matrixArg.getNumRowsLocal(), matrixArg.getNumColumnsGlobal(), matrixArg.getNumRowsGlobal(), true);
#ifdef TIMER
    size_t index1 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,StructureArg,MatrixStructureRectangle>::Serialize(matrixToSerialize, matrixDest);
#ifdef TIMER
    timer.setEndTime("Serializer", index1);
#endif
  }
  else
  {
    // If code path gets here, StructureArg must be a LT or UT, so we need to serialize into a Square, not a Rectangle
#ifdef TIMER
    size_t index1 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,StructureArg,MatrixStructureRectangle>::Serialize(matrixArg, matrixDest);
#ifdef TIMER
    timer.setEndTime("Serializer", index1);
#endif
  }
}


template<typename T, typename U, template<typename,typename> class blasEngine>							// Defaulted to cblasEngine
template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
Matrix<T,U,StructureArg,Distribution> MM3D<T,U,blasEngine>::getSubMatrix(
#ifdef TIMER
                pTimer& timer,
#endif
								Matrix<T,U,StructureArg, Distribution>& srcMatrix,
								U matrixArgColumnStart,
								U matrixArgColumnEnd,
								U matrixArgRowStart,
								U matrixArgRowEnd,
		            int pGridDimensionSize, 
    						bool getSub
						       )
{
  if (getSub)
  {
    U numColumns = matrixArgColumnEnd - matrixArgColumnStart;
    U numRows = matrixArgRowEnd - matrixArgRowStart;
    Matrix<T,U,StructureArg,Distribution> fillMatrix(std::vector<T>(), numColumns, numRows, numColumns*pGridDimensionSize, numRows*pGridDimensionSize);
#ifdef TIMER
    size_t index1 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,StructureArg,StructureArg>::Serialize(srcMatrix, fillMatrix,
      matrixArgColumnStart, matrixArgColumnEnd, matrixArgRowStart, matrixArgRowEnd);
#ifdef TIMER
    timer.setEndTime("Serializer", index1);
#endif
    return fillMatrix;			// I am returning an rvalue
  }
  else
  {
    // return cheap garbage.
    return Matrix<T,U,StructureArg,Distribution>(0,0,1,1);
  }
}
