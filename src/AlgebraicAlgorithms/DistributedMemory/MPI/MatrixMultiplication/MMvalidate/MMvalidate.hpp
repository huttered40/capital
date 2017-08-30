/* Author: Edward Hutter */

template<typename T, typename U, template<typename,typename> class blasEngine>
template<
          template<typename,typename, template<typename,typename,int> class> class Structure,
          template<typename,typename,int> class Distribution
        >
T MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,Structure,Distribution>& matrixSol,
                        U dimensionX,
                        U dimensionY,
                        U dimensionZ,
                        int blasEngineInfo
                      )
{
/*  
T* matrixAtoSerialize = matrixA.getData(); 
  T* matrixBtoSerialize = matrixB.getData();
  T* matrixAforEngine = nullptr;
  T* matrixBforEngine = nullptr;
  int infoA = 0;
  int infoB = 0;
  Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixAtoSerialize, matrixAforEngine, dimensionX, dimensionY, infoA);
  Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixBtoSerialize, matrixBforEngine, dimensionY, dimensionZ, infoB);

  // infoA and infoB have information as to what kind of memory allocations occured within Serializer

  T* matrixCforEngine = matrixC.getData();
  U numElems = matrixC.getNumElems();

  // Assume for now that first 2 bits give 4 possibilies
  //   0 -> _gemm
  //   1 -> _trmm
  //   2 -> ?
  //   3 -> ?
  bool helper1 = blasEngineInfo&0x1;
  blasEngineInfo >>= 1;
  bool helper2 = blasEngineInfo&0x1;
  blasEngineInfo >>= 1;
  int whichRoutine = static_cast<int>(helper2)*2 + static_cast<int>(helper1);
  switch (whichRoutine)
  {
    case 0:
    {
      blasEngine<T,U>::_gemm(matrixAforEngine, matrixBforEngine, matrixCforEngine, dimensionX, dimensionY,
        dimensionX, dimensionZ, dimensionY, dimensionZ, 1., 1., dimensionY, dimensionX, dimensionY, blasEngineInfo);
      break;
    }
    case 1:
    {
      int checkOrder = (0x2 & (blasEngineInfo>>1));		// check the 2nd bit to see if square matrix is on left or right
      blasEngine<T,U>::_trmm(matrixAforEngine, matrixBforEngine, (checkOrder ? dimensionX : dimensionY),
        (checkOrder ? dimensionZ : dimensionX), 1., (checkOrder ? dimensionY : dimensionX), (checkOrder ? dimensionX : dimensionY), blasEngineInfo);
      break;
    }
    case 2:
    {
      break;
    }
    case 3:
    {
      break;
    }
  }

  if (infoA == 2)
  {
    delete[] matrixAforEngine;
  }
  if (infoB == 2)
  {
    delete[] matrixBforEngine;
  }
*/
}

template<typename T, typename U, template<typename,typename> class blasEngine>
template<
          template<typename,typename, template<typename,typename,int> class> class Structure,
          template<typename,typename,int> class Distribution
        >
T MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,Structure,Distribution>& matrixSol,
                        U matrixAcutXstart,
                        U matrixAcutXend,
                        U matrixAcutYstart,
                        U matrixAcutYend,
                        U matrixBcutXstart,
                        U matrixBcutXend,
                        U matrixBcutZstart,
                        U matrixBcutZend,
                        U matrixCcutYstart,
                        U matrixCcutYend,
                        U matrixCcutZstart,
                        U matrixCcutZend,
                        int blasEngineInfo
                      )
{
/*
  U rangeA_x = matrixAcutXend-matrixAcutXstart;
  U rangeA_y = matrixAcutYend-matrixAcutYstart;
  U rangeB_x = matrixBcutXend-matrixBcutXstart;
  U rangeB_z = matrixBcutZend-matrixBcutZstart;

  T* dataA;
  T* dataB;
  U sizeA = matrixA.getNumElems(rangeA_x, rangeA_y);
  U sizeB = matrixB.getNumElems(rangeB_x, rangeB_z);

  // No clear way to prevent a needless copy if the cut dimensions of a matrix are full.
  dataA = new T[sizeA];
  T* matAsource = matrixA.getData();
  int infoA1 = 0;
  Serializer<T,U,StructureA,StructureA>::Serialize(matAsource, dataA, matrixAcutXstart,
    matrixAcutXend, matrixAcutYstart, matrixAcutYend, infoA1);
  // Now, dataA is set and ready to be communicated

  dataB = new T[sizeB];
  T* matBsource = matrixB.getData();
  int infoB1 = 0;
  Serializer<T,U,StructureB,StructureB>::Serialize(matBsource, dataB, matrixBcutXstart,
    matrixBcutXend, matrixBcutZstart, matrixBcutZend, infoB1);
  // Now, dataB is set and ready to be communicated

  // Now need to perform the cblas call via Summa3DEngine (to use the right cblas call based on the structure combo)
  // Need to call serialize blindly, even if we are going from square to square
  //   This is annoyingly required for cblas calls. For now, just abide by the rules.
  // We also must create an interface to serialize from vectors to vectors to avoid instantiating temporary matrices.
  // These can be made static methods in the Matrix class to the MatrixSerialize class.
  // Its just another option for the user.

  T* matrixAtoSerialize = dataA;
  T* matrixBtoSerialize = dataB;
  T* matrixAforEngine = nullptr;
  T* matrixBforEngine = nullptr;
  int infoA2 = 0;
  int infoB2 = 0;
  Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixAtoSerialize, matrixAforEngine, 0, rangeA_x, 0, rangeA_y, infoA2);
  Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixBtoSerialize, matrixBforEngine, 0, rangeB_x, 0, rangeB_z, infoB2);

  T* matrixCforEngine = matrixC.getData();
  U numElems = matrixC.getNumElems();

  U rangeC_y = matrixCcutYend - matrixCcutYstart; 
  U rangeC_z = matrixCcutZend - matrixCcutZstart;

  // The BLAS call below needs modifed because we need to allow for transpose or triangular structure AND allow for dtrmm instead of dgemm
  blasEngine<T,U>::_gemm(matrixAforEngine, matrixBforEngine, matrixCforEngine, rangeA_x, rangeA_y,
    rangeB_x, rangeB_z, rangeC_y, rangeC_z, blasEngineInfo);
  // Assume for now that first 2 bits give 4 possibilies
  //   0 -> _gemm
  //   1 -> _trmm
  //   2 -> ?
  //   3 -> ?
  bool helper1 = blasEngineInfo&0x1;
  blasEngineInfo >>= 1;
  bool helper2 = blasEngineInfo&0x1;
  blasEngineInfo >>= 1;
  int whichRoutine = static_cast<int>(helper2)*2 + static_cast<int>(helper1);
  switch (whichRoutine)
  {
    case 0:
    {
      blasEngine<T,U>::_gemm(matrixAforEngine, matrixBforEngine, matrixCforEngine, rangeA_x, rangeA_y,
        rangeB_x, rangeB_z, rangeC_y, rangeC_z, 1., 1., rangeA_y, rangeB_x, rangeC_y, blasEngineInfo);
      break;
    }
    case 1:
    {
      int checkOrder = (0x2 & (blasEngineInfo>>1));		// check the 2nd bit to see if square matrix is on left or right
      blasEngine<T,U>::_trmm(matrixAforEngine, matrixBforEngine, (checkOrder ? rangeB_x : rangeA_y),
        (checkOrder ? rangeB_z : rangeA_x), 1., (checkOrder ? rangeA_y : rangeB_x), (checkOrder ? rangeB_x : rangeA_y), blasEngineInfo);
      break;
    }
    case 2:
    {
      break;
    }
    case 3:
    {
      break;
    }
  }

  if (infoA1 == 2)
  {
    delete[] dataA;
  }
  if (infoB1 == 2)
  {
    delete[] dataB;
  }
  if (infoA2 == 2)
  {
    delete[] matrixAforEngine;
  }
  if (infoB2 == 2)
  {
    delete[] matrixBforEngine;
  }
*/
}
