/* Author: Edward Hutter */

template<typename T, typename U, template<typename,typename> class blasEngine>
template<
          template<typename,typename, template<typename,typename,int> class> class StructureA,
          template<typename,typename, template<typename,typename,int> class> class StructureB,
          template<typename,typename, template<typename,typename,int> class> class StructureC,
          template<typename,typename,int> class Distribution
        >
void MMlocal<T,U,blasEngine>::multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        Matrix<T,U,StructureC,Distribution>& matrixC,
                        U dimensionX,
                        U dimensionY,
                        U dimensionZ,
                        int blasEngineInfo
                      )
{
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
}

template<typename T, typename U, template<typename,typename> class blasEngine>
template<
          template<typename,typename, template<typename,typename,int> class> class StructureA,
          template<typename,typename, template<typename,typename,int> class> class StructureB,
          template<typename,typename, template<typename,typename,int> class> class StructureC,
          template<typename,typename,int> class Distribution
        >
void MMlocal<T,U,blasEngine>::multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        Matrix<T,U,StructureC,Distribution>& matrixC,
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
  std::cout << "I am Joe\n";
}
