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
  std::cout << "I am Ed\n";
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
