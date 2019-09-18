#ifndef CHOLESKY__POLICY__CHOLINV
#define CHOLESKY__POLICY__CHOLINV

namespace cholesky{
namespace policy{
namespace cholinv{

// Policy classes for the policy describing whether or not to serialize from symmetric Gram matrix
//   to triangular matrix before AllReduction.
class GemmUpdate;
class TrmmUpdate;

/*
template<class PolicyClass>
class ReduceSymmetricMatrix{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& Matrix, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    using U = typename MatrixType::DimensionType;
    U localDimensionN = Matrix.getNumColumnsLocal();
    MPI_Allreduce(MPI_IN_PLACE, Matrix.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, CommInfo.world);
    return;
  }
};

template<>
class ReduceSymmetricMatrix<SerializeSymmetricToTriangle>{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& Matrix, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    using U = typename MatrixType::DimensionType;
    using Distribution = typename MatrixType::DistributionType;
    using Offload = typename MatrixType::OffloadType;
    U localDimensionN = Matrix.getNumColumnsLocal();
    U globalDimensionN = Matrix.getNumColumnsGlobal();
    matrix<T,U,uppertri,Distribution,Offload> Packed(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(Matrix, Packed);
    MPI_Allreduce(MPI_IN_PLACE, Packed.getRawData(), Packed.getNumElems(), mpi_type<T>::type, MPI_SUM, CommInfo.world);
    serialize<uppertri,square>::invoke(Packed, Matrix);
    return;
  }
};
*/

};
};
};
#endif // CHOLESKY__POLICY__CHOLINV
