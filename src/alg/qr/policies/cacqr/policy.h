#ifndef QR__POLICY__CACQR
#define QR__POLICY__CACQR

namespace qr{
namespace policy{
namespace cacqr{

// Policy classes for the policy describing whether or not to serialize from symmetric Gram matrix
//   to triangular matrix before AllReduction.
class SerializeSymmetricToTriangle;
class NoSerializeSymmetricToTriangle;

template<class PolicyClass>
class SerializeSymmetricPolicyClass{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& Matrix, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    using U = typename MatrixType::DimensionType;
    U localDimensionN = Matrix.num_columns_local();
    MPI_Allreduce(MPI_IN_PLACE, Matrix.data(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, CommInfo.world);
    return;
  }
};

template<>
class SerializeSymmetricPolicyClass<SerializeSymmetricToTriangle>{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& Matrix, CommType&& CommInfo){
    using T = typename MatrixType::ScalarType;
    using U = typename MatrixType::DimensionType;
    using Distribution = typename MatrixType::DistributionType;
    using Offload = typename MatrixType::OffloadType;
    U localDimensionN = Matrix.num_columns_local();
    U globalDimensionN = Matrix.num_columns_global();
    //TODO: Note that this can be further optimized if necessary to reduce number of allocations from each invocation of cacqr to just 1.
    matrix<T,U,uppertri,Distribution,Offload> Packed(globalDimensionN, globalDimensionN, CommInfo.c, CommInfo.c);
    serialize<square,uppertri>::invoke(Matrix, Packed);
    MPI_Allreduce(MPI_IN_PLACE, Packed.data(), Packed.getNumElems(), mpi_type<T>::type, MPI_SUM, CommInfo.world);
    serialize<uppertri,square>::invoke(Packed, Matrix);
    return;
  }
};

};
};
};
#endif // QR__POLICY__CACQR
