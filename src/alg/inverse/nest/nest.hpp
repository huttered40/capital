/* Author: Edward Hutter */

namespace inverse{

template<typename MatrixType, typename CommType>
void nest::invoke(MatrixType& matrix, CommType&& CommInfo){
  // `matrix` is modified in-place
  TAU_FSTART(nest::invoke);
  // TODO: Assuming CommType is an instance of SquareTopo
  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  assert(matrix.getNumColumnsGlobal() == matrix.getNumRowsGlobal());

  //TODO: Call Strassen with a Newton base case

  TAU_FSTOP(nest::invoke);
}
}
