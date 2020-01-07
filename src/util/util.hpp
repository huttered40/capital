/* Author: Edward Hutter */

template<typename MatrixType, typename CommType>
typename MatrixType::ScalarType util::get_identity_residual(MatrixType& Matrix, CommType&& CommInfo, MPI_Comm comm){
  // Should be offloaded to Matrix definition, which knows how best to iterate over matrix?
  // Should this be made more general so that user can supply a Lambda and more than the max can be obtained in this interface?
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType;
  T res = 0; U localNumRows = Matrix.num_rows_local(); U localNumColumns = Matrix.num_columns_local();
  U globalX = CommInfo.x; U globalY = CommInfo.y; U index=0;
  for (int64_t i=0; i<localNumColumns; i++){
    globalY = CommInfo.y;// reset
    for (int64_t j=0; j<localNumRows; j++){
      if ((globalX<globalNumRows) && (globalY<globalNumColumns)){
        if (globalX == globalY){ res += 1.-Matrix.data()[index++]; }
        else { res += Matrix.data()[index++]; }
      }
      globalY += CommInfo.d;
    }
    globalX += CommInfo.d;//TODO: This will not work for a rectangular grid (i.e. CommType is RectTopo)
  }
  MPI_Allreduce(MPI_IN_PLACE,&res,1,mpi_type<T>::type, MPI_SUM, comm);
  return res;
}

template<typename MatrixType, typename RefMatrixType, typename LambdaType>
typename MatrixType::ScalarType
util::residual_local(MatrixType& Matrix, RefMatrixType& RefMatrix, LambdaType&& Lambda, MPI_Comm slice, int64_t sliceX, int64_t sliceY, int64_t sliceDimX, int64_t sliceDimY){
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType;
  int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank); T error = 0; T control = 0;
  U localNumRows = Matrix.num_rows_local(); U globalNumRows = Matrix.num_rows_global();
  U localNumColumns = Matrix.num_columns_local(); U globalNumColumns = Matrix.num_columns_global();
  U globalX = sliceX; U globalY = sliceY;

  for (int64_t i=0; i<localNumColumns; i++){
    globalY = sliceY;    // reset
    for (int64_t j=0; j<localNumRows; j++){
      if ((globalX<globalNumRows) && (globalY<globalNumColumns)){
        auto info = Lambda(Matrix, RefMatrix, i*localNumRows+j,globalX, globalY);
        error += std::abs(info.first*info.first); control += std::abs(info.second*info.second);
        /*int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if (info.first >= 1.e-8 && rank==0){std::cout << "current error - " << error << " global index - " << i*localNumRows+j << " local index - (" << i << "," << j << ") global index - ("\
                                                      << globalX << "," << globalY << ")\n\t\t\tlocal dimensions - (" << localNumColumns << "," << localNumRows << ") global dimensions - (" << globalNumColumns << "," << globalNumRows << ")\n\t\t\t error - " << info.first << " A value - " << info.second << " process grid id - (" << sliceX << "," << sliceY << ")\n";}*/
      }
      globalY += sliceDimY;
    }
    globalX += sliceDimX;
  }

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, mpi_type<T>::type, MPI_SUM, slice);
  MPI_Allreduce(MPI_IN_PLACE, &control, 1, mpi_type<T>::type, MPI_SUM, slice);
  error = std::sqrt(error) / std::sqrt(control);
  return error;
}

// Note: this method differs from the one below it because blockedData is in packed storage
template<typename ScalarType>
void util::block_to_cyclic(std::vector<ScalarType>& blockedData, ScalarType* cyclicData, int64_t localDimensionRows, int64_t localDimensionColumns, int64_t sliceDim, char dir){

  int64_t aggregNumRows = localDimensionRows*sliceDim;
  int64_t aggregNumColumns = localDimensionColumns*sliceDim;
  int64_t aggregSize = aggregNumRows*aggregNumColumns;
  int64_t numCyclicBlocksPerRow = localDimensionRows;
  int64_t numCyclicBlocksPerCol = localDimensionColumns;
  int64_t write_idx=0; int64_t read_idx=0;

  write_idx = 0; int64_t recvDataOffset = blockedData.size()/(sliceDim*sliceDim);
  int64_t off1 = 0; int64_t off3 = sliceDim*recvDataOffset;
  // MACRO loop over all cyclic "blocks" (dimensionX direction)
  for (int64_t i=0; i<numCyclicBlocksPerCol; i++){
    off1 += i;
    // Inner loop over all columns in a cyclic "block"
    for (int64_t j=0; j<sliceDim; j++){
      int64_t off2 = j*recvDataOffset + off1;
      write_idx = ((i*sliceDim)+j)*aggregNumRows;    //  reset each time
      // Inner loop over all cyclic "blocks"
      for (int64_t k=0; k<i; k++){
        int64_t off4 = off2;
        // Inner loop over all elements along columns
        for (int64_t z=0; z<sliceDim; z++){
          read_idx = off4 + z*off3 + k;
          cyclicData[write_idx++] = blockedData[read_idx];
        }
      }
      // Special final block
      int64_t off4 = off2;
      // Inner loop over all elements along columns
      for (int64_t z=0; z<=j; z++){
        read_idx = off4 + z*off3 + i; cyclicData[write_idx++] = blockedData[read_idx];
      }
    }
  }
}

template<typename ScalarType>
void util::block_to_cyclic(ScalarType* blockedData, ScalarType* cyclicData, int64_t localDimensionRows, int64_t localDimensionColumns, int64_t sliceDim){
  int64_t write_idx = 0; int64_t read_idx = 0;
  int64_t readDataOffset = localDimensionRows*localDimensionColumns;
  for (int64_t i=0; i<localDimensionColumns; i++){
    for (int64_t j=0; j<sliceDim; j++){
      for (int64_t k=0; k<localDimensionRows; k++){
        for (int64_t z=0; z<sliceDim; z++){
          read_idx = z*readDataOffset*sliceDim + k + j*readDataOffset + i*localDimensionRows;
          cyclicData[write_idx++] = blockedData[read_idx];
        }
      }
    }
  }
}

// This method can be called from Lower and Upper with one tweak, but note that currently, we iterate over the entire square,
//   when we are really only writing to a triangle. So there is a source of optimization here at least in terms of
//   number of flops, but in terms of memory accesses and cache lines, not sure. Note that with this optimization,
//   we may need to separate into two different functions
template<typename ScalarType>
void util::cyclic_to_local(ScalarType* storeT, ScalarType* storeTI, int64_t localDimension, int64_t globalDimension, int64_t bcDimension, int64_t sliceDim, int64_t rankSlice){

  int64_t writeIndex,readIndexCol,readIndexRow;
  int64_t rowOffsetWithinBlock = rankSlice / sliceDim;
  int64_t columnOffsetWithinBlock = rankSlice % sliceDim;
  // MACRO loop over all cyclic "blocks"
  for (int64_t i=0; i<localDimension; i++){
    // We know which row corresponds to our processor in each cyclic "block"
    // Inner loop over all cyclic "blocks" partitioning up the columns
    for (int64_t j=0; j<localDimension; j++){
      readIndexCol = i*sliceDim + columnOffsetWithinBlock;
      readIndexRow = j*sliceDim + rowOffsetWithinBlock;
      writeIndex = i*bcDimension+j;
      if (readIndexCol >= readIndexRow){
        storeT[writeIndex] = storeT[readIndexCol*bcDimension + readIndexRow];
        storeTI[writeIndex] = storeTI[readIndexCol*bcDimension + readIndexRow];
      } else{
        storeT[writeIndex] = 0.; storeTI[writeIndex] = 0.;
      }
      writeIndex++;
    }
  }
}

template<typename MatrixType, typename CommType>
void util::transpose(MatrixType& mat, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType;
  int64_t SquareFaceSize = CommInfo.c*CommInfo.d;
  int64_t transposePartner = CommInfo.x*SquareFaceSize + CommInfo.y*CommInfo.c + CommInfo.z;
    MPI_Sendrecv_replace(mat.data(), mat.num_elems(), mpi_type<T>::type, transposePartner, 0, transposePartner, 0, CommInfo.world, MPI_STATUS_IGNORE);
    // Note: the received data that now resides in mat is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.
}

int64_t util::get_next_power2(int64_t localShift){

  if ((localShift & (localShift-1)) != 0){
    // move localShift up to the next power of 2
    localShift--;
    localShift |= (localShift >> 1);
    localShift |= (localShift >> 2);
    localShift |= (localShift >> 4);
    localShift |= (localShift >> 8);
    localShift |= (localShift >> 16);
    // corner case: if dealing with 64-bit integers, shift the 32
    localShift |= (localShift >> 32);
    localShift++;
  }
  return localShift;
}

template<typename MatrixType>
void util::remove_triangle(MatrixType& matrix, int64_t sliceX, int64_t sliceY, int64_t sliceDim, char dir){
  using U = typename MatrixType::DimensionType; using T = typename MatrixType::ScalarType;
  T* data = std::is_same<typename MatrixType::StructureType,rect>::value ? matrix.data() : matrix.pad();

  U globalDimVert = sliceY; U globalDimHoriz = sliceX;
  U localVert = matrix.num_rows_local();
  U localHoriz = matrix.num_columns_local();
  for (U i=0; i<localHoriz; i++){
    globalDimVert = sliceY;    //   reset
    for (U j=0; j<localVert; j++){
      if ((globalDimVert < globalDimHoriz) && (dir == 'L')){
        data[i*localVert + j] = 0;
      }
      if ((globalDimVert > globalDimHoriz) && (dir == 'U')){
        data[i*localVert + j] = 0;
      }
      globalDimVert += sliceDim;
    }
    globalDimHoriz += sliceDim;
  }
}

template<typename MatrixType>
void util::remove_triangle_local(MatrixType& matrix, int64_t sliceX, int64_t sliceY, int64_t sliceDim, char dir){
  using U = typename MatrixType::DimensionType; using T = typename MatrixType::ScalarType;
  T* data = std::is_same<typename MatrixType::StructureType,rect>::value ? matrix.data() : matrix.pad();

  U localVert = matrix.num_rows_local();
  U localHoriz = matrix.num_columns_local();
  for (U i=0; i<localHoriz; i++){
    for (U j=0; j<localVert; j++){
      if ((j < localHoriz) && (dir == 'L')){
        data[i*localVert + j] = 0;
      }
      if ((j > localHoriz) && (dir == 'U')){
        data[i*localVert + j] = 0;
      }
    }
  }
}
