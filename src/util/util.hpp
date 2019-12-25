/* Author: Edward Hutter */

template<typename MatrixType, typename CommType>
typename MatrixType::ScalarType util::get_identity_residual(MatrixType& Matrix, CommType&& CommInfo, MPI_Comm comm){
  // Should be offloaded to Matrix definition, which knows how best to iterate over matrix?
  // Should this be made more general so that user can supply a Lambda and more than the max can be obtained in this interface?
  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  T res = 0;
  U localNumRows = Matrix.num_rows_local();
  U localNumColumns = Matrix.num_columns_local();
  U globalX = CommInfo.x;
  U globalY = CommInfo.y;
  U index=0;
  for (int64_t i=0; i<localNumColumns; i++){
    globalY = CommInfo.y;    // reset
    for (int64_t j=0; j<localNumRows; j++){
      if (globalX == globalY){
        res += 1.-Matrix.data()[index++];
      } else{
        res += Matrix.data()[index++];
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
  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  T error = 0;
  T control = 0;
  U localNumRows = Matrix.num_rows_local();
  U localNumColumns = Matrix.num_columns_local();
  U globalX = sliceX;
  U globalY = sliceY;

  for (int64_t i=0; i<localNumColumns; i++){
    globalY = sliceY;    // reset
    for (int64_t j=0; j<localNumRows; j++){
      auto info = Lambda(Matrix, RefMatrix, i*localNumRows+j,globalX, globalY);
      error += std::abs(info.first*info.first);
      control += std::abs(info.second*info.second);
      globalY += sliceDimY;
      //if (rank == 0) std::cout << val << " " << i << " " << j << std::endl;
    }
    globalX += sliceDimX;
  }

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, mpi_type<T>::type, MPI_SUM, slice);
  MPI_Allreduce(MPI_IN_PLACE, &control, 1, mpi_type<T>::type, MPI_SUM, slice);
  error = std::sqrt(error) / std::sqrt(control);
  return error;
}

// Note: this method differs from the one below it because blockedData is in packed storage
template<typename T, typename U>
void util::block_to_cyclic(std::vector<T>& blockedData, std::vector<T>& cyclicData, U localDimensionRows, U localDimensionColumns, int64_t sliceDim, char dir){

  U aggregNumRows = localDimensionRows*sliceDim;
  U aggregNumColumns = localDimensionColumns*sliceDim;
  U aggregSize = aggregNumRows*aggregNumColumns;
  U numCyclicBlocksPerRow = localDimensionRows;
  U numCyclicBlocksPerCol = localDimensionColumns;
  U write_idx=0;
  U read_idx=0;

  if (dir == 'L'){
    write_idx = 0;
    U recvDataOffset = blockedData.size()/(sliceDim*sliceDim);
    U off1 = (-1)*numCyclicBlocksPerRow-1;
    U off3 = sliceDim*recvDataOffset;
    // MACRO loop over all cyclic "blocks" (dimensionX direction)
    for (U i=0; i<numCyclicBlocksPerCol; i++){
      off1 += (numCyclicBlocksPerRow-i+1);
      // Inner loop over all columns in a cyclic "block"
      for (U j=0; j<sliceDim; j++){
        U off2 = j*recvDataOffset + off1;
        write_idx += (i*sliceDim) + j;
        // Treat first block separately
        for (U z=j; z<sliceDim; z++){
          read_idx = off2 + z*off3;
          cyclicData[write_idx++] = blockedData[read_idx];
        }
        // Inner loop over all cyclic "blocks"
        for (U k=(i+1); k<numCyclicBlocksPerRow; k++){
          U off4 = off2;
          // Inner loop over all elements along columns
          for (U z=0; z<sliceDim; z++){
            read_idx = off4 + z*off3 + (k-i);
            cyclicData[write_idx++] = blockedData[read_idx];
          }
        }
      }
    }
  }
  else{
    write_idx = 0;
    U recvDataOffset = blockedData.size()/(sliceDim*sliceDim);
    U off1 = 0;
    U off3 = sliceDim*recvDataOffset;
    // MACRO loop over all cyclic "blocks" (dimensionX direction)
    for (U i=0; i<numCyclicBlocksPerCol; i++){
      off1 += i;
      // Inner loop over all columns in a cyclic "block"
      for (U j=0; j<sliceDim; j++){
        U off2 = j*recvDataOffset + off1;
        write_idx = ((i*sliceDim)+j)*aggregNumRows;    //  reset each time
        // Inner loop over all cyclic "blocks"
        for (U k=0; k<i; k++){
          U off4 = off2;
          // Inner loop over all elements along columns
          for (U z=0; z<sliceDim; z++){
            read_idx = off4 + z*off3 + k;
            cyclicData[write_idx++] = blockedData[read_idx];
          }
        }
        
        // Special final block
        U off4 = off2;
        // Inner loop over all elements along columns
        for (U z=0; z<=j; z++){
          read_idx = off4 + z*off3 + i;
          cyclicData[write_idx++] = blockedData[read_idx];
        }
      }
    }
  }
}

template<typename T, typename U>
void util::block_to_cyclic(T* blockedData, T* cyclicData, U localDimensionRows, U localDimensionColumns, int64_t sliceDim){
  U write_idx = 0;
  U read_idx = 0;
  U readDataOffset = localDimensionRows*localDimensionColumns;
  for (U i=0; i<localDimensionColumns; i++){
    for (U j=0; j<sliceDim; j++){
      for (U k=0; k<localDimensionRows; k++){
        for (U z=0; z<sliceDim; z++){
          read_idx = z*readDataOffset*sliceDim + k + j*readDataOffset + i*localDimensionRows;
          cyclicData[write_idx++] = blockedData[read_idx];
        }
      }
    }
  }
}

template<typename MatrixType, typename CommType>
void util::transpose(MatrixType& mat, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType;
  int64_t SquareFaceSize = CommInfo.c*CommInfo.d;
  int64_t transposePartner = CommInfo.x*SquareFaceSize + CommInfo.y*CommInfo.c + CommInfo.z;
  //if (myRank != transposeRank)
  //{
    // Transfer with transpose rank
    MPI_Sendrecv_replace(mat.data(), mat.num_elems(), mpi_type<T>::type, transposePartner, 0, transposePartner, 0, CommInfo.world, MPI_STATUS_IGNORE);

    // Note: the received data that now resides in mat is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.

  //}
}

template<typename U>
U util::get_next_power2(U localShift){

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
  using U = typename MatrixType::DimensionType;

  U globalDimVert = sliceY;
  U globalDimHoriz = sliceX;
  U localVert = matrix.num_rows_local();
  U localHoriz = matrix.num_columns_local();
  for (U i=0; i<localHoriz; i++){
    globalDimVert = sliceY;    //   reset
    for (U j=0; j<localVert; j++){
      if ((globalDimVert < globalDimHoriz) && (dir == 'L')){
        matrix.data()[i*localVert + j] = 0;
      }
      if ((globalDimVert > globalDimHoriz) && (dir == 'U')){
        matrix.data()[i*localVert + j] = 0;
      }
      globalDimVert += sliceDim;
    }
    globalDimHoriz += sliceDim;
  }
}
