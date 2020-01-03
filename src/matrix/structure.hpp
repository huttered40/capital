/* Author: Edward Hutter */

template<typename ScalarType, typename DimensionType>
void rect::_assemble(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType& matrixNumElems, DimensionType dimensionX, DimensionType dimensionY){
  matrix.resize(dimensionX);
  matrixNumElems = dimensionX * dimensionY;
  data = new ScalarType[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename ScalarType, typename DimensionType>
void rect::_assemble_matrix(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY){
  DimensionType matrixNumElems = dimensionX * dimensionY;
  scratch = new ScalarType[matrixNumElems];
  pad = nullptr;
  DimensionType offset{0};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += dimensionY;
  }
}

template<typename ScalarType, typename DimensionType>
void rect::_copy(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, ScalarType* const & source, DimensionType dimensionX, DimensionType dimensionY){
  DimensionType numElems = 0;
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(ScalarType));
}

template<typename ScalarType, typename DimensionType>
void rect::_print(const std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY){
  for (DimensionType i=0; i<dimensionY; i++){
    for (DimensionType j=0; j<dimensionX; j++){
      std::cout << " " << matrix[j][i];
    }
    std::cout << std::endl;
  }
}

template<typename ScalarType, typename DimensionType>
void rect::_distribute_identity(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, ScalarType val){
  // Note: this is not fully implemented yet, as I have not decided on whether I need to perform a local transpose
  //       or local (but distributed based on the values each processor gets) transpose.

  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int64_t padYlen = (((globalDimensionY % globalPgridDimY != 0) && ((dimensionY-1)*globalPgridDimY + localPgridDimY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  DimensionType saveGlobalPositionX = localPgridDimX;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  DimensionType saveGlobalPositionY = localPgridDimY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (DimensionType i=0; i<padXlen; i++){
    saveGlobalPositionY = localPgridDimY;
    for (DimensionType j=0; j<padYlen; j++){
      matrix[i][j] = 0;
      if ((saveGlobalPositionX == saveGlobalPositionY) && (i==j)){
        matrix[i][j] += val;	// X or Y, should not matter
      }
      saveGlobalPositionY += globalPgridDimY;
    }
    // check for padding
    if (padYlen != dimensionY) { matrix[i][dimensionY-1] = 0; }
    saveGlobalPositionX += globalPgridDimX;
  }
  // check for padding
  if (padXlen != dimensionX){
    for (DimensionType j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}

template<typename ScalarType, typename DimensionType>
void rect::_distribute_symmetric(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, bool diagonallyDominant){

  srand48(key);
  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int64_t padYlen = (((globalDimensionY % globalPgridDimY != 0) && ((dimensionY-1)*globalPgridDimY + localPgridDimY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  DimensionType saveGlobalPositionX = localPgridDimX;
  DimensionType saveGlobalPositionY = localPgridDimY;
  for (DimensionType i=0; i<padXlen; i++){
    saveGlobalPositionY = localPgridDimY;
    for (DimensionType j=0; j<padYlen; j++){
      if (saveGlobalPositionX > saveGlobalPositionY){
        srand48(saveGlobalPositionX + globalDimensionY*saveGlobalPositionY);
      }
      else{
        srand48(saveGlobalPositionY + globalDimensionY*saveGlobalPositionX);
      }
      matrix[i][j] = drand48();			// Change this later.
      if ((diagonallyDominant) && (saveGlobalPositionX == saveGlobalPositionY) && (i==j)){
        matrix[i][j] += globalDimensionX;		// X or Y, should not matter
      }
      saveGlobalPositionY += globalPgridDimY;
    }
    // check for padding
    if (padYlen != dimensionY) { matrix[i][dimensionY-1] = 0; }
    saveGlobalPositionX += globalPgridDimX;
  }
  // check for padding
  if (padXlen != dimensionX){
    for (DimensionType j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}

template<typename ScalarType, typename DimensionType>
void rect::_distribute_random(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key){
  srand48(key);
  DimensionType saveGlobalPosition = localPgridDimY + localPgridDimX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int64_t padYlen = (((globalDimensionY % globalPgridDimY != 0) && ((dimensionY-1)*globalPgridDimY + localPgridDimY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  for (DimensionType i=0; i<padXlen; i++){
    DimensionType globalPosition = saveGlobalPosition;
    for (DimensionType j=0; j<padYlen; j++){
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridDimY;
    }
    // check for padding
    if (padYlen != dimensionY) { matrix[i][dimensionY-1] = 0; }
    saveGlobalPosition += (globalPgridDimX*globalDimensionY);
  }
  // check for padding
  if (padXlen != dimensionX){
    for (DimensionType j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}


template<typename ScalarType, typename DimensionType>
void uppertri::_assemble(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType& matrixNumElems, DimensionType dimensionX, DimensionType dimensionY){
  matrix.resize(dimensionY);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  data = new ScalarType[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename ScalarType, typename DimensionType>
void uppertri::_assemble_matrix(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY){
  DimensionType nonPackedNumElems = dimensionX*dimensionY;
  DimensionType matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  scratch = new ScalarType[matrixNumElems];
  pad = new ScalarType[nonPackedNumElems];	// we give full non-packed size here to account for need for summa to use nonpacked layout
  DimensionType offset{0};
  DimensionType counter{1};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter++;
  }
}

template<typename ScalarType, typename DimensionType>
void uppertri::_copy(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, ScalarType* const & source, DimensionType dimensionX, DimensionType dimensionY){
  DimensionType numElems = 0;
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(ScalarType));
}

template<typename ScalarType, typename DimensionType>
void uppertri::_print(const std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY){
  DimensionType startIter = 0;
  for (DimensionType i=0; i<dimensionY; i++){
    // print spaces to represent the lower triangular zeros
    for (DimensionType j=0; j<i; j++){
      std::cout << "    ";
    }

    for (DimensionType j=startIter; j<dimensionX; j++){
      std::cout << " " << matrix[j][i];
    }
    startIter++;
    std::cout << std::endl;
  }
}

template<typename ScalarType, typename DimensionType>
void uppertri::_distribute_random(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
    int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key){
  srand48(key);
  DimensionType saveGlobalPosition = localPgridDimY + localPgridDimX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  DimensionType counter{1};
  DimensionType startIter;
  DimensionType endIter;
  for (DimensionType i=0; i<padXlen; i++){
    DimensionType globalPosition = saveGlobalPosition;
    startIter = 0;
    endIter = counter;
    for (DimensionType j=startIter; j<endIter; j++){
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridDimY;
    }
    // Special corner case: If a processor's first data on each row is out of bounds of the DimensionTypeT structure, then give a 0 value
    if (localPgridDimY > localPgridDimX){
      matrix[i][endIter-1] = 0;			// reset this to 0 instead of whatever was set in the last iteration of the loop above.
    }
    counter++;
    saveGlobalPosition += (globalPgridDimX*globalDimensionY);
  }
  if (padXlen != dimensionX){
    // fill in the last column with zeros
    for (DimensionType j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}


template<typename ScalarType, typename DimensionType>
void lowertri::_assemble(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType& matrixNumElems, DimensionType dimensionX, DimensionType dimensionY){
  matrix.resize(dimensionX);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);
  data = new ScalarType[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename ScalarType, typename DimensionType>
void lowertri::_assemble_matrix(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY){
  DimensionType nonPackedNumElems = dimensionX*dimensionY;
  DimensionType matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  scratch = new ScalarType[matrixNumElems];
  pad = new ScalarType[nonPackedNumElems];	// we give full non-packed size here to account for need for summa to use nonpacked layout
  DimensionType offset{0};
  DimensionType counter{dimensionY};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter--;
  }
}

template<typename ScalarType, typename DimensionType>
void lowertri::_copy(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, ScalarType* const & source, DimensionType dimensionX, DimensionType dimensionY){
  DimensionType numElems = 0;
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename ScalarType, typename DimensionType>
void lowertri::_print(const std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY){
  for (DimensionType i=0; i<dimensionY; i++){
    for (DimensionType j=0; j<=i; j++){
      std::cout << matrix[j][i-j] << " ";
    }
    std::cout << "\n";
  }
}

template<typename ScalarType, typename DimensionType>
void lowertri::_distribute_random(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
    int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key){
  srand48(key);
  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int64_t padYlen = (((globalDimensionY % globalPgridDimY != 0) && ((dimensionY-1)*globalPgridDimY + localPgridDimY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  DimensionType saveGlobalPosX = localPgridDimX;
  DimensionType saveGlobalPosY = localPgridDimY;
  DimensionType counter{0};
  for (DimensionType i=0; i<padXlen; i++){
    // Special corner case: If a processor's last data on each row is out of bounds of the LT structure, then give a 0 value
    counter = 0;		// reset
    saveGlobalPosY += (i*globalPgridDimY);
    if (localPgridDimY < localPgridDimX){
      // Set the last position in row
      matrix[i][0] = 0;
      counter++;
      saveGlobalPosY += globalPgridDimY;
    }
    for (DimensionType j=counter; j<(padYlen-i); j++){
      // Maybe in the future, get rid of this inner if statementand try something else? If statements in inner
      //   nested loops can be very expensive.
      if (saveGlobalPosX == saveGlobalPosY){
        // Set the first position in row to a 1 -> special only to lowertri matrices.
        matrix[i][j] = 1.;
      }
      else{
        matrix[i][j] = drand48();			// Change this later.
      }
      // check padding
      if (padXlen != dimensionX) { matrix[i][dimensionY-i] = 0; }
      saveGlobalPosY += globalPgridDimY;
    }
    // check padding
    if (padXlen != dimensionX) { matrix[dimensionX-1][0] = 0; }
    saveGlobalPosY = localPgridDimY;			// reset
    saveGlobalPosX += globalPgridDimX;
  }
  return;
}
