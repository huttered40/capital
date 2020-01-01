/* Author: Edward Hutter */

template<typename T, typename U>
void rect::_assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  matrix.resize(dimensionX);
  matrixNumElems = dimensionX * dimensionY;
  data = new T[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void rect::_assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U matrixNumElems = dimensionX * dimensionY;
  scratch = new T[matrixNumElems];
  pad = nullptr;
  U offset{0};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += dimensionY;
  }
}

template<typename T, typename U>
void rect::_copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY){
  U numElems = 0;
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void rect::_print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  for (U i=0; i<dimensionY; i++){
    for (U j=0; j<dimensionX; j++){
      std::cout << " " << matrix[j][i];
    }
    std::cout << std::endl;
  }
}

template<typename T, typename U>
void rect::_distribute_identity(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, T val){
  // Note: this is not fully implemented yet, as I have not decided on whether I need to perform a local transpose
  //       or local (but distributed based on the values each processor gets) transpose.

  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int64_t padYlen = (((globalDimensionY % globalPgridDimY != 0) && ((dimensionY-1)*globalPgridDimY + localPgridDimY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  U saveGlobalPositionX = localPgridDimX;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  U saveGlobalPositionY = localPgridDimY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<padXlen; i++){
    saveGlobalPositionY = localPgridDimY;
    for (U j=0; j<padYlen; j++){
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
    for (U j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}

template<typename T, typename U>
void rect::_distribute_symmetric(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, bool diagonallyDominant){

  srand48(key);
  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int64_t padYlen = (((globalDimensionY % globalPgridDimY != 0) && ((dimensionY-1)*globalPgridDimY + localPgridDimY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  U saveGlobalPositionX = localPgridDimX;
  U saveGlobalPositionY = localPgridDimY;
  for (U i=0; i<padXlen; i++){
    saveGlobalPositionY = localPgridDimY;
    for (U j=0; j<padYlen; j++){
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
    for (U j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}

template<typename T, typename U>
void rect::_distribute_random(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key){
  srand48(key);
  U saveGlobalPosition = localPgridDimY + localPgridDimX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int64_t padYlen = (((globalDimensionY % globalPgridDimY != 0) && ((dimensionY-1)*globalPgridDimY + localPgridDimY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  for (U i=0; i<padXlen; i++){
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<padYlen; j++){
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridDimY;
    }
    // check for padding
    if (padYlen != dimensionY) { matrix[i][dimensionY-1] = 0; }
    saveGlobalPosition += (globalPgridDimX*globalDimensionY);
  }
  // check for padding
  if (padXlen != dimensionX){
    for (U j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}


template<typename T, typename U>
void uppertri::_assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionY);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  data = new T[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void uppertri::_assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U nonPackedNumElems = dimensionX*dimensionY;
  U matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  scratch = new T[matrixNumElems];
  pad = new T[nonPackedNumElems];	// we give full non-packed size here to account for need for summa to use nonpacked layout
  U offset{0};
  U counter{1};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter++;
  }
}

template<typename T, typename U>
void uppertri::_copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY){
  U numElems = 0;
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void uppertri::_print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U startIter = 0;
  for (U i=0; i<dimensionY; i++){
    // print spaces to represent the lower triangular zeros
    for (U j=0; j<i; j++){
      std::cout << "    ";
    }

    for (U j=startIter; j<dimensionX; j++){
      std::cout << " " << matrix[j][i];
    }
    startIter++;
    std::cout << std::endl;
  }
}

template<typename T, typename U>
void uppertri::_distribute_random(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
    int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key){
  srand48(key);
  U saveGlobalPosition = localPgridDimY + localPgridDimX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  U counter{1};
  U startIter;
  U endIter;
  for (U i=0; i<padXlen; i++){
    U globalPosition = saveGlobalPosition;
    startIter = 0;
    endIter = counter;
    for (U j=startIter; j<endIter; j++){
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridDimY;
    }
    // Special corner case: If a processor's first data on each row is out of bounds of the UT structure, then give a 0 value
    if (localPgridDimY > localPgridDimX){
      matrix[i][endIter-1] = 0;			// reset this to 0 instead of whatever was set in the last iteration of the loop above.
    }
    counter++;
    saveGlobalPosition += (globalPgridDimX*globalDimensionY);
  }
  if (padXlen != dimensionX){
    // fill in the last column with zeros
    for (U j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}


template<typename T, typename U>
void lowertri::_assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);
  data = new T[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void lowertri::_assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U nonPackedNumElems = dimensionX*dimensionY;
  U matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  scratch = new T[matrixNumElems];
  pad = new T[nonPackedNumElems];	// we give full non-packed size here to account for need for summa to use nonpacked layout
  U offset{0};
  U counter{dimensionY};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter--;
  }
}

template<typename T, typename U>
void lowertri::_copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY){
  U numElems = 0;
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void lowertri::_print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  for (U i=0; i<dimensionY; i++){
    for (U j=0; j<=i; j++){
      std::cout << matrix[j][i-j] << " ";
    }
    std::cout << "\n";
  }
}

template<typename T, typename U>
void lowertri::_distribute_random(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
    int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key){
  srand48(key);
  int64_t padXlen = (((globalDimensionX % globalPgridDimX != 0) && ((dimensionX-1)*globalPgridDimX + localPgridDimX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int64_t padYlen = (((globalDimensionY % globalPgridDimY != 0) && ((dimensionY-1)*globalPgridDimY + localPgridDimY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  U saveGlobalPosX = localPgridDimX;
  U saveGlobalPosY = localPgridDimY;
  U counter{0};
  for (U i=0; i<padXlen; i++){
    // Special corner case: If a processor's last data on each row is out of bounds of the LT structure, then give a 0 value
    counter = 0;		// reset
    saveGlobalPosY += (i*globalPgridDimY);
    if (localPgridDimY < localPgridDimX){
      // Set the last position in row
      matrix[i][0] = 0;
      counter++;
      saveGlobalPosY += globalPgridDimY;
    }
    for (U j=counter; j<(padYlen-i); j++){
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
