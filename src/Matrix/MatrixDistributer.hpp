/* Author: Edward Hutter */

// Try this to get it to compile: provide a declaration of a templated constructor,
//   and then define/implement a specialized method.


// MatrixStructureSquare
template<typename T, typename U>
void MatrixDistributerCyclic<T,U,0>::DistributeRandom
                (
		   std::vector<T*>& matrix,
		   U dimensionX,
		   U dimensionY,
                   U globalDimensionX,
                   U globalDimensionY,
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY,
		   U key
		 ){

  srand48(key);
  U saveGlobalPosition = localPgridY + localPgridX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  int padXlen = (((globalDimensionX % globalPgridX != 0) && ((dimensionX-1)*globalPgridX + localPgridX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int padYlen = (((globalDimensionY % globalPgridY != 0) && ((dimensionY-1)*globalPgridY + localPgridY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  for (U i=0; i<padXlen; i++){
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<padYlen; j++){
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    // check for padding
    if (padYlen != dimensionY) { matrix[i][dimensionY-1] = 0; }
    saveGlobalPosition += (globalPgridX*globalDimensionY);
  }
  // check for padding
  if (padXlen != dimensionX){
    for (U j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}

// MatrixStructureSquare
template<typename T, typename U>
void MatrixDistributerCyclic<T,U,0>::DistributeSymmetric
                (
		   std::vector<T*>& matrix,
		   U dimensionX,
		   U dimensionY,
                   U globalDimensionX,
                   U globalDimensionY,
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY,
		   U key,
                   bool diagonallyDominant
		 ){
  // Note: this is not fully implemented yet, as I have not decided on whether I need to perform a local transpose
  //       or local (but distributed based on the values each processor gets) transpose.

  srand48(key);
  int padXlen = (((globalDimensionX % globalPgridX != 0) && ((dimensionX-1)*globalPgridX + localPgridX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int padYlen = (((globalDimensionY % globalPgridY != 0) && ((dimensionY-1)*globalPgridY + localPgridY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  U saveGlobalPositionX = localPgridX;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  U saveGlobalPositionY = localPgridY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<padXlen; i++){
    saveGlobalPositionY = localPgridY;
    for (U j=0; j<padYlen; j++){
/*
      if (saveGlobalPositionX > saveGlobalPositionY)
      {
        srand48(saveGlobalPositionX + globalDimensionY*saveGlobalPositionY);
      }
      else
      {
        srand48(saveGlobalPositionY + globalDimensionY*saveGlobalPositionX);
      }
*/
      matrix[i][j] = drand48();			// Change this later.
      if ((diagonallyDominant) && (saveGlobalPositionX == saveGlobalPositionY) && (i==j)){
        matrix[i][j] += globalDimensionX;		// X or Y, should not matter
      }
      saveGlobalPositionY += globalPgridY;
    }
    // check for padding
    if (padYlen != dimensionY) { matrix[i][dimensionY-1] = 0; }
    saveGlobalPositionX += globalPgridX;
  }
  // check for padding
  if (padXlen != dimensionX){
    for (U j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}

// MatrixStructureRectangle
template<typename T, typename U>
void MatrixDistributerCyclic<T,U,1>::DistributeRandom
                (
		   std::vector<T*>& matrix,
		   U dimensionX,
		   U dimensionY,
                   U globalDimensionX,
                   U globalDimensionY,
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY,
		   U key
		 ){
  srand48(key);
  U saveGlobalPosition = localPgridY + localPgridX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  int padXlen = (((globalDimensionX % globalPgridX != 0) && ((dimensionX-1)*globalPgridX + localPgridX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int padYlen = (((globalDimensionY % globalPgridY != 0) && ((dimensionY-1)*globalPgridY + localPgridY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  for (U i=0; i<padXlen; i++){
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<padYlen; j++){
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    // check for padding
    if (padYlen != dimensionY) { matrix[i][dimensionY-1] = 0; }
    saveGlobalPosition += (globalPgridX*globalDimensionY);
  }
  // check for padding
  if (padXlen != dimensionX){
    for (U j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}

// MatrixStructureUpperTriangular
template<typename T, typename U>
void MatrixDistributerCyclic<T,U,2>::DistributeRandom
                (
		   std::vector<T*>& matrix,
		   U dimensionX,
		   U dimensionY,
                   U globalDimensionX,
                   U globalDimensionY,
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY,
		   U key
		 ){
  srand48(key);
  U saveGlobalPosition = localPgridY + localPgridX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  int padXlen = (((globalDimensionX % globalPgridX != 0) && ((dimensionX-1)*globalPgridX + localPgridX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int padYlen = (((globalDimensionY % globalPgridY != 0) && ((dimensionY-1)*globalPgridY + localPgridY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  U counter{1};
  U startIter;
  U endIter;
  for (U i=0; i<padXlen; i++){
    U globalPosition = saveGlobalPosition;
    startIter = 0;
    endIter = counter;
    for (U j=startIter; j<endIter; j++){
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    // Special corner case: If a processor's first data on each row is out of bounds of the UT structure, then give a 0 value
    if (localPgridY > localPgridX){
      matrix[i][endIter-1] = 0;			// reset this to 0 instead of whatever was set in the last iteration of the loop above.
    }
    counter++;
    saveGlobalPosition += (globalPgridX*globalDimensionY);
  }
  if (padXlen != dimensionX){
    // fill in the last column with zeros
    for (U j=0; j<dimensionY; j++){
      matrix[dimensionX-1][j] = 0;
    }
  }
  return;
}

// MatrixStructureLowerTriangular
template<typename T, typename U>
void MatrixDistributerCyclic<T,U,3>::DistributeRandom
                (
		   std::vector<T*>& matrix,
		   U dimensionX,
		   U dimensionY,
                   U globalDimensionX,
                   U globalDimensionY,
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY,
		   U key
		 ){
  srand48(key);
  int padXlen = (((globalDimensionX % globalPgridX != 0) && ((dimensionX-1)*globalPgridX + localPgridX >= globalDimensionX)) ? dimensionX-1 : dimensionX);
  int padYlen = (((globalDimensionY % globalPgridY != 0) && ((dimensionY-1)*globalPgridY + localPgridY >= globalDimensionY)) ? dimensionY-1 : dimensionY);
  U saveGlobalPosX = localPgridX;
  U saveGlobalPosY = localPgridY;
  U counter{0};
  for (U i=0; i<padXlen; i++){
    // Special corner case: If a processor's last data on each row is out of bounds of the LT structure, then give a 0 value
    counter = 0;		// reset
    saveGlobalPosY += (i*globalPgridY);
    if (localPgridY < localPgridX){
      // Set the last position in row
      matrix[i][0] = 0;
      counter++;
      saveGlobalPosY += globalPgridY;
    }
    for (U j=counter; j<(padYlen-i); j++){
      // Maybe in the future, get rid of this inner if statementand try something else? If statements in inner
      //   nested loops can be very expensive.
      if (saveGlobalPosX == saveGlobalPosY){
        // Set the first position in row to a 1 -> special only to LowerTriangular matrices.
        matrix[i][j] = 1.;
      }
      else{
        U globalPos = saveGlobalPosY + saveGlobalPosX*globalDimensionY;
        matrix[i][j] = drand48();			// Change this later.
      }
      // check padding
      if (padXlen != dimensionX) { matrix[i][dimensionY-i] = 0; }
      saveGlobalPosY += globalPgridY;
    }
    // check padding
    if (padXlen != dimensionX) { matrix[dimensionX-1][0] = 0; }
    saveGlobalPosY = localPgridY;			// reset
    saveGlobalPosX += globalPgridX;
  }
  return;
}
