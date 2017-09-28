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
		   U globalPgridY
		 )
{

  U saveGlobalPosition = localPgridY*globalDimensionX+localPgridX;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<dimensionY; i++)
  {
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<dimensionX; j++)
    {
      srand48(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridX;
    }
    saveGlobalPosition += (globalPgridY*globalDimensionX);
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
                   bool diagonallyDominant
		 )
{
  // Note: this is not fully implemented yet, as I have not decided on whether I need to perform a local transpose
  //       or local (but distributed based on the values each processor gets) transpose.

  std::cout << "HEREHEREHERE\n";

  U saveGlobalPosition = localPgridY*globalDimensionX+localPgridX;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  U trackX = localPgridX;
  U trackY = localPgridY;
  for (U i=0; i<dimensionY; i++)
  {
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<dimensionX; j++)
    {
      srand48(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      if ((diagonallyDominant) && (trackX == trackY) && (i==j))
      {
        matrix[i][j] += globalDimensionX;		// X or Y, should not matter
      }
      globalPosition += globalPgridX;
      trackX += globalPgridX;
    }
    saveGlobalPosition += (globalPgridY*globalDimensionX);
    trackY += globalPgridY;
    trackX = localPgridX;		// reset
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
		   U globalPgridY
		 )
{

  U saveGlobalPosition = localPgridY*globalDimensionX+localPgridX;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<dimensionY; i++)
  {
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<dimensionX; j++)
    {
      srand48(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridX;
    }
    saveGlobalPosition += (globalPgridY*globalDimensionX);
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
		   U globalPgridY
		 )
{
  U saveGlobalPosition = localPgridY*globalDimensionX+localPgridX;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  U counter{0};
  U startIter;
  U endIter;
  for (U i=0; i<dimensionY; i++)
  {
    U globalPosition = saveGlobalPosition;
    startIter = 0;
    endIter = dimensionX-counter;
    // Special corner case: If a processor's first data on each row is out of bounds of the UT structure, then give a 0 value
    if (localPgridY > localPgridX)
    {
      matrix[i][0] = 0;
      startIter++;
      globalPosition += globalPgridX;
    }
    for (U j=startIter; j<endIter; j++)
    {
      srand48(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridX;
    }
    counter++;
    saveGlobalPosition += (globalPgridY*globalDimensionX);
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
		   U globalPgridY
		 )
{
  U saveGlobalPosX = localPgridX;
  U saveGlobalPosY = localPgridY;
  U counter{1};
  for (U i=0; i<dimensionY; i++)
  {
    counter=i+1;
    // Special corner case: If a processor's last data on each row is out of bounds of the UT structure, then give a 0 value
    if (localPgridY < localPgridX)
    {
      // Set the last position in row
      matrix[i][counter-1] = 0;
      counter--;
    }
    for (U j=0; j<counter; j++)
    {
      // Maybe in the future, get rid of this inner if statementand try something else? If statements in inner
      //   nested loops can be very expensive.
      if (saveGlobalPosX == saveGlobalPosY)
      {
        // Set the first position in row to a 1 -> special only to LowerTriangular matrices.
        matrix[i][j] = 1.;
      }
      else if (saveGlobalPosY < saveGlobalPosX)
      {
        matrix[i][j] = 0;
      }
      else
      {
        U globalPos = saveGlobalPosY*globalDimensionX+saveGlobalPosX;
        srand48(globalPos);
        matrix[i][j] = drand48();			// Change this later.
      }
      saveGlobalPosX += globalPgridX;
    }
    saveGlobalPosY += globalPgridY;
    saveGlobalPosX = localPgridX;
  }
  return;
}
