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

  U saveGlobalPosition = localPgridY + localPgridX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<dimensionY; j++)
    {
      srand48(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    saveGlobalPosition += (globalPgridX*globalDimensionY);
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


  U saveGlobalPositionX = localPgridX;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  U saveGlobalPositionY = localPgridY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<dimensionX; i++)
  {
    saveGlobalPositionY = localPgridY;
    for (U j=0; j<dimensionY; j++)
    {
      if (saveGlobalPositionX > saveGlobalPositionY)
      {
        srand48(saveGlobalPositionX + globalDimensionY*saveGlobalPositionY);
      }
      else
      {
        srand48(saveGlobalPositionY + globalDimensionY*saveGlobalPositionX);
      }
      matrix[i][j] = drand48();			// Change this later.
      if ((diagonallyDominant) && (saveGlobalPositionX == saveGlobalPositionY) && (i==j))
      {
        matrix[i][j] += globalDimensionX;		// X or Y, should not matter
      }
      saveGlobalPositionY += globalPgridY;
    }
    saveGlobalPositionX += globalPgridX;
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

  U saveGlobalPosition = localPgridY + localPgridX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<dimensionY; j++)
    {
      srand48(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    saveGlobalPosition += (globalPgridX*globalDimensionY);
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
  U saveGlobalPosition = localPgridY + localPgridX*globalDimensionY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  U counter{1};
  U startIter;
  U endIter;
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    startIter = 0;
    endIter = counter;
    for (U j=startIter; j<endIter; j++)
    {
      srand48(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    // Special corner case: If a processor's first data on each row is out of bounds of the UT structure, then give a 0 value
    if (localPgridY > localPgridX)
    {
      matrix[i][endIter-1] = 0;			// reset this to 0 instead of whatever was set in the last iteration of the loop above.
    }
    counter++;
    saveGlobalPosition += (globalPgridX*globalDimensionY);
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
  U counter{0};
  for (U i=0; i<dimensionX; i++)
  {
    // Special corner case: If a processor's last data on each row is out of bounds of the LT structure, then give a 0 value
    counter = 0;		// reset
    saveGlobalPosY += (i*globalPgridY);
    if (localPgridY < localPgridX)
    {
      // Set the last position in row
      matrix[i][0] = 0;
      counter++;
      saveGlobalPosY += globalPgridY;
    }
    for (U j=counter; j<(dimensionY-i); j++)
    {
      // Maybe in the future, get rid of this inner if statementand try something else? If statements in inner
      //   nested loops can be very expensive.
      if (saveGlobalPosX == saveGlobalPosY)
      {
        // Set the first position in row to a 1 -> special only to LowerTriangular matrices.
        matrix[i][j] = 1.;
      }
      else
      {
        U globalPos = saveGlobalPosY + saveGlobalPosX*globalDimensionY;
        srand48(globalPos);
        matrix[i][j] = drand48();			// Change this later.
      }
      saveGlobalPosY += globalPgridY;
    }
    saveGlobalPosY = localPgridY;			// reset
    saveGlobalPosX += globalPgridX;
  }
  return;
}
