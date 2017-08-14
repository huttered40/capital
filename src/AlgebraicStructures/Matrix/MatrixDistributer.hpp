/* Author: Edward Hutter */

// Try this to get it to compile: provide a declaration of a templated constructor,
//   and then define/implement a specialized method.


// MatrixStructureSquare
template<typename T, typename U>
void MatrixDistributerCyclic<T,U,0>::Distribute
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

  U saveGlobalPosition = localPgridX*globalDimensionY+localPgridY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<dimensionX; j++)
    {
      srand(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    saveGlobalPosition += (globalPgridX*globalDimensionY);
  }
  return;
}

// MatrixStructureRectangle
template<typename T, typename U>
void MatrixDistributerCyclic<T,U,1>::Distribute
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

  U saveGlobalPosition = localPgridX*globalDimensionY+localPgridY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<dimensionY; j++)
    {
      srand(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    saveGlobalPosition += (globalPgridX*globalDimensionY);
  }
  return;
}

// MatrixStructureUpperTriangular
template<typename T, typename U>
void MatrixDistributerCyclic<T,U,2>::Distribute
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
  U saveGlobalPosition = localPgridX*globalDimensionY+localPgridY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  U counter{0};
  U startIter;
  U endIter;
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    startIter = 0;
    endIter = dimensionY-counter;
    // Special corner case: If a processor's first data on each row is out of bounds of the UT structure, then give a 0 value
    if (localPgridX > localPgridY)
    {
      matrix[i][0] = 0;
      startIter++;
      globalPosition += globalPgridY;
    }
    for (U j=startIter; j<endIter; j++)
    {
      srand(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    counter++;
    saveGlobalPosition += (globalPgridX*globalDimensionY);
  }
  return;
}

// MatrixStructureLowerTriangular
template<typename T, typename U>
void MatrixDistributerCyclic<T,U,3>::Distribute
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
  for (U i=0; i<dimensionX; i++)
  {
    counter=i+1;
    // Special corner case: If a processor's last data on each row is out of bounds of the UT structure, then give a 0 value
    if (localPgridX < localPgridY)
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
      else if (saveGlobalPosX < saveGlobalPosY)
      {
        matrix[i][j] = 0;
      }
      else
      {
        U globalPos = saveGlobalPosX*globalDimensionY+saveGlobalPosY;
        srand(globalPos);
        matrix[i][j] = drand48();			// Change this later.
      }
      saveGlobalPosY += globalPgridY;
    }
    saveGlobalPosX += globalPgridX;
    saveGlobalPosY = localPgridY;
  }
  return;
}
