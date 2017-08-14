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
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY
		 )
{

  U saveGlobalPosition = localPgridX*dimensionY+localPgridY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<dimensionX; j++)
    {
      srand(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    saveGlobalPosition += (globalPgridX*dimensionY);
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
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY
		 )
{

  U saveGlobalPosition = localPgridX*dimensionY+localPgridY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    for (U j=0; j<dimensionY; j++)
    {
      srand(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    saveGlobalPosition += (globalPgridX*dimensionY);
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
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY
		 )
{
  U saveGlobalPosition = localPgridX*dimensionY+localPgridY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  U counter{0};
  U startIter;
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    // Special corner case: If a processor's first data on each row is out of bounds of the UT structure, then give a 0 value
    if (localPgridX > localPgridY)
    {
      matrix[i][0] = 0;
      startIter = counter+1;
      globalPosition += globalPgridY;
    }
    else
    {
      startIter = counter;
    }
    for (U j=startIter; j<dimensionY; j++)
    {
      srand(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    counter++;
    saveGlobalPosition += (globalPgridX*dimensionY);
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
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY
		 )
{
  U saveGlobalPosition = localPgridX*dimensionY+localPgridY;		// Watch for 64-bit problems later with temporaries being implicitely casted.
  U counter{1};
  U startIter;
  for (U i=0; i<dimensionX; i++)
  {
    U globalPosition = saveGlobalPosition;
    counter=i+1;
    // Special corner case: If a processor's last data on each row is out of bounds of the UT structure, then give a 0 value
    if (localPgridX < localPgridY)
    {
      // Set the last position in row
      matrix[i][counter-1] = 0;
      counter--;
      startIter = 0;
    }
    else if (localPgridX == localPgridY)
    {
      // Set the first position in row to a 1 -> special only to LowerTriangular matrices.
      matrix[i][0] = 1.;
      startIter = 1;
      globalPosition += globalPgridY;
    }
    else
    {
      startIter = 0;
    }
    for (U j=startIter; j<counter; j++)
    {
      srand(globalPosition);
      matrix[i][j] = drand48();			// Change this later.
      globalPosition += globalPgridY;
    }
    saveGlobalPosition += (globalPgridX*dimensionY);
  }
  return;
}
