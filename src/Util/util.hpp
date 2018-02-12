/* Author: Edward Hutter */

template<typename T, typename U>
std::vector<T> util<T,U>::blockedToCyclic(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, int pGridDimensionSize)
{
  U aggregNumRows = localDimensionRows*pGridDimensionSize;
  U aggregNumColumns = localDimensionColumns*pGridDimensionSize;
  U aggregSize = aggregNumRows*aggregNumColumns;
  std::vector<T> cyclicData(aggregSize);
  U numCyclicBlocksPerRow = localDimensionRows;
  U numCyclicBlocksPerCol = localDimensionColumns;
  U writeIndex = 0;
  U recvDataOffset = localDimensionRows*localDimensionColumns;
  // MACRO loop over all cyclic "blocks" (dimensionX direction)
  for (U i=0; i<numCyclicBlocksPerCol; i++)
  {
    // Inner loop over all columns in a cyclic "block"
    for (U j=0; j<pGridDimensionSize; j++)
    {
      // Inner loop over all cyclic "blocks"
      for (U k=0; k<numCyclicBlocksPerRow; k++)
      {
        // Inner loop over all elements along columns
        for (U z=0; z<pGridDimensionSize; z++)
        {
          U readIndex = i*numCyclicBlocksPerRow + j*recvDataOffset + k + z*pGridDimensionSize*recvDataOffset;
          cyclicData[writeIndex++] = blockedData[readIndex];
        }
      }
    }
  }

  // Should be quick pass-by-value via move semantics, since we are effectively returning a localvariable that is going to lose its scope anyways,
  //   so the compiler should be smart enough to use the move constructor for the vector in the caller function.
  return cyclicData;

}
