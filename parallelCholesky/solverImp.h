//#include "solver.h" careful here

#define DEBUGGING

template<typename T>
solver<T>::solver(int rank, int size, int nDims, int matrixDimSize)
{
  this->worldRank = rank;
  this->worldSize = size;
  this->nDims = nDims;						// should always be 3
  this->matrixDimSize = matrixDimSize;

  /*
	Lets precompute a list of cubes for quick lookUp of cube dimensions based on processor size
  */
  for (int i=1; i<500; i++)
  {
    this->gridSizeLookUp[i*i*i] = i; 				// quick lookup for (grid size, 2d sheet size)
  }
}

template <typename T>
void solver<T>::startUp(bool &flag)
{
  /*
	Look up to see if the total processor size is a cube. If not, then return.
	We can reduce this restriction after we have it working for a perfect cubic processor grid
	If found, this->processorGridDimSize will represent the number of processors along a row or column (P^(1/3))
  */

  if (this->gridSizeLookUp.find(this->worldSize) == gridSizeLookUp.end())
  {
    flag = true;
    return;
  }

  this->processorGridDimSize = this->gridSizeLookUp[this->worldSize];

  /*
	this->baseCaseSize gives us a size in which to stop recursing in the LURecurse method
	= n/P^(2/3). The math behind it is written about in my report and other papers
  */
  this->baseCaseSize = this->matrixDimSize/(this->processorGridDimSize*this->processorGridDimSize);

  this->gridDims.resize(this->nDims,this->processorGridDimSize);
  this->gridCoords.resize(this->nDims); 
  
  /*
	Prepare to create some communicators that will be needed later
	The Cart Communicator is mainly used to distribute the random data in a block-cyclic fashion
	The other communicators are created for communication reasons. Will come back to this later
  */
  std::vector<int> boolVec(3,0);
  MPI_Cart_create(MPI_COMM_WORLD, this->nDims, &this->gridDims[0], &boolVec[0], false, &this->grid3D);
  MPI_Comm_rank(this->grid3D, &this->grid3DRank);
  MPI_Comm_size(this->grid3D, &this->grid3DSize);

  /*
	Remember that this call, like the 2 above it, are collective, meaning that every rank present in the communicator must call it. So MPI_Cart_coords gives a different 3-tuple location to each rank/processor in the cartesian communicator.
  */
  MPI_Cart_coords(this->grid3D, this->grid3DRank, this->nDims, &this->gridCoords[0]);

  /*
    Secret: Before I create row and column communicators, I need to split grid3D into a separate communicator for its layers.
  */

  // 2D (xy) Layer Communicator (split by z coordinate)
  MPI_Comm_split(this->grid3D, this->gridCoords[2],this->grid3DRank,&this->layerComm);
  MPI_Comm_rank(this->layerComm,&layerCommRank);
  MPI_Comm_size(this->layerComm,&layerCommSize);
  // Row Communicator
  MPI_Comm_split(this->layerComm, this->gridCoords[0],this->gridCoords[1],&this->rowComm);
  MPI_Comm_rank(this->rowComm,&this->rowCommRank);
  MPI_Comm_size(this->rowComm,&this->rowCommSize);
  // column Communicator
  MPI_Comm_split(this->layerComm, this->gridCoords[1],this->gridCoords[0],&this->colComm);
  MPI_Comm_rank(this->colComm,&this->colCommRank);
  MPI_Comm_size(this->colComm,&this->colCommSize);
  // Depth Communicator
  MPI_Comm_split(this->grid3D,this->gridCoords[0]*this->processorGridDimSize+this->gridCoords[1],this->gridCoords[2],&this->depthComm);
  MPI_Comm_rank(this->depthComm,&this->depthCommRank);
  MPI_Comm_size(this->depthComm,&this->depthCommSize);
}


/*
  Lets focus only on a cyclic distribution for now.
*/

template <typename T>
void solver<T>::collectDataCyclic()
{
  /*
	When writing this code to distribute the values, you have to think of who lives there. If we think using the Cartesian rank coordinates, then we dont care what layer (in the z-direction) we are in. We care about the (x,y,?) coordinates. Sublayers exist in a block-cyclic distribution, so that in the first sublayer of the matrix, all processors are used, therefore, we cycle them from (0,1,2,0,1,2,0,1,2,....) in the x-dimension of te cube (when traveling down from the top-front-left corner) to the bottom-front-left corner. And then for each of those, there are the processors in the y-direction as well, which are representing via the nested loop.

*/

  // Note that because matrix A is supposed to be symmetric, I will only give out the lower-triangular portion of A
  // Note that this would require a change in my algorithm, because I do reference uper part of A.
  // Instead, I will only give values for the upper-portion of A. At a later date, I can change the code to only
  // allocate a triangular portion instead of storing half as zeros.

  for (int i=this->gridCoords[0]; i<this->matrixDimSize; i+=this->processorGridDimSize)
  {
    this->localSize = 0;	// Its fine to keep resetting this
    for (int j=this->gridCoords[1]; j<this->matrixDimSize; j+=this->processorGridDimSize)
    {
      srand(i*this->matrixDimSize + j);			// this is very important. Needs to be the same for all processors
    							// completely based on where we are in entire matrix
      if (i < j)  // if lower-triangular, I can just give a zero
      {
        matrixA.push_back(0.);
      }
      else
      {
        matrixA.push_back((rand()%100)*1./100.);
      }
      if (i==j)
      {
        matrixA[matrixA.size()-1]+=10.;		// we want all diagonals to be dominant.
      }
      this->localSize++;
    }
  }
  for (int i=0; i<this->localSize; i++)
  {
    for (int j=0; j<=i; j++)
    {
      this->matrixA[j*this->localSize+i] = this->matrixA[i*this->localSize+j];
    }
  }

  int temp = this->matrixA.size()-((this->localSize*(this->localSize-1))>>1);                   // n^2 - n(n-1)/2
  this->matrixL.resize(temp);
  this->matrixLInverse.resize(temp);
/*
  matrixB.resize(temp);						// I could fill this up with A or just start it off at zeros. I reset it anyway 
*/

  this->matrixB.resize(this->matrixA.size(),0.);
}


/********************************************************************************************************************************************/
// This divides the program. If anything above this point is incorrect, then everything below will be incorrect as well.
/********************************************************************************************************************************************/



/*
  The solve method will initiate the solving of this LU Factorization algorithm
  All ranks better be ready to go in order to solve this problem.
*/
template<typename T>
void solver<T>::solve()
{
  /*
	Note: I changed the arguments from this->matrixDimSize, which doesnt help given the fact that we are dealing only with the 2-d vectors
		that each process holds, to this->matrixA.size(), which does represent what we are cutting up
  
	Check the final argument. For safety reasons, I changed it. Be careful!
  */

  LURecurse(0,this->localSize,0,this->localSize,this->localSize,this->matrixDimSize,0);  // start on track 1
}

/*
  This function is where I can recurse within each of the P^{1/3} layers separately
  But after that, we perform matrix multiplication with all P processors in the 3-d grid.
*/
template<typename T>
void solver<T>::LURecurse(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int matrixSize, int matrixTrack)
{

  /*
    Recursive base case. I debugged this and it is correct.
  */
  if (matrixSize == this->baseCaseSize)
  {
    LURecurseBaseCase(dimXstart,dimXend,dimYstart,dimYend,matrixWindow,matrixSize,matrixTrack);
    return;
  }

  /*
	Recursive case -> Keep recursing on the top-left half
  */

  int shift = matrixWindow>>1;
  LURecurse(dimXstart,dimXend-shift,dimYstart,dimYend-shift,shift,(matrixSize>>1),matrixTrack);
  
  MM(dimXstart,dimXend-shift,dimYstart,dimYend-shift,dimXstart,dimXend-shift,dimYstart+shift,dimYend,dimXstart+shift,dimXend,dimYstart,dimYend-shift,shift,(matrixSize>>1),0,matrixTrack);

  //Below is the tricky one. We need to create a vector via MM(..), then subtract it from A via some sort easy method...
  MM(dimXstart+shift,dimXend,dimYstart,dimYend-shift,dimXstart+shift,dimXend,dimYstart,dimYend-shift,dimXstart,dimXend-shift,dimYstart,dimYend-shift,shift,(matrixSize>>1),1,matrixTrack);

  // Big question: LURecurseBaseCase relies on taking from matrixA, but in this case, we would have to take from a modified matrix via the Schur Complement.

  // Note that the below code has room for optimization via some call to LAPACK, but I would have to loop over and put into a 1d array first and then
  // pack it back into my 2d array. So in general, I need to look at whether modifying the code to use 1D vectors instead of 2D vectors is worth it.

  // REPLACE THE BELOW WITH LAPACK SYRK ONCE I FINISH FIXING CORRECTNESS ERRORS
 

  int temp = (dimXstart+shift)*this->localSize + dimYstart+shift;
  for (int i=0; i<shift; i++)
  {
    int save = i*shift;
    for (int j=0; j<shift; j++)
    {
      this->matrixB[temp+j] = this->matrixA[temp+j] - this->holdMatrix[save+j];
    }
    //temp += matrixWindow;
    temp += this->localSize;		// remember that matrixB is localSize x localsize, so when we write to a square portion
						// of it, our indices have to reflect that.
  }

  LURecurse(dimXstart+shift,dimXend,dimYstart+shift,dimYend,shift,(matrixSize>>1),1);  // HALT! Change the track!

  // These last 4 matrix multiplications are for building up our LInverse and UInverse matrices
  MM(dimXstart+shift,dimXend,dimYstart,dimYend-shift,dimXstart,dimXend-shift,dimYstart,dimYend-shift,dimXstart,dimXend-shift,dimYstart,dimYend-shift,shift,(matrixSize>>1),2,matrixTrack);

  MM(dimXstart+shift,dimXend,dimYstart+shift,dimYend,dimXstart,dimXend-shift,dimYstart,dimYend-shift,dimXstart+shift,dimXend,dimYstart,dimYend-shift,shift,(matrixSize>>1),3,matrixTrack);

}

/*
  Recursive Matrix Multiplication -> needs arguments that makes sense.

  This routine requires that matrix data is duplicated across all layers of the 3D Processor Grid
*/
template<typename T>
void solver<T>::MM(int dimXstartA,int dimXendA,int dimYstartA,int dimYendA,int dimXstartB,int dimXendB,int dimYstartB,int dimYendB,int dimXstartC, int dimXendC, int dimYstartC, int dimYendC, int matrixWindow,int matrixSize, int key, int matrixTrack)
{
  /*
    I need two broadcasts, then an AllReduce
  */

  /*
    I think there are only 2 possibilities here. Either size == matrixWindow*(matrixWindow)/2 if the first element fits
						     or size == matrixWindow*(matrixWindow)/2 - 1 if the first element does not fit.
  */
  std::vector<T> buffer1;
  std::vector<T> buffer2;

  if (this->rowCommRank == this->gridCoords[2])     // different matches on each of the P^(1/3) layers
  {
    switch (key)
    {
      case 0:
      {
        // Triangular -> (lower,square) -> Note for further optimization.
        //                              -> This copy loop may not be needed. I might be able to purely send in &this->matrixLInverse[0] into broadcast
        buffer1.resize(matrixWindow*(matrixWindow+1)>>1,0.);
        int startOffset = (dimXstartA*(dimXstartA+1)>>1);
        int index1 = 0;
        int index2 = startOffset + dimYstartA;
        for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<=i; j++)
          {
            buffer1[index1++] = this->matrixLInverse[index2++];
          }
          index2 += dimYstartA;
        }
        break;
      }
      case 1:
      {
        // Not triangular
        buffer1.resize(matrixWindow*matrixWindow);
        int startOffset = (dimXstartA*(dimXstartA+1))>>1;
        int index1 = 0;
        int index2 = startOffset + dimYstartA;
        for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<matrixWindow; j++)
          {
            buffer1[index1++] = this->matrixL[index2++];
          }
          index2 += dimYstartA;
        }
        break;
      }
      case 2:		// Part of the Inverse L calculation
      {
        // Not Triangular
        buffer1.resize(matrixWindow*matrixWindow);
        int startOffset = (dimXstartA*(dimXstartA+1))>>1;
        int index1 = 0;
        int index2 = startOffset + dimYstartA;
        for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<matrixWindow; j++)
          {
            buffer1[index1++] = this->matrixL[index2++];
          }
          index2 += dimYstartA;
        }
        break;
      }
      case 3:
      {
        // Triangular -> Lower
        // As noted above, size depends on whether or not the gridCoords lie in the lower-triangular portion of first block
        buffer1.resize((matrixWindow*(matrixWindow+1))>>1);
        int startOffset = (dimXstartA*(dimXstartA+1))>>1;
        int index1 = 0;
        int index2 = startOffset + dimYstartA;
        for (int i=0; i<matrixWindow; i++)		// start can take on 2 values corresponding to the situation above: 1 or 0
        {
          for (int j=0; j<i+1; j++)
          {
            buffer1[index1++] = this->matrixLInverse[index2++];
          }
          index2 += dimYstartA;
        }
        break;
      }        
    }
    
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }
  else
  {
    switch (key)
    {
      case 0:
      {
        buffer1.resize((matrixWindow*(matrixWindow+1))>>1);
        break;
      }
      case 1:
      {
        buffer1.resize(matrixWindow*matrixWindow);
        break;
      }
      case 2:
      {
        buffer1.resize(matrixWindow*matrixWindow);
        break;
      }
      case 3:
      {
        buffer1.resize((matrixWindow*(matrixWindow+1))>>1);
        break;
      }
    }

    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }

  if (this->colCommRank == this->gridCoords[2])    // May want to recheck this later if bugs occur.
  {
    switch (key)
    {
      case 0:
      {
        //Not Triangular
        buffer2.resize(matrixWindow*matrixWindow);
        int startOffset = dimXstartB*this->localSize;	// Problem! startOffset needs to be based off of this->localRank, not anything else
        int index1 = 0;
        int index2 = startOffset + dimYstartB;
        for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<matrixWindow; j++)
          {
            buffer2[index1++] = (matrixTrack ? this->matrixB[index2+j] : this->matrixA[index2+j]);
          }
          index2 += this->localSize;
        }
        break;
      }
      case 1:
      {
        // Not Triangular, but transpose
        buffer2.resize(matrixWindow*matrixWindow);
        int startOffset = (dimXstartB*(dimXstartB+1))>>1;
        int index1 = 0;
        for (int i=0; i<matrixWindow; i++)
        {
          int index2 = startOffset + dimYstartB + i;
          int temp = /*dimYstartB+*/1+matrixWindow;			// I added the +1 back in
          for (int j=0; j<matrixWindow; j++)
          {
            buffer2[index1++] = this->matrixL[index2];                  // Transpose has poor spatial locality
            index2 += temp+j+dimYstartB;
          }
        }
        break;
      }
      case 2:
      {
        // Triangular -> Lower
        // As noted above, size depends on whether or not the gridCoords lie in the lower-triangular portion of first block
        buffer2.resize((matrixWindow*(matrixWindow+1))>>1);	// change to bitshift later
        int startOffset = (dimXstartB*(dimXstartB+1))>>1;
        int index1 = 0;
        int index2 = startOffset + dimYstartB;
        for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<=i; j++)
          {
            buffer2[index1++] = this->matrixLInverse[index2++];
          }
          index2 += dimYstartB;
        }
        break;
      }
      case 3:
      {
        // Not Triangular, but this requires a special traversal because we are using this->holdMatrix
        buffer2.resize(matrixWindow*matrixWindow);
        //int startOffset = (dimXstartB*(dimXstartB-1))>>1;
        int index1 = 0;
        int index2 = 0;
        //int index2 = startOffset + dimYstartB;
        for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<matrixWindow; j++)
          {
            buffer2[index1++] = (-1)*this->holdMatrix[index2++];	// I just changed this to use a (-1)
          }
          //index2 += dimYstartB;
        }
        break;
      }
    }
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->colComm);
  }
  else
  {
    switch (key)
    {
      case 0:
      {
        // What I can do here is receive with a buffer of size matrixWindow*(matrixWindow+1)/2 and then iterate over the last part to see what kind it is
        // i was trying to do it with gridCoords, but then I would have needed this->rowRank, stuff like that so this is a bit easier
        // Maybe I can fix this later to make it more stable
        buffer2.resize(matrixWindow*matrixWindow);	// this is a special trick
        break;
      }
      case 1:
      {
        buffer2.resize(matrixWindow*matrixWindow);
        break;
      }
      case 2:
      {
        // What I can do here is receive with a buffer of size matrixWindow*(matrixWindow+1)/2 and then iterate over the last part to see what kind it is
        // i was trying to do it with gridCoords, but then I would have needed this->rowRank, stuff like that so this is a bit easier
        // Maybe I can fix this later to make it more stable
        buffer2.resize((matrixWindow*(matrixWindow+1))>>1);	// this is a special trick
        break;
      }
      case 3:
      {
        buffer2.resize(matrixWindow*matrixWindow);
        break;
      }
    }
    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->colComm);
  }

  /*
    Now once here, I have the row data and the column data residing in buffer1 and buffer2
    I need to use the data in these buffers correctly to calculate partial sums for each place that I own. And store them together in order to
      properly reduce the sums. So use another temporary buffer. Only update this->matrixU or this->matrixL after the reduction if root or after the final
        broadcast if non-root.
  */

  // So iterate over everything that we own and calculate its partial sum that will be used in reduction
  // buffer3 will be used to hold the partial matrix sum result that will contribute to the reduction.
  std::vector<T> buffer4(matrixWindow*matrixWindow,0.);	// need to give this outside scope

  switch (key)
  {
    case 0:
    {
      // Triangular Multiplicaton -> (lower,square)
      std::vector<T> updatedVector(matrixWindow*matrixWindow,0.);
      int index = 0;
      for (int i = 0; i<matrixWindow; i++)
      {
        int temp = i*matrixWindow;
        for (int j = 0; j <= i; j++)
        {
          updatedVector[temp+j] = buffer1[index++];
        }
      }
      cblas_dtrmm(CblasRowMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,matrixWindow,matrixWindow,1.,&updatedVector[0],matrixWindow,&buffer2[0],matrixWindow);
      MPI_Allreduce(&buffer2[0],&buffer4[0],buffer2.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);
      break;
    }
    case 1:
    {
      // Matrix Multiplication -> (square,square)
      std::vector<T> buffer3(matrixWindow*matrixWindow);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,matrixWindow,matrixWindow,matrixWindow,1.,&buffer1[0],matrixWindow,&buffer2[0],matrixWindow,1.,&buffer3[0],matrixWindow);
      MPI_Allreduce(&buffer3[0],&buffer4[0],buffer3.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);
      break;
    }
    case 2:
    {
      // Triangular Multiplication -> (square,lower)
      std::vector<T> updatedVector(matrixWindow*matrixWindow,0.);
      int index = 0;
      for (int i = 0; i<matrixWindow; i++)
      {
        int temp = i*matrixWindow;
        for (int j = 0; j <= i; j++)	// this could be wrong ???
        {
          updatedVector[temp+j] = buffer2[index++];
        }
      }
      
      cblas_dtrmm(CblasRowMajor,CblasRight,CblasLower,CblasNoTrans,CblasNonUnit,matrixWindow,matrixWindow,1.,&updatedVector[0],matrixWindow,&buffer1[0],matrixWindow);
      MPI_Allreduce(&buffer1[0],&buffer4[0],buffer1.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);	// use buffer4
      break;
    }
    case 3:
    {
      // Triangular Multiplication -> (lower,square)
      std::vector<T> updatedVector(matrixWindow*matrixWindow,0.);
      int index = 0;
      for (int i = 0; i<matrixWindow; i++)
      {
        int temp = i*matrixWindow;
        for (int j = 0; j <= i; j++)	// This could be wrong?
        {
          updatedVector[temp+j] = buffer1[index++];
        }
      }

      cblas_dtrmm(CblasRowMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,matrixWindow,matrixWindow,1.,&updatedVector[0],matrixWindow,&buffer2[0],matrixWindow);
      MPI_Allreduce(&buffer2[0],&buffer4[0],buffer2.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);
      break;
    }
  }

  /*
    Now I need a case statement of where to put the guy that was just broadasted.
  */

  this->holdMatrix.resize(matrixWindow*matrixWindow,0.);
  switch (key)
  {
    case 0:	// Beware, this is a traspose
    {
      // Square and Transpose
      int startOffset = (dimXstartC*(dimXstartC+1))>>1;
      int index1 = 0;
      for (int i=0; i<matrixWindow; i++)
      {
        int index2 = startOffset + dimYstartC + i;
        int temp = 1+matrixWindow;		// added the 1 back in. See if this works
        for (int j=0; j<matrixWindow; j++)
        {
          this->matrixL[index2] = buffer4[index1++];                    // Transpose has poor spatial locality. Could fix later if need be
          index2 += temp + j+dimYstartC;		// j captures the idea of each further row having an additional column
        }
      }
      break;
    }
    case 1:
    {
      int index1 = 0;
      for (int i=0; i<matrixWindow; i++)
      {
        int temp = i*matrixWindow;
        for (int j=0; j<matrixWindow; j++)
        {
          this->holdMatrix[temp+j] = buffer4[index1++];
        }
      }
      break;
    }
    case 2:
    {
      //int startOffset = (dimXstartC*(dimXstartC+1))>>1;
      int index1 = 0;
      //int index2 = startOffset + dimYstartC;
      int index2 = 0;
      for (int i=0; i<matrixWindow; i++)
      {
        for (int j=0; j<matrixWindow; j++)
        {
          this->holdMatrix[index2++] = buffer4[index1++];
        }
        //index2 += dimYstartC;
      }
      break;
    }
    case 3:
    {
      int startOffset = (dimXstartC*(dimXstartC+1))>>1;
      int index1 = 0;
      int index2 = startOffset + dimYstartC;
      for (int i=0; i<matrixWindow; i++)
      {
        for (int j=0; j<matrixWindow; j++)
        {
          this->matrixLInverse[index2++] = buffer4[index1++];
        }
        index2 += dimYstartC;
      }
      break;
    }
  }

}


/*
  Base case of LURecurse
*/
template<typename T>
void solver<T>::LURecurseBaseCase(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int matrixSize, int matrixTrack)
{
  /*
	1) AllGather onto a single processor, which has to be chosen carefully
	2) Sequential BLAS-3 routines to solve for LU and inverses

	For now, I can just create a 1-d buffer of the data that I want to send first.
	Later, I can try to optimize this to avoid needless copying just for a collective

	Also remember the importance of using the corect communicator here. We need the sheet communicator of size P^{2/3}
	This avoids traffic and is the only way it could work

	I think that I will need to receive the data in a buffer, and then use another buffer to line up the data properly
	This is extra computational cost, but for the moment, I don't see any other option

	This data will be used to solve for L,U,L-I, and U-I. So it is very important that the input buffer be correct for BLAS-3
	Then we will work on communicating the correct data to the correct places so that RMM operates correctly.
 
	So below, I need to transfer a 2-d vector into a 1-d vector
  */

  std::vector<T> sendBuffer((matrixWindow*(matrixWindow+1))>>1,0.);

  /* Note that for now, we are accessing matrix A as a 1d square vector, NOT a triangular vector, so startOffset is not sufficient
  int startOffset = (dimXstart*(dimXstart+1))>>1;
  */
  int index1 = 0;
  //int index2 = startOffset + dimYstart;
  int index2 = dimXstart*this->localSize + dimYstart;
  if (matrixTrack == 0)
  {
    for (int i=0; i<matrixWindow; i++)
    {
      for (int j=0; j<=i; j++)			// Note that because A is upper-triangular (shouldnt really matter), I must
						// access A differently than before
      {
        sendBuffer[index1++] = this->matrixA[index2++];
      }
      index2 += (this->localSize - matrixWindow+1);					// this should be correct
    }
  }
  else /* matrixTrack == 1 */
  {
    for (int i=0; i<matrixWindow; i++)
    {
      for (int j=0; j<=i; j++)
      {
        sendBuffer[index1++] = this->matrixB[index2++];
      }
      index2 += (this->localSize - matrixWindow+1);					// this should be correct
    }
  }

  std::vector<T> recvBuffer(sendBuffer.size()*this->processorGridDimSize*this->processorGridDimSize,0);
  MPI_Allgather(&sendBuffer[0], sendBuffer.size(),MPI_DOUBLE,&recvBuffer[0],sendBuffer.size(),MPI_DOUBLE,this->layerComm);

  /*
	After the all-gather, I need to format recvBuffer so that it works with OpenBLAS routines!
  */


  // I still need to fix the below. The way an allgather returns data is in blocks
  int count = 0;
  sendBuffer.clear();				// This is new. Edgar says to use capacity??
  sendBuffer.resize(matrixSize*matrixSize,0.);
  for (int i=0; i<this->processorGridDimSize*this->processorGridDimSize; i++)  // MACRO loop over all processes' data (stored contiguously)
  {
    for (int j=0; j<matrixWindow; j++)
    {
      for (int k=0; k<=j; k++)		// I changed this. It should be correct.
      {
        int index = j*this->processorGridDimSize*matrixSize+(i/this->processorGridDimSize)*matrixSize + k*this->processorGridDimSize+(i%this->processorGridDimSize);  // remember that recvBuffer is stored as P^(2/3) consecutive blocks of matrix data pertaining to each p
        sendBuffer[index] = recvBuffer[count++];
      }
    }
  }

  // Cholesky factorization

  // HOLD UP. Why isnt this Cholesky returning L?
  // I will try modifying the upper-triangular part and seeing what happens. Nothing should change
  LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',matrixSize,&sendBuffer[0],matrixSize);

  //recvBuffer.resize(sendBuffer.size(),0.);			// I am just re-using vectors. They change form as to what data they hold all the time
 
  recvBuffer = sendBuffer;

  // I am assuming that the diagonals are ok (maybe they arent all 1s, but that should be ok right?) 
  // Hold on. I think that i JUST FOUND my bug. recvBuffer is all zeros. It has no valid info. I think I needed to copy
  // sendBuffer into recvBuffer first. Check on this after I fix the seg fault.
  LAPACKE_dtrtri(LAPACK_ROW_MAJOR,'L','N',matrixSize,&recvBuffer[0],matrixSize);

  // Hold up. We need to discuss this. I am traversing this matrixSize x matrixSize matrix redundantly,
    // meaning that 
  int pIndex = this->gridCoords[0]*this->processorGridDimSize+this->gridCoords[1];
  int startOffset = (dimXstart*(dimXstart+1))>>1;
  int rowCounter = 0;
  for (int i=0; i<matrixWindow; i++)
  {
    int temp = startOffset + (i+1)*dimYstart + rowCounter++;
    for (int j=0; j<=i; j++)	// for matrixWindow==2, j should always go to 
    {
      // I need to use dimXstart and dimXend and dimYstart and dimYend ...
      int index = i*this->processorGridDimSize*matrixSize+(pIndex/this->processorGridDimSize)*matrixSize + j*this->processorGridDimSize+(pIndex%this->processorGridDimSize);
      this->matrixL[temp+j] = sendBuffer[index];
      this->matrixLInverse[temp+j] = recvBuffer[index];
    }
  }
}

template<typename T>
void solver<T>::printL()
{
  // We only need to print out a single layer, due to replication of L on each layer
  if ((this->gridCoords[2] == 0) && (this->gridCoords[1] == 0) && (this->gridCoords[0] == 0))
  {
    int tracker = 0;
    for (int i=0; i<this->localSize; i++)
    {
      for (int j=0; j<this->localSize; j++)
      {
        if (i >= j)
        {
          std::cout << this->matrixL[tracker++] << " ";
        }
        else
        {
          std::cout << 0 << " ";
        }
      }
      std::cout << "\n";
    }
  }
}

template<typename T>
void solver<T>::lapackTest(int n)
{
  std::vector<T> data(n*n);
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      srand(i*n+j);
      data[i*n+j] = (rand()%100)*1./100.;
      if (i==j)
      {
        data[i*n+j] += 10.;
      }
    }
  }

  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      std::cout << data[i*n+j] << " ";
    }
    std::cout << "\n";
  }

  LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',n,&data[0],n);
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      std::cout << data[i*n+j] << " ";
    }
    std::cout << "\n";
  }
  return;
}

template<typename T>
void solver<T>::solveScalapack()
{
  //PDGETRF(&this->matrixA.size(), &this->matrixA.size(), &this->matrixA[...][...],....... lots of more stuff t keep track of);
}
