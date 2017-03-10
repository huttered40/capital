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
  MPI_Comm_rank(this->layerComm,&this->layerCommRank);
  MPI_Comm_size(this->layerComm,&this->layerCommSize);
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

  // New comment (March 1)
 
/*
  if (this->gridCoords[2] == 0)
  {
    std::cout << "check - " << this->layerCommRank << "        " << this->gridCoords[0] << " " << this->gridCoords[1] << std::endl;
  }
*/

  for (int i=this->gridCoords[0]; i<this->matrixDimSize; i+=this->processorGridDimSize)
  {
    this->localSize = 0;	// Its fine to keep resetting this
    for (int j=this->gridCoords[1]; j<this->matrixDimSize; j+=this->processorGridDimSize)
    {

      if (i > j)
      {
        srand(i*this->matrixDimSize + j);			// this is very important. Needs to be the same for all processors
      }
      else							// wont matter either way
      {
        srand(j*this->matrixDimSize + i);			// this is very important. Needs to be the same for all processors
      }
    							// completely based on where we are in entire matrix
//      if (i < j)  // if upper-triangular, I can just give a zero
//      {
//        matrixA.push_back(0.);
//      }
//      else
//      {
      matrixA.push_back((rand()%100)*1./100.);
//      if ((i==1) && (j==2))
//      {
//        std::cout << "(" << i << "," << j << ") - " << this->matrixA[matrixA.size()-1] << std::endl;
//      }
//      if ((i==2) && (j==1))
//      {
//        std::cout << "(" << i << "," << j << ") - " << this->matrixA[matrixA.size()-1] << std::endl;
//      }
//      }
//      if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 1) && (this->gridCoords[2] == 0)) { std::cout << "what - " << i << "," << j << " " << this->matrixA[this->matrixA.size()-1] << std::endl; }
      if (i==j)
      {
        matrixA[matrixA.size()-1]+=10.;		// we want all diagonals to be dominant.
      }
      this->localSize++;
    }
  }
  // This next nested loop section is special because it guarantees that matrixA is symmetric.
  // This is a slow, but safe way to do it. Can be optimizes later, but it doesnt count in the algorithm runtime

/*
  if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 1) && (this->gridCoords[2] == 0))
  {
    std::cout << "About to print what processor (0,1,0) owns\n";
    for (int i=0; i<this->localSize; i++)
    {
      for (int j=0; j<this->localSize; j++)
      {
        std::cout << this->matrixA[i*this->localSize+j] << " ";
      }
      std::cout << "\n";
    }
  }
*/
  
  int temp = ((this->localSize*(this->localSize+1))>>1);                   // n^2 - n(n-1)/2
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
  
  // Add MPI_SendRecv in here
  fillTranspose(dimXstart, dimXend-shift, dimYstart, dimYend-shift, shift, 0);

  MM(dimXstart+shift,dimXend,dimYstart,dimYend-shift,0,0,0,0,dimXstart+shift,dimXend,dimYstart,dimYend-shift,shift,(matrixSize>>1),0,matrixTrack);

  fillTranspose(dimXstart+shift, dimXend, dimYstart, dimYend-shift, shift, 1);
  

  //Below is the tricky one. We need to create a vector via MM(..), then subtract it from A via some sort easy method...
  MM(dimXstart+shift,dimXend,dimYstart,dimYend-shift,0,0,0,0,dimXstart,dimXend-shift,dimYstart,dimYend-shift,shift,(matrixSize>>1),1,matrixTrack);

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
  std::vector<T> buffer1;  // use capacity() or reserve() methods here?
  std::vector<T> buffer2;

  if (this->rowCommRank == this->gridCoords[2])     // different matches on each of the P^(1/3) layers
  {
    switch (key)
    {
      case 0:
      {
        // (square,lower) -> Note for further optimization.
        // -> This copy loop may not be needed. I might be able to purely send in &this->matrixLInverse[0] into broadcast
        buffer1.resize(matrixWindow*matrixWindow,0.);
        int startOffset = dimXstartA*this->localSize;
        int index1 = 0;
        int index2 = startOffset + dimYstartA;
        for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<matrixWindow; j++)
          {
            buffer1[index1++] = (matrixTrack ? this->matrixB[index2++] : this->matrixA[index2++]);
//            if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 0) && (this->gridCoords[2] == 0))
//            {
//              std::cout << "what was just set - " << buffer1[index1-1] << std::endl;
//            }
          }
          index2 += (this->localSize-matrixWindow);				// see if that works
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
          index2 += dimYstartA;						// Is this really correct?
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
        buffer1.resize(matrixWindow*matrixWindow);
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
//    if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 1) && (this->gridCoords[2] == 1)) { for (int i=0; i<buffer1.size(); i++) { std::cout << "Inv2? - " << buffer1[i] << " of size - " << buffer1.size() << std::endl; } std::cout << "\n"; }
  }

  if (this->colCommRank == this->gridCoords[2])    // May want to recheck this later if bugs occur.
  {
    switch (key)
    {
      case 0:
      {
/*
        // Triangular
        buffer2.resize(matrixWindow*matrixWindow);
        int startOffset = dimXstartB*this->localSize;	// Problem! startOffset needs to be based off of this->localRank, not anything else
        int index1 = 0;
        int index2 = startOffset + dimYstartB;
        for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<matrixWindow; j++)
          {
            buffer2[index1++] = (matrixTrack ? this->matrixB[index2+j] : this->matrixA[index2+j]);
//            if ((this->gridCoords[0] == 1) && (this->gridCoords[1] == 1) && (this->gridCoords[2] == 1)) {std::cout << "GAY - " << index2+j << " " << buffer2[index1-1] << std::endl; }
          }
          index2 += this->localSize;
        }
*/
        buffer2 = this->holdTransposeL;			// Is this copy necessary?
        break;
      }
      case 1:
      {
        // Not Triangular, but transpose
        buffer2.resize(matrixWindow*matrixWindow);
/*
        int startOffset = (dimXstartB*(dimXstartB+1))>>1;
        int index1 = 0;
        for (int i=0; i<matrixWindow; i++)
        {
          int index2 = startOffset + dimYstartB + i;
          int temp = //dimYstartB+//1+matrixWindow;			// I added the +1 back in
          for (int j=0; j<matrixWindow; j++)
          {
            buffer2[index1++] = this->matrixL[index2];                  // Transpose has poor spatial locality
            index2 += temp+j+dimYstartB;
          }
        }
*/
        // Simple transpose into buffer2 using this->holdTransposeL, but we can make this more efficient later via 2D tiling to reduce cache misses
       for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<matrixWindow; j++)
          {
            buffer2[j*matrixWindow+i] = this->holdTransposeL[i*matrixWindow+j];
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
        buffer2.resize((matrixWindow*(matrixWindow+1))>>1);	// this is a special trick
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
//    if ((this->gridCoords[0] == 1) && (this->gridCoords[1] == 1) && (this->gridCoords[2] == 1)) { for (int i=0; i<buffer2.size(); i++) { std::cout << "GAY2 - " << buffer2[i] << " of size - " << buffer2.size() << std::endl; } std::cout << "\n"; }
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
      int counter = 1;
      for (int i = 0; i<matrixWindow; i++)
      {
        for (int j = i; j <matrixWindow; j++)
        {
          int temp = i*matrixWindow+j;				// JUST CHANGED THIS!
          updatedVector[temp] = buffer2[index]; 
          index += counter++;
        }
        index = (((i+2)*(i+3))>>1)-1;
        counter = i+2;
      }

/*
      if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 0) && (this->gridCoords[2] == 0))
      {
        for (int i=0; i<updatedVector.size(); i++)
        {
          std::cout << "check1 - " << updatedVector[i] << std::endl;
        }
        for (int i=0; i<buffer1.size(); i++)
        {
          std::cout << "check2 - " << buffer1[i] << std::endl;
        }
      } 
*/
      cblas_dtrmm(CblasRowMajor,CblasRight,CblasLower,CblasNoTrans,CblasNonUnit,matrixWindow,matrixWindow,1.,&updatedVector[0], matrixWindow, &buffer1[0], matrixWindow);

/*
      if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 0) && (this->gridCoords[2] == 0))
      {
        for (int i=0; i<buffer1.size(); i++)
        {
          std::cout << "check4 - " << buffer1[i] << std::endl;
        }
      } 
*/
      //if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 0) && (this->gridCoords[2] == 1)) { for (int i=0; i<buffer2.size(); i++) {std::cout << "check - " << tryMe[i] << " and size - " << buffer2.size() << std::endl; } std::cout << std::endl; }

      MPI_Allreduce(&buffer1[0],&buffer4[0],buffer1.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);

/*
      if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 0) && (this->gridCoords[2] == 0))
      {
        for (int i=0; i<matrixWindow; i++)
        {
          for (int j=0; j<matrixWindow; j++)
          {
            std::cout << "check - " << buffer4[i*matrixWindow+j] << std::endl;
          }
        }
      }
*/
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
    case 0:
    {
      // Square
      int startOffset = (dimXstartC*(dimXstartC+1))>>1;
      int index1 = 0;
      int index2 = startOffset + dimYstartC;
      for (int i=0; i<matrixWindow; i++)
      {
        for (int j=0; j<matrixWindow; j++)
        {
/*          
          if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 0) && (this->gridCoords[2] == 0))
          {
            std::cout << "buffer4[index1] - " << buffer4[index1] << " goes into matrixL[] - " << index2 << std::endl;
          }
*/
          this->matrixL[index2++] = buffer4[index1++];                    // Transpose has poor spatial locality. Could fix later if need be
        }
        index2 += (dimYstartC+i+1);
      }
      break;
    }
    case 1:						// Can this just be done with a simple copy statement??
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
    case 2:						// Can this just be done with a simple copy statement??
    {
      int index1 = 0;
      int index2 = 0;
      for (int i=0; i<matrixWindow; i++)
      {
        for (int j=0; j<matrixWindow; j++)
        {
          this->holdMatrix[index2++] = buffer4[index1++];
        }
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
//          if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 1) && (this->gridCoords[2] == 0)) {std::cout << "check index2 - " << index2-1 << " " << this->matrixLInverse[index2-1] << std::endl; }
          
        }
        index2 += (dimYstartC+i+1);		// THIS COULD BE WRONG!
      }
      break;
    }
  }
}


/*
  Transpose Communication Helper Function
*/
template<typename T>
void solver<T>::fillTranspose(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int dir)
{
  switch (dir)
  {
    case 0:
    {
      // copy L11-inverse data into holdTransposeInverse to be send/received
      this->holdTransposeL.clear();
      this->holdTransposeL.resize((matrixWindow*(matrixWindow+1))>>1, 0.);
      int startOffset = ((dimXstart*(dimXstart+1))>>1);
      int index1 = 0;
      int index2 = startOffset + dimXstart;
      for (int i=0; i<matrixWindow; i++)
      {
        for (int j=0; j<=i; j++)
        {
          this->holdTransposeL[index1++] = this->matrixLInverse[index2++];
        }
        index2 += dimYstart;
      }

      if ((this->gridCoords[0] != this->gridCoords[1]))
      {
        // perform MPI_SendRecv_replace
        int destRank = -1;
        int rankArray[3] = {this->gridCoords[1], this->gridCoords[0], this->gridCoords[2]};
        MPI_Cart_rank(this->grid3D, &rankArray[0], &destRank);
        MPI_Status stat;
        MPI_Sendrecv_replace(&this->holdTransposeL[0], this->holdTransposeL.size(), MPI_DOUBLE,
          destRank, 0, destRank, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        // anything else?
      }
      break;
    }
    case 1:
    {
      // copy L11-inverse data into holdTransposeInverse to be send/received
      this->holdTransposeL.clear();
      this->holdTransposeL.resize(matrixWindow*matrixWindow, 0.);
      int startOffset = ((dimXstart*(dimXstart+1))>>1);
      int index1 = 0;
      int index2 = startOffset + dimXstart;
      for (int i=0; i<matrixWindow; i++)
      {
        for (int j=0; j<matrixWindow; j++)
        {
          this->holdTransposeL[index1++] = this->matrixL[index2++];
        }
        index2 += dimYstart;
      }

      if ((this->gridCoords[0] != this->gridCoords[1]))
      {
        // perform MPI_SendRecv_replace
        MPI_Status stat;
        int destRank = -1;
        int rankArray[3] = {this->gridCoords[1], this->gridCoords[0], this->gridCoords[2]};
        MPI_Cart_rank(this->grid3D, &rankArray[0], &destRank);
        MPI_Sendrecv_replace(&this->holdTransposeL[0], this->holdTransposeL.size(), MPI_DOUBLE,
          destRank, 0, destRank, 0, MPI_COMM_WORLD, &stat);

        // anything else?
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
        sendBuffer[index1++] = this->matrixA[index2+j];
        //if (this->worldRank == 0) {std::cout << "BC index - " << index2+j << std::endl; }
      }
      index2 += this->localSize;
    }
  }
  else /* matrixTrack == 1 */
  {
    for (int i=0; i<matrixWindow; i++)
    {
      for (int j=0; j<=i; j++)
      {
        sendBuffer[index1++] = this->matrixB[index2+j];
        //if (this->worldRank == 0) {std::cout << "BC index - " << index2+j << std::endl; }
      }
      index2 += this->localSize;
    }
  }

  std::vector<T> recvBuffer(sendBuffer.size()*this->processorGridDimSize*this->processorGridDimSize,0);
  MPI_Allgather(&sendBuffer[0], sendBuffer.size(),MPI_DOUBLE,&recvBuffer[0],sendBuffer.size(),MPI_DOUBLE,this->layerComm);


  /*
	After the all-gather, I need to format recvBuffer so that it works with OpenBLAS routines!
  
        Word of caution: How do I know the ordering of the blocks? It should be by processor rank within the communicator used, but that processor
	  rank has no correlation to the 2D gridCoords that were used to distribute the matrix.
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
  LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',matrixSize,&sendBuffer[0],matrixSize);

  //recvBuffer.resize(sendBuffer.size(),0.);			// I am just re-using vectors. They change form as to what data they hold all the time
 
  recvBuffer = sendBuffer;

  // I am assuming that the diagonals are ok (maybe they arent all 1s, but that should be ok right?) 
  LAPACKE_dtrtri(LAPACK_ROW_MAJOR,'L','N',matrixSize,&recvBuffer[0],matrixSize);

  int pIndex = this->gridCoords[0]*this->processorGridDimSize+this->gridCoords[1];
  int startOffset = (dimXstart*(dimXstart+1))>>1;
  int rowCounter = 0;
  for (int i=0; i<matrixWindow; i++)
  {
    int temp = startOffset + (i+1)*dimYstart + rowCounter;
    for (int j=0; j<=i; j++)	// for matrixWindow==2, j should always go to 
    {
      // I need to use dimXstart and dimXend and dimYstart and dimYend ...
      int index = i*this->processorGridDimSize*matrixSize+(pIndex/this->processorGridDimSize)*matrixSize + j*this->processorGridDimSize+(pIndex%this->processorGridDimSize);
      this->matrixL[temp+j] = sendBuffer[index];
      this->matrixLInverse[temp+j] = recvBuffer[index];
    }
    rowCounter += (i+1);
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
void solver<T>::lapackTest(std::vector<T> &data, std::vector<T> &dataInverse, int n)
{
  //std::vector<T> data(n*n); Assume that space has been allocated for data vector on the caller side.
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      if (i > j)
      {
        srand(i*n+j);
      }
      else
      {
        srand(j*n+i);
      }
      data[i*n+j] = (rand()%100)*1./100.;
      //std::cout << "hoogie - " << i*n+j << " " << data[i*n+j] << std::endl;
      if (i==j)
      {
        data[i*n+j] += 10.;
      }
    }
  }
  
  std::cout << "*************************************************************************************************************\n";
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      std::cout << data[i*n+j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "*************************************************************************************************************\n";

  LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',n,&data[0],n);
  dataInverse = data;				// expensive copy
  LAPACKE_dtrtri(LAPACK_ROW_MAJOR,'L','N',n,&dataInverse[0],n);
/*
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    {
      std::cout << data[i*n+j] << " ";
    }
    std::cout << "\n";
  }
*/
  return;
}

template<typename T>
void solver<T>::solveScalapack()
{
/*

  // Scalapack Cholesky.
  vector<int> desc(9);
  int info,icontxt;
  char order = 'R';
  int npRow = 2;
  int npCol = 2;		// these two can change obviously
  int myCol = this->worldSize%npRow;
  int myRow = this->worldSize/npRow;
  BLACS_GET(0,0,&icontxt);
  BLACS_GRIDINIT(&icontxt,order,npRow,npCol);
  BLACS_GRIDINFO(icontxt,&npRow,&npCol,&myRow,&myCol);	// the last 4 arguments are apparently output arguments

  // Set up the descriptor vector
  desc[0] = 1;
  desc[1] = iscontxt;
  desc[2] = this->matrixDimSize;
  desc[3] = this->matrixDimSize;
  desc[4] = ;
  desc[5] = ;
  desc[6] = ;
  desc[7] = ;
  desc[8] = ;

  // Now I guess I would distribute the input matrix over the P processors.

  PDPOTRF('L',this->matrixDimSize,....,1,1,&desc[0],&info);

*/
}

template<typename T>
void solver<T>::compareSolutions()
{
  /*
	We want to perform a reduction on the data on one of the P^{1/3} layers, then call lapackTest with a single
	processor. Then we can, in the right order, compare these solutions for correctness.
  */

  if ((this->gridCoords[0] == 0) && (this->gridCoords[1] == 1))
  {
    std::cout << "\n";
    for (int i=0; i<this->matrixL.size(); i++)
    {
      std::cout << this->matrixL[i] << std::endl;
    }
    std::cout << "\n\n";
  }



  if (this->gridCoords[2] == 0)		// 1st layer
  {
    //std::vector<T> sendData(this->matrixDimSize * this->matrixDimSize);
    std::vector<T> recvData(this->processorGridDimSize*this->processorGridDimSize*this->matrixL.size());	// only the bottom half, remember?
    MPI_Gather(&this->matrixL[0],this->matrixL.size(), MPI_DOUBLE, &recvData[0],this->matrixL.size(), MPI_DOUBLE,
	0, this->layerComm);		// use 0 as root rank, as it needs to be the same for all calls

    if (this->layerCommRank == 0)
    {
      /*
	Now on this specific rank, we currently have all the algorithm data that we need to compare, but now we must call the sequential
        lapackTest method in order to get what we want to compare the Gathered data against.
      */
      std::vector<T> lapackData(this->matrixDimSize*this->matrixDimSize);
      std::vector<T> lapackDataInverse(this->matrixDimSize*this->matrixDimSize);
      lapackTest(lapackData, lapackDataInverse, this->matrixDimSize);		// pass this vector in by reference and have it get filled up

      // Now we can start comparing this->matrixL with data
      // Lets just first start out by printing everything separately too the screen to make sure its correct up till here
      
      int index = 0;
      std::vector<int> pCounters(this->processorGridDimSize*this->processorGridDimSize,0);		// start each off at zero
      std::cout << "\n";
      for (int i=0; i<this->matrixDimSize; i++)
      {
        for (int j=0; j<=i; j++)
	{
          int PE = (j%this->processorGridDimSize) + (i%this->processorGridDimSize)*this->processorGridDimSize;
	  std::cout << lapackData[index] << " " << recvData[PE*this->matrixL.size() + pCounters[PE]] << " " << index << std::endl;
	  std::cout << lapackData[index++] - recvData[PE*this->matrixL.size() + pCounters[PE]++] << std::endl;
        }
        if (i%2==0)
        {
          pCounters[1]++;		// this is a serious edge case due to the way I handled the actual code
        }
        index = this->matrixDimSize*(i+1);			// try this. We skip the non lower triangular elements
        std::cout << std::endl;
      }
    }
  }
}

template<typename T>
void solver<T>::printInputA()
{
  if (this->gridCoords[2] == 0)		// 1st layer
  {
    //std::vector<T> sendData(this->matrixDimSize * this->matrixDimSize);
    std::vector<T> recvData(this->processorGridDimSize*this->processorGridDimSize*this->matrixA.size());	// only the bottom half, remember?
    MPI_Gather(&this->matrixA[0],this->matrixA.size(), MPI_DOUBLE, &recvData[0],this->matrixA.size(), MPI_DOUBLE,
	0, this->layerComm);		// use 0 as root rank, as it needs to be the same for all calls

    if (this->layerCommRank == 0)
    {
      /*
	Now on this specific rank, we currently have all the algorithm data that we need to compare, but now we must call the sequential
        lapackTest method in order to get what we want to compare the Gathered data against.
      */
      
      for (int i=0; i<recvData.size(); i++)
      {
        std::cout << recvData[i] << " ";
        if (i == 31)
        { std::cout << "\n"; }
      }

      std::cout << "\n\n";

      std::vector<int> pCounters(this->processorGridDimSize*this->processorGridDimSize,0);		// start each off at zero
      for (int i=0; i<this->matrixDimSize; i++)
      {
        for (int j=0; j < this->matrixDimSize; j++)
	{
          int PE = (j%this->processorGridDimSize) + (i%this->processorGridDimSize)*this->processorGridDimSize;
          std::cout << recvData[PE*this->matrixA.size() + pCounters[PE]++] << " ";
        }
        if (i%2==0)
        {
          pCounters[1]++;		// this is a serious edge case due to the way I handled the actual code
        }
        std::cout << std::endl;
      }
    }
  }
}
