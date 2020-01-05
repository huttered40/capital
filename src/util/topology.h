/* Author: Edward Hutter */

#ifndef TOPOLOGY_H_
#define TOPOLOGY_H_

/*
  Note: topo does not quite serve as a policy. Algorithms have a specific topo that is needed,
        as usually these algorithm class templates are not written in such a way that a summa3d/2d is performed
	if a square3d/2d instance is passed as an argument. That would be interesting, but that is not needed now.

  Note: square/Rect refers to the shape of the face of the processor grid
*/

namespace topo{

class rect{
public:
  rect(MPI_Comm comm, size_t c, size_t num_chunks=0){

    this->num_chunks = num_chunks;
    MPI_Comm column;
    int rank, size, columnRank, cubeRank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    size_t SubCubeSize = c*c*c;
    size_t SubCubeSliceSize = c*c;
    MPI_Comm_split(comm, rank/SubCubeSize, rank, &this->cube);
    MPI_Comm_rank(this->cube, &cubeRank);
    size_t temp1 = (cubeRank%c) + c*(cubeRank/SubCubeSliceSize);
    size_t temp2 = rank % SubCubeSliceSize;
    MPI_Comm_split(this->cube, cubeRank/c, cubeRank, &this->depth);
    MPI_Comm_split(this->cube, temp1, cubeRank, &this->row);
    // Note: columnComm in the tunable grid is of size d, not size c like in the 3D grid
    MPI_Comm_split(comm, temp2, rank, &column);
    MPI_Comm_split(comm, rank%c, rank, &this->slice);
    MPI_Comm_rank(column, &columnRank);
    MPI_Comm_split(column, columnRank/c, columnRank, &this->column_contig);
    MPI_Comm_split(column, columnRank%c, columnRank, &this->column_alt); 
    if (comm != MPI_COMM_WORLD){
      MPI_Comm_dup(comm,&this->world);
    }
    else{
      this->world=comm;
    }
    this->c = c;
    this->d = size / (this->c*this->c);
    this->z = rank%this->c;
    this->y = rank/SubCubeSliceSize;
    this->x = (rank%SubCubeSliceSize)/c;
    MPI_Comm_free(&column);
  }
  ~rect(){
    MPI_Comm_free(&this->row);
    MPI_Comm_free(&this->column_contig);
    MPI_Comm_free(&this->column_alt);
    MPI_Comm_free(&this->depth);
    MPI_Comm_free(&this->slice);
    MPI_Comm_free(&this->cube);
  }

  MPI_Comm world,row,column_contig,column_alt,depth,slice,cube;
  size_t c,d,x,y,z,num_chunks;
};

class square{
public:
  square(MPI_Comm comm, size_t c, size_t num_chunks=0){

    this->num_chunks = num_chunks;
    int rank,size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    this->c = c;
    this->d = std::nearbyint(std::ceil(pow(size/c,1./2.)));
    size_t TopFaceSize = this->d*this->d;
    this->z = rank%c;
    this->y = rank/TopFaceSize;
    this->x = (rank%TopFaceSize)/this->c;

    // First, split the 3D Cube processor grid communicator into groups based on what 2D slice they are located on.
    // Then, subdivide further into row groups and column groups
    MPI_Comm_split(comm, rank/this->c, rank, &this->depth);
    MPI_Comm_split(comm, this->z, rank, &this->slice);
    MPI_Comm_split(this->slice, this->y, this->x, &this->row);
    MPI_Comm_split(this->slice, this->x, this->y, &this->column);

    if (comm != MPI_COMM_WORLD){
      MPI_Comm_dup(comm,&this->world);
    }
    else{
      this->world=comm;
    }

  }
  ~square(){
    MPI_Comm_free(&this->row);
    MPI_Comm_free(&this->column);
    MPI_Comm_free(&this->slice);
    MPI_Comm_free(&this->depth);
  }

  MPI_Comm world,row,column,slice,depth;
  size_t c,d,x,y,z,num_chunks;
};
}

#endif /*TOPOLOGY_H_*/
