/* Author: Edward Hutter */

#ifndef TOPOLOGY_H_
#define TOPOLOGY_H_

/*
  Note: topology does not quite serve as a policy. Algorithms have a specific topology that is needed,
        as usually these algorithm class templates are not written in such a way that a summa3d/2d is performed
	if a Square3d/2d instance is passed as an argument. That would be interesting, but that is not needed now.
*/

namespace topology{

class Rect3D{
public:
  Rect3D(MPI_Comm comm, size_t c){
    TAU_FSTART(topology::Rect3D);

    int rank, size, columnRank, cubeRank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    size_t SubCubeSize = c*c*c;
    size_t SubCubeSliceSize = c*c;
    MPI_Comm_split(comm, worldRank/SubCubeSize, worldRank, &this->cube);
    MPI_Comm_rank(miniCubeComm, &cubeRank);
    size_t temp1 = (cubeRank%c) + c*(cubeRank/SubCubeSliceSize);
    size_t temp2 = worldRank % SubCubeSliceSize;
    MPI_Comm_split(this->cube, cubeRank/c, cubeRank, &this->depth);
    MPI_Comm_split(this->cube, temp1, cubeRank, &this->row);
    // Note: columnComm in the tunable grid is of size d, not size c like in the 3D grid
    MPI_Comm_split(comm, temp2, worldRank, &this->column);
    MPI_Comm_split(comm, worldRank%c, worldRank, &this->slice);
    MPI_Comm_rank(this->column, &columnRank);
    MPI_Comm_split(this->column, columnRank/c, columnRank, &this->column_contig);
    MPI_Comm_split(this->column, columnRank%c, columnRank, &this->column_alt); 
    this->world=comm;
    this->c = c;
    this->d = size / (this->c*this->c);
    this->z = rank%this->c;
    this->x = (rank%SubCubeSliceSize)/c;
    MPI_Comm_free(&this->column);
    TAU_FSTOP(topology::Rect3D);
  }
  ~Rect3D(){
    TAU_FSTART(topology::~Rect3D);
    MPI_Comm_free(&std::get<0>(commInfoTunable));
    MPI_Comm_free(&std::get<1>(commInfoTunable));
    MPI_Comm_free(&std::get<2>(commInfoTunable));
    MPI_Comm_free(&std::get<3>(commInfoTunable));
    MPI_Comm_free(&std::get<4>(commInfoTunable));
    MPI_Comm_free(&std::get<5>(commInfoTunable));
    TAU_FSTOP(topology::~Rect3D);
  }

  MPI_Comm world,row,column_contig,column_alt,depth,slice,cube;
  size_t c,d,x,z; // 'y' not included because that dimension is partitioned
};

class Square3D{
public:
  Square3D(MPI_Comm comm){
    TAU_FSTART(topology::Square3D);

    int rank,size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t pGridDimensionSize = std::nearbyint(std::ceil(pow(size,1./3.)));
    size_t helper = pGridDimensionSize;
    helper *= helper;
    this->z = rank%pGridDimensionSize;
    this->y = rank/helper;
    this->x = (rank%helper)/pGridDimensionSize;

    // First, split the 3D Cube processor grid communicator into groups based on what 2D slice they are located on.
    // Then, subdivide further into row groups and column groups
    MPI_Comm_split(comm, rank/pGridDimensionSize, rank, &this->depth);
    MPI_Comm_split(comm, this->z, rank, &this->slice);
    MPI_Comm_split(this->slice, this->y, this->x, &this->row);
    MPI_Comm_split(this->slice, this->x, this->y, &this->column);
    this->world=comm;

    TAU_FSTOP(topology::Square3D);
  }
  ~Square3D(){
    TAU_FSTART(topology::~Square3D);
    MPI_Comm_free(&this->row);
    MPI_Comm_free(&this->column);
    MPI_Comm_free(&this->slice);
    MPI_Comm_free(&this->depth);
    TAU_FSTOP(topology::~Square3D);
  }

  MPI_Comm world,row,column,slice,depth;
  size_t x,y,z;
};

class Square2D{
public:
  Square2D(MPI_Comm comm){
  }
  ~Square2D(){
  }

  MPI_Comm row,column;
  size_t x,y;
};
}

#include "topology.hpp"
#endif /*TOPOLOGY_H_*/
