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
  rect(MPI_Comm comm, size_t c, size_t layout = 0, size_t num_chunks=0){

    this->layout = layout;
    this->num_chunks = num_chunks;
    MPI_Comm column;
    int columnRank, cubeRank;
    MPI_Comm_rank(comm, &this->rank);
    MPI_Comm_size(comm, &this->size);
    size_t SubCubeSize = c*c*c;
    size_t SubCubeSliceSize = c*c;
    MPI_Comm_split(comm, this->rank/SubCubeSize, this->rank, &this->cube);
    MPI_Comm_rank(this->cube, &cubeRank);
    size_t temp1 = (cubeRank%c) + c*(cubeRank/SubCubeSliceSize);
    size_t temp2 = this->rank % SubCubeSliceSize;
    MPI_Comm_split(this->cube, cubeRank/c, cubeRank, &this->depth);
    MPI_Comm_split(this->cube, temp1, cubeRank, &this->row);
    // Note: columnComm in the tunable grid is of size d, not size c like in the 3D grid
    MPI_Comm_split(comm, temp2, this->rank, &column);
    MPI_Comm_split(comm, this->rank%c, this->rank, &this->slice);
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
    this->d = this->size/(this->c*this->c);
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
  int rank,size;
  size_t c,d,x,y,z,layout,num_chunks;
};

class square{
public:
  square(MPI_Comm comm, size_t c, size_t layout=0, size_t num_chunks=0){

    this->layout = layout;
    this->num_chunks = num_chunks;
    MPI_Comm_rank(comm, &this->rank);
    MPI_Comm_size(comm, &this->size);

    this->c = c;
    this->d = std::nearbyint(std::ceil(pow(this->size/c,1./2.)));
    size_t TopFaceSize = this->d*this->c;
    size_t FrontFaceSize = this->d*this->d;
    if (layout==0){
      this->z = this->rank%c;
      this->y = this->rank/TopFaceSize;
      this->x = (this->rank%TopFaceSize)/this->c;
      MPI_Comm_split(comm, this->rank/this->c, this->rank, &this->depth);
      MPI_Comm_split(comm, this->z, this->rank, &this->slice);
      //MPI_Comm_split(comm, this->rank%TopFaceSize, this->rank, &this->row);
      //MPI_Comm_split(comm, this->rank%this->c + (this->rank/TopFaceSize)*this->c, this->rank, &this->column);

      // debug
      //MPI_Comm new_slice1; MPI_Comm_split(comm,this->y,this->rank,&new_slice1);
      //MPI_Comm new_slice2; MPI_Comm_split(comm,this->x,this->rank,&new_slice2);

      MPI_Comm_split(this->slice, this->y, this->x, &this->row);
      MPI_Comm_split(this->slice, this->x, this->y, &this->column);

    } else if (layout == 1){
      this->y = this->rank%d;
      this->x = (this->rank%FrontFaceSize)/this->d;
      this->z = this->rank/FrontFaceSize;
      MPI_Comm_split(comm, this->y+this->x*this->d, this->rank, &this->depth);
      MPI_Comm_split(comm, this->z, this->rank, &this->slice);
      MPI_Comm_split(this->slice, this->y, this->x, &this->row);
      MPI_Comm_split(this->slice, this->x, this->y, &this->column);
    } else if (layout == 2){
      int subcube_size = std::min(this->size,64);
      int subcube_slice_size = std::nearbyint(std::ceil(pow(subcube_size,2./3.)));
      int subcube_dim_size = std::nearbyint(std::ceil(pow(subcube_size,1./3.)));
      int rank_mod = this->rank%subcube_size;
      int rank_div = this->rank/subcube_size;
      int local_x = (rank_mod%subcube_slice_size)/subcube_dim_size;
      int local_y = rank_mod%subcube_dim_size;
      int local_z = rank_mod/subcube_slice_size;
      int global_x = TopFaceSize>=subcube_slice_size ? (rank_div%(TopFaceSize/subcube_slice_size))/(this->c/subcube_dim_size) : 0;
      int global_y = rank_div%(this->c/subcube_dim_size);
      int global_z = TopFaceSize>=subcube_slice_size ? rank_div/(TopFaceSize/subcube_slice_size) : 0;
      this->x = global_x*subcube_dim_size + local_x;
      this->y = global_y*subcube_dim_size + local_y;
      this->z = global_z*subcube_dim_size + local_z;
      MPI_Comm_split(comm, this->y+this->d*this->x, this->rank, &this->depth);
      MPI_Comm_split(comm, this->z, this->rank, &this->slice);
      MPI_Comm_split(this->slice, this->y, this->x, &this->row);
      MPI_Comm_split(this->slice, this->x, this->y, &this->column);
    }

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
  int rank,size;
  size_t c,d,x,y,z,layout,num_chunks;
};
}

#endif /*TOPOLOGY_H_*/
