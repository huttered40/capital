#ifndef PROCESSORGRID_H_
#define PROCESSORGRID_H_

/*
  System Includes
*/
#include <vector>


class ProcessorGrid
{
public:
  ProcessorGrid();
  ~ProcessorGrid();

  struct SubCommunicatorInfo
  {
    int commSize;
    int commRank;
    // What else?
  }

  /*
    Could use an argument into MatMatMultiplication that could idenify what kind of matrix (dense, square, triangular dense, sparse, data distribution, etc) that
    could, via a switch statement call a private method that handles that specific kind of MatMatMultiplication with the correct arguments
  */
  virtual void MatMatMultiplication() = 0;
  virtual void MatVecMultiplication() = 0;

  virtual void DistributeDataCyclic() = 0;
  virtual void DistributeDataBlockCyclic() = 0;
  virtual void DistributeDataBlocked() = 0;
protected:
  std::vector<SubCommunicatorInfo> SubCommunicatorTable;		// will return the index into the table, giving the information about the subcommunicator needed
};

class ProcessorGrid2D : public ProcessorGrid
{
  ProcessorGrid2D();
  ~ProcessorGrid2D();

  virtual void MatMatMultiplication();
  virtual void MatVecMultiplication();
  
  virtual void DistributeDataCyclic();
  virtual void DistributeDataBlockCyclic();
  virtual void DistributeDataBlocked();
};

class ProcessorGrid3D : public ProcessorGrid
{
  ProcessorGrid3D();
  ~ProcessorGrid3D();

  virtual void MatMatMultiplication3D();
  virtual void MatVecMultiplication3D();

  virtual void DistributeDataCyclic();
  virtual void DistributeDataBlockCyclic();
  virtual void DistributeDataBlocked();
};

#endif /*PROCESSORGRID_H_*/
