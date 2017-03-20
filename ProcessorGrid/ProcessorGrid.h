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

  virtual MatMatMultiplication() = 0;
  virtual MatVecMultiplication() = 0;

protected:
  std::vector<SubCommunicatorInfo> SubCommunicatorTable;		// will return the index into the table, giving the information about the subcommunicator needed
};

class ProcessorGrid2D : public ProcessorGrid
{
  ProcessorGrid2D();
  ~ProcessorGrid2D();

  virtual MatMatMultiplication();
  virtual MatVecMultiplication();
};

class ProcessorGrid3D : public ProcessorGrid
{
  ProcessorGrid3D();
  ~ProcessorGrid3D();

  virtual MatMatMultiplication3D();
  virtual MatVecMultiplication3D();
};

#endif /*PROCESSORGRID_H_*/
