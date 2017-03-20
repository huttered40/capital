#ifndef PROCESSORGRID_H_
#define PROCESSORGRID_H_

class ProcessorGrid
{
  ProcessorGrid();
  ~ProcessorGrid();

  virtual MatMatMultiplication() = 0;
  virtual MatVecMultiplication() = 0;
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
