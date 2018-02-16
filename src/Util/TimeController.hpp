/* Author: Edward Hutter */


template<
  typename T,
  typename U,
  template<typename,typename,template<typename,typename,int> class> class Structure,
  template<typename, typename,int> class Distributer,
  template<typename,typename> class blasEngine
>
void TimeController<T,U,Structure,Distributer,blasEngine>::initTimers()
{
}

template<
  typename T,
  typename U,
  template<typename,typename,template<typename,typename,int> class> class Structure,
  template<typename, typename,int> class Distributer,
  template<typename,typename> class blasEngine
>
void TimeController<T,U,Structure,Distributer,blasEngine>::displayResults()
{
  // Evaluate all possible classes using the template parameters.
  // First I need to finalize all of the timers
/*
  CFR3D<T,U,blasEngine>::timer.finalize(MPI_COMM_WORLD);
  TRSM3D<T,U,blasEngine>::timer.finalize(MPI_COMM_WORLD);
  MM3D<T,U,blasEngine>::timer.finalize(MPI_COMM_WORLD);
  CholeskyQR2<T,U,blasEngine>::timer.finalize(MPI_COMM_WORLD);
  cblasEngine<T,U>::timer.finalize(MPI_COMM_WORLD);
  cblasHelper::timer.finalize(MPI_COMM_WORLD);
  Matrix<T,U,Structure,Distributer>::timer.finalize(MPI_COMM_WORLD);
  util<T,U>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureSquare, MatrixStructureSquare>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureSquare, MatrixStructureRectangle>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureRectangle, MatrixStructureSquare>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureRectangle, MatrixStructureRectangle>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureRectangle, MatrixStructureUpperTriangular>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureRectangle, MatrixStructureLowerTriangular>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureRectangle>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureRectangle>::timer.finalize(MPI_COMM_WORLD);
  Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>::timer.finalize(MPI_COMM_WORLD);
  this->ConvertTimerInfo(CFR3D<T,U,blasEngine>::timer);
  this->ConvertTimerInfo(TRSM3D<T,U,blasEngine>::timer);
  this->ConvertTimerInfo(MM3D<T,U,blasEngine>::timer);
  this->ConvertTimerInfo(CholeskyQR2<T,U,blasEngine>::timer);
  this->ConvertTimerInfo(cblasEngine<T,U>::timer);
  this->ConvertTimerInfo(cblasHelper::timer);
  this->ConvertTimerInfo(Matrix<T,U,Structure,Distributer>::timer);
  this->ConvertTimerInfo(util<T,U>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureSquare, MatrixStructureSquare>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureSquare, MatrixStructureRectangle>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureRectangle, MatrixStructureSquare>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureRectangle, MatrixStructureRectangle>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureRectangle, MatrixStructureUpperTriangular>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureRectangle, MatrixStructureLowerTriangular>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureRectangle>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureRectangle>::timer);
  this->ConvertTimerInfo(Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>::timer);
*/
}


template<
  typename T,
  typename U,
  template<typename,typename,template<typename,typename,int> class> class Structure,
  template<typename, typename,int> class Distributer,
  template<typename,typename> class blasEngine
>
void TimeController<T,U,Structure,Distributer,blasEngine>::ConvertTimerInfo(pTimer& timer)
{
  for (size_t i=0; i<timer.functionTimes.size(); i++)
  {
    if (this->saveIndices.find(timer.functionNames[i]) == this->saveIndices.end())
    {
      std::tuple<std::string,int,double> temp = std::make_tuple(timer.functionNames[i], timer.functionNames.size(), timer.functionTimes[i]);
      this->timerInfo.push_back(temp);
    }
    else
    {
      size_t index = this->saveIndices[timer.functionNames[i]];
      std::get<1>(this->timerInfo[index]) += timer.functionNames.size();
      std::get<2>(this->timerInfo[index]) += timer.functionTimes[i];
    }
  }
}
