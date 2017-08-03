/* Author: Edward Hutter */

#ifndef MPIWORLD_H_
#define MPIWORLD_H_

/* system includes */
#include <mpi>

/* local includes */

/*
  Idea: each algorithm can interface with this. This will allow us to generalize higher level algorithms that
        can use MPI, c++11 threads, hybrid MPI_OpenMP, and OneSided.

  Use case: In an matrix multiplication code, instead of it owning lots of random communicators,
            it can use one instance of this class to hold all of that information and just reference it for the
	    communication details and MPI objects that it needs.

	    Another thing I can envision is a class for say moving data among shapes efficients between processors.
	      In this case, it can use an MpiWorld and an algebraic structure and go to work.

  A specific matrix multiplication class will be templated to take in a data type, an algebric structure,
    and a specific communication class object such as MpiWorld. Then, it should be able to do things in such a
    way that it doesn't know that MPI is the workhorse underneath. It could be threaded for all it cares.


   Still, this may be unnecessary abstraction. I can always delete this. The point was to
         create different algorithms with different communication protocols.
	 I wanted to implement algorithms using MPI or c++11 threads, or Hybrid communication or OneSided communication
*/
class MpiWorld
{
public:
  MpiWorld();
  MpiWorld(const MpiWorld& rhs);
  ~MpiWorld();



private:
  MPI_Comm _communicator;

};

#endif /* MPIWORLD_H_ */
