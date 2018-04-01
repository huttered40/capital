/* Author: Edgar Solomonik */

#ifndef CTF_TIMER_H_
#define CTF_TIMER_H_

// System includes
#include <string>
#include <cstring>

// Local includes
#include "../Util/shared.h"

namespace CTF {

  using namespace std;
/**
 * Timing and cost measurement
 */
  #define MAX_NAME_LENGTH 53
      
  /**
   * Tracks the timer of a specific symbol
   */
  class Function_timer{
    public:
      std::string name;
      double start_time;
      double start_excl_time;
      double acc_time;
      double acc_excl_time;
      int calls;

      double total_time;
      double total_excl_time;
      int total_calls;

    public: 
      Function_timer(const std::string& name_,
                     double start_time_,
                     double start_excl_time_);
      //Function_timer(Function_timer const & other);
      //~Function_timer();
      void compute_totals(MPI_Comm comm);
      bool operator<(const Function_timer& w) const ;
      void print(FILE* output, FILE* fptr,
                 MPI_Comm comm, 
                 int rank,
                 int np);
  };


  /**
   * Tracks the local process walltime measurement
   */
  class Timer{
    public:
      std::string timer_name;
      int index;
      int exited;
      int original;
      bool printBool;
    
    public:
      Timer(const std::string& name);
      ~Timer();
      int stop(FILE* fptr=nullptr, int numIter=-1);
      void start();
      int exit(FILE* fptr=nullptr, int numIter=-1);
  };

  /**
   * Defines the epoch during which to measure timers
   */
  class Timer_epoch{
    private:
      Timer* tmr_inner;
      Timer* tmr_outer;
      double save_excl_time;
      double save_complete_time; 
      std::vector<Function_timer> saved_function_timers;
    public:
      std::string name;
      //create epoch called name
      Timer_epoch(const std::string& name_);

      ~Timer_epoch(){
        saved_function_timers.clear();
      }
      
      //clears timers and begins epoch
      void begin();

      //prints timers and clears them
      void end();
  };


  /**
   * \brief a term is an abstract object representing some expression of tensors
   */

  /**
   * \brief measures flops done in a code region
   */
  class Flop_counter{
    public:
      int64_t  start_count;

    public:
      /**
       * \brief constructor, starts counter
       */
      Flop_counter();
      ~Flop_counter();

      /**
       * \brief restarts counter
       */
      void zero();

      /**
       * \brief get total flop count over all counters in comm
       */
      int64_t count(MPI_Comm comm = MPI_COMM_SELF);

  };

  void set_main_args(int argc, const char * const * argv);

/**
 * @}
 */

}

#define TAU_FSTART(ARG)                                           \
  do { CTF::Timer t(#ARG); t.start(); } while (0);

#define TAU_FSTOP(ARG)                                            \
  do { CTF::Timer t(#ARG); t.stop(); } while (0);

#define TAU_FSTOP_FILE(ARG1,ARG2,ARG3,ARG4)                       \
  do { CTF::Timer t(#ARG1); ARG4 = t.stop(ARG2,ARG3); } while (0);

#define TAU_PROFILE_TIMER(ARG1, ARG2, ARG3, ARG4)                 

#define TAU_PROFILE_INIT(argc, argv)                              \
  CTF::set_main_args(argc, argv);

#define TAU_PROFILE_SET_NODE(ARG)

#define TAU_PROFILE_START(ARG)                                    \
  CTF::Timer __CTF::Timer##ARG(#ARG);

#define TAU_PROFILE_STOP(ARG)                                     \
 __CTF::Timer##ARG.stop();

#define TAU_PROFILE_SET_CONTEXT(ARG)                              \
  if (ARG==0) CTF::set_context(MPI_COMM_WORLD);                    \
  else CTF::set_context((MPI_Comm)ARG);

#endif /*CTF_TIMER_H_*/
