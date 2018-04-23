/* Author: Edgar Solomonik */

#include <assert.h>
#include <map>
#include <tuple>

//#include "CTFtimer.h"

namespace CTF{
  #define MAX_TOT_SYMBOLS_LEN 1000000

  int main_argc = 0;
  const char * const * main_argv;
  MPI_Comm comm;
  double excl_time;
  double complete_time;
  int set_contxt = 0;
  int output_file_counter = 0;
  std::map<std::string,std::tuple<int,int,int,int,int,int,int,int,int> > saveFunctionInfo;

  Function_timer::Function_timer(const std::string& name_, 
                                 double start_time_,
                                 double start_excl_time_)
  {
    name = name_;
    start_time = start_time_;
    start_excl_time = start_excl_time_;
    acc_time = 0.0;
    acc_excl_time = 0.0;
    calls = 0;
  }

/*
  Function_timer::Function_timer(Function_timer const & other){
    start_time = other.start_time;
    start_excl_time = other.start_excl_time;
    acc_time = other.acc_time;
    calls = other.calls;
    total_time = other.total_time;
    total_excl_time = other.total_excl_time;
    total_calls = other.total_calls;
    name = (char*)CTF_int::alloc(strlen(other.name)+1);
    strcpy(name, other.name);
    
  }
  
  Function_timer::~Function_timer(){
    cdealloc(name);
  }
*/
  void Function_timer::compute_totals(MPI_Comm comm){ 
    PMPI_Allreduce(&acc_time, &total_time, 1, 
                  MPI_DOUBLE, MPI_SUM, comm);
    PMPI_Allreduce(&acc_excl_time, &total_excl_time, 1, 
                  MPI_DOUBLE, MPI_SUM, comm);
    PMPI_Allreduce(&calls, &total_calls, 1, 
                  MPI_INT, MPI_SUM, comm);
  }

  bool Function_timer::operator<(Function_timer const & w) const {
    return total_time > w.total_time;
  }

  void Function_timer::print(FILE* output, std::ofstream& fptr,
                             MPI_Comm comm, 
                             int rank,
                             int np)
  {
    int i;
    if (rank == 0)
    {
      fprintf(output, "%s", name.c_str());
      std::string space(MAX_NAME_LENGTH-strlen(name.c_str())+1, ' ');
      for (i=0; i<(MAX_NAME_LENGTH-(int)strlen(name.c_str())); i++)
      {
        space[i] = ' ';
      }
      space[i] = '\0';
      fprintf(output, "%s", space.c_str());
      fprintf(output,"%5d   %3d.%03d   %3d.%02d  %3d.%03d   %3d.%02d\n",
              total_calls/np,
              (int)(total_time/np),
              ((int)(1000.*(total_time)/np))%1000,
              (int)(100.*(total_time)/complete_time),
              ((int)(10000.*(total_time)/complete_time))%100,
              (int)(total_excl_time/np),
              ((int)(1000.*(total_excl_time)/np))%1000,
              (int)(100.*(total_excl_time)/complete_time),
              ((int)(10000.*(total_excl_time)/complete_time))%100);
      
      saveFunctionInfo[name.c_str()] = std::make_tuple(total_calls/np, (int)(total_time/np), ((int)(1000.*(total_time)/np))%1000, (int)(100.*(total_time)/complete_time), ((int)(10000.*(total_time)/complete_time))%100, (int)(total_excl_time/np), ((int)(1000.*(total_excl_time)/np))%1000, (int)(100.*(total_excl_time)/complete_time), ((int)(10000.*(total_excl_time)/complete_time))%100);
    } 
  }

  bool comp_name(const Function_timer& w1, const Function_timer& w2) {
    return strcmp(w1.name.c_str(), w2.name.c_str())>0;
  }

  // function_timers is static, defined within the CTF namespace, means that any instance of any class can access it and modify it. For us, that means that any Timer instance
  //   can access and modify it. Only one instance exists, created and initialized
  //   before program starts in the global/static part of memory.
  static std::vector<Function_timer>* function_timers = NULL;

  // Timer is local to each MPI process, so there is no overlap or anything. If threading is added, might need to be careful, but we probably won't do
  //  task-thread-parallelism anyways so it should never really matter.
  Timer::Timer(const std::string& name){
  #ifdef PROFILE
    int i;

    // Special addition so that only output for functions with the tag "Total" are used for output.
    if (name == "Total")
    {
      printBool = true;
    }
    else {printBool = false;}

    // This test should only pass once, on the very first instance of Timer
    if (function_timers == NULL)
    {
      // NOTE: I don't know why this check is necessary, I may just get rid of it.
      if (name[0] == 'M' && name[1] == 'P' && 
          name[2] == 'I' && name[3] == '_'){
        exited = 2;
        original = 0;
        return;
      }
      original = 1;
      index = 0;
      excl_time = 0.0;
      function_timers = new std::vector<Function_timer>();
      function_timers->push_back(Function_timer(name, MPI_Wtime(), 0.0)); 
    } else {
      for (i=0; i<(int)function_timers->size(); i++){
        if ((*function_timers)[i].name == name){
          break;
        }
      }
      index = i;
      original = (index==0);
    }
    // Below: if it did not find a match in the higher-level vector, then create a new timer element for the function
    if (index == (int)function_timers->size()) {
      function_timers->push_back(Function_timer(name, MPI_Wtime(), excl_time)); 
    }
    timer_name = name;
    exited = 0;
  #endif
  }
    
  void Timer::start(){
  #ifdef PROFILE
    // Make sure that we don't time an MPI routine (for some weird reason, I may just change this, doesnt make much sense to me)
    // Index is set from the constructor of the Timer instance
    if (exited != 2){
      exited = 0;
      (*function_timers)[index].start_time = MPI_Wtime();
      (*function_timers)[index].start_excl_time = excl_time;
    }
  #endif
  }

  int Timer::stop(std::ofstream& fptr, int numIter){
  #ifdef PROFILE
    // Note that when we started the timer, as long as exited wasn't equal to 2, we set exited<-0, so this should really pass most of the time
    int numFuncs;
    if (exited == 0){
      int is_fin;
      MPI_Finalized(&is_fin);
      if (!is_fin){
        double delta_time = MPI_Wtime() - (*function_timers)[index].start_time;
        (*function_timers)[index].acc_time += delta_time;
        (*function_timers)[index].acc_excl_time += delta_time - 
              (excl_time- (*function_timers)[index].start_excl_time); 
        excl_time = (*function_timers)[index].start_excl_time + delta_time;
        // Only increment calls when we have a start ^ stop match
        (*function_timers)[index].calls++;
      }
      numFuncs = exitTimer(fptr, numIter);
      exited = 1;
    }
    return numFuncs;
  #endif
  return 0;
  }
  
  void Timer::stopFunction(){
  #ifdef PROFILE
    // Note that when we started the timer, as long as exited wasn't equal to 2, we set exited<-0, so this should really pass most of the time
    int numFuncs;
    if (exited == 0){
      int is_fin;
      MPI_Finalized(&is_fin);
      if (!is_fin){
        double delta_time = MPI_Wtime() - (*function_timers)[index].start_time;
        (*function_timers)[index].acc_time += delta_time;
        (*function_timers)[index].acc_excl_time += delta_time - 
              (excl_time- (*function_timers)[index].start_excl_time); 
        excl_time = (*function_timers)[index].start_excl_time + delta_time;
        // Only increment calls when we have a start ^ stop match
        (*function_timers)[index].calls++;
      }
      exitTimerFunction();
      exited = 1;
    }
  #endif
  }

  Timer::~Timer(){ }

  int print_timers(const string& name, std::ofstream& fptr, int iterNum){
    int rank, np, i, j, len_symbols, nrecv_symbols;
    int numFuncs = 0;

    int is_fin = 0;
    MPI_Finalized(&is_fin);
    if (is_fin) return 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);


    std::string all_symbols(MAX_TOT_SYMBOLS_LEN, ' ');
    std::string recv_symbols(MAX_TOT_SYMBOLS_LEN, ' ');
    FILE* output = NULL;

//  Lets comment out all of the 'model' stuff, I don't think I need it
//    CTF_int::update_all_models(comm);
    if (rank == 0){
//      CTF_int::print_all_models();

      char filename[300];
      char part[300];

      sprintf(filename, "profile.%s.", name.c_str());
      srand(time(NULL));
      sprintf(filename+strlen(filename), "%d.", output_file_counter);
      output_file_counter++;

      int off;
      if (main_argc > 0){
        for (i=0; i<main_argc; i++){
          for (off=strlen(main_argv[i]); off>=1; off--){
            if (main_argv[i][off-1] == '/') break;
          }
          sprintf(filename+strlen(filename), "%s.", main_argv[i]+off);
        }
      } 
      sprintf(filename+strlen(filename), "-p%d.out", np);
      
      
      output = stdout;// fopen(filename, "w");
      printf("%s\n",filename);
      char heading[MAX_NAME_LENGTH+200];
      for (i=0; i<MAX_NAME_LENGTH; i++){
        part[i] = ' ';
      }
      part[i] = '\0';
      sprintf(heading,"%s",part);
      //sprintf(part,"calls   total sec   exclusive sec\n");
      sprintf(part,"       inclusive         exclusive\n");
      strcat(heading,part);
      fprintf(output, "%s", heading);
      for (i=0; i<MAX_NAME_LENGTH; i++){
        part[i] = ' ';
      }
      part[i] = '\0';
      sprintf(heading,"%s",part);
      sprintf(part, "calls        sec       %%"); 
      strcat(heading,part);
      sprintf(part, "       sec       %%\n"); 
      strcat(heading,part);
      fprintf(output, "%s", heading);

    }
    len_symbols = 0;
    for (i=0; i<(int)function_timers->size(); i++){
      sprintf(&all_symbols[0]+len_symbols, "%s", (*function_timers)[i].name.c_str());
      len_symbols += strlen((*function_timers)[i].name.c_str())+1;
    }
    if (np > 1){
      for (int lp=1; lp<log2(np)+1; lp++){
        int gap = 1<<lp;
        if (rank%gap == gap/2){
          PMPI_Send(&len_symbols, 1, MPI_INT, rank-gap/2, 1, comm);
          PMPI_Send(all_symbols.c_str(), len_symbols, MPI_CHAR, rank-gap/2, 2, comm);
        }
        if (rank%gap==0 && rank+gap/2<np){
          MPI_Status stat;
          PMPI_Recv(&nrecv_symbols, 1, MPI_INT, rank+gap/2, 1, comm, &stat);
          PMPI_Recv(&recv_symbols[0], nrecv_symbols, MPI_CHAR, rank+gap/2, 2, comm, &stat);
          for (i=0; i<nrecv_symbols; i+=strlen(recv_symbols.c_str()+i)+1){
            j=0;
            while (j<len_symbols && strcmp(all_symbols.c_str()+j, recv_symbols.c_str()+i) != 0){
              j+=strlen(all_symbols.c_str()+j)+1;
            }
            
            if (j>=len_symbols){
              sprintf(&all_symbols[0]+len_symbols, "%s", recv_symbols.c_str()+i);
              len_symbols += strlen(recv_symbols.c_str()+i)+1;
            }
          }
        }
      }
      PMPI_Bcast(&len_symbols, 1, MPI_INT, 0, comm);
      PMPI_Bcast(&all_symbols[0], len_symbols, MPI_CHAR, 0, comm);
      j=0;
      while (j<len_symbols){
        Timer t(all_symbols.c_str()+j);
        j+=strlen(all_symbols.c_str()+j)+1;
      }
    }
    assert(len_symbols <= MAX_TOT_SYMBOLS_LEN);

    std::sort(function_timers->begin(), function_timers->end(),comp_name);
    for (i=0; i<(int)function_timers->size(); i++){
      (*function_timers)[i].compute_totals(comm);
    }
    std::sort(function_timers->begin(), function_timers->end());
    complete_time = (*function_timers)[0].total_time;
    if (rank == 0){
      for (i=0; i<(int)function_timers->size(); i++){
        (*function_timers)[i].print(output,fptr,comm,rank,np);
	numFuncs++;
      }
      if (iterNum == 0)
      {
        fptr << "Input";
	for (auto& funcName : saveFunctionInfo)
	{
	  fptr << "\t" << funcName.first;
	}
      }
      
      // For each iteration, we have 5 lines to print out.
      fptr << "\n" << iterNum;
      for (auto& funcName : saveFunctionInfo)
      {
        fptr << "\t" << std::get<0>(funcName.second);
      }
      
      fptr << "\n" << iterNum;
      for (auto& funcName : saveFunctionInfo)
      {
        fptr << "\t" << std::get<1>(funcName.second) << "." << std::get<2>(funcName.second);
      }
      
      fptr << "\n" << iterNum;
      for (auto& funcName : saveFunctionInfo)
      {
        fptr << "\t" << std::get<3>(funcName.second) << "." << std::get<4>(funcName.second);
      }
      
      fptr << "\n" << iterNum;
      for (auto& funcName : saveFunctionInfo)
      {
        fptr << "\t" << std::get<5>(funcName.second) << "." << std::get<6>(funcName.second);
      }
      
      fptr << "\n" << iterNum;
      for (auto& funcName : saveFunctionInfo)
      {
        fptr << "\t" << std::get<7>(funcName.second) << "." << std::get<8>(funcName.second);
      }
    }
    saveFunctionInfo.clear();		// clear after each iteration
    return numFuncs;
  }

  int Timer::exitTimer(std::ofstream& fptr, int numIter){
  #ifdef PROFILE
    int numFuncs;
    if (set_contxt && original && !exited) {
      if (comm != MPI_COMM_WORLD){
        return 0;
      }
      if (printBool)
      {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
          printf("\nPROFILE\n");
        }
        numFuncs = print_timers("all", fptr, numIter);  
      }
      function_timers->clear();
      delete function_timers;
      function_timers = NULL;
    }
    return numFuncs;
    #endif
    return 0;
  }
  
  void Timer::exitTimerFunction(){
  #ifdef PROFILE
    int numFuncs;
    if (set_contxt && original && !exited) {
      if (comm != MPI_COMM_WORLD){
        return;
      }
      if (printBool)
      {
        assert(0);			// In this function, we should never get in here
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
          printf("\nPROFILE\n");
        }
        //numFuncs = print_timers("all", fptr, numIter);  
      }
      function_timers->clear();
      delete function_timers;
      function_timers = NULL;
    }
    #endif
  }

  void set_main_args(int argc, const char * const * argv){
    main_argv = argv;
    main_argc = argc;
  }

  void set_context(MPI_Comm ctxt){
    if (!set_contxt)
      comm = ctxt;
    set_contxt = 1;
  }

  Timer_epoch::Timer_epoch(const string& name_){
  #ifdef PROFILE
    name = name_;
  #endif
  }

  void Timer_epoch::begin(){
  #ifdef PROFILE
    tmr_outer = new Timer(name);
    tmr_outer->start();
    saved_function_timers = *function_timers;
    save_excl_time = excl_time;
    excl_time = 0.0;
    function_timers->clear();
    tmr_inner = new Timer(name);
    tmr_inner->start();
  #endif
  }

  void Timer_epoch::end(){
  #ifdef PROFILE
    tmr_inner->stopFunction();
    if (function_timers != NULL){
      function_timers->clear();
      delete function_timers;
    }
    function_timers = new std::vector<Function_timer>();
    *function_timers = saved_function_timers;
    excl_time = save_excl_time;
    tmr_outer->stopFunction();
    //delete tmr_inner;
    delete tmr_outer;
  #endif
  }
}
