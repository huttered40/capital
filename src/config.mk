HOST=$(shell hostname)
PORTER=$(shell echo $(HOST) |grep 'porter')
BGQ=$(shell echo $(HOST) |grep 'mira\|cetus')
THETA=$(shell echo $(HOST) |grep 'theta')
STAMPEDE2=$(shell echo $(HOST) |grep 'stampede2')
BLUEWATERS=$(shell echo $(HOST) |grep 'h2o')
BIN=$(HOME)/camfs/src/bin/

# If make is called outside of the batch script generator, a few environment variables need to be set by default (double,int)
ifeq ($(MPITYPE),)
  MPITYPE=MPI_TYPE
endif

ifneq ($(PORTER),)
  MACHINE=PORTER
  CCMPI = ~/hutter2/external/MPICH/installDir/bin/mpicxx
  ifeq ($(MPITYPE),"MPI_TYPE")
    CCMPI = ~/hutter2/external/MPICH/installDir/bin/mpicxx
  endif
  ifeq ($(MPITYPE),"AMPI_TYPE")
    CCMPI = ~/hutter2/external/CHARM/charm/bin/ampicxx
  endif
  CFLAGS = -g -Wall -O0 -std=c++14
  LIB_PATH=-L/home/hutter2/hutter2/critter/lib
  LIBS=~/hutter2/external/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread -lcritter

endif

ifneq ($(BGQ),)
  MACHINE=BGQ
  CCMPI = mpic++11
  CFLAGS = -std=c++11 -g
  INCLUDES=-I/soft/libraries/essl/current/essl/5.1/include/
  LIB_PATH=-L../../../../CritterBlueGene/critter/lib -L/soft/libraries/alcf/current/xl/LAPACK/lib -L/soft/libraries/alcf/current/xl/BLAS/lib -L/soft/libraries/alcf/current/xl/CBLAS/lib -L/soft/libraries/essl/current/essl/5.1/lib64/ -L/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/lib64
  LIBS=-llapack -lcblas -lblas -lesslbg -lgfortran -lxlopt -lxlf90_r -lxlfmath -lxl -lcritter

endif

ifneq ($(THETA),)
  MACHINE=THETA
  CCMPI = CC
  CFLAGS = -g -Wall -fast -std=c++11 -mkl=parallel -xMIC-AVX512
  LIB_PATH=-L../../../../CritterXC40/critter/lib
  LIBS=-lcritter

endif

ifneq ($(STAMPEDE2),)
  critter_dir=/home1/05608/tg849075/critter
  MACHINE=STAMPEDE2
  CCMPI = mpicxx
  CFLAGS = -g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512 -I$(critter_dir)/src
  LIB_PATH=-L$(critter_dir)/lib
  LIBS=-lcritter
endif
