HOST=$(shell hostname)
PORTER=$(shell echo $(HOST) |grep 'porter')
BGQ=$(shell echo $(HOST) |grep 'mira\|cetus')
THETA=$(shell echo $(HOST) |grep 'theta')
STAMPEDE2=$(shell echo $(HOST) |grep 'stampede2')
BLUEWATERS=$(shell echo $(HOST) |grep 'h2o')
BIN=$(HOME)/camfs/bin/

ifneq ($(STAMPEDE2),)
  critter_dir=$(HOME)/critter
  MACHINE=STAMPEDE2
  #CCMPI=scorep mpicxx
  CCMPI=mpicxx
  INCLUDES=-I$(critter_dir)/include
  #DEFS=-DCRITTER -DALGORITHMIC_SYMBOLS# -DCOLLECTIVE_CONCURRENCY_SOLO
  DEFS=-DMKL -DCRITTER -DFUNCTION_SYMBOLS# -DCOLLECTIVE_CONCURRENCY_SOLO
  CFLAGS=-g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512 ${DEFS} ${INCLUDES}
  #CFLAGS=-g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512 ${INCLUDES}
  LIB_PATH=-L$(critter_dir)/lib
  LIBS=-lcritter
endif