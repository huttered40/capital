HOST=$(shell hostname)
PORTER=$(shell echo $(HOST) |grep 'porter')
BGQ=$(shell echo $(HOST) |grep 'mira\|cetus')
THETA=$(shell echo $(HOST) |grep 'theta')
STAMPEDE2=$(shell echo $(HOST) |grep 'stampede2')
BLUEWATERS=$(shell echo $(HOST) |grep 'h2o')
BIN=$(HOME)/capital/bin/

ifneq ($(STAMPEDE2),)
  critter_dir=$(HOME)/critter
  MACHINE=STAMPEDE2
  #CCMPI=scorep mpicxx
  CCMPI=mpicxx
  #CCMPI=ampicxx
  INCLUDES=-I$(critter_dir)/include
  DEFS=-DMKL -DCRITTER -DALGORITHMIC_SYMBOLS
  #DEFS=-DMKL
  CFLAGS=-g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512 ${DEFS} ${INCLUDES}
  LIB_PATH=-L$(critter_dir)/lib
  LIBS=-lcritter
endif
