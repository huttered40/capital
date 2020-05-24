HOST=$(shell hostname)
PORTER=$(shell echo $(HOST) |grep 'porter')
BGQ=$(shell echo $(HOST) |grep 'mira\|cetus')
THETA=$(shell echo $(HOST) |grep 'theta')
STAMPEDE2=$(shell echo $(HOST) |grep 'stampede2')
BLUEWATERS=$(shell echo $(HOST) |grep 'h2o')
BIN=$(HOME)/camfs/src/bin/

ifneq ($(STAMPEDE2),)
  critter_dir=/home1/05608/tg849075/critter
  MACHINE=STAMPEDE2
  CCMPI=mpicxx
  INCLUDES=-I$(critter_dir)/src
  DEFS=-DCRITTER -DALGORITHMIC_SYMBOLS
  CFLAGS=-g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512 ${DEFS} ${INCLUDES}
  LIB_PATH=-L$(critter_dir)/lib
  LIBS=-lcritter
endif
