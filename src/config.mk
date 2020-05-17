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
  CCMPI = mpicxx
  CFLAGS = -g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512 -I$(critter_dir)/src
  LIB_PATH=-L$(critter_dir)/lib
  LIBS=-lcritter
endif
