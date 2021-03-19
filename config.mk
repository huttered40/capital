BIN=$(HOME)/capital/bin/

## For Stampede2
#critter_dir=$(HOME)/critter
#CCMPI=mpicxx
CCMPI=
#INCLUDES=-I$(critter_dir)/include
INCLUDES=
#DEFS=-DMKL -DCRITTER -DALGORITHMIC_SYMBOLS
DEFS=
#CFLAGS=-g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512 ${DEFS} ${INCLUDES}
CFLAGS=${DEFS} ${INCLUDES}
#LIB_PATH=-L$(critter_dir)/lib
LIB_PATH=
#LIBS=-lcritter
LIBS=
