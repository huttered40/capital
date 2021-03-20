#!/bin/bash

export MKL_NUM_THREADS=1
export CRITTER_MODE=1
export CRITTER_MECHANISM=0
export CRITTER_TRACK_SYNCHRONIZATION=0
export CRITTER_TRACK_BLAS1=1
export CRITTER_TRACK_BLAS2=1
export CRITTER_TRACK_BLAS3=1
export CRITTER_TRACK_LAPACK=1
export CRITTER_INCLUDE_BARRIER_TIME=0
export CRITTER_EAGER_LIMIT=32768
export CRITTER_DELETE_COMM=1
export CRITTER_COST_MODEL=1
export CRITTER_PATH_DECOMPOSITION=2
export CRITTER_PATH_SELECT=000001
export CRITTER_PATH_MEASURE_SELECT=0000001
export CRITTER_VIZ_FILE=critter