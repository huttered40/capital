#!/bin/bash

export CRITTER_MODE=1
export CRITTER_TEST=baseline_variants
export CRITTER_MECHANISM=0
#export CRITTER_TRACK_COLLECTIVE=1
#export CRITTER_TRACK_P2P=1
#export CRITTER_TRACK_BLAS=0
#export CRITTER_TRACK_LAPACK=0
#export CRITTER_SCHEDULE_KERNELS=1
export CRITTER_VIZ_FILE=Test-${CRITTER_TEST}__mode-${CRITTER_MODE}__mechanism-${CRITTER_MECHANISM}
