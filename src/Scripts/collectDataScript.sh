#!/bin/bash

if [ "$(hostname |grep "porter")" != "" ]
then
  export SCRATCH=../../../PAA_data
  export RESULTSPATH=../../../PAA_data
elif [ "$(hostname |grep "mira")" != "" ] || [ "$(hostname |grep "cetus")" != "" ]
then
  export SCRATCH=/projects/QMCat/huttered
  export RESULTSPATH=../../../PAA_data
elif [ "$(hostname |grep "theta")" != "" ]
then
  export SCRATCH=/projects/QMCat/huttered
  export RESULTSPATH=../../../PAA_data
elif [ "$(hostname |grep "stampede2")" != "" ]
then
fi

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " resultsDir
cp ${SCRATCH}/${resultsDir}/* ${RESULTSPATH}/${resultsDir}/

cd ${RESULTSPATH}

# Get rid of the binaries before making the tarball
rm -rf ${resultsDir}/bin
tar -cvf ${resultsDir}.tar ${resultsDir}/*
