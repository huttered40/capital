#!/bin/bash

# Note: must be in PAA/src/Scripts on my local machine to run this

if [ "$(hostname |grep "porter")" != "" ]
then
  export SCRATCH=../../../PAA_data
elif [ "$(hostname |grep "mira")" != "" ] || [ "$(hostname |grep "cetus")" != "" ]
then
  export SCRATCH=/projects/QMCat/huttered
elif [ "$(hostname |grep "theta")" != "" ]
then
  export SCRATCH=/projects/QMCat/huttered
elif [ "$(hostname |grep "stampede2")" != "" ]
then
fi

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " resultsDir
cd ../../../../myData/

if [ "$(hostname |grep "porter")" != "" ]
then
  scp -r hutter2@porter.cs.illinois.edu:~/hutter2/PAA_data/${resultsDir} .
elif [ "$(hostname |grep "mira")" != "" ] || [ "$(hostname |grep "cetus")" != "" ]
then
  scp -r huttered@miradtn.alcf.anl.gov:~/hutter2/PAA_data/${resultsDir} .
elif [ "$(hostname |grep "theta")" != "" ]
then
  scp -r huttered@thetadtn.alcf.anl.gov:~/hutter2/PAA_data/${resultsDir} .
elif [ "$(hostname |grep "stampede2")" != "" ]
then
fi

tar -xvf ${resultsDir}.tar
cd -

scaplotDir=../../../scaplot/
scriptDir=$(pwd)

# Generate the Makefile for Scaplot
cd ${scaplotDir}
bash ${scriptDir}/../Results/${resultsDir}/plotInstructions.sh | bash MakePlotScript.sh
