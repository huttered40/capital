#!/bin/bash

# Note: must be in CAMFS/src/Scripts on my local machine to run this

if [ "$(hostname |grep "porter")" != "" ]
then
  export SCRATCH=../../../CAMFS_data
elif [ "$(hostname |grep "mira")" != "" ] || [ "$(hostname |grep "cetus")" != "" ]
then
  export SCRATCH=/projects/QMCat/huttered
elif [ "$(hostname |grep "theta")" != "" ]
then
  export SCRATCH=/projects/QMCat/huttered
elif [ "$(hostname |grep "stampede2")" != "" ]
then
  export SCRATCH=/scratch/05608/tg849075
elif [ "$(hostname |grep "h2o")" != "" ]
then
  export SCRATCH=/scratch/sciteam/hutter
fi

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " resultsDir
read -p "Enter the machine from which you want to grab data (PORTER,MIRA,CETUS,THETA,STAMPEDE2,BLUEWATERS): " machineName
cd ../../../../myData/

if [ "${machineName}" == "PORTER" ]
then
  scp -r hutter2@porter.cs.illinois.edu:~/hutter2/CAMFS_data/${resultsDir}.tar .
elif [ "${machineName}" == "MIRA" ]
then
  scp -r huttered@miradtn.alcf.anl.gov:~/hutter2/CAMFS_data/${resultsDir}.tar .
elif [ "${machineName}" == "CETUS" ]
then
  scp -r huttered@cetusdtn.alcf.anl.gov:~/hutter2/CAMFS_data/${resultsDir}.tar .
elif [ "${machineName}" == "THETA" ]
then
  scp -r huttered@thetadtn.alcf.anl.gov:~/hutter2/CAMFS_data/${resultsDir}.tar .
elif [ "${machineName}" == "STAMPEDE2" ]
then
  scp -r tg849075@stampede2.tacc.utexas.edu:~/CAMFS_data/${resultsDir}.tar .
elif [ "${machineName}" == "BLUEWATERS" ]
then
  scp -r hutter@bw.ncsa.illinois.edu:~/CAMFS_data/${resultsDir}.tar .
fi

tar -xvf ${resultsDir}.tar
cd -

scaplotDir=../../../scaplot/
scriptDir=$(pwd)

# Generate the Makefile for Scaplot
cd ${scaplotDir}
bash ../../ResearchData/${resultsDir}/plotInstructions.sh | bash MakePlotScript.sh