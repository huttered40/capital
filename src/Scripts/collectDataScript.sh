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
  echo "dog"
fi

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " resultsDir
read -p "Enter machine name: " machineName 

if [ "${machineName}" != "PORTER" ]
then
  cp -r ${SCRATCH}/${resultsDir}/* ${RESULTSPATH}/${resultsDir}/
fi

g++ fileTransfer.cpp -o fileTransfer
read -p "Enter number of tests: " numTests
for ((i=0; i<${numTests}; i++))
do
  read -p "Enter number of configurations (files to write to): " numConfigFiles
  for ((j=0; j<${numConfigFiles}; j++))
  do
    read -p "Enter performance/profiling/critter file to write to: " configFilePerf
    read -p "Enter numerics file to write to: " configFileNumerics
    read -p "Enter number of files to read from: " numInputFiles
    for ((k=0; k<${numInputFiles}; k++))
    do
      # Currently, every other input file will be performance, so that is how this inner-loop code will be structured
      read -p "Enter file to read from: " InputFile
      ./fileTransfer ${RESULTSPATH}/${resultsDir}/${configFilePerf} ${RESULTSPATH}/${resultsDir}/${InputFile}
      rm ${RESULTSPATH}/${resultsDir}/${InputFile}
      read -p "Enter file to read from: " InputFile
      ./fileTransfer ${RESULTSPATH}/${resultsDir}/${configFileNumerics} ${RESULTSPATH}/${resultsDir}/${InputFile}
      rm ${RESULTSPATH}/${resultsDir}/${InputFile}
    done
  done
done

cd ${RESULTSPATH}

# Get rid of the binaries before making the tarball
rm -rf ${resultsDir}/bin
tar -cvf ${resultsDir}.tar ${resultsDir}/*

cd -
rm fileTransfer
# Push the changes, which should just a single file - collectInstructions.sh
git add -A && git commit -m "Commiting updated collectInstructions.sh, which contains useful info for plotting on local machine."
git push origin master
