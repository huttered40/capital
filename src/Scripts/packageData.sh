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
  export RESULTSPATH=../../../PAA_data
fi

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " resultsDir
read -p "Enter machine name: " machineName 
read -p "Enter run type: " profType

if [ "${machineName}" != "PORTER" ]
then
  mkdir ${RESULTSPATH}/${resultsDir}/
  cp -r ${SCRATCH}/${resultsDir}/* ${RESULTSPATH}/${resultsDir}/
fi

g++ fileTransfer.cpp -o fileTransfer
read -p "Enter number of tests: " numTests
for ((i=0; i<${numTests}; i++))
do
  read -p "Enter number of configurations (files to write to): " numConfigFiles
  for ((j=0; j<${numConfigFiles}; j++))
  do
    read -p "Enter binary tag: " binaryTag
    read -p "Enter performance/NoFormQ file to write to: " configFilePerf
    read -p "Enter numerics/FormQ file to write to: " configFileNumerics
    
    configFileCritter="" 
    configFileTimer=""
    if [ "${profType}" == "A"  ]
    then
      read -p "Enter critter file to write to: " configFileCritter
      read -p "Enter profiling file to write to: " configFileTimer
    fi

    read -p "Enter number of files to read from: " numInputFiles
    echo "tell me these 4 things - ${binaryTag} ${configFilePerf} ${configFileNumerics} ${numInputFiles}"
    for ((k=0; k<${numInputFiles}; k++))
    do
      # First, performance
      # Currently, every other input file will be performance (if profType=="P", if =="A", then every 4 is performance), so that is how this inner-loop code will be structured
      read -p "Enter file to read from: " InputFile
      ./fileTransfer ${RESULTSPATH}/${resultsDir}/${configFilePerf} ${RESULTSPATH}/${resultsDir}/${InputFile} ${binaryTag} 1 ${k}
      #rm ${RESULTSPATH}/${resultsDir}/${InputFile}
      
      # Second, numerics
      read -p "Enter file to read from: " InputFile
      ./fileTransfer ${RESULTSPATH}/${resultsDir}/${configFileNumerics} ${RESULTSPATH}/${resultsDir}/${InputFile} ${binaryTag} 2 ${k}
      #rm ${RESULTSPATH}/${resultsDir}/${InputFile}

      if [ "${profType}" == "A"  ]
      then
        echo "Tell me k - ${k}"
        # Third, critter
	read -p "Enter file to read from: " InputFile
        ./fileTransfer ${RESULTSPATH}/${resultsDir}/${configFileCritter} ${RESULTSPATH}/${resultsDir}/${InputFile} ${binaryTag} 3 ${k}
        #rm ${RESULTSPATH}/${resultsDir}/${InputFile}
        
	# Fourth, timer
        read -p "Enter file to read from: " InputFile
        ./fileTransfer ${RESULTSPATH}/${resultsDir}/${configFileTimer} ${RESULTSPATH}/${resultsDir}/${InputFile} ${binaryTag} 4 ${k}
        #rm ${RESULTSPATH}/${resultsDir}/${InputFile}
      fi

    done
  done
done

cd ${RESULTSPATH}

# Get rid of the binaries before making the tarball
rm -rf ${resultsDir}/bin
tar -cvf ${resultsDir}.tar ${resultsDir}/*

cd -
rm fileTransfer
