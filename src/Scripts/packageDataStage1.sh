#!/bin/bash

# Function that simply appends each line in the data files to the 'Pre' files that collect all the data over all rounds
WriteToPre() {
  srcFile=${1}
  destFile=${2}

  while read -r line
  do
    echo "${line}" >> ${destFile}
  done < "${srcFile}"
}

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
elif [ "$(hostname |grep "h2o")" != "" ]
then
  export SCRATCH=/scratch/sciteam/hutter/
  export RESULTSPATH=../../../PAA_data
elif [ "$(hostname |grep "stampede2")" != "" ]
then
  export RESULTSPATH=../../../PAA_data
fi

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " srcDir
read -p "Enter the directory name to which to write updated results: " destDir
read -p "Enter machine name: " machineName 
read -p "Enter run type: " profType
read -p "Enter node scaling factor: " nodeScaleFactor

if [ "${machineName}" != "PORTER" ]
then
  # Check if the directory already exists, which will be the case when running this for rounds 2+
  IsRound1=0
  if [ ! -d "${RESULTSPATH}/${destDir}" ];
  then
    IsRound1=1
    mkdir ${RESULTSPATH}/${destDir}/
    mkdir ${RESULTSPATH}/${destDir}/Pre		# Pre is where we are appending these datafiles to, regardless of round ID.
    mkdir ${RESULTSPATH}/${destDir}/Post	# Post is where the analyzed datafiles are written to. This does not happen in this file
    mkdir ${RESULTSPATH}/${destDir}/Post/Raw
    mkdir ${RESULTSPATH}/${destDir}/Post/Stats
    echo "bash ${RESULTSPATH}/${destDir}/collectInstructionsStage2.sh | bash packageDataStage2.sh" > collectDataStage2.sh
  fi
  mkdir ${RESULTSPATH}/${destDir}/${srcDir}/
  cp -r ${SCRATCH}/${srcDir}/* ${RESULTSPATH}/${destDir}/${srcDir}/
  # If round 1, then copy collectInstructionsStage2 into the main directory, for use in Stage2. Otherwise, it assumes no changes and serious changes will have to be done manually
  if [ ${IsRound1} -eq 1 ];
  then
    cp ${SCRATCH}/${srcDir}/collectInstructionsStage2.sh ${RESULTSPATH}/${destDir}/collectInstructionsStage2.sh
    cp ${SCRATCH}/${srcDir}/plotInstructions.sh ${RESULTSPATH}/${destDir}/plotInstructions.sh
  fi
fi

#g++ fileTransfer.cpp -o fileTransfer
read -p "Enter number of tests: " numTests

for ((i=0; i<${numTests}; i++))
do
  while [ 1 -eq 1 ];
  do
    read -p "Enter method ID: " methodID
    # break-case is methodID == 4, otherwise methodID is valid. Read it in
    if [ ${methodID} -ne 4 ];
    then
      while [ 1 -eq 1 ];
      do
        read -p "Enter binary ID: " binaryID
        # break-case is binaryID == 1, otherwise binaryID is valid. Read it in
        if [ ${binaryID} -eq 0 ];
        then
          read -p "Enter binary tag: " binaryTag

          read -p "Enter performance/NoFormQ file to write to: " preFilePerf
          preFileNumerics=""
          if [ "${binaryTag}" != "bench_scala_cholesky" ];
          then
	    read -p "Enter numerics/FormQ file to write to: " preFileNumerics
          fi    

	  preFileCritter="" 
	  preFileTimer=""
	  if [ "${profType}" == "PC"  ] || [ "${profType}" == "PCT" ];
	  then
	    read -p "Enter critter file to write to: " preFileCritter
	  fi
	  if [ "${profType}" == "PT"  ] || [ "${profType}" == "PCT" ];
	  then
	    read -p "Enter profiling file to write to: " preFileTimer
	  fi

	  # First, performance
	  # Currently, every other input file will be performance (if profType=="P", if =="A", then every 4 is performance), so that is how this inner-loop code will be structured
	  read -p "Enter file to read from: " InputFile
#	  ./fileTransfer ${RESULTSPATH}/${srcDir}/${configFilePerf} ${RESULTSPATH}/${srcDir}/${InputFile} ${binaryTag} 1 ${k}
          WriteToPre ${SCRATCH}/${srcDir}/${InputFile} ${RESULTSPATH}/${destDir}/Pre/${preFilePerf}.txt

	  if [ "${binaryTag}" != "bench_scala_cholesky" ];
	  then 
	    # Second, numerics
	    read -p "Enter file to read from: " InputFile
#	    ./fileTransfer ${RESULTSPATH}/${srcDir}/${configFileNumerics} ${RESULTSPATH}/${srcDir}/${InputFile} ${binaryTag} 2 ${k}
            WriteToPre ${SCRATCH}/${srcDir}/${InputFile} ${RESULTSPATH}/${destDir}/Pre/${preFileNumerics}.txt
	  fi

	  if [ "${profType}" == "PC"  ] || [ "${profType}" == "PCT" ];
	  then
	    # Third, critter
	    read -p "Enter file to read from: " InputFile
#	    ./fileTransfer ${RESULTSPATH}/${srcDir}/${configFileCritter} ${RESULTSPATH}/${srcDir}/${InputFile} ${binaryTag} 3 ${k}
            WriteToPre ${SCRATCH}/${srcDir}/${InputFile} ${RESULTSPATH}/${destDir}/Pre/${preFileCritter}.txt
	  fi
	  if [ "${profType}" == "PT"  ] || [ "${profType}" == "PCT" ];
	  then
	    # Fourth, timer
	    read -p "Enter file to read from: " InputFile
#	    ./fileTransfer ${RESULTSPATH}/${srcDir}/${configFileTimer} ${RESULTSPATH}/${srcDir}/${InputFile} ${binaryTag} 4 ${k}
            WriteToPre ${SCRATCH}/${srcDir}/${InputFile} ${RESULTSPATH}/${destDir}/Pre/${preFileTimer}.txt
	  fi
        else
          break
        fi
      done
    else
      break
    fi
  done
done

# Get rid of the binaries
cd ${RESULTSPATH}
rm -rf ${destDir}/${srcDir}/bin
cd -
