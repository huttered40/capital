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
  # Clear Post/ each time
  rm ${RESULTSPATH}/${destDir}/Post/Raw/*
  rm ${RESULTSPATH}/${destDir}/Post/Stats/*
fi

g++ fileTransfer.cpp -o fileTransfer
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
          read -p "Enter performance/NoFormQ file to write to: " postFilePerf
          preFileNumerics=""
          if [ "${binaryTag}" != "bench_scala_cholesky" ];
          then
	    read -p "Enter numerics/FormQ file to write to: " postFileNumerics
          fi    

	  preFileCritter="" 
	  preFileTimer=""
	  if [ "${profType}" == "PC"  ] || [ "${profType}" == "PCT" ];
	  then
	    read -p "Enter critter file to write to: " postFileCritter
	  fi
	  if [ "${profType}" == "PT"  ] || [ "${profType}" == "PCT" ];
	  then
	    read -p "Enter profiling file to write to: " postFileTimer
	  fi

	  # First, performance
	  # Currently, every other input file will be performance (if profType=="P", if =="A", then every 4 is performance), so that is how this inner-loop code will be structured
	  read -p "Enter file to read from: " InputFile
	  ./fileTransfer ${RESULTSPATH}/${destDir}/Post/ ${postFilePerf} ${RESULTSPATH}/${destDir}/Pre/${InputFile} ${binaryTag} 1 0

	  if [ "${binaryTag}" != "bench_scala_cholesky" ];
	  then 
	    # Second, numerics
	    read -p "Enter file to read from: " InputFile
	    ./fileTransfer ${RESULTSPATH}/${destDir}/Post/ ${postFileNumerics} ${RESULTSPATH}/${destDir}/Pre/${InputFile} ${binaryTag} 2 0
	  fi

	  if [ "${profType}" == "PC"  ] || [ "${profType}" == "PCT" ];
	  then
	    # Third, critter
	    read -p "Enter file to read from: " InputFile
	    ./fileTransfer ${RESULTSPATH}/${destDir}/Post/ ${postFileCritter} ${RESULTSPATH}/${destDir}/Pre/${InputFile} ${binaryTag} 3 0
	  fi
	  if [ "${profType}" == "PT"  ] || [ "${profType}" == "PCT" ];
	  then
	    # Fourth, timer
	    read -p "Enter file to read from: " InputFile
	    ./fileTransfer ${RESULTSPATH}/${destDir}/Post/ ${postFileTimer} ${RESULTSPATH}/${destDir}/Pre/${InputFile} ${binaryTag} 4 0
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

# Write to a new file that will offer opportunity to mutate plotInstructions in order to analyze the node usage in scaling studies.
echo "cp ${RESULTSPATH}/${destDir}/plotInstructions.sh plotInstructions.sh" > updatePlotScript.sh
echo "bash UpdatePlotInstructions.sh ${RESULTSPATH} ${RESULTSPATH}/${destDir}/ ${destDir} \${1}" >> updatePlotScript.sh

cd ${RESULTSPATH}
tar -cvf ${destDir}.tar ${destDir}/*
cd -
rm fileTransfer
