#!/bin/bash

# Create new file by modifying plotInstructions
cp plotInstructions.sh plotInstructionsCopy.sh
sed -i 's/echo "//g' plotInstructionsCopy.sh
sed -i 's/"//g' plotInstructionsCopy.sh

WriteFile="plotInstructionsNodeUsage.sh"

numTestsTotal=0
numPlotTagsTotal=0	# Does not have to be unique
numTestsPerMacroTest=()
savePlotTagsGlobal=()

NewTestInfo=()		# Saves the static parameters for each new test, once its found
NewTestData=()		# Saves the details of each plotTag, whether unique or not, to avoid having to read in file repeatedly. Just scan this array repeatedly.
saveMethodID=()

# Fill these in below
createMakefileDecision=""
fileName=""
numTests=""
machineName=""
profType=""
nodeScalingFactor=""

while [ 1 -eq 1 ];
do
  SavePlotID=()				# Reset for each MACRO test
  # Read in first 7 header data
  read createMakefileDecision
  read fileName
  read numTests
  read machineName
  read profType
  read nodeScalingFactor

  for ((i=0;i<${numTests};i++));
  do
    # Read in the information static to each test
    read scaleType
    read NodeCount
    saveNodeNum=()
    for ((i=0;i<${NodeCount};i++));
    do
      read nodeNum
      saveNodeNum+=(${nodeNum})
      echo "What is nodeNum - ${nodeNum} ${saveNodeNum[@]}"
    done
    read MatDimM
    read MatDimN

    numTestsLocal=0
    savePlotTagsLocal=()
    while [ 1 -eq 1 ];
    do
      read MethodID
      if [ ${MethodID} -ne 4 ];
      then
        while [ 1 -eq 1 ];
        do
          read binaryID
          if [ ${binaryID} -ne 1 ];
          then
            read PlotTag
            numPlotTagsTotal=$(( ${numPlotTagsTotal} + 1 ))
            # Check to see if this PlotTag is new
            isUnique=1
            for ((j=0;j<${numTestsLocal}; j++));
            do
              plotTagFromArray=${savePlotTagsLocal[${j}]}
              if [ "${plotTagFromArray}" == "${PlotTag}" ];
              then
                isUnique=0
                break
              fi
            done
            # If isUnique is still of value 1, then this tag is unique and we can save it!
            if [ ${isUnique} -eq 1 ];
            then
              numTestsLocal=$(( ${numTestsLocal} + 1 ))
              savePlotTagsLocal+=(${PlotTag})
              savePlotTagsGlobal+=(${PlotTag})

              # Save the ScaleType,NodeCount,NodeNumbers,MatrixDimensions for use later
              NewTestInfo+=(${scaleType})
              NewTestInfo+=(${NodeCount})
              for ((j=0;j<${NodeCount};j++));
              do
                NewTestInfo+=(${saveNodeNum[${j}]})
              done
              NewTestInfo+=(${MatDimM})
              NewTestInfo+=(${MatDimN})
            fi

            # Either way, the rest of the arguments needs to be read in as garbage before we analyze the next plotTag
            read binaryTag
            numGarbageReads=0
            if [ "${binaryTag}" == "cqr2" ];
            then
              numGarbageReads=10
              if [ ${isUnique} -eq 1 ];
              then
                saveMethodID+=(0)
              fi
            elif [ "${binaryTag}" == "bsqr" ] || [ "${binaryTag}" == "rsqr" ];
            then
              numGarbageReads=9
              if [ ${isUnique} -eq 1 ];
              then
                saveMethodID+=(1)
              fi
            elif [ "${binaryTag}" == "cfr3d" ];
            then
              # TODO: This isn't write. Fix when doing first Cholesky run
              numGarbageReads=10
              if [ ${isUnique} -eq 1 ];
              then
                saveMethodID+=(2)
              fi
            elif [ "${binaryTag}" == "bscf" ];
            then
              # TODO: This isn't write. Fix when doing first Cholesky run
              numGarbageReads=10
              if [ ${isUnique} -eq 1 ];
              then
                saveMethodID+=(3)
              fi
            fi

            if [ "${binaryTag}" == "cqr2" ] && [ "${scaleType}" == "WS" ];
            then
              incrAmount=$(( ${NodeCount} * 3 ))
              incrAmount=$(( ${incrAmount} + 1 ))		# factors in the extra nodeCount that we don't need anymore
              numGarbageReads=$(( ${numGarbageReads} + ${incrAmount} ))
            fi

            NewTestData+=(${PlotTag})
            NewTestData+=(${binaryTag})
            NewTestData+=(${numGarbageReads})

            for ((j=0;j<${numGarbageReads};j++));
            do
              read garbage
              NewTestData+=(${garbage})
            done
          else
            break
          fi
        done
      else
        break
      fi
    done

    # Save the number of tests found in this original 'test'
    numTestsPerMacroTest+=(${numTestsLocal})
    numTestsTotal=$(( ${numTestsTotal} + ${numTestsLocal} ))
  done

  break
done < plotInstructionsCopy.sh

echo "${savePlotTagsGlobal[@]}"
echo "${NewTestInfo[@]}"
echo "${NewTestData[@]}"

# Now we start pass #2, where we need to rewrite another copy of plotInstructions
# First, write header
echo "echo \"${createMakefileDecision}\"" > ${WriteFile}	# This is not an append. It erases whatever was left behind.
echo "echo \"${fileName}\"" >> ${WriteFile}
echo "echo \"${numTestsTotal}\"" >> ${WriteFile}
echo "echo \"${machineName}\"" >> ${WriteFile}
echo "echo \"${profType}\"" >> ${WriteFile}
echo "echo \"${nodeScalingFactor}\"" >> ${WriteFile}

NewTestInfoIndex=0
for ((i=0;i<${numTestsTotal};i++));
do
  # For each new test, we need to write the scaleType, nodeCount, and nodes of the original test from which the new test in question was formed
  echo "echo \"${NewTestInfo[${NewTestInfoIndex}]}\"" >> ${WriteFile}	# ScaleType
  NewTestInfoIndex=$(( ${NewTestInfoIndex} + 1 ))
  NodeCount=${NewTestInfo[${NewTestInfoIndex}]}
  echo "echo \"${NodeCount}\"" >> ${WriteFile}
  NewTestInfoIndex=$(( ${NewTestInfoIndex} + 1 ))
  for ((j=0;j<${NodeCount};j++));
  do
    echo "echo \"${NewTestInfo[${NewTestInfoIndex}]}\"" >> ${WriteFile}	# Number of nodes
    NewTestInfoIndex=$(( ${NewTestInfoIndex} + 1 ))
  done
  echo "echo \"${NewTestInfo[${NewTestInfoIndex}]}\"" >> ${WriteFile}	# MatDimM
  NewTestInfoIndex=$(( ${NewTestInfoIndex} + 1 ))
  echo "echo \"${NewTestInfo[${NewTestInfoIndex}]}\"" >> ${WriteFile}	# MatDimN
  NewTestInfoIndex=$(( ${NewTestInfoIndex} + 1 ))

  echo "echo \"${saveMethodID[${i}]}\"" >> ${WriteFile}		# This simplified for us, because in this NodeUsage scheme, we know that each test will test the same methodID

  # Now starts where we need to traverse the entire original file, and find matches with the corresponding plotTag
  curPlotTag=${savePlotTagsGlobal[${i}]}

  NewTestDataIndex=0
  for ((j=0;j<${numPlotTagsTotal};j++));
  do
    pt=${NewTestData[${NewTestDataIndex}]}
    echo "What is plotTag and what is pt - ${curPlotTag} ${pt}"
    echo "What is NewTestDataIndex - ${NewTestDataIndex}"
    echo "What are outer loop indices - ${i} ${j}"
    NewTestDataIndex=$(( ${NewTestDataIndex} + 1 ))
    if [ "${pt}" == "${curPlotTag}" ];
    then
      echo "echo \"0\"" >> ${WriteFile}
      echo "echo \"${curPlotTag}\"" >> ${WriteFile}
      echo "echo \"${NewTestData[${NewTestDataIndex}]}\"" >> ${WriteFile}	# Write binaryTag
      NewTestDataIndex=$(( ${NewTestDataIndex} + 1 ))
      NumIter=${NewTestData[${NewTestDataIndex}]}
      NewTestDataIndex=$(( ${NewTestDataIndex} + 1 ))
      for ((k=0;k<${NumIter};k++));
      do
        echo "echo \"${NewTestData[${NewTestDataIndex}]}\"" >> ${WriteFile}
        NewTestDataIndex=$(( ${NewTestDataIndex} + 1 ))
      done
    else
      NewTestDataIndex=$(( ${NewTestDataIndex} + 1 ))
      NumIter=${NewTestData[${NewTestDataIndex}]}
      NewTestDataIndex=$(( ${NewTestDataIndex} + 1 ))
      NewTestDataIndex=$(( ${NewTestDataIndex} + ${NumIter} ))
    fi
    echo "What is NumIter - ${NumIter}"
  done
  # Mark down that we are finished with this test. This simplified for us, because in this NodeUsage scheme, we know that each test will test the same methodID
  echo "echo \"1\"" >> ${WriteFile}
  echo "echo \"4\"" >> ${WriteFile}
done

rm plotInstructionsCopy.sh
mv ${WriteFile} ${2}${WriteFile}

# Redo the tar now that plotInstructionsNodeUsage.sh is included
cd ${1}
rm ${3}.tar
tar -cvf ${3}.tar ${3}/*
cd -
