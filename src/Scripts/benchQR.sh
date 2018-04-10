#!/bin/bash

tag1='cqr2'
tag2='bench_scala_qr'

# Make sure that the src/bin directory is created, or else compilation won't work
if [ ! -d "../bin" ];
then
  mkdir ../bin
fi

# Lets remove the old instructions
rm -f fileTransfer
rm -f collectInstructions.sh

scalaDir=""
machineName=""
if [ "$(hostname |grep "porter")" != "" ]
then
  machineName=PORTER
  scalaDir=~/hutter2/ExternalLibraries/CANDMC/CANDMC
  scaplotDir=~/hutter2/ExternalLibraries/SCAPLOT/scaplot
  read -p "Do you want to use MPI[mpi] or AMPI[ampi]? " mpiType
  if [ "${mpiType}" == "mpi" ]
  then
    export MPITYPE=MPI_TYPE
  elif [ "${mpiType}" == "ampi" ]
  then
    export MPITYPE=AMPI_TYPE
  fi
elif [ "$(hostname |grep "mira")" != "" ] || [ "$(hostname |grep "cetus")" != "" ]
then
  machineName=BGQ
  scalaDir=~/scratch/CANDMC
  export MPITYPE=MPI_TYPE
elif [ "$(hostname |grep "theta")" != "" ]
then
  machineName=THETA
  scalaDir=~/scratch/CANDMC
  export MPITYPE=MPI_TYPE
elif [ "$(hostname |grep "stampede2")" != "" ]
then
  machineName=STAMPEDE2
  scalaDir=""                # Fill in soon
  export MPITYPE=MPI_TYPE
fi

read -p "Do you want Profiling/Timer[T] output, Critter[C] output, or absolute performance[P] output? Note that AMPI only allows [P]: " profType
if [ "${profType}" == "T" ]
then
  export PROFTYPE=TIMER_TYPE
elif [ "${profType}" == "C" ]
then
  export PROFTYPE=CRITTER_TYPE
elif [ "${profType}" == "P" ]
then
  export PROFTYPE=PERF_TYPE
fi

# Note: leaving this code inactive for now. I can always manually remove data files once it gets too large
#read -p "Should we delete the items in Results directory? Yes[1], No[0] " delDecision1
#if [ "${delDecision1}" == "1" ]
#then
#  rm -rf ../Results/*
#fi

#read -p "Enter machine name [BGQ (cetus,mira), THETA, BW, STAMPEDE2, PORTER]: " machineName
#read -p "Enter the Date (MM_DD_YYYY): " dateStr
dateStr=$(date +%Y-%m-%d-%H_%M_%S)
read -p "Enter ID of auto-generated file this program will create: " fileID
read -p "Enter minimum number of nodes requested: " minNumNodes
read -p "Enter maximum number of nodes requested: " maxNumNodes
read -p "Enter ppn: " ppn
read -p "Enter number of tests (equal to number of strong scaling or weak scaling tests that will be run): " numTests

if [ "${machineName}" == "BW" ]
then
  read -p "Enter number of hours of job: " numHours
  read -p "Enter number of minutes of job: " numMinutes
  read -p "Enter number of seconds of job: " numSeconds
elif [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ]
then
  read -p "Enter number of minutes of job: " numMinutes
fi

numPEs=$((ppn*numNodes))
fileName=benchQR${fileID}_${dateStr}_${machineName}_${profType}

read -p "What datatype? float[0], double[1], complex<float>[2], complex<double>[3]: " dataType
read -p "What integer type? int[0], int64_t[1]: " intType
if [ ${dataType} == 0 ]
then
  export DATATYPE=FLOAT_TYPE
elif [ ${dataType} == 1 ]
then
  export DATATYPE=DOUBLE_TYPE
elif [ ${dataType} == 2 ]
then
  export DATATYPE=COMPLEX_FLOAT_TYPE
elif [ ${dataType} == 3 ]
then
  export DATATYPE=COMPLEX_DOUBLE_TYPE
fi
if [ ${intType} == 0 ]
then
  export INTTYPE=INT_TYPE
elif [ ${intType} == 1 ]
then
  export INTTYPE=INT64_T_TYPE
fi

# Build PAA code
make -C./.. clean
make -C./.. cqr2_${mpiType}

# Build CANDMC code
if [ "${machineName}" == "THETA" ]
then
  cd ${scalaDir}
  ./configure
  make clean
  make bench_scala_qr
  cd -
  mv ${scalaDir}/bin/benchmarks/bench_scala_qr ${scalaDir}/bin/benchmarks/bench_scala_qr_${machineName}
  mv ${scalaDir}/bin/benchmarks/* ../bin/
fi

if [ "${machineName}" == "BGQ" ]
then
  export SCRATCH=/projects/QMCat/huttered
elif [ "${machineName}" == "BW" ]
then
  echo "dog"
elif [ "${machineName}" == "THETA" ]
then
  export SCRATCH=/projects/QMCat/huttered
  export BINPATH=${SCRATCH}/${fileName}/bin/
elif [ "${machineName}" == "STAMPEDE2" ]
then
  echo "dog"
elif [ "${machineName}" == "PORTER" ]
then
  export SCRATCH=../../../PAA_data
  export BINPATH=./../bin/
fi

cat <<-EOF > $SCRATCH/${fileName}.sh
scriptName=$SCRATCH/${fileName}/script.sh
mkdir $SCRATCH/${fileName}/
mkdir $SCRATCH/${fileName}/results

# Loop over all scripts - log(P) of them
curNumNodes=${minNumNodes}
while [ \${curNumNodes} -le ${maxNumNodes} ];
do
  scriptName=$SCRATCH/${fileName}/script\${curNumNodes}.sh
  if [ "${machineName}" == "BGQ" ]
  then
    echo "#!/bin/sh" > \${scriptName}
  elif [ "${machineName}" == "BW" ]
  then
    echo "#!/bin/bash" > \$scriptName
    echo "#PBS -l nodes=$numNodes:ppn=${ppn}:xe" >> \$scriptName
    echo "#PBS -l walltime=${numHours}:${numMinutes}:${numSeconds}" >> \$scriptName
    echo "#PBS -N ${numNodes}" >> \$scriptName
    echo "#PBS -e \$PBS_JOBID.err" >> \$scriptName
    echo "#PBS -o \$PBS_JOBID.out" >> \$scriptName
    echo "##PBS -m Ed" >> \$scriptName
    echo "##PBS -M hutter2@illinois.edu" >> \$scriptName
    echo "##PBS -A xyz" >> \$scriptName
    echo "#PBS -W umask=0027" >> \$scriptName
    echo "cd \$PBS_O_WORKDIR" >> \$scriptName
    echo "#module load craype-hugepages2M  perftools" >> \$scriptName
    echo "#export APRUN_XFER_LIMITS=1  # to transfer shell limits to the executable" >> \$scriptName
  elif [ "${machineName}" == "THETA" ]
  then
    echo "#!/bin/bash" > \$scriptName
    echo "#COBALT -t ${numMinutes}" >> \$scriptName
    echo "#COBALT -n ${numNodes}" >> \$scriptName
    echo "#COBALT --attrs mcdram=cache:numa=quad" >> \$scriptName
    echo "#COBALT -A QMCat" >> \${scriptName}
    echo "export n_nodes=${numNodes}" >> \$scriptName
    echo "export n_mpi_ranks_per_node=${ppn}" >> \$scriptName
    echo "export n_mpi_ranks=\$((${numNodes} * ${ppn}))" >> \$scriptName
    read -p "Enter number of OpenMP threads per rank: " numOMPthreadsPerRank
    read -p "Enter number of hyperthreads per core: " numHyperThreadsPerCore
    read -p "Enter number of hyperthreads skipped per rank: " numHyperThreadsSkippedPerRank
    echo "export n_openmp_threads_per_rank=\${numOMPthreadsPerRank}" >> \$scriptName
    echo "export n_hyperthreads_per_core=\${numHyperThreadsPerCore}" >> \$scriptName
    echo "export n_hyperthreads_skipped_between_ranks=\${numHyperThreadsSkippedPerRank}" >> \$scriptName
  elif [ "${machineName}" == "STAMPEDE22" ]
  then
    echo "dog" > \$scriptName
  fi
  curNumNodes=\$(( \${curNumNodes} * 2 ))   # Its always going to be 2 for test. Don't overcomplicate and generalize this
done

# Now I need to use a variable for the command-line prompt, since it will change based on the binary executable,
#   for example, scalapack QR has multiple special inputs that take up comm-line prompts that others dont
#   I want special functions in the inner-loop to handle this

updateCounter () {
  local counter=\${1}
  if [ \${2} -eq 1 ]
  then
    counter=\$((\${counter} + \${3})) 
  elif [ \${2} -eq 2 ]
  then
   counter=\$((\${counter} - \${3})) 
  elif [ \${2} -eq 3 ]
  then
    counter=\$((\${counter} * \${3})) 
  elif [ \${2} -eq 4 ]
  then
    counter=\$((\${counter} / \${3})) 
  fi
  echo "\${counter}"
}

# Note: this function is only used for finding the number of dependencies for each binary run
# Therefore, I can multiply the output by 2 (total and average) and it won't affect anything else
# New note: I am getting rid of the *2, since that will be understood by all other scripts, including MakePlotScript.sh
findCountLength () {
  local curr=\${1}
  local counter=0
  while [ \${curr} -le \${2} ];
  do
    curr=\$(updateCounter \${curr} \${3} \${4})
    counter=\$(( counter+1 ))
  done
  echo "\${counter}"
}

log2 () {
    local x=0
    for (( y=\${1}-1 ; \${y} > 0; y >>= 1 )) ; do
        let x=\${x}+1
    done
    echo \${x}
}

# Will need to figure out how to get this to work with CANDMC (only when Porter runs fully work)
writePlotFileName() {
  if [ "${profType}" == "T" ]
  then
    echo "echo \"\${1}_timer.txt\"" >> \${2}
  elif [ "${profType}" == "C" ]
  then
    echo "echo \"\${1}_critter.txt\"" >> \${2}
  elif [ "${profType}" == "P" ]
  then
    echo "echo \"\${1}_perf.txt\"" >> \${2}
    if [ "\${3}" == "1" ]
    then
      echo "echo \"\${1}_perf_median.txt\"" >> \${2}
    fi
  fi
  echo "echo \"\${1}_numerics.txt\"" >> \${2}
  if [ "\${3}" == "1" ]
  then
    echo "echo \"\${1}_numerics_median.txt\"" >> \${2}
  fi
}

# Functions that write the actual script, depending on machine
launchJobs () {
  local numProcesses=\$((\${2} * $ppn))
  if [ "$machineName" == "BGQ" ]
  then
    echo "runjob --np \$numProcesses -p $ppn --block \$COBALT_PARTNAME --verbose=INFO : \${@:3:\$#}" >> $SCRATCH/${fileName}/script\${2}.sh
  elif [ "$machineName" == "BW" ]
  then
    echo "Note: this is probably wrong, and I need to check this once I get BW access"
    echo "aprun -n \$numProcesses \$@" >> $SCRATCH/${fileName}/script\${2}.sh
  elif [ "$machineName" == "THETA" ]
  then
    echo "aprun -n \${numProcesses} -N ${ppn} --env OMP_NUM_THREADS=\${numOMPthreadsPerRank} -cc depth -d \${numHyperThreadsSkippedPerRank} -j \${numHyperThreadsPerCore} \${@:3:\$#}" >> $SCRATCH/${fileName}/script\${2}.sh
  elif [ "$machineName" == "STAMPEDE2" ]
  then
    echo "dog" >> $SCRATCH/${fileName}/script\${2}.sh
  elif [ "$machineName" == "PORTER" ]
  then
    if [ "${mpiType}" == "mpi" ]
    then
      mpiexec -n \$numProcesses \${@:3:\$#}
    elif [ "${mpiType}" == "ampi" ]
    then
      ${BINPATH}charmrun +p1 +vp\${numProcesses} \${@:3:\$#}
    fi
  fi
  writePlotFileName \${@:1:1} collectInstructions.sh 0
}

launch$tag1 () {
  # launch CQR2
  local startNumNodes=\${4}
  local endNumNodes=\${5}
  local matrixDimM=\${8}
  local startPdimD=\${10}
  while [ \$startNumNodes -le \$endNumNodes ];
  do
    local fileString="results/results_${tag1}_\${1}_\${startNumNodes}nodes_\${matrixDimM}dimM_\${9}dimN_\${12}inverseCutOffMult_0bcMult_0panelDimMult_\${startPdimD}pDimD_\${11}pDimC"
    launchJobs \${fileString} \$startNumNodes \${2} \${matrixDimM} \${9} 0 \${12} 0 \${startPdimD} \${11} \${3} $SCRATCH/${fileName}/\${fileString}
    startNumNodes=\$(updateCounter \${startNumNodes} \${7} \${6})
    startPdimD=\$(updateCounter \${startPdimD} \${7} \${6})
    if [ "\${1}" == "WS" ];
    then
      matrixDimM=\$(updateCounter \${matrixDimM} \${7} \${6})
    fi
  done
}

launch$tag2 () {
  # launch scaLAPACK_QR
  local startNumNodes=\$4
  local endNumNodes=\$5
  local matrixDimM=\${8}
  local matrixDimN=\${9}
  local numProws=\${10}
  while [ \${startNumNodes} -le \${endNumNodes} ];
  do
    startBlockSize=1
    div1=\$(( \${matrixDimM} / \${numProws} ))
    curNumProcesses=\$(( \${startNumNodes}*${ppn} ))
    numPcols=\$(( \${curNumProcesses} / \${numProws} ))
    div2=\$(( \${matrixDimN} / \${numPcols} ))
    endBlockSize=\$(( \${div1}>\${div2}?\${div1}:\${div2} ))
    while [ \${startBlockSize} -le \${endBlockSize} ];
    do
      local fileString="results/results_${tag2}_\$1_\${startNumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${startBlockSize}blockSize_\${numProws}numProws"
      launchJobs \${fileString} \$startNumNodes \${2} \${matrixDimM} \${matrixDimN} \${startBlockSize} \${3} 0 \${numProws} 1 0 $SCRATCH/${fileName}/\${fileString}
      startBlockSize=\$(( \$startBlockSize * 2 ))
    done
    startNumNodes=\$(updateCounter \${startNumNodes} \$7 \$6)
    numProws=\$(updateCounter \${numProws} \$7 \$6)
    if [ "\${1}" == "WS" ];
    then
      matrixDimM=\$(updateCounter \${marixDimM} \${7} \${6})
    fi
  done
}


# Note: in future, I may want to decouple numBinaries and numPlotTargets, but only when I find it necessary
# Write to Plot Instructions file, for use by SCAPLOT makefile generator
echo "echo \"1\"" > $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${fileName}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${numTests}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${machineName}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${profType}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${ppn}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

# Echo for data collection from remote machine (not porter) to PAA/src/Results
# This temporary file will be deleted while collectScript.sh is called.
echo "echo \"${fileName}\"" > collectInstructions.sh
echo "echo \"${machineName}\"" >> collectInstructions.sh

echo "echo \"${numTests}\"" >> collectInstructions.sh

for ((i=1; i<=${numTests}; i++))
do
  echo -e "\nTest #\${i}\n"
  read -p "Enter scaling type [SS,WS]: " scale
  read -p "Enter number of different configurations/binaries which will be used for this test: " numBinaries

  echo "echo \"\${numBinaries}\"" >> collectInstructions.sh

  # Echo for SCAPLOT makefile generator
  echo "echo \"\${scale}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
  echo "echo \"\${numBinaries}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
  read -p "Enter number of distinct nodes across all binaries of this test. Afterward, list them in increasing order: " nodeCount
  echo "echo \"\${nodeCount}\" " >> $SCRATCH/${fileName}/plotInstructions.sh
  for ((j=0; j<\${nodeCount}; j++))
  do
    read -p "Enter number of nodes (\${j} of \${nodeCount}): " nodeNumber
    echo "echo \"\${nodeNumber}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
  done

  read -p "Enter starting number of nodes for this test: " startNumNodes
  read -p "Enter ending number of nodes for this test: " endNumNodes
  read -p "Enter factor by which to increase the number of nodes: " jumpNumNodes
  read -p "Enter arithmetic operator by which to increase the number of nodes by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpNumNodesoperator

  for ((j=1; j<=\${numBinaries}; j++))
  do
    echo -e "\nStage #\${j}"

    # Echo for SCAPLOT makefile generator
    read -p "Enter binary tag [cqr2,bench_scala_qr]: " binaryTag
    binaryPath=${BINPATH}\${binaryTag}_${machineName}
    if [ "${machineName}" == "PORTER" ]
    then
      binaryPath=\${binaryPath}_${mpiType}
    fi
    echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

    read -p "Enter number of iterations: " numIterations
    if [ \${binaryTag} == 'cqr2' ]
    then
      read -p "Enter the inverseCutOff multiplier, 0 indicates that CFR3D will use the explicit inverse, 1 indicates that top recursive level will avoid calculating inverse, etc.: " inverseCutOffMult
      read -p "In this strong scaling test for CQR2, enter matrix dimension m: " matrixDimM
      read -p "In this strong scaling test for CQR2, enter matrix dimension n: " matrixDimN
      read -p "In this strong scaling test for CQR2, enter starting tunable processor grid dimension d: " pDimD
      read -p "In this strong scaling test for CQR2, enter static tunable processor grid dimension c: " pDimC
      
      # Write to plotInstructions file
      echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${binaryTag}\"" >> collectInstructions.sh
      echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_perf\"" >> collectInstructions.sh
      echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_numerics\"" >> collectInstructions.sh
      echo "echo \"\$(findCountLength \${startNumNodes} \${endNumNodes} \${jumpNumNodesoperator} \${jumpNumNodes})\"" >>collectInstructions.sh
      # Write to plotInstructions file
      echo "echo \"\${matrixDimM}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${matrixDimN}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${pDimD}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${pDimC}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${inverseCutOffMult}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      writePlotFileName \${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC} $SCRATCH/${fileName}/plotInstructions.sh 1
  
      launch\${binaryTag} \${scale} \${binaryPath} \${numIterations} \${startNumNodes} \${endNumNodes} \${jumpNumNodes} \${jumpNumNodesoperator} \${matrixDimM} \${matrixDimN} \${pDimD} \${pDimC} \${inverseCutOffMult}
    elif [ \${binaryTag} == 'bench_scala_qr' ]
    then
      read -p "In this strong scaling test for Scalapack QR, enter matrix dimension m: " matrixDimM
      read -p "In this strong scaling test for Scalapack QR, enter matrix dimension n: " matrixDimN
      read -p "In this strong scaling test for Scalapack QR, enter the starting number of processor rows: " numProws
      
      # Write to plotInstructions file
      echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${binaryTag}\"" >> collectInstructions.sh
      echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_NoFormQ\"" >> collectInstructions.sh
      echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_FormQ\"" >> collectInstructions.sh
      # This is where the last tricky part is: how many files do we need, because blockSize must be precomputed basically, and then multiplied by findCountLength
      # Write to plotInstructions file
      echo "echo \"\${matrixDimM}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${matrixDimN}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${numProws}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\$(findCountLength \${startNumNodes} \${endNumNodes} \${jumpNumNodesoperator} \${jumpNumNodes})\"" >> collectInstructions.sh
      #.. write arguments to plot. what about blockSize?
      writePlotFileName \${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC} $SCRATCH/${fileName}/plotInstructions.sh 1
      launch\${binaryTag} \${scale} \${binaryPath} \${numIterations} \${startNumNodes} \${endNumNodes} \${jumpNumNodes} \${jumpNumNodesoperator} \${matrixDimM} \${matrixDimN} \${numProws}
    fi
  done
done
EOF



bash $SCRATCH/${fileName}.sh
#rm $SCRATCH/${fileName}.sh

# Note that for Porter, no need to do this, since we are submitting to a queue
if [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ]
then
  mkdir $SCRATCH/${fileName}/bin
  mv ../bin/* $SCRATCH/${fileName}/bin
  #mv ${scalaDir}/bin/benchmarks/* $SCRATCH/${fileName}/bin  # move all scalapack benchmarks to same place before job is submitted
  cd $SCRATCH

  # Submit all scripts
  curNumNodes=${minNumNodes}
  while [ \${curNumNodes} -le ${maxNumNodes} ];
  do
    chmod +x ${fileName}/script${curNumNodes}.sh
    qsub ${fileName}/script${curNumNodes}.sh
    curNumNodes=$(( ${curNumNodes} * 2 ))
  done
fi
