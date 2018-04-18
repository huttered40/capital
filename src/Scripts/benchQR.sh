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

scalaDir=""
machineName=""
mpiType=""
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
  mpiType=mpi
elif [ "$(hostname |grep "theta")" != "" ]
then
  machineName=THETA
  scalaDir=~/scratch/CANDMC
  export MPITYPE=MPI_TYPE
  mpiType=mpi
elif [ "$(hostname |grep "stampede2")" != "" ]
then
  machineName=STAMPEDE2
  scalaDir=~/CANDMC
  export MPITYPE=MPI_TYPE
  mpiType=mpi
fi

dateStr=$(date +%Y-%m-%d-%H_%M_%S)
read -p "Enter ID of auto-generated file this program will create: " fileID
read -p "Enter minimum number of nodes requested: " minNumNodes
read -p "Enter maximum number of nodes requested: " maxNumNodes
read -p "Enter ppn: " ppn

numThreadsPerRankMin=""
numThreadsPerRankMax=""
if [ "${machineName}" == "STAMPEDE2" ]
then
  read -p "Enter minimum number of MKL threads per MPI rank: " numThreadsPerRankMin
  read -p "Enter maximum number of MKL threads per MPI rank: " numThreadsPerRankMax
fi

read -p "Enter number of tests (equal to number of strong scaling or weak scaling tests that will be run): " numTests

numHours=""
numMinutes=""
numSeconds=""
if [ "${machineName}" == "BW" ] || [ "${machineName}" == "STAMPEDE2" ]
then
  read -p "Enter number of hours of job: " numHours
  read -p "Enter number of minutes of job: " numMinutes
  read -p "Enter number of seconds of job: " numSeconds
elif [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ]
then
  read -p "Enter number of minutes of job: " numMinutes
fi

numPEs=$((ppn*numNodes))
fileName=benchQR${fileID}_${dateStr}_${machineName}
if [ "${machineName}" == "STAMPEDE2" ]   # Will allow me to run multiple jobs with different numThreadsPerRank without the fileName aliasing.
then
  fileName=${fileName}_${numThreadsPerRankMin}_${numThreadsPerRankMax}
fi

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
# Build separately for performance runs, critter runs, and profiling runs. To properly analyze, all 3 are necessary.
# Any one without the other two renders it meaningless.
if [ "${machineName}" == "STAMPEDE2" ]
then
  echo "Loading Intel MPI module"
  module load impi
fi
make -C./.. clean
export PROFTYPE=PERFORMANCE
make -C./.. cqr2_${mpiType}
if [ "${machineName}" != "PORTER" ]
then
  export PROFTYPE=PROFILE
  make -C./.. cqr2_${mpiType}
  export PROFTYPE=CRITTER
  make -C./.. cqr2_${mpiType}
fi
# For now, set profType=P, but tomorrow, note that it will be gone, as will anything that depends on it
profType=P


# Build CANDMC code
if [ "${machineName}" == "THETA" ] || [ "${machineName}" == "STAMPEDE2" ]
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
  export BINPATH=${SCRATCH}/${fileName}/bin/
elif [ "${machineName}" == "PORTER" ]
then
  export SCRATCH=../../../PAA_data
  export BINPATH=./../bin/
fi

# collectData.sh will always be a single line, just a necessary intermediate step.
echo "bash $SCRATCH/${fileName}/collectInstructions.sh | bash packageData.sh" > collectData.sh
# plotData.sh will always be a single line, just a necessary intermediate step.
echo "bash collectInstructions.sh | bash plotScript.sh" > plotData.sh

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
    echo "#!/bin/bash" > \${scriptName}
    echo "#PBS -l nodes=$numNodes:ppn=${ppn}:xe" >> \${scriptName}
    echo "#PBS -l walltime=${numHours}:${numMinutes}:${numSeconds}" >> \${scriptName}
    echo "#PBS -N ${numNodes}" >> \${scriptName}
    echo "#PBS -e \$PBS_JOBID.err" >> \${scriptName}
    echo "#PBS -o \$PBS_JOBID.out" >> \${scriptName}
    echo "##PBS -m Ed" >> \${scriptName}
    echo "##PBS -M hutter2@illinois.edu" >> \${scriptName}
    echo "##PBS -A xyz" >> \${scriptName}
    echo "#PBS -W umask=0027" >> \${scriptName}
    echo "cd \$PBS_O_WORKDIR" >> \${scriptName}
    echo "#module load craype-hugepages2M  perftools" >> \${scriptName}
    echo "#export APRUN_XFER_LIMITS=1  # to transfer shell limits to the executable" >> \${scriptName}
  elif [ "${machineName}" == "THETA" ]
  then
    echo "#!/bin/bash" > \${scriptName}
    echo "#COBALT -t ${numMinutes}" >> \${scriptName}
    echo "#COBALT -n \${curNumNodes}" >> \${scriptName}
    echo "#COBALT --attrs mcdram=cache:numa=quad" >> \${scriptName}
    echo "#COBALT -A QMCat" >> \${scriptName}
    echo "export n_nodes=\${curNumNodes}" >> \${scriptName}
    echo "export n_mpi_ranks_per_node=${ppn}" >> \${scriptName}
    echo "export n_mpi_ranks=\$((\${curNumNodes} * ${ppn}))" >> \${scriptName}
    read -p "Enter number of OpenMP threads per rank: " numOMPthreadsPerRank
    read -p "Enter number of hyperthreads per core: " numHyperThreadsPerCore
    read -p "Enter number of hyperthreads skipped per rank: " numHyperThreadsSkippedPerRank
    echo "export n_openmp_threads_per_rank=\${numOMPthreadsPerRank}" >> \${scriptName}
    echo "export n_hyperthreads_per_core=\${numHyperThreadsPerCore}" >> \${scriptName}
    echo "export n_hyperthreads_skipped_between_ranks=\${numHyperThreadsSkippedPerRank}" >> \${scriptName}
  elif [ "${machineName}" == "STAMPEDE2" ]
  then
    echo "#!/bin/bash" > \${scriptName}
    echo "#SBATCH -J myjob_\${curNumNodes}" >> \${scriptName}
    echo "#SBATCH -o myjob_\${curNumNodes}.o%j" >> \${scriptName}
    echo "#SBATCH -e myjob_\${curNumNodes}.e%j" >> \${scriptName}
    if [ \${curNumNodes} -le 256 ]
    then
      echo "#SBATCH -p normal" >> \${scriptName}
    else
      echo "#SBATCH -p large" >> \${scriptName}
    fi
    echo "#SBATCH -N \${curNumNodes}" >> \${scriptName}
    echo "#SBATCH -n \$((\${curNumNodes} * ${ppn}))" >> \${scriptName}
    echo "#SBATCH -t ${numHours}:${numMinutes}:${numSeconds}" >> \${scriptName}
    echo "export MKL_NUM_THREADS=${numThreadsPerRankMin}" >> \${scriptName}
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

# Only for cqr2
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

# Only for bench_scala_qr -- only necessary for Performance now. Might want to use Critter later, but not Profiler
writePlotFileNameScalapack() {
  if [ "${profType}" == "P" ]
  then
    echo "echo \"\${1}_NoFormQ.txt\"" >> \${2}
    if [ "\${3}" == "1" ]
    then
      echo "echo \"\${1}_NoFormQ_median.txt\"" >> \${2}
    fi
    echo "echo \"\${1}_FormQ.txt\"" >> \${2}
    if [ "\${3}" == "1" ]
    then
      echo "echo \"\${1}_FormQ_median.txt\"" >> \${2}
    fi
  fi
}

# Functions that write the actual script, depending on machine
launchJobs () {
  local numProcesses=\$((\${3} * $ppn))
  if [ "$machineName" == "BGQ" ]
  then
    echo "runjob --np \${numProcesses} -p ${ppn} --block \$COBALT_PARTNAME --verbose=INFO : \${@:4:\$#}" >> $SCRATCH/${fileName}/script\${3}.sh
  elif [ "$machineName" == "BW" ]
  then
    echo "Note: this is probably wrong, and I need to check this once I get BW access"
    echo "aprun -n \$numProcesses \$@" >> $SCRATCH/${fileName}/script\${2}.sh
  elif [ "$machineName" == "THETA" ]
  then
    echo "aprun -n \${numProcesses} -N ${ppn} --env OMP_NUM_THREADS=\${numOMPthreadsPerRank} -cc depth -d \${numHyperThreadsSkippedPerRank} -j \${numHyperThreadsPerCore} \${@:4:\$#}" >> $SCRATCH/${fileName}/script\${3}.sh
  elif [ "$machineName" == "STAMPEDE2" ]
  then
    echo "ibrun \${@:4:\$#}" >> $SCRATCH/${fileName}/script\${3}.sh
  elif [ "$machineName" == "PORTER" ]
  then
    if [ "${mpiType}" == "mpi" ]
    then
      mpiexec -n \$numProcesses \${@:4:\$#}
    elif [ "${mpiType}" == "ampi" ]
    then
      ${BINPATH}charmrun +p1 +vp\${numProcesses} \${@:4:\$#}
    fi
  fi
  if [ "\${1}" == "cqr2" ]
  then
    writePlotFileName \${@:2:1} $SCRATCH/${fileName}/collectInstructions.sh 0
  elif [ "\${1}" == "bench_scala_qr" ]
  then
    writePlotFileNameScalapack \${@:2:1} $SCRATCH/${fileName}/collectInstructions.sh 0
  fi
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
    launchJobs ${tag1} \${fileString} \$startNumNodes \${2} \${matrixDimM} \${9} 0 \${12} 0 \${startPdimD} \${11} \${3} $SCRATCH/${fileName}/\${fileString}
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
  local startNumNodes=\${4}
  local endNumNodes=\${5}
  local matrixDimM=\${8}
  local matrixDimN=\${9}
  local numProws=\${10}
  while [ \${startNumNodes} -le \${endNumNodes} ];
  do
    local fileString="results/results_${tag2}_\$1_\${startNumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${numProws}numProws_\${11}bSize"
    launchJobs ${tag2} \${fileString} \$startNumNodes \${2} \${matrixDimM} \${matrixDimN} \${11} \${3} 0 \${numProws} 1 0 $SCRATCH/${fileName}/\${fileString}
    startNumNodes=\$(updateCounter \${startNumNodes} \$7 \$6)
    numProws=\$(updateCounter \${numProws} \$7 \$6)
    if [ "\${1}" == "WS" ];
    then
      matrixDimM=\$(updateCounter \${matrixDimM} \${7} \${6})
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
echo "echo \"${fileName}\"" > $SCRATCH/${fileName}/collectInstructions.sh
echo "echo \"${machineName}\"" >> $SCRATCH/${fileName}/collectInstructions.sh

echo "echo \"${numTests}\"" >> $SCRATCH/${fileName}/collectInstructions.sh

for ((i=1; i<=${numTests}; i++))
do
  echo -e "\nTest #\${i}\n"
  read -p "Enter scaling type [SS,WS]: " scale
  read -p "Enter number of different configurations/binaries which will be used for this test: " numBinaries

  echo "echo \"\${numBinaries}\"" >> $SCRATCH/${fileName}/collectInstructions.sh

  # Echo for SCAPLOT makefile generator
  echo "echo \"\${scale}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
  echo "echo \"\${numBinaries}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

  read -p "Enter starting number of nodes for this test: " startNumNodes
  read -p "Enter ending number of nodes for this test: " endNumNodes
  
  # Assume for now that we always jump up by a power of 2
  #read -p "Enter factor by which to increase the number of nodes: " jumpNumNodes
  #read -p "Enter arithmetic operator by which to increase the number of nodes by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpNumNodesoperator
  jumpNumNodes=2
  jumpNumNodesoperator=3

  nodeCount=\$(findCountLength \${startNumNodes} \${endNumNodes} \${jumpNumNodesoperator} \${jumpNumNodes})
  echo "echo \"\${nodeCount}\" " >> $SCRATCH/${fileName}/plotInstructions.sh

  curNumNodes=\${startNumNodes}
  for ((j=0; j<\${nodeCount}; j++))
  do
    echo "echo \"\${curNumNodes}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
    curNumNodes=\$(( \${curNumNodes} * 2 ))
  done

  read -p "Enter matrix dimension m: " matrixDimM
  read -p "Enter matrix dimension n: " matrixDimN
  echo "echo \"\${matrixDimM}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
  echo "echo \"\${matrixDimN}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

  j=1
  while [ \${j} -le \${numBinaries} ];
  do
    echo -e "\nStage #\${j}"

    # Echo for SCAPLOT makefile generator
    read -p "Enter binary tag [0 for cqr2,1 for bench_scala_qr]: " binaryTagChoice

    binaryTag=""
    if [ \${binaryTagChoice} == 0 ]
    then
      binaryTag=cqr2
    else
      binaryTag=bench_scala_qr
    fi

    binaryPath=${BINPATH}\${binaryTag}_${machineName}
    if [ "${machineName}" == "PORTER" ]
    then
      binaryPath=\${binaryPath}_${mpiType}
    fi

    read -p "Enter number of iterations: " numIterations
    if [ \${binaryTag} == 'cqr2' ]
    then
      echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      
      read -p "Enter the inverseCutOff multiplier, 0 indicates that CFR3D will use the explicit inverse, 1 indicates that top recursive level will avoid calculating inverse, etc.: " inverseCutOffMult
      read -p "In this strong scaling test for CQR2, enter starting tunable processor grid dimension d: " pDimD
      read -p "In this strong scaling test for CQR2, enter static tunable processor grid dimension c: " pDimC
      
      # Write to plotInstructions file
      echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/collectInstructions.sh
      echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_perf\"" >> $SCRATCH/${fileName}/collectInstructions.sh
      echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_numerics\"" >> $SCRATCH/${fileName}/collectInstructions.sh
      echo "echo \"\$(findCountLength \${startNumNodes} \${endNumNodes} \${jumpNumNodesoperator} \${jumpNumNodes})\"" >> $SCRATCH/${fileName}/collectInstructions.sh
      # Write to plotInstructions file
      echo "echo \"\${pDimD}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${pDimC}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${inverseCutOffMult}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      writePlotFileName \${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC} $SCRATCH/${fileName}/plotInstructions.sh 1  
      launch\${binaryTag} \${scale} \${binaryPath}_PERFORMANCE \${numIterations} \${startNumNodes} \${endNumNodes} \${jumpNumNodes} \${jumpNumNodesoperator} \${matrixDimM} \${matrixDimN} \${pDimD} \${pDimC} \${inverseCutOffMult}
      j=\$(( \${j} + 1 ))
    elif [ \${binaryTag} == 'bench_scala_qr' ]
    then
      read -p "Enter the starting number of processor rows: " numProws
      read -p "Enter the minimum block size: " minBlockSize
      read -p "Enter the maximum block size: " maxBlockSize

      for ((k=\${minBlockSize}; k<=\${maxBlockSize}; k*=2))
      do
        # Write to plotInstructions file
        echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_\${k}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/collectInstructions.sh
        echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_\${k}_NoFormQ\"" >> $SCRATCH/${fileName}/collectInstructions.sh
        echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_\${k}_FormQ\"" >> $SCRATCH/${fileName}/collectInstructions.sh
        # This is where the last tricky part is: how many files do we need, because blockSize must be precomputed basically, and then multiplied by findCountLength
        # Write to plotInstructions file
        echo "echo \"\${numProws}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        echo "echo \"\${k}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        echo "echo \"\$(findCountLength \${startNumNodes} \${endNumNodes} \${jumpNumNodesoperator} \${jumpNumNodes})\"" >> $SCRATCH/${fileName}/collectInstructions.sh
        writePlotFileNameScalapack \${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_\${k} $SCRATCH/${fileName}/plotInstructions.sh 1
        launch\${binaryTag} \${scale} \${binaryPath} \${numIterations} \${startNumNodes} \${endNumNodes} \${jumpNumNodes} \${jumpNumNodesoperator} \${matrixDimM} \${matrixDimN} \${numProws} \${k}
        j=\$(( \${j} + 1 ))
      done
    fi
  done
done
EOF



bash $SCRATCH/${fileName}.sh
#rm $SCRATCH/${fileName}.sh

# Copy a local version to Scripts directory so that it can be used on the local side to generate plots.
# But its important that we keep a backup in SCRATCH/fileName in case we overwrite collectInstructions.sh, we can always write it back.
cp $SCRATCH/${fileName}/collectInstructions.sh collectInstructions.sh

# Note that for Porter, no need to do this, since we are submitting to a queue
if [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ] || [ "${machineName}" == "STAMPEDE2" ]
then
  mkdir $SCRATCH/${fileName}/bin
  mv ../bin/* $SCRATCH/${fileName}/bin
  #mv ${scalaDir}/bin/benchmarks/* $SCRATCH/${fileName}/bin  # move all scalapack benchmarks to same place before job is submitted
  cd $SCRATCH

  # Submit all scripts
  curNumNodes=${minNumNodes}
  while [ ${curNumNodes} -le ${maxNumNodes} ];
  do
    chmod +x ${fileName}/script${curNumNodes}.sh
    if [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ]
    then
      qsub ${fileName}/script${curNumNodes}.sh
    else
      echo "Dog"
      sbatch ${fileName}/script${curNumNodes}.sh
    fi
    curNumNodes=$(( ${curNumNodes} * 2 ))
  done
fi
