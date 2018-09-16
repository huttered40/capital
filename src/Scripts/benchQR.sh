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
if [ "$(hostname |grep "porter")" != "" ];
then
  machineName=PORTER
  scalaDir=~/hutter2/ExternalLibraries/CANDMC/CANDMC
  scaplotDir=~/hutter2/ExternalLibraries/SCAPLOT/scaplot
  read -p "Do you want to use MPI[mpi] or AMPI[ampi]? " mpiType
  if [ "${mpiType}" == "mpi" ];
  then
    export MPITYPE=MPI_TYPE
  elif [ "${mpiType}" == "ampi" ];
  then
    export MPITYPE=AMPI_TYPE
  fi
elif [ "$(hostname |grep "mira")" != "" ] || [ "$(hostname |grep "cetus")" != "" ];
then
  machineName=BGQ
  scalaDir=~/scratch/CANDMC
  export MPITYPE=MPI_TYPE
  mpiType=mpi
elif [ "$(hostname |grep "theta")" != "" ];
then
  machineName=THETA
  scalaDir=~/scratch/CANDMC
  export MPITYPE=MPI_TYPE
  mpiType=mpi
elif [ "$(hostname |grep "stampede2")" != "" ];
then
  machineName=STAMPEDE2
  scalaDir=~/CANDMC
  export MPITYPE=MPI_TYPE
  mpiType=mpi
elif [ "$(hostname |grep "h2o")" != "" ];
then
  machineName=BLUEWATERS
  scalaDir=~/scratch/CANDMC
  export MPITYPE=MPI_TYPE
  mpiType=mpi
fi

dateStr=$(date +%Y-%m-%d-%H_%M_%S)
read -p "Enter ID of auto-generated file this program will create: " fileID
read -p "Enter minimum number of nodes requested: " minNumNodes
read -p "Enter maximum number of nodes requested: " maxNumNodes
read -p "Enter Scaling regime:\
	[0 -> Weak scaling && scale nodes by 2, m by 2, n by 1, d by 2, c by 1
	 1 -> Weak scaling && scale nodes by 16, m by 4, n by 2, d by 4, c by 2
	 2 -> Strong scaling && scale nodes by 2, m by 1, n by 1, d by 2, c by 1]: " scaleRegime
read -p "Also enter factor to scale number of nodes: " nodeScaleFactor
read -p "Enter ppn: " ppn

# Default setting is 1
numThreadsPerRankMin=1
numThreadsPerRankMax=1
# For now, do not allow Blue Waters to try to thread the BLAS calls
if [ "${machineName}" == "STAMPEDE2" ];
then
  read -p "Enter minimum number of MKL threads per MPI rank: " numThreadsPerRankMin
  read -p "Enter maximum number of MKL threads per MPI rank: " numThreadsPerRankMax
fi

read -p "Enter number of tests (equal to number of strong scaling or weak scaling tests that will be run): " numTests

numHours=""
numMinutes=""
numSeconds=""
if [ "${machineName}" == "BLUEWATERS" ] || [ "${machineName}" == "STAMPEDE2" ];
then
  read -p "Enter number of hours of job: " numHours
  read -p "Enter number of minutes of job: " numMinutes
  read -p "Enter number of seconds of job: " numSeconds
elif [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ];
then
  read -p "Enter number of minutes of job: " numMinutes
fi

numPEs=$((ppn*numNodes))
fileName=benchQR${fileID}_${dateStr}_${machineName}
if [ "${machineName}" == "STAMPEDE2" ];   # Will allow me to run multiple jobs with different numThreadsPerRank without the fileName aliasing.
then
  fileName=${fileName}_${numThreadsPerRankMin}_${numThreadsPerRankMax}
fi

read -p "What datatype? float[0], double[1], complex<float>[2], complex<double>[3]: " dataType
read -p "What integer type? int[0], int64_t[1]: " intType
if [ ${dataType} == 0 ];
then
  export DATATYPE=FLOAT_TYPE
elif [ ${dataType} == 1 ];
then
  export DATATYPE=DOUBLE_TYPE
elif [ ${dataType} == 2 ];
then
  export DATATYPE=COMPLEX_FLOAT_TYPE
elif [ ${dataType} == 3 ];
then
  export DATATYPE=COMPLEX_DOUBLE_TYPE
fi
if [ ${intType} == 0 ];
then
  export INTTYPE=INT_TYPE
elif [ ${intType} == 1 ];
then
  export INTTYPE=INT64_T_TYPE
fi

# Build PAA code
# Build separately for performance runs, critter runs, and profiling runs. To properly analyze, all 3 are necessary.
# Any one without the other two renders it meaningless.

# Choice of compiler for Blue Waters (assumes Cray compiler is loaded by default)
if [ "${machineName}" == "BLUEWATERS" ];
then
  read -p "Do you want the Intel Programming Environment (I) or the Cray Programming Environment (C): " bwPrgEnv
  if [ "${bwPrgEnv}" == "I" ];
  then
    if [ "${PE_ENV}" == "CRAY" ];
    then
      module swap PrgEnv-cray PrgEnv-intel
    fi
  elif [ "${bwPrgEnv}" == "C" ];
  then
    if [ "${PE_ENV}" == "INTEL" ];
    then
      module swap PrgEnv-intel PrgEnv-cray
    fi
  fi
fi

read -p "Do you want to analyze these tests? Yes[1], No[0]: " analyzeDecision
if [ "${machineName}" == "STAMPEDE2" ];
then
  echo "Loading Intel MPI module"
  module load impi
  module unload gcc
  module load intel
fi
make -C./.. clean
export PROFTYPE=PERFORMANCE
make -C./.. cqr2_${mpiType}
profType=P
if [ ${analyzeDecision} == 1 ];
then
  profType=A					#  A for all 3 (performance, profiling, critical path analysis)
  export PROFTYPE=PROFILE
  make -C./.. cqr2_${mpiType}
  export PROFTYPE=CRITTER
  make -C./.. cqr2_${mpiType}
fi


# Build CANDMC code
if [ "${machineName}" == "THETA" ] || [ "${machineName}" == "STAMPEDE2" ] || [ "${machineName}" == "BLUEWATERS" ];
then
  if [ ${profType} == "P" ];
  then
    cd ${scalaDir}
    ./configure
    make clean
    make bench_scala_qr
    cd -
    mv ${scalaDir}/bin/benchmarks/bench_scala_qr ${scalaDir}/bin/benchmarks/bench_scala_qr_${machineName}
    mv ${scalaDir}/bin/benchmarks/* ../bin/
  fi
fi

if [ "${machineName}" == "BGQ" ];
then
  export SCRATCH=/projects/QMCat/huttered
elif [ "${machineName}" == "THETA" ];
then
  export SCRATCH=/projects/QMCat/huttered
  export BINPATH=${SCRATCH}/${fileName}/bin/
elif [ "${machineName}" == "STAMPEDE2" ];
then
  export BINPATH=${SCRATCH}/${fileName}/bin/
elif [ "${machineName}" == "BLUEWATERS" ];
then
  export SCRATCH=/scratch/sciteam/hutter
  export BINPATH=${SCRATCH}/${fileName}/bin/
elif [ "${machineName}" == "PORTER" ];
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
  if [ "${machineName}" == "BGQ" ];
  then
    scriptName=$SCRATCH/${fileName}/script\${curNumNodes}.sh
    echo "#!/bin/sh" > \${scriptName}
  elif [ "${machineName}" == "BLUEWATERS" ];
  then
    scriptName=$SCRATCH/${fileName}/script\${curNumNodes}.sh
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
  elif [ "${machineName}" == "THETA" ];
  then
    scriptName=$SCRATCH/${fileName}/script\${curNumNodes}.sh
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
  elif [ "${machineName}" == "STAMPEDE2" ];
  then
    curNumThreadsPerRank=${numThreadsPerRankMin}
    while [ \${curNumThreadsPerRank} -le ${numThreadsPerRankMax} ];
    do
      scriptName=$SCRATCH/${fileName}/script\${curNumNodes}_\${curNumThreadsPerRank}.sh
      echo "bash script name: \${scriptName}"
      echo "#!/bin/bash" > \${scriptName}
      echo "#SBATCH -J myjob_\${curNumNodes}_\${curNumThreadsPerRank}" >> \${scriptName}
      echo "#SBATCH -o myjob_\${curNumNodes}_\${curNumThreadsPerRank}.o%j" >> \${scriptName}
      echo "#SBATCH -e myjob_\${curNumNodes}_\${curNumThreadsPerRank}.e%j" >> \${scriptName}
      if [ \${curNumNodes} -le 256 ];
      then
        echo "#SBATCH -p normal" >> \${scriptName}
      else
        echo "#SBATCH -p large" >> \${scriptName}
      fi
      echo "#SBATCH -N \${curNumNodes}" >> \${scriptName}
      echo "#SBATCH -n \$((\${curNumNodes} * ${ppn}))" >> \${scriptName}
      echo "#SBATCH -t ${numHours}:${numMinutes}:${numSeconds}" >> \${scriptName}
      echo "export MKL_NUM_THREADS=\${curNumThreadsPerRank}" >> \${scriptName}
      curNumThreadsPerRank=\$(( \${curNumThreadsPerRank} * 2 ))
    done 
  fi
  curNumNodes=\$(( \${curNumNodes} * ${nodeScaleFactor} ))   # So far, only use cases for nodeScaleFactor are 2 and 16.
done

# Now I need to use a variable for the command-line prompt, since it will change based on the binary executable,
#   for example, scalapack QR has multiple special inputs that take up comm-line prompts that others dont
#   I want special functions in the inner-loop to handle this

updateCounter () {
  local counter=\${1}
  if [ \${2} -eq 1 ];
  then
    counter=\$((\${counter} + \${3})) 
  elif [ \${2} -eq 2 ];
  then
   counter=\$((\${counter} - \${3})) 
  elif [ \${2} -eq 3 ];
  then
    counter=\$((\${counter} * \${3})) 
  elif [ \${2} -eq 4 ];
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
  # Performance runs will always run, so no reason for an if-statement here
  echo "echo \"\${1}_perf.txt\"" >> \${2}
  if [ "\${3}" == "1" ];
  then
    echo "echo \"\${1}_perf_median.txt\"" >> \${2}
  fi
  
  echo "echo \"\${1}_numerics.txt\"" >> \${2}
  if [ "\${3}" == "1" ];
  then
    echo "echo \"\${1}_numerics_median.txt\"" >> \${2}
  fi

  if [ "${profType}" == "A" ];
  then
    echo "echo \"\${1}_critter.txt\"" >> \${2}
    if [ "\${3}" == "1" ];
    then
      echo "echo \"\${1}_critter_breakdown.txt\"" >> \${2}
    fi
    echo "echo \"\${1}_timer.txt\"" >> \${2}
  fi
}

# Only for bench_scala_qr -- only necessary for Performance now. Might want to use Critter later, but not Profiler
writePlotFileNameScalapack() {
  echo "echo \"\${1}_NoFormQ.txt\"" >> \${2}
  if [ "\${3}" == "1" ];
  then
    echo "echo \"\${1}_NoFormQ_median.txt\"" >> \${2}
  fi
  echo "echo \"\${1}_FormQ.txt\"" >> \${2}
  if [ "\${3}" == "1" ];
  then
    echo "echo \"\${1}_FormQ_median.txt\"" >> \${2}
  fi
}

# Functions that write the actual script, depending on machine
launchJobs () {
  local numProcesses=\$((\${3} * $ppn))
  if [ "$machineName" == "BGQ" ];
  then
    echo "runjob --np \${numProcesses} -p ${ppn} --block \$COBALT_PARTNAME --verbose=INFO : \${@:4:\$#}" >> $SCRATCH/${fileName}/script\${3}.sh
  elif [ "$machineName" == "BLUEWATERS" ];
  then
    # Assume (for now) that we want a process mapped to each Bulldozer core (1 per 2 integer cores)
    echo "aprun -n \${numProcesses} -d 2 \${@:5:\$#}" >> $SCRATCH/${fileName}/script\${2}.sh
  elif [ "$machineName" == "THETA" ];
  then
    echo "aprun -n \${numProcesses} -N ${ppn} --env OMP_NUM_THREADS=\${numOMPthreadsPerRank} -cc depth -d \${numHyperThreadsSkippedPerRank} -j \${numHyperThreadsPerCore} \${@:4:\$#}" >> $SCRATCH/${fileName}/script\${3}.sh
  elif [ "$machineName" == "STAMPEDE2" ];
  then
    echo "ibrun \${@:5:\$#}" >> $SCRATCH/${fileName}/script\${3}_\${4}.sh
  elif [ "$machineName" == "PORTER" ];
  then
    if [ "${mpiType}" == "mpi" ];
    then
      mpiexec -n \${numProcesses} \${@:5:\$#}
    elif [ "${mpiType}" == "ampi" ];
    then
      ${BINPATH}charmrun +p1 +vp\${numProcesses} \${@:5:\$#}
    fi
  fi
}

launch$tag1 () {
  # launch CQR2
  local startNumNodes=\${4}
  local endNumNodes=\${5}
  local matrixDimM=\${6}
  local matrixDimN=\${7}
  local startPdimD=\${8}
  local startPdimC=\${9}
  local bcDim=0
  while [ \$startNumNodes -le \$endNumNodes ];
  do
    local fileString="results/results_${tag1}_\${1}_\${startNumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${10}inverseCutOffMult_0bcMult_0panelDimMult_\${startPdimD}pDimD_\${startPdimC}pDimC_\${11}tpk"
    # Launch performance job always.
    launchJobs ${tag1} \${fileString} \$startNumNodes \${11} \${2}_PERFORMANCE \${matrixDimM} \${matrixDimN} \${bcDim} \${10} 0 \${startPdimD} \${startPdimC} \${3} $SCRATCH/${fileName}/\${fileString}

    # If analysis is turned on, launch Profiling job and Critter job.
    if [ "${profType}" == "A" ];
    then
      launchJobs ${tag1} \${fileString} \$startNumNodes \${11} \${2}_CRITTER \${matrixDimM} \${matrixDimN} \${bcDim} \${10} 0 \${startPdimD} \${startPdimC} \${3} $SCRATCH/${fileName}/\${fileString}
      launchJobs ${tag1} \${fileString} \$startNumNodes \${11} \${2}_PROFILE \${matrixDimM} \${matrixDimN} \${bcDim} \${10} 0 \${startPdimD} \${startPdimC} \${3} $SCRATCH/${fileName}/\${fileString}
    fi

    writePlotFileName \${fileString} $SCRATCH/${fileName}/collectInstructions.sh 0

    # Rest of scaling decisions are made based on scaleRegime
    if [ ${scaleRegime} == 0 ];
    then
      startPdimD=\$(( \${startPdimD} * 2 ))
      matrixDimM=\$(( \${matrixDimM} * 2 ))
    elif [ ${scaleRegime} == 1 ];
    then
      startPdimD=\$(( \${startPdimD} * 4 ))
      startPdimC=\$(( \${startPdimC} * 2 ))
      matrixDimM=\$(( \${matrixDimM} * 4 ))
      matrixDimN=\$(( \${matrixDimN} * 2 ))
    elif [ ${scaleRegime} == 2 ];
    then
      startPdimD=\$(( \${startPdimD} * 2 ))
    fi
    startNumNodes=\$((\${startNumNodes} * ${nodeScaleFactor} ))
  done
}

launch$tag2 () {
  # launch scaLAPACK_QR
  local startNumNodes=\${4}
  local endNumNodes=\${5}
  local matrixDimM=\${6}
  local matrixDimN=\${7}
  local numProws=\${8}
  while [ \${startNumNodes} -le \${endNumNodes} ];
  do
    local fileString="results/results_${tag2}_\$1_\${startNumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${numProws}numProws_\${9}bSize_\${10}tpk"
    launchJobs ${tag2} \${fileString} \$startNumNodes \${10} \${2} \${matrixDimM} \${matrixDimN} \${9} \${3} 0 \${numProws} 1 0 $SCRATCH/${fileName}/\${fileString}

    writePlotFileNameScalapack \${fileString} $SCRATCH/${fileName}/collectInstructions.sh 0

    # Rest of scaling decisions are made based on scaleRegime
    if [ ${scaleRegime} == 0 ];
    then
      numProws=\$(( \${numProws} * 2 ))
      matrixDimM=\$(( \${matrixDimM} * 2 ))
    elif [ ${scaleRegime} == 1 ];
    then
      numPEs=\$(( \${startNumNodes} * ${ppn} ))
      numPcols=\$(( \${numPEs} / \${numProws} ))
      num=\$(( \${matrixDimM} / \${numProws} ))
      denom=\$(( \${matrixDimN} / \${numPcols} ))

      if [ \${num} -gt \${denom} ];
      then
        numProws=\$(( \${numProws} * 8 ))
        matrixDimM=\$(( \${matrixDimM} * 4 ))
        matrixDimN=\$(( \${matrixDimN} * 2 ))
      else
        numProws=\$(( \${numProws} * 4 ))
        matrixDimM=\$(( \${matrixDimM} * 4 ))
        matrixDimN=\$(( \${matrixDimN} * 2 ))
      fi
    elif [ ${scaleRegime} == 2 ];
    then
      numProws=\$(( \${numProws} * 2 ))
    fi
    startNumNodes=\$(( \${startNumNodes} * ${nodeScaleFactor} ))
  done
}


# Note: in future, I may want to decouple numBinaries and numPlotTargets, but only when I find it necessary
# Write to Plot Instructions file, for use by SCAPLOT makefile generator
echo "echo \"1\"" > $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${fileName}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${numTests}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${machineName}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${profType}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${scaleRegime}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${nodeScaleFactor}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${ppn}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

# Echo for data collection from remote machine (not porter) to PAA/src/Results
# This temporary file will be deleted while collectScript.sh is called.
echo "echo \"${fileName}\"" > $SCRATCH/${fileName}/collectInstructions.sh
echo "echo \"${machineName}\"" >> $SCRATCH/${fileName}/collectInstructions.sh
echo "echo \"${profType}\"" >> $SCRATCH/${fileName}/collectInstructions.sh
echo "echo \"${scaleRegime}\"" >> $SCRATCH/${fileName}/collectInstructions.sh
echo "echo \"${nodeScaleFactor}\"" >> $SCRATCH/${fileName}/collectInstructions.sh

echo "echo \"${numTests}\"" >> $SCRATCH/${fileName}/collectInstructions.sh

for ((i=1; i<=${numTests}; i++))
do
  echo -e "\nTest #\${i}\n"
  read -p "Enter scaling type [SS,WS]: " scale
  read -p "Enter number of different configurations/binaries which will be used for this test: " numBinaries

  # Echo for SCAPLOT makefile generator
  echo "echo \"\${scale}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

  # Nodes
  read -p "Enter starting number of nodes for this test: " startNumNodes
  read -p "Enter ending number of nodes for this test: " endNumNodes

  nodeCount=\$(findCountLength \${startNumNodes} \${endNumNodes} 3 ${nodeScaleFactor})
  echo "echo \"\${nodeCount}\" " >> $SCRATCH/${fileName}/plotInstructions.sh

  curNumNodes=\${startNumNodes}
  for ((j=0; j<\${nodeCount}; j++))
  do
    echo "echo \"\${curNumNodes}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
    curNumNodes=\$(( \${curNumNodes} * ${nodeScaleFactor} ))
  done

  # Threads
  startNumTPR=1
  endNumTPR=1
  if [ "${machineName}" == "STAMPEDE2" ];
  then
    read -p "Enter starting number of threads-per-rank for this test: " startNumTPR
    read -p "Enter ending number of threads-per-rank for this test: " endNumTPR
  fi
  # Assume for now that we always jump up by a power of 2
  TPRcount=\$(findCountLength \${startNumTPR} \${endNumTPR} 3 2)

  totalNumConfigs=\$((\${TPRcount} * \${numBinaries} ))
  echo "echo \"\${totalNumConfigs}\"" >> $SCRATCH/${fileName}/collectInstructions.sh
  echo "echo \"\${totalNumConfigs}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

  read -p "Enter matrix dimension m: " matrixDimM
  read -p "Enter matrix dimension n: " matrixDimN

  j=1
  while [ \${j} -le \${numBinaries} ];
  do
    echo -e "\nStage #\${j}"

    # Echo for SCAPLOT makefile generator
    read -p "Enter binary tag [0 for cqr2,1 for bench_scala_qr]: " binaryTagChoice

    binaryTag=""
    if [ \${binaryTagChoice} == 0 ];
    then
      binaryTag=cqr2
    else
      binaryTag=bench_scala_qr
    fi

    binaryPath=${BINPATH}\${binaryTag}_${machineName}
    if [ "${machineName}" == "PORTER" ];
    then
      binaryPath=\${binaryPath}_${mpiType}
    fi

    read -p "Enter number of iterations: " numIterations
    if [ \${binaryTag} == 'cqr2' ];
    then
      read -p "Enter the inverseCutOff multiplier, 0 indicates that CFR3D will use the explicit inverse, 1 indicates that top recursive level will avoid calculating inverse, etc.: " inverseCutOffMult
      read -p "Enter starting tunable processor grid dimension d: " pDimD
      read -p "Enter static tunable processor grid dimension c: " pDimC

      curNumThreadsPerRank=${numThreadsPerRankMin}
      while [ \${curNumThreadsPerRank} -le ${numThreadsPerRankMax} ];
      do
        # Write to plotInstructions file
        echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        
        # Special thing in order to allow MakePlotScript.sh to work with both CQR2 and CFR3D. Only print on 1st iteration
        if [ \${j} == 1 ] && [ \${curNumThreadsPerRank} == ${numThreadsPerRankMin} ];
        then
          echo "echo \"\${matrixDimM}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
          echo "echo \"\${matrixDimN}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi

        echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_\${curNumThreadsPerRank}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        # Write to collectInstructions file
        echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/collectInstructions.sh
        echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_\${curNumThreadsPerRank}_perf\"" >> $SCRATCH/${fileName}/collectInstructions.sh
        echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_\${curNumThreadsPerRank}_numerics\"" >> $SCRATCH/${fileName}/collectInstructions.sh

        if [ "${profType}" == "A" ];
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_\${curNumThreadsPerRank}_critter\"" >> $SCRATCH/${fileName}/collectInstructions.sh
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_\${curNumThreadsPerRank}_timer\"" >> $SCRATCH/${fileName}/collectInstructions.sh
	fi
        
	echo "echo \"\$(findCountLength \${startNumNodes} \${endNumNodes} 3 ${nodeScaleFactor})\"" >> $SCRATCH/${fileName}/collectInstructions.sh
        # Write to plotInstructions file
        echo "echo \"\${pDimD}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        echo "echo \"\${pDimC}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        echo "echo \"\${inverseCutOffMult}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        echo "echo \"\${curNumThreadsPerRank}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        writePlotFileName \${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}_\${curNumThreadsPerRank} $SCRATCH/${fileName}/plotInstructions.sh 1  
        launch\${binaryTag} \${scale} \${binaryPath} \${numIterations} \${startNumNodes} \${endNumNodes} \${matrixDimM} \${matrixDimN} \${pDimD} \${pDimC} \${inverseCutOffMult} \${curNumThreadsPerRank}
        curNumThreadsPerRank=\$(( \${curNumThreadsPerRank} * 2 ))
      done
      j=\$(( \${j} + 1 ))
    elif [ \${binaryTag} == 'bench_scala_qr' ];
    then
      read -p "Enter the starting number of processor rows: " numProws
      read -p "Enter the minimum block size: " minBlockSize
      read -p "Enter the maximum block size: " maxBlockSize

      for ((k=\${minBlockSize}; k<=\${maxBlockSize}; k*=2))
      do
        curNumThreadsPerRank=${numThreadsPerRankMin}
        while [ \${curNumThreadsPerRank} -le ${numThreadsPerRankMax} ];
        do
          # Write to plotInstructions file
          echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        
          # Special thing in order to allow MakePlotScript.sh to work with both CQR2 and CFR3D. Only print on 1st iteration
          if [ \${j} == 1 ] && [ \${k} == \${minBlockSize} ] && [ \${curNumThreadsPerRank} == ${numThreadsPerRankMin} ];
          then
            echo "echo \"\${matrixDimM}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
            echo "echo \"\${matrixDimN}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
          fi

          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_\${k}_\${curNumThreadsPerRank}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
          echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/collectInstructions.sh
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_\${k}_\${curNumThreadsPerRank}_NoFormQ\"" >> $SCRATCH/${fileName}/collectInstructions.sh
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_\${k}_\${curNumThreadsPerRank}_FormQ\"" >> $SCRATCH/${fileName}/collectInstructions.sh
          # This is where the last tricky part is: how many files do we need, because blockSize must be precomputed basically, and then multiplied by findCountLength
          # Write to plotInstructions file
          echo "echo \"\${numProws}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
          echo "echo \"\${k}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
          echo "echo \"\${curNumThreadsPerRank}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
          echo "echo \"\$(findCountLength \${startNumNodes} \${endNumNodes} 3 ${nodeScaleFactor})\"" >> $SCRATCH/${fileName}/collectInstructions.sh
          writePlotFileNameScalapack \${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}_\${k}_\${curNumThreadsPerRank} $SCRATCH/${fileName}/plotInstructions.sh 1
          launch\${binaryTag} \${scale} \${binaryPath} \${numIterations} \${startNumNodes} \${endNumNodes} \${matrixDimM} \${matrixDimN} \${numProws} \${k} \${curNumThreadsPerRank}
          curNumThreadsPerRank=\$(( \${curNumThreadsPerRank} * 2 ))
        done
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
if [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ] || [ "${machineName}" == "STAMPEDE2" ] || [ "${machineName}" == "BLUEWATERS" ];
then
  mkdir $SCRATCH/${fileName}/bin
  mv ../bin/* $SCRATCH/${fileName}/bin
  #mv ${scalaDir}/bin/benchmarks/* $SCRATCH/${fileName}/bin  # move all scalapack benchmarks to same place before job is submitted
  cd $SCRATCH

  # Submit all scripts
  curNumNodes=${minNumNodes}
  while [ ${curNumNodes} -le ${maxNumNodes} ];
  do
    curNumThreadsPerRank=${numThreadsPerRankMin}
    while [ ${curNumThreadsPerRank} -le ${numThreadsPerRankMax} ];
    do
      chmod +x ${fileName}/script${curNumNodes}_${curNumThreadsPerRank}.sh
      if [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ] || [ "${machineName}" == "BLUEWATERS" ];
      then
        qsub ${fileName}/script${curNumNodes}.sh
      else
        sbatch ${fileName}/script${curNumNodes}_${curNumThreadsPerRank}.sh
      fi
      curNumThreadsPerRank=$(( ${curNumThreadsPerRank} * 2 ))
    done
    curNumNodes=$(( ${curNumNodes} * ${nodeScaleFactor} ))
  done
fi
