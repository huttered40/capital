#!/bin/bash

tag1='cqr2'
tag2='bench_scala_qr'
tag3='cfr3d'
tag4='bench_scala_cholesky'

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
  scalaDir=~/CANDMC
  export MPITYPE=MPI_TYPE
  mpiType=mpi
fi

dateStr=$(date +%Y-%m-%d-%H_%M_%S)
read -p "Enter ID of auto-generated file this program will create: " fileID
read -p "What round is this? " roundID
read -p "Enter minimum number of nodes requested: " minNumNodes
read -p "Enter maximum number of nodes requested: " maxNumNodes
read -p "Also enter factor to scale number of nodes: " nodeScaleFactor
read -p "Enter number of launches per binary: " NumLaunchesPerBinary

ppnMinList=()
ppnMaxList=()
tprMinList=()
tprMaxList=()

curNumNodes=${minNumNodes}
while [ ${curNumNodes} -le ${maxNumNodes} ];
do
  read -p "Enter min ppn for node count ${curNumNodes}: " ppnMin
  read -p "Enter max ppn for node count ${curNumNodes}: " ppnMax
  read -p "Enter min tpr for node count ${curNumNodes}: " tprMin
  read -p "Enter max tpr for node count ${curNumNodes}: " tprMax

  ppnMinList+=(${ppnMin})
  ppnMaxList+=(${ppnMax})
  tprMinList+=(${tprMin})
  tprMaxList+=(${tprMax})

  curNumNodes=$(( ${curNumNodes} * ${nodeScaleFactor} ))   # So far, only use cases for nodeScaleFactor are 2 and 16.
done

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

fileName=benchQR_launch${fileID}_${dateStr}_${machineName}_round${roundID}
fileNameToProcess=benchQR_launch${fileID}_${machineName}	# Name of the corresponding directory in PAA_data. Allows for appending multiple runs

# Below: might delete. Not sure if this is really necessary, but don't currently want to delete it before I'm sure.
#if [ "${machineName}" == "STAMPEDE2" ];   # Will allow me to run multiple jobs with different numThreadsPerRank without the fileName aliasing.
#then
#  fileName=${fileName}_${numThreadsPerRankMin}_${numThreadsPerRankMax}
#fi

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
  read -p "Do you want the Intel Programming Environment (I) or the GNU Programming Environment (G): " bwPrgEnv
  if [ "${bwPrgEnv}" == "I" ];
  then
    if [ "${PE_ENV}" == "GNU" ];
    then
      module swap PrgEnv-gnu PrgEnv-intel
      module load cblas
    elif [ "${PE_ENV}" == "CRAY" ];
    then
      module swap PrgEnv-cray PrgEnv-intel
      module load cblas
    elif [ "${PE_ENV}" == "INTEL" ];
    then
      module load cblas
    fi
  elif [ "${bwPrgEnv}" == "G" ];
  then
    if [ "${PE_ENV}" == "INTEL" ];
    then
      module swap PrgEnv-intel PrgEnv-gnu
      module load cblas
    elif [ "${PE_ENV}" == "CRAY" ];
    then
      module swap PrgEnv-cray PrgEnv-gnu
      module load cblas
    elif [ "${PE_ENV}" == "GNU" ];
    then
      module load cblas
    fi
  fi
fi

read -p "Do you want to analyze these tests with Critter? Yes[1], No[0]: " analyzeDecision1
read -p "Do you want to analyze these tests with TAU? Yes[1], No[0]: " analyzeDecision2
make -C./.. clean
export PROFTYPE=PERFORMANCE
make -C./.. cqr2_${mpiType}
profType=P
if [ ${analyzeDecision1} == 1 ];
then
  profType=${profType}C
  export PROFTYPE=CRITTER
  make -C./.. cqr2_${mpiType}
fi
if [ ${analyzeDecision2} == 1 ];
then
  profType=${profType}T
  export PROFTYPE=PROFILE
  make -C./.. cqr2_${mpiType}
fi


# Build CANDMC code (only if testing performance, not for profiling)
if [ "${profType}" == "P" ];
then
  if [ "${machineName}" == "THETA" ] || [ "${machineName}" == "STAMPEDE2" ] || [ "${machineName}" == "BLUEWATERS" ];
  then
    # ScaLAPACK should now work for both analyzing (critter only) and performance
    cd ${scalaDir}
    make clean
    rm config.mk
    export PROFTYPE=PERFORMANCE
    profType=P
    ./configure
    make bench_scala_qr
    cd -
    mv ${scalaDir}/bin/benchmarks/bench_scala_qr ${scalaDir}/bin/benchmarks/bench_scala_qr_${machineName}_${PROFTYPE}
    mv ${scalaDir}/bin/benchmarks/bench_scala_qr_${machineName}_${PROFTYPE} ../bin/
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
echo "bash $SCRATCH/${fileName}/collectInstructionsStage1.sh | bash packageDataStage1.sh" > collectDataStage1.sh
# plotData.sh will always be a single line, just a necessary intermediate step.
echo "bash collectInstructionsStage1.sh | bash plotScript.sh" > plotData.sh	# plotScript only needs the 2nd and 3rd items in collectInstructionsStage1.sh, so overwritting it doesn't matter.

cat <<-EOF > $SCRATCH/${fileName}.sh
scriptName=$SCRATCH/${fileName}/script.sh
mkdir $SCRATCH/${fileName}/
mkdir $SCRATCH/${fileName}/DataFiles

# Need to re-build ppn/tpr lists (for each node count) because I cannot access the pre-time list with run-time indices
ppnMinListRunTime=()
ppnMaxListRunTime=()
tprMinListRunTime=()
tprMaxListRunTime=()

curNumNodes=${minNumNodes}
while [ \${curNumNodes} -le ${maxNumNodes} ];
do
  read -p "Enter min ppn for node count \${curNumNodes}: " ppnMin
  read -p "Enter max ppn for node count \${curNumNodes}: " ppnMax
  read -p "Enter min tpr for node count \${curNumNodes}: " tprMin
  read -p "Enter max tpr for node count \${curNumNodes}: " tprMax

  ppnMinListRunTime+=(\${ppnMin})
  ppnMaxListRunTime+=(\${ppnMax})
  tprMinListRunTime+=(\${tprMin})
  tprMaxListRunTime+=(\${tprMax})

  curNumNodes=\$(( \${curNumNodes} * ${nodeScaleFactor} ))   # So far, only use cases for nodeScaleFactor are 2 and 16.
done

curLaunchID=1
while [ \${curLaunchID} -le ${NumLaunchesPerBinary} ];
do
  # Loop over all scripts - log(P) of them
  curNumNodes=${minNumNodes}
  ppnIndex=0
  while [ \${curNumNodes} -le ${maxNumNodes} ];
  do
    minPPN=\${ppnMinListRunTime[\${ppnIndex}]}
    maxPPN=\${ppnMaxListRunTime[\${ppnIndex}]}
    curPPN=\${minPPN}
    tprIndex=0
    while [ \${curPPN} -le \${maxPPN} ];
    do
      minTPR=\${tprMinListRunTime[\${tprIndex}]}
      maxTPR=\${tprMaxListRunTime[\${tprIndex}]}
      curTPR=\${minTPR}
      while [ \${curTPR} -le \${maxTPR} ];
      do
        scriptName=$SCRATCH/${fileName}/script_${fileID}id_${roundID}round_\${curLaunchID}launchID_\${curNumNodes}nodes_\${curPPN}ppn_\${curTPR}tpr.sh
        if [ "${machineName}" == "BGQ" ];
        then
          echo "#!/bin/sh" > \${scriptName}
        elif [ "${machineName}" == "BLUEWATERS" ];
        then
          echo "#!/bin/bash" > \${scriptName}
          echo "#PBS -l nodes=\${curNumNodes}:ppn=\${curPPN}:xe" >> \${scriptName}
          echo "#PBS -l walltime=${numHours}:${numMinutes}:${numSeconds}" >> \${scriptName}
          echo "#PBS -N testjob" >> \${scriptName}
          echo "#PBS -e ${fileName}_\${curNumNodes}_\${curTPR}.err" >> \${scriptName}
          echo "#PBS -o ${fileName}_\${curNumNodes}_\${curTPR}.out" >> \${scriptName}
          echo "##PBS -m Ed" >> \${scriptName}
          echo "##PBS -M hutter2@illinois.edu" >> \${scriptName}
          echo "##PBS -A xyz" >> \${scriptName}
          echo "#PBS -W umask=0027" >> \${scriptName}
#          echo "cd \${PBS_O_WORKDIR}" >> \${scriptName}
          echo "#module load craype-hugepages2M  perftools" >> \${scriptName}
          echo "#export APRUN_XFER_LIMITS=1  # to transfer shell limits to the executable" >> \${scriptName}
        elif [ "${machineName}" == "THETA" ];
        then
          echo "#!/bin/bash" > \${scriptName}
          echo "#COBALT -t ${numMinutes}" >> \${scriptName}
          echo "#COBALT -n \${curNumNodes}" >> \${scriptName}
          echo "#COBALT --attrs mcdram=cache:numa=quad" >> \${scriptName}
          echo "#COBALT -A QMCat" >> \${scriptName}
          echo "export n_nodes=\${curNumNodes}" >> \${scriptName}
          echo "export n_mpi_ranks_per_node=\${curPPN}" >> \${scriptName}
          echo "export n_mpi_ranks=\$((\${curNumNodes} * \${curPPN}))" >> \${scriptName}
          read -p "Enter number of OpenMP threads per rank: " numOMPthreadsPerRank
          read -p "Enter number of hyperthreads per core: " numHyperThreadsPerCore
          read -p "Enter number of hyperthreads skipped per rank: " numHyperThreadsSkippedPerRank
          echo "export n_openmp_threads_per_rank=\${numOMPthreadsPerRank}" >> \${scriptName}
          echo "export n_hyperthreads_per_core=\${numHyperThreadsPerCore}" >> \${scriptName}
          echo "export n_hyperthreads_skipped_between_ranks=\${numHyperThreadsSkippedPerRank}" >> \${scriptName}
        elif [ "${machineName}" == "STAMPEDE2" ];
        then
          echo "bash script name: \${scriptName}"
          echo "#!/bin/bash" > \${scriptName}
          echo "#SBATCH -J myjob_${fileID}id_${roundID}round_\${curNumNodes}nodes_\${curPPN}ppn_\${curTPR}tpr" >> \${scriptName}
          echo "#SBATCH -o myjob_${fileID}id_${roundID}round_\${curNumNodes}nodes_\${curPPN}ppn_\${curTPR}tpr.o%j" >> \${scriptName}
          echo "#SBATCH -e myjob_${fileID}id_${roundID}round_\${curNumNodes}nodes_\${curPPN}ppn_\${curTPR}tpr.e%j" >> \${scriptName}
          if [ \${curNumNodes} -le 256 ];
          then
            echo "#SBATCH -p normal" >> \${scriptName}
          else
            echo "#SBATCH -p large" >> \${scriptName}
          fi
          echo "#SBATCH -N \${curNumNodes}" >> \${scriptName}
          echo "#SBATCH -n \$((\${curNumNodes} * \${curPPN}))" >> \${scriptName}
          echo "#SBATCH -t ${numHours}:${numMinutes}:${numSeconds}" >> \${scriptName}
          echo "export MKL_NUM_THREADS=\${curTPR}" >> \${scriptName}
        fi
        curTPR=\$(( \${curTPR} * 2 ))
      done
      curPPN=\$(( \${curPPN} * 2 ))
      tprIndex=\$(( \${tprIndex} + 1 ))
    done
    curNumNodes=\$(( \${curNumNodes} * ${nodeScaleFactor} ))   # So far, only use cases for nodeScaleFactor are 2 and 16.
    ppnIndex=\$(( \${ppnIndex} + 1 ))
  done
  curLaunchID=\$(( \${curLaunchID} + 1 ))
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


# Writes the beginning of collectInstructionsStage1 and collectInstructionsStage2
WriteHeaderForCollection () {
  echo "echo \"${fileName}\"" > \${1}
  echo "echo \"${fileNameToProcess}\"" >> \${1}
  echo "echo \"${machineName}\"" >> \${1}
  echo "echo \"${profType}\"" >> \${1}
  echo "echo \"${nodeScaleFactor}\"" >> \${1}
  echo "echo \"${numTests}\"" >> \${1}
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

  if [ "${profType}" == "PC" ] || [ "${profType}" == "PCT" ];
  then
    echo "echo \"\${1}_critter.txt\"" >> \${2}
    if [ "\${3}" == "1" ];
    then
      echo "echo \"\${1}_critter_breakdown.txt\"" >> \${2}
    fi
  fi
  if [ "${profType}" == "PT" ] || [ "${profType}" == "PCT" ];
  then
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
  local launchID=\${3}
  local numNodes=\${4}
  local ppn=\${5}
  local tpr=\${6}
  local numProcesses=\$((\${numNodes} * \${ppn}))
  local scriptName=script_${fileID}id_${roundID}round_\${launchID}launchID_\${numNodes}nodes_\${ppn}ppn_\${tpr}tpr
  if [ "$machineName" == "BGQ" ];
  then
    echo "runjob --np \${numProcesses} -p \${ppn} --block \$COBALT_PARTNAME --verbose=INFO : \${@:7:\$#}" >> $SCRATCH/${fileName}/\${scriptName}.sh
  elif [ "$machineName" == "BLUEWATERS" ];
  then
    # Assume (for now) that we want a process mapped to each Bulldozer core (1 per 2 integer cores)
    echo "aprun -n \${numProcesses} -N \${ppn} -d 2 \${@:7:\$#}" >> $SCRATCH/${fileName}/\${scriptName}.pbs
  elif [ "$machineName" == "THETA" ];
  then
    echo "aprun -n \${numProcesses} -N \${ppn} --env OMP_NUM_THREADS=\${numOMPthreadsPerRank} -cc depth -d \${numHyperThreadsSkippedPerRank} -j \${numHyperThreadsPerCore} \${@:7:\$#}" >> $SCRATCH/${fileName}/\${scriptName}.sh
  elif [ "$machineName" == "STAMPEDE2" ];
  then
    echo "ibrun \${@:7:\$#}" >> $SCRATCH/${fileName}/\${scriptName}.sh
  elif [ "$machineName" == "PORTER" ];
  then
    if [ "${mpiType}" == "mpi" ];
    then
      mpiexec -n \${numProcesses} \${@:7:\$#}
    elif [ "${mpiType}" == "ampi" ];
    then
      ${BINPATH}charmrun +p1 +vp\${numProcesses} \${@:7:\$#}
    fi
  fi
}


WriteMethodDataForPlotting () {
  for arg in "\${@}"
  do
    echo "echo \"\${arg}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
  done
}

WriteMethodDataForCollectingStage1 () {
  local MethodTag=\${1}
  local FileNameBase=\${2}
  local FileName1=\${3}
  local FileName2=\${4}
  local WriteFile=\${5}

  echo "echo \"0\"" >> \${WriteFile}
  echo "echo \"\${MethodTag}\"" >> \${WriteFile}
  echo "echo \"\${FileName1}\"" >> \${WriteFile}
  if [ "\${MethodTag}" != "bench_scala_cholesky" ];
  then
    echo "echo \"\${FileName2}\"" >> \${WriteFile}
  fi
  if [ "${profType}" == "PC" ] || [ "${profType}" == "PCT" ];
  then
    echo "echo \"\${FileNameBase}_critter\"" >> \${WriteFile}
  fi
  if [ "${profType}" == "PT" ] || [ "${profType}" == "PCT" ];
  then
    echo "echo \"\${FileNameBase}_timer\"" >> \${WriteFile}
  fi
}

WriteMethodDataForCollectingStage2 () {
  # Because 'Pre' (Stage1) collapses the NumLaunchesPerBinary, we do not want to overcount.
  local launchID=\${1}
  if [ \${launchID} -eq 1 ];
  then
    local MethodTag=\${2}
    local PreFileNameBase=\${3}
    local PreFileName1=\${4}
    local PreFileName2=\${5}
    local PostFileNameBase=\${6}
    local PostFileName1=\${7}
    local PostFileName2=\${8}
    local WriteFile=\${9}

    echo "echo \"0\"" >> \${WriteFile}
    echo "echo \"\${MethodTag}\"" >> \${WriteFile}
    echo "echo \"\${PostFileName1}\"" >> \${WriteFile}
    if [ "\${MethodTag}" != "bench_scala_cholesky" ];
    then
      echo "echo \"\${PostFileName2}\"" >> \${WriteFile}
    fi
    if [ "${profType}" == "PC" ] || [ "${profType}" == "PCT" ];
    then
      echo "echo \"\${PostFileNameBase}_critter\"" >> \${WriteFile}
    fi
    if [ "${profType}" == "PT" ] || [ "${profType}" == "PCT" ];
    then
      echo "echo \"\${PostFileNameBase}_timer\"" >> \${WriteFile}
    fi
    echo "echo \"\${PreFileName1}\"" >> \${WriteFile}
    if [ "\${MethodTag}" != "bench_scala_cholesky" ];
    then
      echo "echo \"\${PreFileName2}\"" >> \${WriteFile}
    fi
    if [ "${profType}" == "PC" ] || [ "${profType}" == "PCT" ];
    then
      echo "echo \"\${PreFileNameBase}_critter\"" >> \${WriteFile}
    fi
    if [ "${profType}" == "PT" ] || [ "${profType}" == "PCT" ];
    then
      echo "echo \"\${PreFileNameBase}_timer\"" >> \${WriteFile}
    fi
  fi
}


launchJobsPortal () {
  # Launch performance job always.
  launchJobs \${@:2:\${#}}

  # If analysis is turned on, launch Profiling job and Critter job.
  if [ "${profType}" == "PC" ] || [ "${profType}" == "PCT" ];
  then
    launchJobs \${@:2:\${7}} \${1}_CRITTER \${@:9:\${#}}
  fi
  if [ "${profType}" == "PT" ] || [ "${profType}" == "PCT" ];
  then
    launchJobs \${@:2:\${7}} \${1}_TIMER \${@:9:\${#}}
  fi
}


###################################################### Method Launches ######################################################

# For CA-CQR2
launch$tag1 () {
  # launch CQR2
  local scale=\${1}
  local binaryPath=\${2}
  local numIterations=\${3}
  local launchID=\${4}
  local NumNodes=\${5}
  local ppn=\${6}
  local tpr=\${7}
  local matrixDimM=\${8}
  local matrixDimN=\${9}
  local pDimDorig=\${10}
  local pDimCorig=\${11}
  local pDimD=\${12}
  local pDimC=\${13}
  local nodeIndex=\${14}
  local scaleRegime=\${15}
  local nodeCount=\${16}
  local bcDim=0

  # Next: Based on pDimC, decide on invCutOff parameter, which will range from 0 to a max of 2 for now
  invCutOffLoopMax=0
  if [ \${pDimC} -le 2 ];
  then
    invCutOffLoopMax=0
  elif [ \${pDimC} -eq 4 ];
  then
    invCutOffLoopMax=1
  else
    invCutOffLoopMax=2
    #invCutOffLoopMax=\$(( \${pDimC} / 2 ))
    #invCutOffLoopMax=\$(( \${invCutOffLoopMax} - 1 ))
  fi

  curInverseCutOffMult=0
  while [ \${curInverseCutOffMult} -le \${invCutOffLoopMax} ];
  do
    # Set up the file string that will store the local benchmarking results
    local fileString="DataFiles/results_${tag1}_\${scale}_\${NumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${curInverseCutOffMult}inverseCutOffMult_0bcMult_0panelDimMult_\${pDimD}pDimD_\${pDimC}pDimC_\${numIterations}numIter_\${ppn}ppn_\${tpr}tpr_\${curLaunchID}launchID"
    # 'PreFile' requires NumNodes specification because in the 'Pre' stage, we want to keep the data for different node counts separate.
    local PreFile="${tag1}_\${scale}_\${matrixDimM}_\${matrixDimN}_\${curInverseCutOffMult}_\${pDimD}_\${pDimC}_\${ppn}_\${tpr}_\${NumNodes}nodes"
    local PostFile="${tag1}_\${scale}_\${matrixDimM}_\${matrixDimN}_\${curInverseCutOffMult}_\${pDimDorig}_\${pDimCorig}_\${ppn}_\${tpr}"

    # Plot instructions only need a single output per scaling study
    if [ \${nodeIndex} == 0 ];
    then
      # New important addition: For special weak scaling, need to print out the number of (d,c) for the binary first, and then each of them in groups of {d,c,(d,c)}
      # Note: still not 100% convinced this is necessary. Need to study scaplot first to make a decision on it.
      # Write to plotInstructions file
      if [ \${scaleRegime} == 2 ];
      then
	echo "echo \"\${nodeCount}\" " >> $SCRATCH/${fileName}/plotInstructions.sh

	curD=\${pDimD}
	curC=\${pDimC}
	trickOffsetTemp=\${trickOffset}
	for ((z=0; z<\${nodeCount}; z++))
	do
	  echo "echo \"\${curD}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	  echo "echo \"\${curC}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	  echo "echo \"(\${curD},\${curC})\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	  trickOffsetTempMod=\$(( trickOffsetTemp % 4 ))
	  if [ \${trickOffsetTempMod} == 0 ];
	  then
	    curD=\$(( \${curD} / 2))
	    curC=\$(( \${curC} * 2))
	  else
	    curD=\$(( \${curD} * 2))
	  fi
	  trickOffsetTemp=\$(( \${trickOffsetTemp} + 1 ))
	done
      fi

      WriteMethodDataForPlotting ${tag1} \${PostFile} \${pDimD} \${pDimC} \${curInverseCutOffMult} \${ppn} \${tpr}
      writePlotFileName \${fileString} $SCRATCH/${fileName}/plotInstructions.sh 1  
    fi

    WriteMethodDataForCollectingStage1 ${tag1} \${PreFile} \${PreFile}_perf \${PreFile}_numerics $SCRATCH/${fileName}/collectInstructionsStage1.sh
    WriteMethodDataForCollectingStage2 \${launchID} ${tag1} \${PreFile} \${PreFile}_perf \${PreFile}_numerics \${PostFile} \${PostFile}_perf \${PostFile}_numerics $SCRATCH/${fileName}/collectInstructionsStage2.sh
    launchJobsPortal \${2} ${tag1} \${fileString} \${curLaunchID} \${NumNodes} \${ppn} \${tpr} \${2}_PERFORMANCE \${matrixDimM} \${matrixDimN} \${bcDim} \${curInverseCutOffMult} 0 \${pDimD} \${pDimC} \${numIterations} $SCRATCH/${fileName}/\${fileString}
    writePlotFileName \${fileString} $SCRATCH/${fileName}/collectInstructionsStage1.sh 0
    curInverseCutOffMult=\$(( \${curInverseCutOffMult} + 1 ))
  done
}




# For ScaLAPACK QR
launch$tag2 () {
  # launch scaLAPACK_QR
  local scale=\${1}
  local binaryPath=\${2}
  local numIterations=\${3}
  local launchID=\${4}
  local NumNodes=\${5}
  local ppn=\${6}
  local tpr=\${7}
  local matrixDimM=\${8}
  local matrixDimN=\${9}
  local numProwsorig=\${10}
  local numProws=\${11}
  local minBlockSize=\${12}
  local maxBlockSize=\${13}
  local nodeIndex=\${14}
  local scaleRegime=\${15}
  local nodeCount=\${16}
  for ((k=\${minBlockSize}; k<=\${maxBlockSize}; k*=2))
  do
    # Set up the file string that will store the local benchmarking results
    local fileString="DataFiles/results_${tag2}_\${1}_\${NumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${numProws}numProws_\${k}bSize_\${numIterations}numIter_\${ppn}ppn_\${tpr}tpr_\${curLaunchID}launchID"
    # 'PreFile' requires NumNodes specification because in the 'Pre' stage, we want to keep the data for different node counts separate.
    local PreFile="${tag2}_\${scale}_\${matrixDimM}_\${matrixDimN}_\${numProws}_\${k}_\${ppn}_\${tpr}_\${NumNodes}nodes"
    local PostFile="${tag2}_\${scale}_\${matrixDimM}_\${matrixDimN}_\${numProwsorig}_\${k}_\${ppn}_\${tpr}"

    # Plot instructions only need a single output per scaling study
    if [ \${nodeIndex} == 0 ];
    then
      WriteMethodDataForPlotting ${tag2} \${PostFile} \${numProws} \${k} \${ppn} \${tpr}
      writePlotFileNameScalapack \${fileString} $SCRATCH/${fileName}/plotInstructions.sh 1
    fi

    WriteMethodDataForCollectingStage1 ${tag2} \${PreFile} \${PreFile}_NoFormQ \${PreFile}_FormQ $SCRATCH/${fileName}/collectInstructionsStage1.sh
    WriteMethodDataForCollectingStage2 \${launchID} ${tag2} \${PreFile} \${PreFile}_NoFormQ \${PreFile}_FormQ \${PostFile} \${PostFile}_NoFormQ \${PostFile}_FormQ $SCRATCH/${fileName}/collectInstructionsStage2.sh
    launchJobsPortal \${2} ${tag2} \${fileString} \${curLaunchID} \${NumNodes} \${ppn} \${tpr} \${2}_PERFORMANCE \${matrixDimM} \${matrixDimN} \${k} \${numIterations} 0 \${numProws} 1 0 $SCRATCH/${fileName}/\${fileString}
    writePlotFileNameScalapack \${fileString} $SCRATCH/${fileName}/collectInstructionsStage1.sh 0
  done
}

# For CFR3D
launch$tag3 () {
  # launch CFR3D
  local scale=\${1}
  local binaryPath=\${2}
  local numIterations=\${3}
  local launchID=\${4}
  local NumNodes=\${5}
  local ppn=\${6}
  local tpr=\${7}
  local matrixDimM=\${8}
  local cubeDimorig=\${9}
  local cubeDim=\${10}
  local nodeIndex=\${11}
  local scaleRegime=\${12}
  local nodeCount=\${13}
  local bcDim=0

  # Next: Based on pDimC, decide on invCutOff parameter, which will range from 0 to a max of 2 for now
  invCutOffLoopMax=0
  if [ \${cubeDim} -le 2 ];
  then
    invCutOffLoopMax=0
  elif [ \${cubeDim} -eq 4 ];
  then
    invCutOffLoopMax=1
  else
    invCutOffLoopMax=2
    #invCutOffLoopMax=\$(( \${cubeDim} / 2 ))
    #invCutOffLoopMax=\$(( \${invCutOffLoopMax} - 1 ))
  fi

  curInverseCutOffMult=0
  while [ \${curInverseCutOffMult} -le \${invCutOffLoopMax} ];
  do
    # Set up the file string that will store the local benchmarking results
    local fileString="DataFiles/results_${tag3}_\${1}_\${NumNodes}nodes_\${matrixDimM}dimM_\${curInverseCutOffMult}inverseCutOffMult_0bcMult_0panelDimMult_\${cubeDim}cubeDim_\${numIterations}numIter_\${ppn}ppn_\${tpr}tpr_\${curLaunchID}launchID"
    # 'PreFile' requires NumNodes specification because in the 'Pre' stage, we want to keep the data for different node counts separate.
    local PreFile="${tag3}_\${scale}_\${matrixDimM}_\${curInverseCutOffMult}_\${cubeDim}_\${ppn}_\${tpr}_\${NumNodes}nodes"
    local PostFile="${tag3}_\${scale}_\${matrixDimM}_\${curInverseCutOffMult}_\${cubeDimorig}_\${ppn}_\${tpr}"

    # Plot instructions only need a single output per scaling study
    if [ \${nodeIndex} == 0 ];
    then
      WriteMethodDataForPlotting ${tag3} \${PostFile} \${cubeDim} \${curInverseCutOffMult} \${ppn} \${tpr}
      writePlotFileName \${fileString} $SCRATCH/${fileName}/plotInstructions.sh 1
    fi

    WriteMethodDataForCollectingStage1 ${tag3} \${PreFile} \${PreFile}_perf \${PreFile}_numerics $SCRATCH/${fileName}/collectInstructionsStage1.sh
    WriteMethodDataForCollectingStage2 \${launchID} ${tag3} \${PreFile} \${PreFile}_perf \${PreFile}_numerics \${PostFile} \${PostFile}_perf \${PostFile}_numerics $SCRATCH/${fileName}/collectInstructionsStage2.sh
    launchJobsPortal \${2} ${tag3} \${fileString} \${curLaunchID} \${NumNodes} \${ppn} \${tpr} \${2}_PERFORMANCE \${matrixDimM} \${bcDim} \${curInverseCutOffMult} 0 \${cubeDim} \${numIterations} $SCRATCH/${fileName}/\${fileString}
    writePlotFileName \${fileString} $SCRATCH/${fileName}/collectInstructionsStage1.sh 0
    curInverseCutOffMult=\$(( \${curInverseCutOffMult} + 1 ))
  done

}

# For ScaLAPACK Cholesky Factorization --- DOESNT CURRENTLY WORK!
launch$tag4 () {
  # launch scaLAPACK_CF
  local scale=\${1}
  local binaryPath=\${2}
  local numIterations=\${3}
  local launchID=\${4}
  local NumNodes=\${5}
  local ppn=\${6}
  local tpr=\${7}
  local matrixDimM=\${8}
  local minBlockSize=\${9}
  local maxBlockSize=\${10}
  local nodeIndex=\${11}
  local scaleRegime=\${12}
  local nodeCount=\${13}
  for ((k=\${minBlockSize}; k<=\${maxBlockSize}; k*=2))
  do
    # Set up the file string that will store the local benchmarking results
    local fileString="DataFiles/results_${tag4}_\${scale}_\${NumNodes}nodes_\${matrixDimM}dimM_\${k}bSize_\${numIterations}numIter_\${ppn}ppn_\${tpr}tpr_\${curLaunchID}launchID"
    # 'PreFile' requires NumNodes specification because in the 'Pre' stage, we want to keep the data for different node counts separate.
    local PreFile="${tag4}_\${scale}_\${matrixDimM}_\${k}_\${ppn}_\${tpr}_\${NumNodes}nodes"
    local PostFile="${tag4}_\${scale}_\${matrixDimM}_\${k}_\${ppn}_\${tpr}"

    if [ \${nodeIndex} == 0 ];
    then
      # Write to plotInstructions file
      WriteMethodDataForPlotting ${tag4} \${PostFile} \${k} \${ppn} \${tpr}
      writePlotFileNameScalapack \${fileString} $SCRATCH/${fileName}/plotInstructions.sh 1
    fi

    WriteMethodDataForCollectingStage1 ${tag4} \${PreFile} \${PreFile} \${PreFile}_blah $SCRATCH/${fileName}/collectInstructionsStage1.sh
    WriteMethodDataForCollectingStage2 \${launchID} ${tag4} \${PreFile} \${PreFile} \${PreFile}_blah \${PostFile} \${PostFile} \${PostFile} $SCRATCH/${fileName}/collectInstructionsStage2.sh
    launchJobsPortal \${2} ${tag4} \${fileString} \${curLaunchID} \${NumNodes} \${ppn} \${tpr} \${2}_PERFORMANCE \${matrixDimM} \${k} \${numIterations} $SCRATCH/${fileName}/\${fileString}
    writePlotFileNameScalapack \${fileString} $SCRATCH/${fileName}/collectInstructionsStage1.sh 0
  done
}


# Note: in future, I may want to decouple numBinaries and numPlotTargets, but only when I find it necessary
# Write to Plot Instructions file, for use by SCAPLOT makefile generator
echo "echo \"1\"" > $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${fileNameToProcess}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${numTests}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${machineName}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${profType}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
echo "echo \"${nodeScaleFactor}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

# Echo for data collection from remote machine (not porter) to PAA/src/Results
# This temporary file will be deleted while collectScript.sh is called.
WriteHeaderForCollection $SCRATCH/${fileName}/collectInstructionsStage1.sh
WriteHeaderForCollection $SCRATCH/${fileName}/collectInstructionsStage2.sh

for ((i=1; i<=${numTests}; i++))
do
  echo -e "\nTest #\${i}\n"

  # Nodes
  read -p "Enter starting number of nodes for this test: " startNumNodes
  read -p "Enter ending number of nodes for this test: " endNumNodes

  # Now allowing for WS and SS of any kind in a single bench job.
  read -p "Enter Scaling regime:\
	   [0 -> Weak scaling with increasingly rectangular matrix/grid for QR (QR only)\
	    1 -> Strong scaling with increasingly rectangular grid for QR, larger cubic grid for CF\
            2 -> Weak scaling with alternating scaling scheme for QR, regular scaling scheme for CF]: " scaleRegime

  scale="WS"
  if [ \${scaleRegime} == "1" ];		# The rest are WS, which is it already set as
  then
    scale="SS"
  fi

  echo "echo \"\${scale}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

  nodeCount=\$(findCountLength \${startNumNodes} \${endNumNodes} 3 ${nodeScaleFactor})
  echo "echo \"\${nodeCount}\" " >> $SCRATCH/${fileName}/plotInstructions.sh

  curNumNodes=\${startNumNodes}
  for ((j=0; j<\${nodeCount}; j++))
  do
    echo "echo \"\${curNumNodes}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
    curNumNodes=\$(( \${curNumNodes} * ${nodeScaleFactor} ))
  done

  read -p "Enter matrix dimension m: " matrixDimM
  read -p "Enter matrix dimension n: " matrixDimN
  read -p "Enter number of iterations (per launch): " numIterations

  j=1
  while [ 1 -eq 1 ];		# Loop iterates until user says stop
  do
    echo -e "\nStage #\${j}"

    # Echo for SCAPLOT makefile generator
    read -p "Enter binary tag [0 for CA-CQR2, 1 for bench_scala_qr, 2 for CFR3D, 3 for bench_scala_cf, 4 for quit]: " binaryTagChoice
    echo "echo \"\${binaryTagChoice}\"" >> $SCRATCH/${fileName}/collectInstructionsStage1.sh
    echo "echo \"\${binaryTagChoice}\"" >> $SCRATCH/${fileName}/collectInstructionsStage2.sh
    echo "echo \"\${binaryTagChoice}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
    # break case
    if [ \${binaryTagChoice} -eq "4" ];
    then
      break
    fi

    binaryTag=""
    if [ \${binaryTagChoice} == 0 ];
    then
      binaryTag=cqr2
    elif [ \${binaryTagChoice} == 1 ];
    then
      binaryTag=bench_scala_qr
    elif [ \${binaryTagChoice} == 2 ];
    then
      binaryTag=cfr3d
    elif [ \${binaryTagChoice} == 3 ];
    then
      binaryTag=bench_scala_cf
    fi

    binaryPath=${BINPATH}\${binaryTag}_${machineName}
    if [ "${machineName}" == "PORTER" ];
    then
      binaryPath=\${binaryPath}_${mpiType}
    fi

    if [ \${j} == 1 ];
    then
      echo "echo \"\${matrixDimM}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
      echo "echo \"\${matrixDimN}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
    fi


    # State variables that scale with nodes must be initialized here. Otherwise, repeated input is needed in loops below
    if [ \${binaryTag} == 'cqr2' ];
    then
      read -p "Enter start range of starting tunable processor grid dimension c: " startStartPdimC
      read -p "Enter end range of starting tunable processor grid dimension c (for any node count * ppn pairing): " endStartPdimC
      # invCutOff shouldn't be asked for. It should, for now, range up to 2 from 0, unless I am seeing a pattern in performance.
    elif [ \${binaryTag} == 'bench_scala_qr' ];
    then
      read -p "Enter the starting number of processor grid columns: " startStartNumPcols
      read -p "Enter the ending number of processor grid columns: " endStartNumPcols
      read -p "Enter the minimum block size: " minBlockSize
      read -p "Enter the maximum block size: " maxBlockSize
      # Anything else? Is above, sufficient?
    elif [ \${binaryTag} == 'cfr3d' ];
    then
      read -p "Enter the starting (cubic) processor grid dimension: " cubeDim
    elif [ \${binaryTag} == 'bench_scala_cholesky' ];
    then
      read -p "Enter the minimum block size: " minBlockSize
      read -p "Enter the maximum block size: " maxBlockSize
      # Anything else? Is above, sufficient?
    fi

    for ((curLaunchID=1; curLaunchID<=${NumLaunchesPerBinary}; curLaunchID+=1));
    do
      # Initialize all possible variables that change with node count
      # shared
      nodeIndex=0
      curMatrixDimM=\${matrixDimM}
      curMatrixDimN=\${matrixDimN}
      WShelpcounter=0
      # cqr2
      pDimCArray=()
      pDimCArrayOrig=()
      rangePdimClen=0
      if [ \${binaryTag} == 'cqr2' ];
      then
        for ((w=\${startStartPdimC}; w<=\${endStartPdimC}; w*=2));
        do
          pDimCArray+=(\${w})
          pDimCArrayOrig+=(\${w})
          rangePdimClen=\$(( \${rangePdimClen} + 1 ))
        done
      fi
      # bench_scala_qr
      numPcolsArray=()
      numPcolsArrayOrig=()
      rangeNumPcolslen=0
      if [ \${binaryTag} == 'bench_scala_qr' ];
      then
        for ((w=\${startStartNumPcols}; w<=\${endStartNumPcols}; w*=2));
        do
          numPcolsArray+=(\${w})
          numPcolsArrayOrig+=(\${w})
          rangeNumPcolslen=\$(( \${rangeNumPcolslen} + 1 ))
        done
      fi
      # cfr3d
      curCubeDim=\${cubeDim}
      for ((curNumNodes=\${startNumNodes}; curNumNodes<=\${endNumNodes}; curNumNodes*=${nodeScaleFactor}));
      do
        minPPN=\${ppnMinListRunTime[\${nodeIndex}]}
        maxPPN=\${ppnMaxListRunTime[\${nodeIndex}]}
        for ((curPPN=\${minPPN}; curPPN<=\${maxPPN}; curPPN*=2));
        do
          numProcesses=\$(( \${curNumNodes} * \${curPPN} ))
          StartingNumProcesses=\$(( \${startNumNodes} * \${curPPN} ))

          minTPR=\${tprMinListRunTime[\${nodeIndex}]}
          maxTPR=\${tprMaxListRunTime[\${nodeIndex}]}
	  for ((curTPR=\${minTPR}; curTPR<=\${maxTPR}; curTPR*=2));
	  do
            # Now decide on a method:

            if [ \${binaryTag} == 'cqr2' ];
            then
	      # Below: note that the STARTING dimC is being changed. The parameters that aren't solely dependent on the node count are
	      #   changed here and not in launchTag***
	      for ((w=0; w<\${rangePdimClen}; w+=1));
	      do
	        pDimC=\${pDimCArray[\${w}]}
                pDimCsquared=\$(( \${pDimC} * \${pDimC} ))
                pDimD=\$(( \${numProcesses} / \${pDimCsquared} ))
	        # Check if pDimC is too big. If so, pDimD will be 0
	        if [ \${pDimD} -ge \${pDimC} ];
	        then
                  originalPdimC=\${pDimCArrayOrig[\${w}]}
                  originalPdimCsquared=\$(( \${originalPdimC} * \${originalPdimC} ))
                  originalPdDimD=\$(( \${StartingNumProcesses} / \${originalPdimCsquared} ))
		  launch\${binaryTag} \${scale} \${binaryPath} \${numIterations} \${curLaunchID} \${curNumNodes} \${curPPN} \${curTPR} \${curMatrixDimM} \${curMatrixDimN} \${originalPdDimD} \${originalPdimC} \${pDimD} \${pDimC} \${nodeIndex} \${scaleRegime} \${nodeCount}
	        fi
              done
	    elif [ \${binaryTag} == 'bench_scala_qr' ];
	    then
	      for ((w=0; w<\${rangeNumPcolslen}; w+=1));
	      do
	        numPcols=\${numPcolsArray[\${w}]}
                numProws=\$(( \${numProcesses} / \${numPcols} ))
	        if [ \${numPcols} -le \${numProws} ];
	        then
                  originalNumPcols=\${numPcolsArrayOrig[\${w}]}
                  originalNumProws=\$(( \${StartingNumProcesses} / \${originalNumPcols} ))
		  launch\${binaryTag} \${scale} \${binaryPath} \${numIterations} \${curLaunchID} \${curNumNodes} \${curPPN} \${curTPR} \${curMatrixDimM} \${curMatrixDimN} \${originalNumProws} \${numProws} \${minBlockSize} \${maxBlockSize} \${nodeIndex} \${scaleRegime} \${nodeCount}
                fi
              done
	    elif [ \${binaryTag} == 'cfr3d' ];
	    then
              launch\${binaryTag} \${scale} \${binaryPath} \${numIterations} \${curLaunchID} \${curNumNodes} \${curPPN} \${curTPR} \${curMatrixDimM} \${cubeDim} \${curCubeDim} \${nodeIndex} \${scaleRegime} \${nodeCount}
	    elif [ \${binaryTag} == 'bench_scala_cholesky' ];
	    then
              launch\${binaryTag} \${scale} \${binaryPath} \${numIterations} \${curLaunchID} \${curNumNodes} \${curPPN} \${curTPR} \${curMatrixDimM} \${minBlockSize} \${maxBlockSize} \${nodeIndex} \${scaleRegime} \${nodeCount}
            fi
          done
        done
        # Update all loop variables for each increasing node count
        # Assuming that pDimD can always just be calculated from pDimC and NumProcesses as we scale
        if [ \${scaleRegime} == 0 ];
        then
	  # below: shared
	  curMatrixDimM=\$(( \${curMatrixDimM} * 2 ))
	  # below: ca-cqr2
	  if [ \${binaryTag} == 'cqr2' ];
          then
	    #pDimD=\$(( \${pDimD} * 2 ))
	    echo "Do nothing"
          fi
          # below: bench scala qr
	  if [ \${binaryTag} == 'bench_scala_qr' ];
          then
	    echo "Do nothing"
	  fi
	  # below: cfr3d
	  if [ \${binaryTag} == 'cfr3d' ];
          then
            curCubeDim=\$(( \${curCubeDim} * 2 ))
          fi
        elif [ \${scaleRegime} == 1 ];
	then
	  # below: ca-cqr2
	  if [ \${binaryTag} == 'cqr2' ];
          then
            #pDimD=\$(( \${pDimD} * 2 ))
	    echo "Do nothing"
	  fi
	  # below: bench scala qr
	  if [ \${binaryTag} == 'bench_scala_qr' ];
          then
	    echo "Do nothing"
	  fi
	  # below: cfr3d
	  if [ \${binaryTag} == 'cfr3d' ];
          then
            curCubeDim=\$(( \${curCubeDim} * 2 ))
          fi
        elif [ \${scaleRegime} == 2 ];
	then
	  immWS=\$(( \${WShelpcounter} % 4 ))
	  if [ \${immWS} == 0 ];
	  then
            # shared
	    curMatrixDimM=\$(( \${curMatrixDimM} / 2 ))
	    curMatrixDimN=\$(( \${curMatrixDimN} * 2 ))
	    # below: ca-cqr2
	    if [ \${binaryTag} == 'cqr2' ];
            then
	      #pDimD=\$(( \${pDimD} / 2 ))
	      for ((w=0; w<\${rangePdimClen}; w+=1));
	      do
	        pDimC=\${pDimCArray[\${w}]}
	        pDimCArray[\${w}]=\$(( \${pDimC} * 2 ))	# update
	      done
            fi
            # below: bench scala qr
	    if [ \${binaryTag} == 'bench_scala_qr' ];
            then
              for ((w=0; w<\${rangeNumPcolslen}; w+=1));
              do
                numPcols=\${numPcolsArray[\${w}]}
                numPcolsArray[\${w}]=\$(( \${numPcols} * 2 ))
              done
            fi
	  else
            # shared
	    curMatrixDimM=\$(( \${curMatrixDimM} * 2 ))
	    # below: ca-cqr2
	    if [ \${binaryTag} == 'cqr2' ];
            then
              #pDimD=\$(( \${pDimD} * 2 ))
	      echo "Do nothing"
	    fi
            # below: bench scala qr
	    if [ \${binaryTag} == 'bench_scala_qr' ];
            then
	      echo "Do nothing"
	    fi
          fi
	  # below: cfr3d
	  if [ \${binaryTag} == 'cfr3d' ];
          then
            curCubeDim=\$(( \${curCubeDim} * 2 ))
          fi
          WShelpcounter=\$(( \${WShelpcounter} + 1 ))
	fi
	nodeIndex=\$(( \${nodeIndex} + 1 ))
      done
    done
    j=\$(( \${j} + 1 ))
    echo "echo \"1\"" >> $SCRATCH/${fileName}/collectInstructionsStage1.sh	# Signals end of the data files for this specific methodID
    echo "echo \"1\"" >> $SCRATCH/${fileName}/collectInstructionsStage2.sh	# Signals end of the data files for this specific methodID
  done
done
EOF


bash $SCRATCH/${fileName}.sh
#rm $SCRATCH/${fileName}.sh

# Copy a local version to Scripts directory so that it can be used on the local side to generate plots.
# But its important that we keep a backup in SCRATCH/fileName in case we overwrite collectInstructionsStage1.sh, we can always write it back.
cp $SCRATCH/${fileName}/collectInstructionsStage1.sh collectInstructionsStage1.sh
# Do not copy collectInstructionsStage2 to local directory.

# Note that for Porter, no need to do this, since we are submitting to a queue
if [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ] || [ "${machineName}" == "STAMPEDE2" ] || [ "${machineName}" == "BLUEWATERS" ];
then
  mkdir $SCRATCH/${fileName}/bin
  mv ../bin/* $SCRATCH/${fileName}/bin
  #mv ${scalaDir}/bin/benchmarks/* $SCRATCH/${fileName}/bin  # move all scalapack benchmarks to same place before job is submitted
  cd $SCRATCH

  # Submit all scripts
  curLaunchID=1
  while [ ${curLaunchID} -le ${NumLaunchesPerBinary} ];
  do
    curNumNodes=${minNumNodes}
    ppnIndex=0
    while [ ${curNumNodes} -le ${maxNumNodes} ];
    do
      minPPN=${ppnMinList[${ppnIndex}]}
      maxPPN=${ppnMaxList[${ppnIndex}]}
      curPPN=${minPPN}
      tprIndex=0
      while [ ${curPPN} -le ${maxPPN} ];
      do
        minTPR=${tprMinList[${tprIndex}]}
        maxTPR=${tprMaxList[${tprIndex}]}
        curTPR=${minTPR}
        while [ ${curTPR} -le ${maxTPR} ];
        do
          if [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ];
          then
            qsub ${fileName}/script_${fileID}id_${roundID}round_${curLaunchID}launchID_${curNumNodes}nodes_${curPPN}ppn_${curTPR}tpr.sh
          elif [ "${machineName}" == "BLUEWATERS" ];
          then
            qsub ${fileName}/script_${fileID}id_${roundID}round_${curLaunchID}launchID_${curNumNodes}nodes_${curPPN}ppn_${curTPR}tpr.pbs
          else
            chmod +x ${fileName}/script_${fileID}id_${roundID}round_${curLaunchID}launchID_${curNumNodes}nodes_${curPPN}ppn_${curTPR}tpr.sh
            sbatch ${fileName}/script_${fileID}id_${roundID}round_${curLaunchID}launchID_${curNumNodes}nodes_${curPPN}ppn_${curTPR}tpr.sh
          fi
          curTPR=$(( ${curTPR} * 2 ))
        done
        curPPN=$(( ${curPPN} * 2 ))
        tprIndex=$(( ${tprIndex} + 1 ))
      done
      curNumNodes=$(( ${curNumNodes} * ${nodeScaleFactor} ))
      ppnIndex=$(( ${ppnIndex} + 1 ))
    done
    curLaunchID=$(( ${curLaunchID} + 1 ))
  done
fi