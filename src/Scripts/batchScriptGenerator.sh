#!/bin/bash

tag1='mm3d'
tag2='cfr3d'
tag3='cqr2'
tag4='bench_scala_qr'
tag5='bench_scala_cf'

if [ $(hostname |grep "porter") != "" ]
then
  machineName=PORTER
  read -p "Do you want to use MPI[mpi] or AMPI[ampi]? Note that use of AMPI forfeits Profiling/Critter output. : " mpiType
  if [ "${mpiType}" == "mpi" ]
  then
    export MPITYPE=MPI_TYPE
  elif [ "${mpiType}" == "ampi" ]
  then
    export MPITYPE=AMPI_TYPE
  fi
elif [ $(hostname |grep "mira") != "" ] || [ $(hostname |grep "cetus") != "" ]
then
  machineName=BGQ
  export MPITYPE=MPI_TYPE
elif [ $(hostname |grep "theta") != "" ]
then
  machineName=THETA
  export MPITYPE=MPI_TYPE
fi

if [ "${mpiType}" == "mpi" ]
then
  read -p "Do you want Profiling/Timer[T] output, Critter[C] output, or absolute performance[P] output? " profType
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
fi

if [ "${mpiType}" == "ampi" ]
then
  export PROFTYPE=PERF_TYPE
fi

read -p "Should we delete the items in Results directory? Yes[1], No[0] " delDecision1
if [ "${delDecision1}" == "1" ]
then
  rm -rf ../Results/*
fi

#read -p "Enter machine name [BGQ (cetus,mira), THETA, BW, STAMPEDE2, PORTER]: " machineName
#read -p "Enter the Date (MM_DD_YYYY): " dateStr
dateStr=$(date +%Y-%m-%d-%H:%M:%S)
read -p "Enter ID of auto-generated file this program will create: " fileID
read -p "Enter maximum number of nodes requested: " numNodes
read -p "Enter ppn: " ppn
read -p "Enter number of tests (equal to number of binaries that will be run): " numBinaries

if [ "${machineName}" == "BW" ]
then
  read -p "Enter number of hours of job (only valid on BW): " numHours
  read -p "Enter number of minutes of job (only valid on BW): " numMinutes
  read -p "Enter number of seconds of job (only valid on BW): " numSeconds
elif [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ]
then
  read -p "Enter number of minutes of job (only valid on BW): " numMinutes
fi

numPEs=$((ppn*numNodes))
fileName=scriptID${fileID}_${dateStr}_${machineName}_${profType}

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

make -C./.. clean
make -C./.. ${mpiType}
export BINPATH=./../bin/
if [ "${machineName}" == "BGQ" ]
then
  export SCRATCH=/projects/QMCat/huttered
elif [ "${machineName}" == "BW" ]
then
  echo "dog"
elif [ "${machineName}" == "THETA" ]
then
  export SCRATCH=/projects/QMCat/huttered/
elif [ "${machineName}" == "STAMPEDE2" ]
then
  echo "dog"
elif [ "${machineName}" == "PORTER" ]
then
  if [ ! -d "../Results" ];
  then
    mkdir ../Results/
  fi
  export SCRATCH=./../Results
fi

cat <<-EOF > $SCRATCH/${fileName}.sh
if [ "${machineName}" != "PORTER" ]
then
  scriptName=$SCRATCH/${fileName}/script.sh
fi
if [ ! -d "$SCRATCH/${fileName}/" ];
then
  mkdir $SCRATCH/${fileName}/
fi
if [ ! -d "$SCRATCH/${fileName}/results" ];
then
  mkdir $SCRATCH/${fileName}/results
fi
if [ "${machineName}" == "cetus" ] || [ "${machineName}" == "mira" ]
then
  echo "#!/bin/sh" > \$scriptName
elif [ "${machineName}" == "bw" ]
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
elif [ "${machineName}" == "theta" ]
then
  echo "#!/bin/sh" > \$scriptName
  echo "#COBALT -t ${numMinutes}" >> \$scriptName
  echo "#COBALT -n ${numNodes}" >> \$scriptName
  echo "#COBALT --attrs mcdram=cache:numa=quad" >> \$scriptName
  echo "#COBALT -A QMCat" >> \$scriptName
  echo "export n_nodes=\$COBALT_JOBSIZE" >> \$scriptName
  echo "export n_mpi_ranks_per_node=${ppn}" >> \$scriptName
  echo "export n_mpi_ranks=\$((${numNodes} * ${ppn}))" >> \$scriptName
  echo "export n_openmp_threads_per_rank=4" >> \$scriptName
  echo "export n_hyperthreads_per_core=2" >> \$scriptName
  echo "export n_hyperthreads_skipped_between_ranks=4" >> \$scriptName
elif [ "${machineName}" == "stampede2" ]
then
  echo "dog" > \$scriptName
fi
# Now I need to use a variable for the command-line prompt, since it will change based on the binary executable,
#   for example, scalapack QR has multiple special inputs that take up comm-line prompts that others dont
#   I want special functions in the inner-loop to handle this


updateCounter () {
  local counter=\$1
  if [ \$2 -eq 1 ]
  then
    counter=\$((\$counter + \$3)) 
  elif [ \$2 -eq 2 ]
  then
   counter=\$((\$counter - \$3)) 
  elif [ \$2 -eq 3 ]
  then
    counter=\$((\$counter * \$3)) 
  elif [ \$2 -eq 4 ]
  then
    counter=\$((\$counter / \$3)) 
  fi
  echo "\$counter"
}

log2 () {
    local x=0
    for (( y=\$1-1 ; \$y > 0; y >>= 1 )) ; do
        let x=\$x+1
    done
    echo \$x
}

# Functions that write the actual script, depending on machine
launchJobs () {
  local numProcesses=\$((\$2 * $ppn))
  if [ "$machineName" == "BGQ" ]
  then
    echo "runjob --np \$numProcesses -p $ppn --block \$COBALT_PARTNAME --verbose=INFO : \${@:3:\$#} > \${@:1:1}.txt" >> \$scriptName
  elif [ "$machineName" == "BW" ]
  then
    echo "aprun -n \$numProcesses \$@" > \$scriptName
  elif [ "$machineName" == "THETA" ]
  then
    echo "#aprun -n $numNodes -N \$n_mpi_ranks_per_node --env OMP_NUM_THREADS=\$n_openmp_threads_per_rank -cc depth -d \$n_hyperthreads_skipped_between_ranks -j \$n_hyperthreads_per_core \${@:3:\$#} > \${@:1:1}.txt" >> \$scriptName
  elif [ "$machineName" == "STAMPEDE2" ]
  then
    echo "dog" >> \$scriptName
  elif [ "$machineName" == "PORTER" ]
  then
    if [ "${mpiType}" == "mpi" ]
    then
      mpiexec -n \$numProcesses \${@:3:\$#} > \${@:1:1}.txt
    elif [ "${mpiType}" == "ampi" ]
    then
      ${BINPATH}charmrun +p1 +vp\${numProcesses} \${@:3:\$#} > \${@:1:1}.txt
    fi
  fi
}

# Functions for launching specific jobs based on certain parameters
launch$tag1 () {
  # launch MM3D
  if [ \$1 == 'SS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        local fileString="$SCRATCH/${fileName}/results/results_${tag1}_\$1_\${startNumNodes}nodes_0_\${11}bcastRoutine_\${8}m_\${9}n_\${10}k_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 0 \${11} \$8 \$9 \${10} \$3 \${fileString}
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
    done
  elif [ \$1 == 'WS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    local startDimensionM=\$8
    local startDimensionN=\${11}
    local startDimensionK=\${14}
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        local fileString="$SCRATCH/${fileName}/results/results_${tag1}_\$1_\${startNumNodes}nodes_0_\${17}bcastRoutine_\${startDimensionM}m_\${startDimensionN}n_\${startDimensionK}k_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 0 \${17} \$startDimensionM \$startDimensionN \$startDimensionK \$3 \${fileString}
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
        startDimensionM=\$(updateCounter \$startDimensionM \${10} \$9)
        startDimensionN=\$(updateCounter \$startDimensionN \${13} \${12})
        startDimensionK=\$(updateCounter \$startDimensionK \${16} \${15})
    done
  fi
}

launch$tag2 () {
  # launch CFR3D
  if [ \$1 == 'SS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        local fileString="$SCRATCH/${fileName}/results/results_${tag2}_\$1_\${startNumNodes}nodes_\${8}side_\${9}dim_0bcMult_\${10}inverseCutOffMult_0panelDimMult_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 \$8 \${9} 0 \${10} 0 \$3 \${fileString}
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
    done
  elif [ \$1 == 'WS' ]
  then
    local startNumNodes=\$4
    local startMatrixDim=\${9}
    local endNumNodes=\$5
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        local fileString="$SCRATCH/${fileName}/results/results_${tag2}_\$1_\${startNumNodes}nodes_\${8}side_\${startMatrixDim}dim_0bcMult_\${12}inverseCutOffMult_0panelDimMult_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 \$8 \${startMatrixDim} 0 \${12} 0 \$3 \${fileString}
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
        startMatrixDim=\$(updateCounter \$startMatrixDim \${11} \${10})
    done
  fi
}

launch$tag3 () {
  # launch CQR2
  if [ \$1 == 'SS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    local startPdimD=\${10}
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        local fileString="$SCRATCH/${fileName}/results/results_${tag3}_\$1_\${startNumNodes}nodes_\${8}dimM_\${9}dimN_\${12}inverseCutOffMult_0bcMult_0panelDimMult_\${startPdimD}pDimD_\${11}pDimC_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 \$8 \${9} \${12} 0 0 \${startPdimD} \${11} \$3 \${fileString}
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
        startPdimD=\$(updateCounter \$startPdimD \$7 \$6)
    done
  elif [ \$1 == 'WS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    local startMatrixDimM=\${8}
    local startPdimD=\${12}
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        local fileString="$SCRATCH/${fileName}/results/results_${tag3}_\$1_\${startNumNodes}nodes_\${startMatrixDimM}dimM_\${11}dimN_\${14}inverseCutOffMult_0bcMult_0panelDimMult_\${startPdimD}pDimD_\${13}pDimC_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 \${8} \${11} \${14} 0 0 \${startPdimD} \${13} \$3 \${fileString}
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
        startMatrixDimM=\$(updateCounter \$startMatrixDimM \${10} \${9})
        startPdimD=\$(updateCounter \$startPdimD \$7 \$6)
    done
  fi
}

launch$tag4 () {
  # launch scaLAPACK_QR
  if [ \$1 == 'SS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    local matrixDimM=\${8}
    local matrixDimN=\${9}
    local numProws=\${10}
    while [ \$startNumNodes -le \$endNumNodes ];
    do
      startBlockSize=1
      div1=\$((\$matrixDimM / \$numProws))
      numPcols=\$((\$startNumNodes / \$numProws))
      div2=\$((\$matrixDimN / \$numPcols))
      endBlockSize=\$((\$div1>\$div2?\$div1:\$div2))
      while [ \$startBlockSize -le \$endBlockSize ];
      do
          local fileString="$SCRATCH/${fileName}/results/results_${tag4}_\$1_\${startNumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${startBlockSize}blockSize_\${numProws}numProws_\${3}numIter"
          launchJobs \${fileString} \$startNumNodes \$2 \${matrixDimM} \${matrixDimN} \${startBlockSize} \${3} 0 \${numProws} 1 0 \${fileString}
          startBlockSize=\$((\$startBlockSize * 2))
      done
      startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
      numProws=\$(updateCounter \$numProws \$7 \$6)
    done
  elif [ \$1 == 'WS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    local matrixDimM=\${8}
    local matrixDimN=\${11}
    local numProws=\${12}
    while [ \$startNumNodes -le \$endNumNodes ];
    do
      startBlockSize=1
      div1=\$((\$matrixDimM / \$numProws))
      numPcols=\$((\$startNumNodes / \$numProws))
      div2=\$((\$matrixDimN / \$numPcols))
      endBlockSize=\$((\$div1>\$div2?\$div1:\$div2))
      while [ \$startBlockSize -le \$endBlockSize ];
      do
          local fileString="$SCRATCH/${fileName}/results/results_${tag4}_\$1_\${startNumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${startBlockSize}blockSize_\${numProws}numProws_\${3}numIter"
          launchJobs \${fileString} \$startNumNodes \$2 \${matrixDimM} \${matrixDimN} \${startBlockSize} \${3} 0 \${numProws} 1 0 \${fileString}
          startBlockSize=\$((\$startBlockSize * 2))
      done
      startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
      numProws=\$(updateCounter \$numProws \$7 \$6)
      matrixDimM=\$(updateCounter \$matrixDimM \${10} \$9)
    done
  fi
}

launch$tag5 () {
  # launch scaLAPACK_CF
  if [ \$1 == 'SS' ]
  then
    local startNumNodesRoot=\$4
    local endNumNodesRoot=\$5
    local matrixDim=\${8}
    local matrixDimRoot=\$(log2 \$matrixDim)
    while [ \$startNumNodesRoot -le \$endNumNodesRoot ];
    do
      temp1=\$(log2 \$startNumNodesRoot)
      temp2=\$((\$matrixDimRoot - \$temp1))
      specialMatDim=\$((1<<\$temp2))
      specialMatDimLog=\$(log2 \$specialMatDim)
      startBlockSize=1
      endBlockSize=\$specialMatDimLog
      curNumNodes=\$((\$startNumNodesRoot * \$startNumNodesRoot))
      while [ \$startBlockSize -le \$endBlockSize ];
      do
          temp3=\$((1<<\$startBlockSize))
          local fileString="$SCRATCH/${fileName}/results/results_${tag5}_\$1_\$((\$startNumNodesRoot * \$startNumNodesRoot))nodes_\${matrixDimRoot}dimM_\${temp3}blockSize_\${3}numIter"
          launchJobs \${fileString} \$curNumNodes \$2 \${matrixDimRoot} \${specialMatDimLog} \${startBlockSize} \${3} \${fileString}
          startBlockSize=\$((\$startBlockSize + 1))
      done
      startNumNodesRoot=\$(updateCounter \$startNumNodesRoot \$7 \$6)
    done
  elif [ \$1 == 'WS' ]
  then
    local startNumNodesRoot=\$4
    local endNumNodesRoot=\$5
    local matrixDim=\${8}
    local matrixDimRoot=\$(log2 \$matrixDim)
    temp1=\$(log2 \$startNumNodesRoot)
    temp2=\$((\$matrixDimRoot - \$temp1))
    local specialMatDim=\$((1<<\$temp2))
    local specialMatDimLog=\$(log2 \$specialMatDim)
    while [ \$startNumNodesRoot -le \$endNumNodesRoot ];
    do
      startBlockSize=1
      endBlockSize=\$specialMatDimLog
      curNumNodes=\$((\$startNumNodesRoot * \$startNumNodesRoot))
      while [ \$startBlockSize -le \$endBlockSize ];
      do
          temp3=\$((1<<\$startBlockSize))
          local fileString="$SCRATCH/${fileName}/results/results_${tag5}_\$1_\$((\$startNumNodesRoot * \$startNumNodesRoot))nodes_\${matrixDimRoot}dimM_\$((\$temp3))blockSize_\${3}numIter"
          launchJobs \${fileString} \$curNumNodes \$2 \${matrixDimRoot} \${specialMatDimLog} \${startBlockSize} \${3} \${fileString}
          startBlockSize=\$((\$startBlockSize + 1))
      done
      startNumNodesRoot=\$(updateCounter \$startNumNodesRoot \$7 \$6)
      matrixDim=\$(updateCounter \$matrixDim \${10} \$9)
      matrixDimRoot=\$(log2 \$matrixDim)
    done
  fi
}


for i in {1..$numBinaries}
do
  read -p "Enter binary tag [mm3d,cfr3d,cqr2,bench_scala_qr,bench_scala_cf]: " binaryTag
  binaryPath=${BINPATH}\${binaryTag}_${machineName}
  if [ "${machineName}" == "PORTER" ]
  then
    binaryPath=\${binaryPath}_${mpiType}
  fi
  read -p "Enter scale [SS,WS]: " scale
  read -p "Enter number of iterations: " numIterations
  if [ \$binaryTag != 'bench_scala_cf' ]
  then
      read -p "Enter starting number of nodes for this test: " startNumNodes
      read -p "Enter ending number of nodes for this test: " endNumNodes
      read -p "Enter factor by which to increase the number of nodes: " jumpNumNodes
      read -p "Enter arithmetic operator by which to increase the number of nodes by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpNumNodesoperator
  fi
  if [ \$binaryTag == 'mm3d' ]
  then
    if [ \$scale == 'SS' ]
    then
      read -p "In this strong scaling test for MM3D, enter dimension m: " DimensionM
      read -p "In this strong scaling test for MM3D, enter dimension n: " DimensionN
      read -p "In this strong scaling test for MM3D, enter dimension k: " DimensionK
      read -p "Do you want an broadcast-based routine[0], or an Allgather-based routine[1]: " bcastVSallgath
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$DimensionM \$DimensionN \$DimensionK \$bcastVSallgath
    elif [ \$scale == 'WS' ]
    then
      read -p "In this weak scaling test for MM3D, enter starting dimension m: " startDimensionM
      read -p "In this weak scaling test for MM3D, enter factor by which to increase dimension m: " jumpDimensionM
      read -p "In this weak scaling test for MM3D, enter arithmetic operator by which to increase dimension m by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpDimensionMoperator
      read -p "In this weak scaling test for MM3D, enter starting dimension n: " startDimensionN
      read -p "In this weak scaling test for MM3D, enter factor by which to increase dimension n: " jumpDimensionN
      read -p "In this weak scaling test for MM3D, enter arithmetic operator by which to increase dimension n by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpDimensionNoperator
      read -p "In this weak scaling test for MM3D, enter starting dimension k: " startDimensionK
      read -p "In this weak scaling test for MM3D, enter factor by which to increase dimension k: " jumpDimensionK
      read -p "In this weak scaling test for MM3D, enter arithmetic operator by which to increase dimension k by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpDimensionKoperator
      read -p "Do you want an broadcast-based routine[0], or an Allgather-based routine[1]: " bcastVSallgath
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$startDimensionM \$jumpDimensionM \$jumpDimensionMoperator \$startDimensionN \$jumpDimensionN \$jumpDimensionNoperator \$startDimensionK \$jumpDimensionK \$jumpDimensionKoperator \$bcastVSallgath
    fi
  elif [ \$binaryTag == 'cfr3d' ]
  then
    read -p "Factor lower[0] or upper[1]: " factorSide
    read -p "Enter the inverseCutOff multiplier, 0 indicates that CFR3D will use the explicit inverse, 1 indicates that top recursive level will avoid calculating inverse, etc.: " inverseCutOffMult
    if [ \$scale == 'SS' ]
    then
      read -p "In this strong scaling test for CFR3D, enter matrix dimension: " matrixDim
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$factorSide \$matrixDim \$inverseCutOffMult
    elif [ \$scale == 'WS' ]
    then
      read -p "In this weak scaling test for CFR3D, enter starting matrix dimension: " startMatrixDim
      read -p "In this weak scaling test for CFR3D, enter factor by which to increase matrix dimension: " jumpMatrixDim
      read -p "In this weak scaling test for CFR3D, enter arithmetic operator by which to increase the matrix dimension by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimoperator
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$factorSide \$startMatrixDim \$jumpMatrixDim \$jumpMatrixDimoperator \$inverseCutOffMult
    fi
  elif [ \$binaryTag == 'cqr2' ]
  then
    read -p "Enter the inverseCutOff multiplier, 0 indicates that CFR3D will use the explicit inverse, 1 indicates that top recursive level will avoid calculating inverse, etc.: " inverseCutOffMult
    if [ \$scale == 'SS' ]
    then
      read -p "In this strong scaling test for CQR2, enter matrix dimension m: " matrixDimM
      read -p "In this strong scaling test for CQR2, enter matrix dimension n: " matrixDimN
      read -p "In this strong scaling test for CQR2, enter starting tunable processor grid dimension d: " pDimD
      read -p "In this strong scaling test for CQR2, enter static tunable processor grid dimension c: " pDimC
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$matrixDimM \$matrixDimN \$pDimD \$pDimC \$inverseCutOffMult
    elif [ \$scale == 'WS' ]
    then
      read -p "In this weak scaling test for CQR2, enter matrix dimension m: " startMatrixDimM
      read -p "In this weak scaling test for CQR2, enter factor by which to increase matrix dimension m: " jumpMatrixDimM
      read -p "In this weak scaling test for CQR2, enter arithmetic operator by which to increase matrix dimension m by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimMoperator
      read -p "In this weak scaling test for CQR2, enter matrix dimension n: " matrixDimN
      read -p "In this weak scaling test for CQR2, enter starting tunable processor grid dimension d: " pDimD
      read -p "In this weak scaling test for CQR2, enter static tunable processor grid dimension c: " pDimC
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$startMatrixDimM \$jumpMatrixDimM \$jumpMatrixDimMoperator \$matrixDimN \$pDimD \$pDimC \$inverseCutOffMult
    fi
  elif [ \$binaryTag == 'bench_scala_qr' ]
  then
    if [ \$scale == 'SS' ]
    then
      read -p "In this strong scaling test for Scalapack QR, enter matrix dimension m: " matrixDimM
      read -p "In this strong scaling test for Scalapack QR, enter matrix dimension n: " matrixDimN
      read -p "In this strong scaling test for Scalapack QR, enter the starting number of processor rows: " numProws
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$matrixDimM \$matrixDimN \$numProws
    elif [ \$scale == 'WS' ]
    then
      read -p "In this weak scaling test for Scalapack QR, enter matrix dimension m: " matrixDimM
      read -p "In this weak scaling test for Scalapack QR, enter factor by which to increase matrix dimension m: " jumpMatrixDimM
      read -p "In this weak scaling test for Scalapack QR, enter arithmetic operator by which to increase matrix dimension m by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimMoperator
      read -p "In this strong scaling test for Scalapack QR, enter matrix dimension n: " matrixDimN
      read -p "In this strong scaling test for Scalapack QR, enter the starting number of processor rows: " numProws
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$matrixDimM \$jumpMatrixDimM \$jumpMatrixDimMoperator \$matrixDimN \$numProws
    fi
  elif [ \$binaryTag == 'bench_scala_cf' ]
  then
    read -p "Enter the starting square root of the number of nodes: " startNumNodesRoot
    read -p "Enter the ending square root of the number of nodes: " endNumNodesRoot
    read -p "Enter factor by which to increase the square root of the number of nodes: " jumpNumNodes
    read -p "Enter arithmetic operator by which to increase the square root of the number of nodes by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpNumNodesoperator
    if [ \$scale == 'SS' ]
    then
      read -p "In this strong scaling test for Scalapack CF, enter the matrix dimension: " matrixDim
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodesRoot \$endNumNodesRoot \$jumpNumNodes \$jumpNumNodesoperator \$matrixDim
    elif [ \$scale == 'WS' ]
    then
      read -p "In this weak scaling test for Scalapack CF, enter the matrix dimension: " matrixDim
      read -p "In this weak scaling test for Scalapack CF, enter factor by which to increase matrix dimension: " jumpMatrixDim
      read -p "In this weak scaling test for Scalapack CF, enter arithmetic operator by which to increase the matrix dimension by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimoperator
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodesRoot \$endNumNodesRoot \$jumpNumNodes \$jumpNumNodesoperator \$matrixDim \$jumpMatrixDim \$jumpMatrixDimoperator
    fi
  fi
done
EOF

chmod +x $SCRATCH/${fileName}.sh
./$SCRATCH/${fileName}.sh
rm $SCRATCH/${fileName}.sh
if [ "${machineName}" != "PORTER" ]
  then
  cd $SCRATCH
  if [ "${machineName}" == "BGQ" ] || [ "${machineName}" == "THETA" ]
  then
    qsub -A QMCat -t ${numMinutes} -n ${numNodes} --mode script ${fileName}/script.sh
  fi
fi
