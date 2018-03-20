#!/bin/bash

tag1='MM3D'
tag2='CFR3D'
tag3='CQR2'
tag4='SCALA_QR'
tag5='SCALA_CF'

read -p "Enter machine name [cetus,mira,theta,bw,stampede2,porter]: " machineName
read -p "Enter the Date (MM_DD_YYYY): " dateStr
read -p "Enter ID of auto-generated file this program will create: " fileID
read -p "Enter maximum number of nodes requested: " numNodes
read -p "Enter ppn: " ppn
read -p "Enter number of tests (equal to number of binaries that will be run): " numBinaries
read -p "Enter number of hours of job (only valid on BW): " numHours
read -p "Enter number of minutes of job (only valid on BW): " numMinutes
read -p "Enter number of seconds of job (only valid on BW): " numSeconds
numPEs=$((ppn*numNodes))
fileName=scriptCreator_${dateStr}_${fileID}_${machineName}

if [ "${machineName}" == "cetus" ] || [ "${machineName}" == "mira" ]
then
  export SCRATCH=/projects/QMCat/huttered
elif [ "${machineName}" == "bw" ]
then
  echo "dog"
elif [ "${machineName}" == "theta" ]
then
  export SCRATCH=/projects/QMCat/huttered/
elif [ "${machineName}" == "stampede2" ]
then
  echo "dog"
elif [ "${machineName}" == "porter" ]
then
  if [ ! -d "../Results" ];
  then
    mkdir ../Results/
  fi
  export SCRATCH=./../Results
fi

cat <<-EOF > $SCRATCH/${fileName}.sh
read -p "Enter name of the file to write the official script to (please include the .sh): " scriptName
if [ ! -d "$SCRATCH/${fileName}/" ];
then
  mkdir $SCRATCH/${fileName}/
fi
if [ ! -d "$SCRATCH/${fileName}/results" ];
then
  mkdir $SCRATCH/${fileName}/results
fi
scriptName=$SCRATCH/${fileName}/\$scriptName
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
elif [ "${machineName}" == "porter" ]
then
  echo "" > \$scriptName
fi
# Now I need to use a variable for the command-line prompt, since it will change based on the binary executable,
#   for example, scalapack QR has multiple special inputs that take up comm-line prompts that others dont
#   I want special functions in the inner-loop to handle this


# List all Binary Tags that this script supports
MM3=1
CFR3D=2
CQR2=3
SCALA_QR=4
SCALA_CF=5

# Each Binary tag has a specific number of command-line arguments that it accepts
let numArgs${tag1}_SS=8
let numArgs${tag2}_SS=1000
let numArgs${tag3}_SS=1000
let numArgs${tag4}_SS=1000
let numArgs${tag5}_SS=1000
let numArgs${tag1}_WS=14
let numArgs${tag2}_WS=1000
let numArgs${tag3}_WS=1000
let numArgs${tag4}_WS=1000
let numArgs${tag5}_WS=1000


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
  local numProcesses=\$((\$1 * $ppn))
  if [ "$machineName" == "mira" ] || [ "$machineName" == "cetus" ]
  then
    echo "runjob --np \$numProcesses -p $ppn --block \$COBALT_PARTNAME --verbose=INFO : \${@:2:\$#}" >> \$scriptName
  elif [ "$machineName" == "bw" ]
  then
    echo "aprun -n \$numProcesses \$@" > \$scriptName
  elif [ "$machineName" == "theta" ]
  then
    echo "#aprun -n $numNodes -N \$n_mpi_ranks_per_node --env OMP_NUM_THREADS=\$n_openmp_threads_per_rank -cc depth -d \$n_hyperthreads_skipped_between_ranks -j \$n_hyperthreads_per_core \${@:2:\$#}" >> \$scriptName
  elif [ "$machineName" == "stampede2" ]
  then
    echo "dog" >> \$scriptName
  elif [ "$machineName" == "porter" ]
  then
    echo "\${@:2:\$#}" >> \$scriptName
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
        local fileString="> \$SCRATCH/${fileName}/results/results_${tag1}_SS_\${startNumNodes}nodes_0_1_\${11}bcastRoutine_\${8}m_\${9}n_\${10}k_\${3}numIter.txt"
        launchJobs \$startNumNodes \$2 0 1 \${11} \$8 \$9 \${10} \$3 \$fileString
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
        local fileString="> $SCRATCH/${fileName}/results/results_${tag1}_SS_\${startNumNodes}nodes_0_1_\${17}bcastRoutine_\${startDimensionM}m_\${startDimensionN}n_\${startDimensionK}k_\${3}numIter.txt"
        launchJobs \$startNumNodes \$2 0 1 \${17} \$startDimensionM \$startDimensionN \$startDimensionK \$3 \$fileString
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
        local fileString="> $SCRATCH/${fileName}/results/results_${tag2}_SS_\${startNumNodes}nodes_\${8}side_\${9}testtype_\${10}dim_0bcMult_\${11}inverseCutOffMult_0panelDimMult_\${3}numIter.txt"
        launchJobs \$startNumNodes \$2 \$8 \$9 \${10} \${11} 0 \$3 \$fileString
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
    done
  elif [ \$1 == 'WS' ]
  then
    local startNumNodes=\$4
    local startMatrixDim=\${10}
    local endNumNodes=\$5
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        local fileString="> $SCRATCH/${fileName}/results/results_${tag2}_WS_\${startNumNodes}nodes_\${8}side_\${9}testtype_\${startMatrixDim}dim_0bcMult_\${13}inverseCutOffMult_0panelDimMult_\${3}numIter.txt"
        launchJobs \$startNumNodes \$2 \$8 \$9 \${10} \${13} 0 \$3 \$fileString
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
        startMatrixDim=\$(updateCounter \$startMatrixDim \$12 \$11)
    done
  fi
}

launch$tag3 () {
  # launch CQR2
  if [ \$1 == 'SS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    local startPdimD=\${11}
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        local fileString="> $SCRATCH/${fileName}/results/results_${tag3}_SS_\${startNumNodes}nodes_\${8}perf_\${9}dimM_\${10}dimN_0bcMult_\${13}inverseCutOffMult_0panelDimMult_\${startPdimD}pDimD_\${12}pDimC_\${3}numIter.txt"
        launchJobs \$startNumNodes \$2 \$8 \$9 \${10} \${13} 0 0 \${startPdimD} \${12} \$3 \$fileString
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
        startPdimD=\$(updateCounter \$startPdimD \$7 \$6)
    done
  elif [ \$1 == 'WS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    local startMatrixDimM=\${9}
    local startPdimD=\${13}
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        local fileString="> $SCRATCH/${fileName}/results/results_${tag3}_WS_\${startNumNodes}nodes_\${8}perf_\${9}dimM_\${12}dimN_0bcMult_\${15}inverseCutOffMult_0panelDimMult_\${startPdimD}pDimD_\${14}pDimC_\${3}numIter.txt"
        launchJobs \$startNumNodes \$2 \$8 \$9 \${12} \${15} 0 0 \${startPdimD} \${14} \$3 \$fileString
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
        startMatrixDimM=\$(updateCounter \$startMatrixDimM \$11 \$10)
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
          local fileString="> $SCRATCH/${fileName}/results/results_${tag4}_SS_\${startNumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${startBlockSize}blockSize_\${numProws}numProws_\${3}numIter.txt"
          launchJobs \$startNumNodes \$2 \${matrixDimM} \${matrixDimN} \${startBlockSize} \${3} 0 \${numProws} 1 0 \$fileString
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
          local fileString="> $SCRATCH/${fileName}/results/results_${tag4}_WS_\${startNumNodes}nodes_\${matrixDimM}dimM_\${matrixDimN}dimN_\${startBlockSize}blockSize_\${numProws}numProws_\${3}numIter.txt"
          launchJobs \$startNumNodes \$2 \${matrixDimM} \${matrixDimN} \${startBlockSize} \${3} 0 \${numProws} 1 0 \$fileString
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
          local fileString="> $SCRATCH/${fileName}/results/results_${tag5}_SS_\$((\$startNumNodesRoot * \$startNumNodesRoot))nodes_\${matrixDimRoot}dimM_\${temp3}blockSize_\${3}numIter.txt"
          launchJobs \$curNumNodes \$2 \${matrixDimRoot} \${specialMatDimLog} \${startBlockSize} \${3} \$fileString
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
          local fileString="> $SCRATCH/${fileName}/results/results_${tag5}_SS_\$((\$startNumNodesRoot * \$startNumNodesRoot))nodes_\${matrixDimRoot}dimM_\$((\$temp3))blockSize_\${3}numIter.txt"
          launchJobs \$curNumNodes \$2 \${matrixDimRoot} \${specialMatDimLog} \${startBlockSize} \${3} \$fileString
          startBlockSize=\$((\$startBlockSize + 1))
      done
      startNumNodesRoot=\$(updateCounter \$startNumNodesRoot \$7 \$6)
      matrixDim=\$(updateCounter \$matrixDim \${10} \$9)
      matrixDimRoot=\$(log2 \$matrixDim)
    done
  fi
}

numArguments${tag1}_SS () {
  echo "\$numArgs${tag1}_SS"
}
numArguments${tag1}_WS () {
  echo "\$numArgs${tag1}_WS"
}
numArguments${tag2}_SS () {
  echo "\${numArgs${tag2}_SS}"
}
numArguments${tag2}_WS () {
  echo "\${numArgs${tag2}_WS}"
}
numArguments${tag3}_SS () {
  echo "\${numArgs${tag3}_SS}"
}
numArguments${tag3}_WS () {
  echo "\${numArgs${tag3}_WS}"
}
numArguments${tag4}_SS () {
  echo "\${numArgs${tag4}_SS}"
}
numArguments${tag4}_WS () {
  echo "\${numArgs${tag4}_WS}"
}
numArguments${tag5}_SS () {
  echo "\${numArgs${tag5}_SS}"
}
numArguments${tag5}_WS () {
  echo "\${numArgs${tag5}_WS}"
}

for i in {1..$numBinaries}
do
  read -p "Enter binary path: " binaryPath
  read -p "Enter binary tag [MM3D,CFR3D,CQR2,SCALA_QR,SCALA_CF]: " binaryTag
  read -p "Enter scale [SS,WS]: " scale
  read -p "Enter number of iterations: " numIterations
  if [ \$binaryTag != 'SCALA_CF' ]
  then
      read -p "Enter starting number of nodes for this test: " startNumNodes
      read -p "Enter ending number of nodes for this test: " endNumNodes
      read -p "Enter factor by which to increase the number of nodes: " jumpNumNodes
      read -p "Enter arithmetic operator by which to increase the number of nodes by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpNumNodesoperator
  fi
  if [ \$binaryTag == 'MM3D' ]
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
  elif [ \$binaryTag == 'CFR3D' ]
  then
    read -p "Factor lower[0] or upper[1]: " factorSide
    read -p "Do you want to test performance[0] or distributed validation[1]: " PerfOrDSV
    read -p "Enter the inverseCutOff multiplier, 0 indicates that CFR3D will use the explicit inverse, 1 indicates that top recursive level will avoid calculating inverse, etc.: " inverseCutOffMult
    if [ \$scale == 'SS' ]
    then
      read -p "In this strong scaling test for CFR3D, enter matrix dimension: " matrixDim
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$factorSide \$PerfOrDSV \$matrixDim \$inverseCutOffMult
    elif [ \$scale == 'WS' ]
    then
      read -p "In this weak scaling test for CFR3D, enter starting matrix dimension: " startMatrixDim
      read -p "In this weak scaling test for CFR3D, enter factor by which to increase matrix dimension: " jumpMatrixDim
      read -p "In this weak scaling test for CFR3D, enter arithmetic operator by which to increase the matrix dimension by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimoperator
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$factorSide \$PerfOrDSV \$startMatrixDim \$jumpMatrixDim \$jumpMatrixDimoperator \$inverseCutOffMult
    fi
  elif [ \$binaryTag == 'CQR2' ]
  then
    read -p "Do you want to test performance[0] or distributed validation[1]: " PerfOrDSV
    read -p "Enter the inverseCutOff multiplier, 0 indicates that CFR3D will use the explicit inverse, 1 indicates that top recursive level will avoid calculating inverse, etc.: " inverseCutOffMult
    if [ \$scale == 'SS' ]
    then
      read -p "In this strong scaling test for CQR2, enter matrix dimension m: " matrixDimM
      read -p "In this strong scaling test for CQR2, enter matrix dimension n: " matrixDimN
      read -p "In this strong scaling test for CQR2, enter starting tunable processor grid dimension d: " pDimD
      read -p "In this strong scaling test for CQR2, enter static tunable processor grid dimension c: " pDimC
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$PerfOrDSV \$matrixDimM \$matrixDimN \$pDimD \$pDimC \$inverseCutOffMult
    elif [ \$scale == 'WS' ]
    then
      read -p "In this weak scaling test for CQR2, enter matrix dimension m: " startMatrixDimM
      read -p "In this weak scaling test for CQR2, enter factor by which to increase matrix dimension m: " jumpMatrixDimM
      read -p "In this weak scaling test for CQR2, enter arithmetic operator by which to increase matrix dimension m by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimMoperator
      read -p "In this weak scaling test for CQR2, enter matrix dimension n: " matrixDimN
      read -p "In this weak scaling test for CQR2, enter starting tunable processor grid dimension d: " pDimD
      read -p "In this weak scaling test for CQR2, enter static tunable processor grid dimension c: " pDimC
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$PerfOrDSV \$startMatrixDimM \$jumpMatrixDimM \$jumpMatrixDimMoperator \$matrixDimN \$pDimD \$pDimC \$inverseCutOffMult
    fi
  elif [ \$binaryTag == 'SCALA_QR' ]
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
  elif [ \$binaryTag == 'SCALA_CF' ]
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
