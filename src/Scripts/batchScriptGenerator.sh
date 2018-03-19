#!/bin/bash

machineName=$1
batchSciptName=$2
numNodes=$3
ppn=$4
numPEs=$((ppn*numNodes))
numBinaries=$5

tag1='MM3D'
tag2='CFR3D'
tag3='CQR2'
tag4='SCALA_QR'
tag5='SCALA_CF'

cat <<-EOF > $2.sh
if [ "${machineName}" == "cetus" ] || [ "${machineName}" == "mira" ]
then
  echo "#!/bin/sh"
elif [ "${machineName}" == "bw" ]
then
  echo "#!/bin/bash"
  echo "#PBS -l nodes=$numNodes:ppn=$3:xe"
  echo "#PBS -l walltime=$6:$7:$8"
  echo "#PBS -N $2"
  echo "#PBS -e \$PBS_JOBID.err"
  echo "#PBS -o \$PBS_JOBID.out"
  echo "##PBS -m Ed"
  echo "##PBS -M hutter2@illinois.edu"
  echo "##PBS -A xyz"
  echo "#PBS -W umask=0027"
  echo "cd \$PBS_O_WORKDIR"
  echo "#module load craype-hugepages2M  perftools"
  echo "#export APRUN_XFER_LIMITS=1  # to transfer shell limits to the executable"
elif [ "${machineName}" == "theta" ]
then
  echo "#!/bin/sh"
  echo "#COBALT -t $6"
  echo "#COBALT -n ${numNodes}"
  echo "#COBALT --attrs mcdram=cache:numa=quad"
  echo "#COBALT -A QMCat"
  echo "export n_nodes=\$COBALT_JOBSIZE"
  echo "export n_mpi_ranks_per_node=${ppn}"
  echo "export n_mpi_ranks=\$((\$n_nodes * \$n_mpi_ranks_per_node))"
  echo "export n_openmp_threads_per_rank=4"
  echo "export n_hyperthreads_per_core=2"
  echo "export n_hyperthreads_skipped_between_ranks=4"
  echo "Starting Cobalt job script"
elif [ "${machineName}" == "stampede2" ]
then
  echo "dog"
fi

# Now I need to use a variable for the command-line prompt, since it will change based on the binary executable,
#   for example, scalapack QR has multiple special inputs that take up comm-line prompts that others dont
#   I want special functions in the inner-loop to handle this


# List all Binary Tags that this script supports
MM3D=1
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

# Functions that write the actual script, depending on machine
launchJobs () {
  local numProcesses=\$((\$1 * $ppn))
  if [ "$machineName" == "mira" ] || [ "$machineName" == "cetus" ]
  then
    echo "runjob --np \$numProcesses -p $ppn --block \$COBALT_PARTNAME --verbose=INFO : \${@:2:9}"
  elif [ "$machineName" == "bw" ]
  then
    echo "aprun -n \$numProcesses \$@"
  elif [ "$machineName" == "theta" ]
  then
    echo "#aprun -n $numNodes -N \$n_mpi_ranks_per_node --env OMP_NUM_THREADS=\$n_openmp_threads_per_rank -cc depth -d \$n_hyperthreads_skipped_between_ranks -j \$n_hyperthreads_per_core "$@""
  elif [ "$machineName" == "stampede2" ]
  then
    echo "dog"
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
        launchJobs \$startNumNodes \$2 0 1 \${11} \$8 \$9 \${10} \$3 
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
    done
  elif [ \$2 == 'WS' ]
  then
    echo "dog"
  fi
}

launch$tag2 () {
  # launch CFR3D
  # Need to loop over the block-size multiplier
  local blockSizeStartRange=\$5
  local blockSizeEndRange=\$6
  while [ \$blockSizeStartRange -le \$blockSizeEndRange ]
  do
    echo "aprun -n \$3 \$1 \$4 \$2 \$blockSizeStartRange"
    blockSizeStartRange=\$((\$blockSizeStartRange+1))
  done
}

launch$tag3 () {
  # launch CholeskyQR2_1D
    echo "aprun -n \$3 \$1 \$4 \$5 \$2"
  echo \$1
}

launch$tag4 () {
  # launch CholeskyQR2_3D
  # Nothing needed besides launch
  echo "aprun -n \$3 \$1 \$4 \$5 \$2"
}

launch$tag5 () {
  # launch CholeskyQR2_Tunable
  local startPgridDimensionD=\$6
  local endPgridDimensionD=\$7
  while [ \$startPgridDimensionD -le \$endPgridDimensionD ]
  do
    local startPgridDimensionC=\${10}
    local endPgridDimensionC=\${11}
    while [ \$startPgridDimensionC -le \$endPgridDimensionC ];
    do
      echo "aprun -n \$3 \$1 \$4 \$5 \$startPgridDimensionD \$startPgridDimensionC \$2"
      startPgridDimensionC=\$(updateCounter \$startPgridDimensionC \${13} \${12})
    done
    startPgridDimensionD=\$(updateCounter \$startPgridDimensionD \$9 \$8)
  done
}

launch$tag6 () {
  # launch scaLAPACK_QR
  local blockSizeStart=\$6
  local blockSizeEnd=\$7
  while [ \$blockSizeStart -le \$blockSizeEnd ]
  do
    local numProwsStart=\${10}
    local numProwsEnd=\${11}
    while [ \$numProwsStart -le \$numProwsEnd ];
    do
      echo "aprun -n \$3 \$1 \$4 \$5 \$blockSizeStart \$2 0 \$numProwsStart 0"
      numProwsStart=\$(updateCounter \$numProwsStart \${13} \${12})
    done
    blockSizeStart=\$(updateCounter \$blockSizeStart \$9 \$8)
  done
}

numArguments${tag1}_SS () {
  echo "\$numArgs${tag1}_SS"
}
numArguments${tag1}_WS () {
  echo "\$numArgs${tag1}_WS"
}
numArguments${tag2}_SS () {
  echo "\$numArgs${tag2}_SS"
}
numArguments${tag2}_WS () {
  echo "\$numArgs${tag2}_WS"
}
numArguments${tag3}_SS () {
  echo "\$numArgs${tag3}_SS"
}
numArguments${tag3}_WS () {
  echo "\$numArgs${tag3}_WS"
}
numArguments${tag4}_SS () {
  echo "\$numArgs${tag4}_SS"
}
numArguments${tag4}_WS () {
  echo "\$numArgs${tag4}_WS"
}
numArguments${tag5}_SS () {
  echo "\$numArgs${tag5}_SS"
}
numArguments${tag5}_WS () {
  echo "\$numArgs${tag5}_WS"
}

commandLineCounter=1
index=1

for i in {1..$numBinaries}
do
  index=\$commandLineCounter
  export binaryPath=\${!index}
  index=\$((\$commandLineCounter+1))
  export binaryTag=\${!index}
  index=\$((\$commandLineCounter+2))
  export scale=\${!index}
  index=\$((\$commandLineCounter+3))
  export numIterations=\${!index}
  index=\$((\$commandLineCounter+4))
  export startNumNodes=\${!index}
  index=\$((\$commandLineCounter+5))
  export endNumNodes=\${!index}
  index=\$((\$commandLineCounter+6))
  export jumpNumNodes=\${!index}
  index=\$((\$commandLineCounter+7))
  export jumpNumNodesoperator=\${!index}
  index=\$((\$commandLineCounter+8))

  if [ \$binaryTag == 'MM3D' ]
  then
    if [ \$scale == 'SS' ]
    then
      export DimensionM=\${!index}
      index=\$((\$commandLineCounter+9))
      export DimensionN=\${!index}
      index=\$((\$commandLineCounter+10))
      export DimensionK=\${!index}
      index=\$((\$commandLineCounter+11))
      export bcastVSallgath=\${!index}
      index=\$((\$commandLineCounter+12))
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$DimensionM \$DimensionN \$DimensionK \$bcastVSallgath
      commandLineCounter=\$((\$commandLineCounter+numArguments\${binaryTag}_\${scale}))
    elif [ \$scale == 'WS' ]
    then
      export startDimensionM=\${!index}
      index=\$((\$commandLineCounter+9))
      export jumpDimensionM=\${!index}
      index=\$((\$commandLineCounter+10))
      export jumpDimensionMoperator=\${!index}
      index=\$((\$commandLineCounter+11))
      export startDimensionN=\${!index}
      index=\$((\$commandLineCounter+12))
      export jumpDimensionN=\${!index}
      index=\$((\$commandLineCounter+13))
      export jumpDimensionNoperator=\${!index}
      index=\$((\$commandLineCounter+14))
      export startDimensionK=\${!index}
      index=\$((\$commandLineCounter+15))
      export jumpDimensionK=\${!index}
      index=\$((\$commandLineCounter+16))
      export jumpDimensionKoperator=\${!index}
      index=\$((\$commandLineCounter+17))
      export bcastVSallgath=\${!index}
      index=\$((\$commandLineCounter+18))
      launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$startDimensionM \$jumpDimensionM \$jumpDimensionMoperator \$startDimensionN \$jumpDimensionN \$jumpDimensionNoperator \$startDimensionK \$jumpDimensionK \$jumpDimensionKoperator \$bcastVSallgath
      commandLineCounter=\$((\$commandLineCounter+numArguments\${binaryTag}_\${scale}))
    fi
  elif [ \$binaryTag == 'CFR3D' ]
  then
    echo "dog"
  elif [ \$binaryTag == 'CQR2' ]
  then
    echo "dog"
  elif [ \$binaryTag == 'SCALA_QR' ]
  then
    echo "dog"
  elif [ \$binaryTag == 'SCALA_CF' ]
  then
    echo "dog"
  fi
done
EOF
