#!/bin/bash

batchSciptName=$1
numNodes=$2
ppn=$3
numPEs=$((ppn*numNodes))
numBinaries=$4

tag1='MM3D'
tag2='CFR3D'
tag3='CholeskyQR2_1D'
tag4='CholeskyQR2_3D'
tag5='CholeskyQR2_Tunable'
tag6='scaLAPACK_QR'

cat <<-EOF > $1.sh
#!/bin/bash
#PBS -l nodes=$numNodes:ppn=$3:xe
#PBS -l walltime=$5:$6:$7
#PBS -N $1
#PBS -e \$PBS_JOBID.err
#PBS -o \$PBS_JOBID.out
##PBS -m Ed
##PBS -M hutter2@illinois.edu
##PBS -A xyz
#PBS -W umask=0027

cd \$PBS_O_WORKDIR

#module load craype-hugepages2M  perftools
# export APRUN_XFER_LIMITS=1  # to transfer shell limits to the executable

# Now I need to use a varibale for the command-line prompt, since it will change based on the binary executable,
#   for example, scalapack QR has multiple special inputs that take up comm-line prompts that others dont
#   I want special functions in the inner-loop to handle this


# List all Binary Tags that this script supports
MM3D=1
CFR3D=2
CholeskyQR2_1D=3
CholeskyQR2_3D=4
CholeskyQR2_Tunable=5
scaLAPACK_QR=6

# Each Binary tag has a specific number of command-line arguments that it accepts
let numArgs${tag1}=19
let numArgs${tag2}=12
let numArgs${tag3}=15
let numArgs${tag4}=15
let numArgs${tag5}=23
let numArgs${tag6}=23


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

# Below: for reference
#aprun -n \${numPEs} ./../../../../../../../CANDMC/bin/benchmarks/bench_scala_qr \${m} \${n} \${bsize} 4 0 \${pr} 1 0  > ScalapackSSresults/$2_scalapack_ss_\${numPEs}_\${m}_\${n}_\${bsize}_\${pr}.out

# Functions for launching specific jobs based on certain parameters
launch$tag1 () {
  # launch MM3D
  local startDimensionK=\$6
  local endDimensionK=\$7
  while [ \$startDimensionK -le \$endDimensionK ];
  do
    echo "aprun -n \$3 \$1 \$4 \$5 \$startDimensionK \$2"
    startDimensionK=\$(updateCounter \$startDimensionK \$9 \$8)
  done
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
  echo $1
}

numArguments$tag1 () {
  echo "\$numArgs$tag1"
}

numArguments$tag2 () {
  echo "\$numArgs$tag2"
}

numArguments$tag3 () {
  echo "\$numArgs$tag3"
}

numArguments$tag4 () {
  echo "\$numArgs$tag4"
}

numArguments$tag5 () {
  echo "\$numArgs$tag5"
}

numArguments$tag6 () {
  echo "\$numArgs$tag6"
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
  export numIterations=\${!index}
  index=\$((\$commandLineCounter+3))
  export startNumPEs=\${!index}
  index=\$((\$commandLineCounter+4))
  export endNumPEs=\${!index}
  index=\$((\$commandLineCounter+5))
  export jumpNumPEs=\${!index}
  index=\$((\$commandLineCounter+6))
  export jumpNumPEsOperator=\${!index}
  index=\$((\$commandLineCounter+7))

  while [ \$startNumPEs -le \$endNumPEs ];
  do
    index=\$((\$commandLineCounter+7))
    export startDimensionM=\${!index}
    index=\$((\$commandLineCounter+8))
    export endDimensionM=\${!index}
    index=\$((\$commandLineCounter+9))
    export jumpDimensionM=\${!index}
    index=\$((\$commandLineCounter+10))
    export jumpDimensionMOperator=\${!index}
    index=\$((\$commandLineCounter+11))

    while [ \$startDimensionM -le \$endDimensionM ];
    do
      if [ \${!binaryTag} -eq \$CFR3D ]
      then
        # call function
        index=\$((\$commandLineCounter+11))
        export blockSizeMultiplierRangeStart=\${!index}
        index=\$((\$commandLineCounter+12))
        export blockSizeMultiplierRangeEnd=\${!index}
        launch\$binaryTag \$binaryPath \$numIterations \$startNumPEs \$startDimensionM \$blockSizeMultiplierRangeStart \$blockSizeMultiplierRangeEnd
      else
        index=\$((\$commandLineCounter+11))
        export startDimensionN=\${!index}
        index=\$((\$commandLineCounter+12))
        export endDimensionN=\${!index}
        index=\$((\$commandLineCounter+13))
        export jumpDimensionN=\${!index}
        index=\$((\$commandLineCounter+14))
        export jumpDimensionNOperator=\${!index}
        index=\$((\$commandLineCounter+15))
        while [ \$startDimensionN -le \$endDimensionN ];
        do
          # call function
          export startRange=\$((\$commandLineCounter+15))
          export lengthTag=\$(numArguments\$binaryTag)
          export lengthRange=\$((\$lengthTag - 15))
          launch\$binaryTag \$binaryPath \$numIterations \$startNumPEs \$startDimensionM \$startDimensionN \${@:\$startRange:\$lengthRange}
          startDimensionN=\$(updateCounter \$startDimensionN \$jumpDimensionNOperator \$jumpDimensionN)
        done
      fi
      startDimensionM=\$(updateCounter \$startDimensionM \$jumpDimensionMOperator \$jumpDimensionM)
    done
    startNumPEs=\$(updateCounter \$startNumPEs \$jumpNumPEsOperator \$jumpNumPEs)
  done
  commandLineCounter=\$((\$commandLineCounter+numArguments\$binaryTag))
done
EOF
