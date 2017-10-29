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


# Functions for launching specific jobs based on certain parameters
launch$tag1 () {
  # do stuff for MM3D
  # Need a loop over parameter K (start, end, jump operator, jump factor)
  echo \$1
}

launch$tag2 () {
  # do stuff for CFR3D
  echo \$1
  #aprun -n \$3 \$1 \$4  
  #aprun -n \${numPEs} ./../../../../../../../CANDMC/bin/benchmarks/bench_scala_qr \${m} \${n} \${bsize} 4 0 \${pr} 1 0  > ScalapackSSresults/$2_scalapack_ss_\${numPEs}_\${m}_\${n}_\${bsize}_\${pr}.out
}

launch$tag3 () {
  # do stuff for CholeskyQR2_1D
  # Nothing needed besides launch
  echo \$1
}

launch$tag4 () {
  # do stuff for CholeskyQR2_3D
  # Nothing needed besides launch
  echo \$1
}

launch$tag5 () {
  # do stuff for CholeskyQR2_Tunable
  echo \$1
}

launch$tag6 () {
  # do stuff for scaLAPACK_QR
  echo \$1
}

numArguments$tag1 () {
  return numArgs$tag1
}

numArguments$tag2 () {
  return numArgs$tag2
}

numArguments$tag3 () {
  return numArgs$tag3
}

numArguments$tag4 () {
  return numArgs$tag4
}

numArguments$tag5 () {
  return numArgs$tag5
}

numArguments$tag6 () {
  return numArgs$tag6
}

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


commandLineCounter=1
index=1

for i in {1..$numBinaries}
do
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
        launch\$binaryTag \$binaryPath \$numIterations \$startNumPEs \$startDimensionM \$((\$commandLineCounter+11))
      else
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
          launch\$binaryTag \$binaryPath \$numIterations \$startNumPEs \$startDimensionM \$startDimensionN \${@:\$((\$commandLineCounter+16)):\$((numArguments\$BinaryTag - 15))}
          startDimensionN=\$(updateCounter \$startDimensionN \$jumpDimensionNOperator \$jumpDimensionN)
        done
      fi
      startDimensionM=\$(updateCounter \$startDimensionM \$jumpDimensionMOperator \$jumpDimensionM)
    done
    startNumPEs=\$(updateCounter \$startNumPEs \$jumpNumPEsOperator \$jumpNumPEs)
  done

  commandLineCounter=\$((\$commandLineCounter+numArguments\$BinaryTag))
done
EOF
