#!/bin/bash

batchSciptName=$1
numNodes=$2
ppn=$3
numPEs=$((ppn*numNodes))
numBinaries=$4

cat > $1.sh <<EOF
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
tag1='MM3D'
tag2='CFR3D'
tag3='CholeskyQR2_1D'
tag4='CholeskyQR2_3D'
tag5='CholeskyQR2_Tunable'
tag6='scaLAPACK_QR'

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


commandLineCounter=8

for i in {1..$numBinaries}
do
  export binaryPath=\$commandlineCounter
  export binaryTag=\$((\$commandlineCounter+1)
  export numIterations=\$((\$commandlineCounter+2)
  export startNumPEs=\$((\$commandLineCounter+3))
  export endNumPEs=\$((\$commandLineCounter+4))
  export jumpNumPEs=\$((\$commandLineCounter+5))
  export jumpNumPEsOperator=\$((\$commandLineCounter+6))

  while [\$startNumPEs -le \$endNumPEs];
  do
    export startDimensionM=\$((\$commandLineCounter+7)) 
    export endDimensionM=\$((\$commandLineCounter+8))
    export jumpDimensionM=\$((\$commandLineCounter+9))
    export jumpDimensionMOperator=\$((\$commandLineCounter+\${10}))

    while [\$startDimensionM -le \$endDimensionM];
    do
      if [\$binaryTag -eq \$tag2]
      do
        # call function
        launch\$binaryTag binaryPath numIterations startNumPEs startDimensionM \$((\$commandLineCounter+\${11}))
      done
      else
      do
        export startDimensionN=\$((\$commandLineCounter+\${11})) 
        export endDimensionN=\$((\$commandLineCounter+\${12}))
        export jumpDimensionN=\$((\$commandLineCounter+\${13}))
        export jumpDimensionNOperator=\$((\$commandLineCounter+\${14}))
        while [\$startDimensionN -le \$endDimensionN];
        do
          # call function
          launch\$binaryTag binaryPath numIterations startNumPEs startDimensionM startDimensionN \${@:\$((\$commandLineCounter+\${14})):\$((numArguments\$BinaryTag - 14))}
          startDimensionN=\$((\$startDimensionN \$jumpDimensionNOperator \$jumpDimensionN)) 
        done
      done
      startDimensionM=\$((\$startDimensionM \$jumpDimensionMOperator \$jumpDimensionM))
    done
    startNumPEs=\$((\$startNumPEs \$jumpNumPEsOperator \$jumpNumPEs))
  done

  commandLineCounter = \$((\$commandLineCounter+numArguments\$BinaryTag))
done


ss_probs00=1
ss_probs01=1024
ss_probs02=1024
ss_probs10=1
ss_probs11=2048
ss_probs12=512
ss_probs20=1
ss_probs21=8192
ss_probs22=128
ss_probs30=64
ss_probs31=8192
ss_probs32=8192
ss_probs40=64
ss_probs41=16384
ss_probs42=4096
ss_probs50=64
ss_probs51=131072
ss_probs52=512

for iprob in 0 1 2 3 4 5
do
  export param1=ss_probs\${iprob}0
  export param2=ss_probs\${iprob}1
  export param3=ss_probs\${iprob}2
  numPEs=\${!param1}
  m=\${!param2}
  n=\${!param3}
  while [ \$numPEs -le $p ];
  do
    for bsize in 1 8 16 32 64
    do
      pr=\$numPEs
      while [ \$((\$pr*\$pr)) -ge \$((\$numPEs*\$m/\$n)) ];
      do
        echo "numPEs - " \$numPEs
        echo "m - " \$m
        echo "n - " \$n
        echo "bsize - " \$bsize
        echo "pr - " \$pr
        echo "LAUNCH STRONG SCALING JOB -- SCALAPACK"
        aprun -n \${numPEs} ./../../../../../../../CANDMC/bin/benchmarks/bench_scala_qr \${m} \${n} \${bsize} 4 0 \${pr} 1 0  > ScalapackSSresults/$2_scalapack_ss_\${numPEs}_\${m}_\${n}_\${bsize}_\${pr}.out
        pr=\$((\$pr/2))
      done
    done
    numPEs=\$((\$numPEs*8))
  done
done
echo "All done with Scalapack strong scaling"

ss_probsLog00=1
ss_probsLog01=10
ss_probsLog02=10
ss_probsLog10=1
ss_probsLog11=11
ss_probsLog12=9
ss_probsLog20=1
ss_probsLog21=13
ss_probsLog22=7
ss_probsLog30=64
ss_probsLog31=13
ss_probsLog32=13
ss_probsLog40=64
ss_probsLog41=14
ss_probsLog42=12
ss_probsLog50=64
ss_probsLog51=17
ss_probsLog52=9

for iprob in 0 1 2 3 4 5
do
  export param1=ss_probsLog\${iprob}0
  export param2=ss_probsLog\${iprob}1
  export param3=ss_probsLog\${iprob}2
  numPEs=\${!param1}
  m=\${!param2}
  n=\${!param3}
  while [ \$numPEs -le $p ];
  do
    d=1
    c=1
    while [ \$((\$d*\$c*\$c)) -lt \$numPEs ]
    do
      d=\$((\$d*2))
      c=\$((\$c*2))
    done
    echo "d - " \$d
    echo "c - " \$c
    while [ \$c -ge 1 ];
    do
      echo "numPEs - " \$numPEs
      echo "m - " \$m
      echo "n - " \$n
      echo "grid dimension d - " \$d
      echo "grid dimension c - " \$c
      echo "LAUNCH STRONG SCALING JOB -- CHOLESKYQR2"
      aprun -n \${numPEs} ./testPerformanceTunable \${m} \${n} \${d} \${c} 4 > TunableSSresults/$2_choleskyqr2_ss_\${numPEs}_\${m}_\${n}_\${d}_\${c}.out
      c=\$((\$c/2))
      d=\$((\$d*4))
    done
    numPEs=\$((\$numPEs*8))
  done
done
echo "All done with Communication-avoiding CholeskyQR2 strong scaling"

ws_probs00=1
ws_probs01=1024
ws_probs02=1024
ws_probs10=8
ws_probs11=2048
ws_probs12=2048
ws_probs20=64
ws_probs21=4096
ws_probs22=4096
ws_probs30=512
ws_probs31=8192
ws_probs32=8192

for iprob in 0 1 2 3
do
  export param1=ws_probs\${iprob}0
  export param2=ws_probs\${iprob}1
  export param3=ws_probs\${iprob}2
  numPEs=\${!param1}
  m=\${!param2}
  n=\${!param3}
  while [ \$numPEs -le $p ];
  do
    for bsize in 1 8 16 32 64
    do
      pr=\$numPEs
      while [ \$((\$pr*\$pr)) -ge \$((\$numPEs*\$m/\$n)) ];
      do
        echo "numPEs - " \$numPEs
        echo "m - " \$m
        echo "n - " \$n
        echo "bsize - " \$bsize
        echo "pr - " \$pr
        echo "LAUNCH WEAK SCALING JOB"
        aprun -n \${numPEs} ./../../../../../../../CANDMC/bin/benchmarks/bench_scala_qr \${m} \${n} \${bsize} 4 0 \${pr} 1 0  > ScalapackWSresults/$2_ws_\${numPEs}_\${m}_\${n}_\${bsize}_\${pr}.out
        pr=\$((\$pr/2))
      done
    done
    numPEs=\$((\$numPEs*8))
    m=\$((\$m*8))
  done
done
echo "All done with Scalapack weak scaling"

ws_probsLog00=1
ws_probsLog01=10
ws_probsLog02=10
ws_probsLog10=8
ws_probsLog11=11
ws_probsLog12=11
ws_probsLog20=64
ws_probsLog21=12
ws_probsLog22=12
ws_probsLog30=512
ws_probsLog31=13
ws_probsLog32=13

for iprob in 0 1 2 3
do
  export param1=ws_probsLog\${iprob}0
  export param2=ws_probsLog\${iprob}1
  export param3=ws_probsLog\${iprob}2
  numPEs=\${!param1}
  m=\${!param2}
  n=\${!param3}
  while [ \$numPEs -le $p ];
  do
    d=1
    c=1
    while [ \$((\$d*\$c*\$c)) -lt \$numPEs ]
    do
      d=\$((\$d*2))
      c=\$((\$c*2))
    done
    echo "d - " \$d
    echo "c - " \$c
    while [ \$c -ge 1 ];
    do
      echo "numPEs - " \$numPEs
      echo "m - " \$m
      echo "n - " \$n
      echo "grid dimension d - " \$d
      echo "grid dimension c - " \$c
      echo "LAUNCH WEAK SCALING JOB -- CHOLESKYQR2"
      aprun -n \${numPEs} ./testPerformanceTunable \${m} \${n} \${d} \${c} 4  > TunableWSresults/$2_choleskyqr2_ws_\${numPEs}_\${m}_\${n}_\${d}_\${c}.out
      c=\$((\$c/2))
      d=\$((\$d*4))
    done
    numPEs=\$((\$numPEs*8))
    m=\$((\$m+3))
  done
done
echo "All done with Communication-avoiding CholeskyQR2 weak scaling"
