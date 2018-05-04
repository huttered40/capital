# Functions for launching specific jobs based on certain parameters
launch$tag1 () {
  # launch MM3D
  if [ \$1 == 'SS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        # Write to plotInstructions file
	echo "echo \"\${8}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	echo "echo \"\${9}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	echo "echo \"\${10}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        echo "echo \"\${11}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

        local fileString="results/results_${tag1}_\$1_\${startNumNodes}nodes_0_\${11}bcastRoutine_\${8}m_\${9}n_\${10}k_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 0 \${11} \$8 \$9 \${10} \$3 $SCRATCH/${fileName}/\${fileString}
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
        # Write to plotInstructions file
	echo "echo \"\${startDimensionM}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	echo "echo \"\${startDimensionN}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	echo "echo \"\${startDimensionK}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	echo "echo \"\${17}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

        local fileString="/results/results_${tag1}_\$1_\${startNumNodes}nodes_0_\${17}bcastRoutine_\${startDimensionM}m_\${startDimensionN}n_\${startDimensionK}k_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 0 \${17} \$startDimensionM \$startDimensionN \$startDimensionK \$3 $SCRATCH/${fileName}/\${fileString}
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
        startDimensionM=\$(updateCounter \$startDimensionM \${10} \$9)
        startDimensionN=\$(updateCounter \$startDimensionN \${13} \${12})
        startDimensionK=\$(updateCounter \$startDimensionK \${16} \${15})
    done
  fi
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


for ((i=1; i<=${numTests}; i++))
do
  echo -e "\nTest #\${i}\n"
  read -p "Enter scaling type [SS,WS]: " scale
  read -p "Enter number of different configurations/binaries which will be used for this test: " numBinaries

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

  for ((j=1; j<=\${numBinaries}; j++))
  do
    echo -e "\nStage #\${j}"

    # Echo for SCAPLOT makefile generator
    read -p "Enter binary tag [mm3d,cfr3d,cqr2,bench_scala_qr,bench_scala_cholesky]: " binaryTag
    binaryPath=${BINPATH}\${binaryTag}_${machineName}
    if [ "${machineName}" == "PORTER" ]
    then
      binaryPath=\${binaryPath}_${mpiType}
    fi
    echo "echo \"\${binaryTag}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

    read -p "Enter number of iterations: " numIterations
    if [ \$binaryTag != 'bench_scala_cholesky' ]
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
        # Write to plotInstructions file
        # Only print out this tag for the first configuration of a test.
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${DimensionM}_\${DimensionN}_\${DimensionK}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLength \$startNumNodes \$endNumNodes \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
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
        # Write to plotInstructions file
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${startDimensionM}_\${startDimensionN}_\${startDimensionK}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLength \$startNumNodes \$endNumNodes \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$startDimensionM \$jumpDimensionM \$jumpDimensionMoperator \$startDimensionN \$jumpDimensionN \$jumpDimensionNoperator \$startDimensionK \$jumpDimensionK \$jumpDimensionKoperator \$bcastVSallgath
      fi
    elif [ \$binaryTag == 'cfr3d' ]
    then
      read -p "Factor lower[0] or upper[1]: " factorSide
      read -p "Enter the inverseCutOff multiplier, 0 indicates that CFR3D will use the explicit inverse, 1 indicates that top recursive level will avoid calculating inverse, etc.: " inverseCutOffMult
      if [ \$scale == 'SS' ]
      then
        read -p "In this strong scaling test for CFR3D, enter matrix dimension: " matrixDim
        # Write to plotInstructions file
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDim}_\${factorSide}_\${inverseCutOffMult}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLength \$startNumNodes \$endNumNodes \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$factorSide \$matrixDim \$inverseCutOffMult
      elif [ \$scale == 'WS' ]
      then
        read -p "In this weak scaling test for CFR3D, enter starting matrix dimension: " startMatrixDim
        read -p "In this weak scaling test for CFR3D, enter factor by which to increase matrix dimension: " jumpMatrixDim
        read -p "In this weak scaling test for CFR3D, enter arithmetic operator by which to increase the matrix dimension by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimoperator
        # Write to plotInstructions file
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${startMatrixDim}_\${factorSide}_\${inverseCutOffMult}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLength \$startNumNodes \$endNumNodes \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
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
        # Write to plotInstructions file
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLength \$startNumNodes \$endNumNodes \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$matrixDimM \$matrixDimN \$pDimD \$pDimC \$inverseCutOffMult
      elif [ \$scale == 'WS' ]
      then
        read -p "In this weak scaling test for CQR2, enter matrix dimension m: " startMatrixDimM
        read -p "In this weak scaling test for CQR2, enter factor by which to increase matrix dimension m: " jumpMatrixDimM
        read -p "In this weak scaling test for CQR2, enter arithmetic operator by which to increase matrix dimension m by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimMoperator
        read -p "In this weak scaling test for CQR2, enter matrix dimension n: " matrixDimN
        read -p "In this weak scaling test for CQR2, enter starting tunable processor grid dimension d: " pDimD
        read -p "In this weak scaling test for CQR2, enter static tunable processor grid dimension c: " pDimC
        # Write to plotInstructions file
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${startMatrixDimM}_\${matrixDimN}_\${inverseCutOffMult}_\${pDimD}_\${pDimC}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLength \$startNumNodes \$endNumNodes \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$startMatrixDimM \$jumpMatrixDimM \$jumpMatrixDimMoperator \$matrixDimN \$pDimD \$pDimC \$inverseCutOffMult
      fi
    elif [ \$binaryTag == 'bench_scala_qr' ]
    then
      if [ \$scale == 'SS' ]
      then
        read -p "In this strong scaling test for Scalapack QR, enter matrix dimension m: " matrixDimM
        read -p "In this strong scaling test for Scalapack QR, enter matrix dimension n: " matrixDimN
        read -p "In this strong scaling test for Scalapack QR, enter the starting number of processor rows: " numProws
        # Write to plotInstructions file
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLengthScalaQR \$startNumNodes \$endNumNodes \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$matrixDimM \$matrixDimN \$numProws
      elif [ \$scale == 'WS' ]
      then
        read -p "In this weak scaling test for Scalapack QR, enter matrix dimension m: " matrixDimM
        read -p "In this weak scaling test for Scalapack QR, enter factor by which to increase matrix dimension m: " jumpMatrixDimM
        read -p "In this weak scaling test for Scalapack QR, enter arithmetic operator by which to increase matrix dimension m by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimMoperator
        read -p "In this strong scaling test for Scalapack QR, enter matrix dimension n: " matrixDimN
        read -p "In this strong scaling test for Scalapack QR, enter the starting number of processor rows: " numProws
        # Write to plotInstructions file
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodes}_\${matrixDimM}_\${matrixDimN}_\${numProws}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLengthScalaQR \$startNumNodes \$endNumNodes \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodes \$endNumNodes \$jumpNumNodes \$jumpNumNodesoperator \$matrixDimM \$jumpMatrixDimM \$jumpMatrixDimMoperator \$matrixDimN \$numProws
      fi
    elif [ \$binaryTag == 'bench_scala_cholesky' ]
    then
      read -p "Enter the starting square root of the number of nodes: " startNumNodesRoot
      read -p "Enter the ending square root of the number of nodes: " endNumNodesRoot
      read -p "Enter factor by which to increase the square root of the number of nodes: " jumpNumNodes
      read -p "Enter arithmetic operator by which to increase the square root of the number of nodes by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpNumNodesoperator
      if [ \$scale == 'SS' ]
      then
        read -p "In this strong scaling test for Scalapack CF, enter the matrix dimension: " matrixDim
        # Write to plotInstructions file
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodesRoot}_\${matrixDim}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLength \$startNumNodesRoot \$endNumNodesRoot \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodesRoot \$endNumNodesRoot \$jumpNumNodes \$jumpNumNodesoperator \$matrixDim
      elif [ \$scale == 'WS' ]
      then
        read -p "In this weak scaling test for Scalapack CF, enter the matrix dimension: " matrixDim
        read -p "In this weak scaling test for Scalapack CF, enter factor by which to increase matrix dimension: " jumpMatrixDim
        read -p "In this weak scaling test for Scalapack CF, enter arithmetic operator by which to increase the matrix dimension by the amount specified above: add[1], subtract[2], multiply[3], divide[4]: " jumpMatrixDimoperator
        # Write to plotInstructions file
	if [ \${j} == 1 ]
	then
          echo "echo \"\${binaryTag}_\${scale}_\${numIterations}_\${startNumNodesRoot}_\${matrixDim}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        fi
	echo "echo \"\$(findCountLength \$startNumNodesRoot \$endNumNodesRoot \$jumpNumNodesoperator \$jumpNumNodes)\"" >> $SCRATCH/${fileName}/plotInstructions.sh
        launch\$binaryTag \$scale \$binaryPath \$numIterations \$startNumNodesRoot \$endNumNodesRoot \$jumpNumNodes \$jumpNumNodesoperator \$matrixDim \$jumpMatrixDim \$jumpMatrixDimoperator
      fi
    fi
  done
done
EOF
