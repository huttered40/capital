launch$tag2 () {
  # launch CFR3D
  if [ \$1 == 'SS' ]
  then
    local startNumNodes=\$4
    local endNumNodes=\$5
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        # Write to plotInstructions file
	echo "echo \"\${9}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	echo "echo \"\${10}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

        local fileString="results/results_${tag2}_\$1_\${startNumNodes}nodes_\${8}side_\${9}dim_0bcMult_\${10}inverseCutOffMult_0panelDimMult_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 \$8 \${9} 0 \${10} 0 \$3 $SCRATCH/${fileName}/\${fileString}
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
    done
  elif [ \$1 == 'WS' ]
  then
    local startNumNodes=\$4
    local startMatrixDim=\${9}
    local endNumNodes=\$5
    while [ \$startNumNodes -le \$endNumNodes ];
    do
        # Write to plotInstructions file
	echo "echo \"\${startMatrixDim}\"" >> $SCRATCH/${fileName}/plotInstructions.sh
	echo "echo \"\${12}\"" >> $SCRATCH/${fileName}/plotInstructions.sh

        local fileString="results/results_${tag2}_\$1_\${startNumNodes}nodes_\${8}side_\${startMatrixDim}dim_0bcMult_\${12}inverseCutOffMult_0panelDimMult_\${3}numIter"
        launchJobs \${fileString} \$startNumNodes \$2 \$8 \${startMatrixDim} 0 \${12} 0 \$3 $SCRATCH/${fileName}/\${fileString}
        startNumNodes=\$(updateCounter \$startNumNodes \$7 \$6)
        startMatrixDim=\$(updateCounter \$startMatrixDim \${11} \${10})
    done
  fi
}

# This will need some fixing
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
          local fileString="results/results_${tag5}_\$1_\$((\$startNumNodesRoot * \$startNumNodesRoot))nodes_\${matrixDimRoot}dimM_\${temp3}blockSize_\${3}numIter"
          launchJobs \${fileString} \$curNumNodes \$2 \${matrixDimRoot} \${specialMatDimLog} \${startBlockSize} \${3} $SCRATCH/${fileName}/\${fileString}
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
          local fileString="results/results_${tag5}_\$1_\$((\$startNumNodesRoot * \$startNumNodesRoot))nodes_\${matrixDimRoot}dimM_\$((\$temp3))blockSize_\${3}numIter"
          launchJobs \${fileString} \$curNumNodes \$2 \${matrixDimRoot} \${specialMatDimLog} \${startBlockSize} \${3} $SCRATCH/${fileName}/\${fileString}
          startBlockSize=\$((\$startBlockSize + 1))
      done
      startNumNodesRoot=\$(updateCounter \$startNumNodesRoot \$7 \$6)
      matrixDim=\$(updateCounter \$matrixDim \${10} \$9)
      matrixDimRoot=\$(log2 \$matrixDim)
    done
  fi
}
    if [ \$binaryTag == 'cfr3d' ]
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
EOF
