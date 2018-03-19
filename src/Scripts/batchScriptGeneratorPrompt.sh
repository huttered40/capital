#!/bin/bash

echo $'\n'
echo "Things to note: With operators, 1 -> +"
echo "                                2 -> -"
echo "                                3 -> *"
echo "                                4 -> /"

echo $'\n'
echo "When generating the script via batchScriptGenerator.sh, the following must be specified as command-line arguments:"
echo "[machine name (cetus,mira,theta,stampede2,bw)] [script name]  [number of nodes]  [processes per node]  [number of binaries to test on] [number of hours for run]"
echo "[number of minutes for run]  [number of seconds for run]"

echo $'\n'

echo "Once script has been generated, the following must be specied as command-line arguments:"
echo "For each of the [number of binaries to test on], you must specify:"
echo "  [binary path and name]"
echo "  Binary Tag [MM3D,CFR3D,CQR2,SCALA_QR,SCALA_CF]"
echo " [SS,WS]"
echo "  [number of iterations for each specific run]"
echo "  [number of processors to start] [number of processors to end] [jump amount] [jump operator (+,-,*,/)]"
echo "  [dimension m to start] [dimension m to end] [jump amount] [jump operator (+,-,*,/)]"
echo "  if not CFR3D, specify: [dimension n to start] [dimension n to end] [jump amount] [jump operator (+,-,*,/)]"
echo $'\n'
echo "    Based on Binary Tag, more options are available."
echo "    If Binary Tag == MM3D"
echo "      specify:"
echo "          [dimension k to start] [dimension k to end] [jump amount] [jump operator (+,-,*,/)]"
echo "          No option to try TRMM-based MM3D, just the normal GEMM"
echo "          Only performance testing option, not sequential validation, is allowed."
echo "          Choice of Broadcast [0] or Allgather [1] based routine"
echo $'\n'
echo "    If Binary Tag == CFR3D",
echo "      specify:"
echo "          [start of range of the number of powers of 2 greater than the theoretically optimal base case] [end of that range]"
echo "    If Binary Tag == CholeskyQR2_1D, specify: nothing is needed"
echo "    If Binary Tag == CholeskyQR2_3D, specify: nothing is needed"
echo "    If Binary Tag == CholeskyQR2_Tunable, specify: [processor grid dimension d to start] [processor grid dimension d to end] [jump amount] [jump operator (+,-,*,/]"
echo "    						 [processor grid dimension c to start] [processor grid dimension c to end] [jump amount] [jump operator (+,-,*,/]"
echo "    If Binary Tag == scaLAPACK_QR, specify:        [blocksize to start] [blocksize to end] [jump amount] [jump operator (+,-,*,/]"
echo "    						 [number of processor rows to start] [number of processor rows to end] [jump amount] [jump operator (+,-,*,/]"
