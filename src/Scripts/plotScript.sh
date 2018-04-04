#!/bin/bash

# Note: must be in PAA/src/Scripts to run this

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " resultsPath

scaplotDir=../../../scaplot/
scriptDir=$(pwd)

# Generate the Makefile for Scaplot
cd ${scaplotDir}
bash ${scriptDir}/../Results/${resultsPath}/plotInstructions.sh | bash MakePlotScript.sh
