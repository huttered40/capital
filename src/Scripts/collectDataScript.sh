#!/bin/bash

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " resultsPath
mkdir ../Results/${resultsPath}/
mv ${SCRATCH}/${resultsPath}/* ../Results/${resultsPath}/
cd ..
git add -A && git commit -m "Added new data files: ${resultsPath}."
git push origin master
