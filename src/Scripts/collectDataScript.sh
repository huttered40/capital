#!/bin/bash

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " resultsPath
if [ ! -d "../Results/${resultsPath}" ];
then
  mkdir ../Results/${resultsPath}/
  mv ${SCRATCH}/${resultsPath}/* ../Results/${resultsPath}/
fi

cd ..
git add -A && git commit -m "Added new data files: ${resultsPath}, and signifying intent to start plotting with recent data."
git push origin master
