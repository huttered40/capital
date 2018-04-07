#!/bin/bash

if [ "$(hostname |grep "porter")" != "" ]
then
  export SCRATCH=../../../PAA_data
  export RESULTSPATH=../../../PAA_data
elif [ "$(hostname |grep "mira")" != "" ] || [ "$(hostname |grep "cetus")" != "" ]
then
  export SCRATCH=/projects/QMCat/huttered
  export RESULTSPATH=../../../PAA_data
elif [ "$(hostname |grep "theta")" != "" ]
then
  export SCRATCH=/projects/QMCat/huttered
  export RESULTSPATH=../../../PAA_data
elif [ "$(hostname |grep "stampede2")" != "" ]
then
  echo "dog"
fi

read -p "Enter the directory name within ${SCRATCH} where the results are hidden: " resultsDir
cp -r ${SCRATCH}/${resultsDir}/* ${RESULTSPATH}/${resultsDir}/

cd ${RESULTSPATH}

# Get rid of the binaries before making the tarball
rm -rf ${resultsDir}/bin
tar -cvf ${resultsDir}.tar ${resultsDir}/*

cd -
# Push the changes, which should just a single file - collectInstructions.sh
git add -A && git commit -m "Commiting updated collectInstructions.sh, which contains useful info for plotting on local machine."
git push origin master
