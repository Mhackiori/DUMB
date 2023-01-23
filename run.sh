#!/bin/sh

script="modelTrainer.py"
if [ $1 = "0" ]
then
  script="modelTrainer.py"
elif [ $1 = "1" ]
then
  script="attackGeneration.py"
elif [ $1 = "2" ]
then
  script="evaluation.py"
else
  echo "[🛑 INVALID PARAMETER] Verrà eseguito lo script di default modelTrainer.py"
fi

echo "\n\n\n[🎈 EXECUTING SCRIPT] $script"

a=0
while [ $a -lt 3 ]
do
  export TASK=$a

  echo "\n[🚀 PERFORMING TASK] $TASK"
  python3 $script

  a=`expr $a + 1`
done