#!/bin/sh

tasks="0 1 2"
scripts="modelTrainer.py attackGeneration.py evaluation.py"

for tsk in $tasks; do
  export TASK=$tsk

  echo "\n\n[🚀 PERFORMING TASK] $TASK"

  for script in $scripts; do
    echo "\n[🎈 EXECUTING SCRIPT] $script"
    python3 $script
  done
done