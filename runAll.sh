#!/bin/sh

a=0
while [ $a -lt 3 ]
do
  ./run.sh $a

  a=`expr $a + 1`
done