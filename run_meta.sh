#!/bin/bash

for n in {1..15}; 
do
    sbatch run.sh $n
    sleep 1
done


sleep 4000


for n in {16..30}; 
do
    sbatch run.sh $n
    sleep 1
done


sleep 4000


for n in {31..45}; 
do
    sbatch run.sh $n
    sleep 1
done


sleep 4000


for n in {46..60}; 
do
    sbatch run.sh $n
    sleep 1
done
