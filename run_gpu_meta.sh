#!/bin/bash


sbatch run_gpu.sh 0.005
sleep 1
sbatch run_gpu.sh 0.0005
