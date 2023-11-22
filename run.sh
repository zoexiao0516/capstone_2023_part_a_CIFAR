#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition ccn
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH -o ./out/%j.out

#SBATCH --ntasks-per-node=8           # number of tasks per node
#SBATCH --cpus-per-task=16            # number of cpus(threads) per task
#SBATCH --constraint=rome,ib          # computers to use at FI

#SBATCH --mail-type=FAIL

# Set up our environment for this SLURM submission
module -q purge                       # purge current modules
module -q load openmpi                # Load openmpi
module list                           # What modules are loaded?

# Helper functions to see what kind of system we are running on, if we have GPUs that are accessible, and other information
lscpu                                 # What cpus do we have?
nvidia-smi                            # Is there gpu information?
numactl -H                            # What is the NUMA layout

# Print some helpful information
echo "Slurm nodes:              ${SLURM_NNODES}"
echo "Slurm ntasks:             ${SLURM_NTASKS}"
echo "Slurm ntasks-per-node:    ${SLURM_NTASKS_PER_NODE}"
echo "Slurm cpus-per-task:      ${SLURM_CPUS_PER_TASK}"

# Arguments: dataset file, out directory, params file
python create_json.py $1


