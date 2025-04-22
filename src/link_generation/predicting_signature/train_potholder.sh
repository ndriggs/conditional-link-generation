#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --job-name=potholder_sig
#SBATCH --output=job_output/potholder_sig_%A_%a.out
#SBATCH --error=job_output/potholder_sig_%A_%a.err
#SBATCH --array=0-3
#SBATCH --mem-per-cpu=32384M   # memory per CPU core

potholder=(5 7 9 11)

index=$((SLURM_ARRAY_TASK_ID % ${#potholder[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m link_generation.predicting_signature.predict_potholder_sig --potholder_size=${potholder[$index]}