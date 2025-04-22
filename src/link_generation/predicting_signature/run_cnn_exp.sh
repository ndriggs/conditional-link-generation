#!/bin/bash

#SBATCH --time=00:45:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --job-name=cnn_batch
#SBATCH --output=job_output/equal_cnn_%A_%a.out
#SBATCH --error=job_output/equal_cnn_%A_%a.err
#SBATCH --array=0
#SBATCH --mem-per-cpu=64384M   # memory per CPU core

# data=("clip" "log")
# classification=("true" "false")
# kernel=(2 3)
# layer=("true" "false")

data=("log")
classification=("false")
kernel=(3)
hidden=(1)


data_index=$((SLURM_ARRAY_TASK_ID % ${#data[@]}))
classification_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} % ${#classification[@]}))
kernel_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} % ${#kernel[@]}))
hidden_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} / ${#kernel[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m link_generation.predicting_signature.predict_signature --model="cnn" --preprocessing=${data[$data_index]} \
        --classification=${classification[$classification_index]} --kernel_size=${kernel[$kernel_index]} \
        --layer_norm="false" --hidden_size=${hidden[$hidden_index]}