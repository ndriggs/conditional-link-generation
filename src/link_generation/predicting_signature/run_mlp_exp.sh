#!/bin/bash

#SBATCH --time=03:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --job-name=mlp_batch
#SBATCH --output=job_output/equal_mlp_%A_%a.out
#SBATCH --error=job_output/equal_mlp_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mem-per-cpu=64384M   # memory per CPU core

# data=("clip" "log")
# classification=("true" "false")
# hidden=(200 500 1000)
# dropout=(0 0.2 0.3)

data=("log")
classification=("false")
hidden=(1950)
dropout=(0.2 0.25 0.3)


data_index=$((SLURM_ARRAY_TASK_ID % ${#data[@]}))
classification_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} % ${#classification[@]}))
hidden_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} % ${#hidden[@]}))
dropout_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} / ${#hidden[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m src.link_generation.predicting_signature.predict_signature --model="mlp" --preprocessing=${data[$data_index]} \
        --classification=${classification[$classification_index]} --hidden_size=${hidden[$hidden_index]} \
        --dropout=${dropout[$dropout_index]}