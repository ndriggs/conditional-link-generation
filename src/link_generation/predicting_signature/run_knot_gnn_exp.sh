#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --job-name=gnn_bidirectional
#SBATCH --output=job_output/equal_laplacian_%A_%a.out
#SBATCH --error=job_output/equal_laplacian_%A_%a.err
#SBATCH --array=0-3
#SBATCH --mem-per-cpu=32384M   # memory per CPU core

k_num=(3)
hidden=(24 28)
n_heads=(3 4)

index=$((SLURM_ARRAY_TASK_ID % ${#k_num[@]}))
hidden_index=$((SLURM_ARRAY_TASK_ID / ${#k_num[@]} % ${#hidden[@]}))
n_heads_index=$((SLURM_ARRAY_TASK_ID / ${#k_num[@]} / ${#hidden[@]} % ${#n_heads[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m link_generation.predicting_signature.predict_signature --model="knot_gnn" --preprocessing="remove_cancellations" \
        --classification="false" --dropout=0 --double_features="true" \
        --ohe_inverses="true" --num_layers=4 --hidden_size=${hidden[$hidden_index]} \
        --nheads=${n_heads[$n_heads_index]} --both="false" --pos_neg="false" --undirected="true" --k=${k_num[$index]}