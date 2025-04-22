#!/bin/bash

#SBATCH --time=5:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --job-name=knot_gnn_batch
#SBATCH --output=job_output/knot_gnn_%A_%a.out
#SBATCH --error=job_output/knot_gnn_%A_%a.err
#SBATCH --array=0-7
#SBATCH --mem-per-cpu=64384M   # memory per CPU core

data=("remove_cancelations" "do_nothing")
classification=("true" "false")
pos_neg=("true" "false")


data_index=$((SLURM_ARRAY_TASK_ID % ${#data[@]}))
classification_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} % ${#classification[@]}))
pos_neg_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m link_generation.predicting_signature.predict_signature --model="knot_gnn" --preprocessing="${data[$data_index]}" \
        --classification=${classification[$classification_index]} --dropout=0 \
        --ohe_inverses="true" --num_layers=5 --hidden_size=16 \
        --nheads=2 --both="false" --pos_neg=${pos_neg[$pos_neg_index]}
