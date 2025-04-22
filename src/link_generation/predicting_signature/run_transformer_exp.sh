#!/bin/bash

# #SBATCH --time=2:00:00   # walltime
#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --job-name=transformer_batch
#SBATCH --output=job_output/equal_transformer_%A_%a.out
#SBATCH --error=job_output/equal_transformer_%A_%a.err
#SBATCH --array=0
#SBATCH --mem-per-cpu=64384M   # memory per CPU core


# data=("remove_cancelations" "do_nothing")
# classification=("true" "false")
# d_model=(16 32 64)
# nheads=(2 4)
# n_layers=(8 24)

data=("do_nothing")
classification=("true")
d_model=(256)
nheads=(2)
n_layers=(3)


data_index=$((SLURM_ARRAY_TASK_ID % ${#data[@]}))
classification_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} % ${#classification[@]}))
d_model_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} % ${#d_model[@]}))
nheads_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} / ${#d_model[@]} % ${#nheads[@]}))
n_layers_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} / ${#d_model[@]} / ${#nheads[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m link_generation.predicting_signature.predict_signature --model="transformer_encoder" \
        --preprocessing=${data[$data_index]} --classification=${classification[$classification_index]} \
        --d_model=${d_model[$d_model_index]} --nheads=${nheads[$nheads_index]} --num_layers=${n_layers[$n_layers_index]}