#!/bin/bash

#SBATCH --time=1:45:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --job-name=reformer_batch
#SBATCH --output=job_output/gnn_%A_%a.out
#SBATCH --error=job_output/gnn_%A_%a.err
#SBATCH --array=0-47
#SBATCH --mem-per-cpu=64384M   # memory per CPU core

data=("remove_cancelations" "do_nothing")
classification=("true" "false")
dropout=(0 0.25 0.4)
ohe_inverses=("true" "false")
n_layers=(3 5)


data_index=$((SLURM_ARRAY_TASK_ID % ${#data[@]}))
classification_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} % ${#classification[@]}))
dropout_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} % ${#dropout[@]}))
ohe_inverses_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} / ${#dropout[@]} % ${#ohe_inverses[@]}))
n_layers_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#classification[@]} / ${#dropout[@]} / ${#ohe_inverses[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m link_generation.predicting_signature.predict_signature --model="gnn" --preprocessing=${data[$data_index]} \
        --classification=${classification[$classification_index]} --dropout=${dropout[$dropout_index]} \
        --ohe_inverses=${ohe_inverses[$ohe_inverses_index]} --num_layers=${n_layers[$n_layers_index]} --hidden_size=16 \
        --nheads=2