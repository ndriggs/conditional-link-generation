#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --job-name=big_batch
#SBATCH --output=job_output/train_%A_%a.out
#SBATCH --error=job_output/train_%A_%a.err
#SBATCH --array=0-23
#SBATCH --mem-per-cpu=16384M   # memory per CPU core
#SBATCH --qos=dw87

data=("scaled" "normalized")
model_types=("mlp" "cnn")
lr_schedules=("poly" "sinexp")
batch_sizes=(32 64 128)

data_index=$((SLURM_ARRAY_TASK_ID % ${#data[@]}))
model_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} % ${#model_types[@]}))
lr_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#model_types[@]} % ${#lr_schedules[@]}))
bs_index=$((SLURM_ARRAY_TASK_ID / ${#data[@]} / ${#model_types[@]} / ${#lr_schedules[@]}))

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -m dimensionless_wildfire.training.train --data ${data[$data_index]} \
        --model ${model_types[$model_index]} --lr_schedule ${lr_schedules[$lr_index]} \
        --batch_size ${batch_sizes[$bs_index]}