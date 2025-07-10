#!/bin/bash


LOGDIR="experiments/measure_distortion/logs"
mkdir -p $LOGDIR 

for k in 5 50 150
do
       sbatch --gres=gpu:1 \
              --qos=normal \
              --cpus-per-task=1 \
              --output=$LOGDIR/slurm_%j.out \
              --wrap="python3 experiments/measure_distortion/ratios/compute_top_k_ratios.py $k"
done

