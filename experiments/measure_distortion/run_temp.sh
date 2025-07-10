#!/bin/bash


LOGDIR="experiments/measure_distortion/logs"
mkdir -p $LOGDIR 

# Arithmetic means
for t in 86 95 98
do
       sbatch --gres=gpu:1 \
              --qos=normal \
              --cpus-per-task=1 \
              --output=$LOGDIR/slurm_%j.out \
              --wrap="python3 experiments/measure_distortion/ratios/compute_temp_ratios.py $t" 
done

# Geometric means
# for t in 87 95 97
# do
#        sbatch --gres=gpu:1 \
#               --qos=normal \
#               --cpus-per-task=1 \
#               --output=$LOGDIR/slurm_%j.out \
#               --wrap="python3 experiments/measure_distortion/ratios/compute_temp_ratios.py $t" 
# done
