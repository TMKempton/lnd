#!/bin/bash


LOGDIR="experiments/measure_distortion/logs"
mkdir -p $LOGDIR 

# Arithmetic means
for p in 65 88 95
do
       sbatch --gres=gpu:1 \
              --qos=normal \
              --cpus-per-task=1 \
              --mem=80G \
              --output=$LOGDIR/slurm_%j.out \
              --wrap="python3 experiments/measure_distortion/ratios/compute_top_p_ratios.py $p"
done


# Geometric means
# for p in 69 88 93
# do
#        sbatch --gres=gpu:1 \
#               --qos=normal \
#               --cpus-per-task=1 \
#               --output=$LOGDIR/slurm_%j.out \
#               --wrap="python3 experiments/measure_distortion/ratios/compute_top_p_ratios.py $p"
# done
