#!/bin/bash

NEW_TOKENS_PER_COMPLETION=15
NUM_COMPLETIONS=30

LOGDIR="experiments/entropy_perplexity_distortion/logs"
mkdir -p $LOGDIR 

for top_k in $(seq 10 200)
do
       sbatch --job-name=ln_top_k \
              --gres=gpu:1 \
              --qos=normal \
              --cpus-per-task=1 \
              --mem=80G \
              --output=$LOGDIR/%x_slurm_%j.out \
              --wrap="python3 experiments/entropy_perplexity_distortion/ln_top_k.py $top_k $NEW_TOKENS_PER_COMPLETION $NUM_COMPLETIONS"
done

for hundred_top_p in $(seq 40 90)
do
       sbatch --job-name=ln_top_p \
              --gres=gpu:1 \
              --qos=normal \
              --cpus-per-task=1 \
              --mem=80G \
              --output=$LOGDIR/%x_slurm_%j.out \
              --wrap="python3 experiments/entropy_perplexity_distortion/ln_top_p.py $hundred_top_p $NEW_TOKENS_PER_COMPLETION $NUM_COMPLETIONS"
done

for hundred_temp in $(seq 85 100)
do
       sbatch --job-name=ln_temp\
              --gres=gpu:1 \
              --qos=normal \
              --cpus-per-task=1 \
              --mem=80G \
              --output=$LOGDIR/%x_slurm_%j.out \
              --wrap="python3 experiments/entropy_perplexity_distortion/ln_temp.py $hundred_temp $NEW_TOKENS_PER_COMPLETION $NUM_COMPLETIONS"
done

for top_k in $(seq 10 200)
do
       sbatch --job-name=gn_top_k_$top_k \
              --gres=gpu:1 \
              --qos=normal \
              --cpus-per-task=1 \
              --mem=80G \
              --output=$LOGDIR/%x_slurm_%j.out \
              --wrap="python3 experiments/entropy_perplexity_distortion/gn_top_k.py $top_k $NEW_TOKENS_PER_COMPLETION $NUM_COMPLETIONS"
done


for hundred_top_p in $(seq 40 90)
do
       sbatch --job-name=global_top_p_rejection \
              --gres=gpu:1 \
              --qos=normal \
              --cpus-per-task=1 \
              --mem=80G \
              --output=$LOGDIR/%x_slurm_%j.out \
              --wrap="python3 experiments/entropy_perplexity_distortion/gn_top_p.py $hundred_top_p $NEW_TOKENS_PER_COMPLETION $NUM_COMPLETIONS"
done


for hundred_temp in $(seq 85 100)
do
       sbatch --job-name=global_temp_rejection \
              --gres=gpu:1 \
              --qos=normal \
              --cpus-per-task=1 \
              --mem=80G \
              --output=$LOGDIR/%x_slurm_%j.out \
              --wrap="python3 experiments/entropy_perplexity_distortion/gn_temp.py $hundred_temp $NEW_TOKENS_PER_COMPLETION $NUM_COMPLETIONS"
done
