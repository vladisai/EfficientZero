#!/bin/bash

#SBATCH --job-name=speed_test
#SBATCH --output=/checkpoint/vladsobal/results/EfficientZero/%j.out
#SBATCH --error=/checkpoint/vladsobal/results/EfficientZero/%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1

#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --gres=gpu:8
#SBATCH --mem=420G
#SBATCH --cpus-per-task=80
#SBATCH --partition=devlab
#SBATCH --constraint=volta32gb

eval "$(conda shell.bash hook)"
conda activate efficient_zero_39

cd /private/home/vladsobal/work/EfficientZero
set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

seed=$1
cpus=$2

python main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --force \
  --num_gpus 8 --num_cpus ${cpus} --cpu_actor 7 --gpu_actor 60 \
  --seed $seed \
  --p_mcts_num 8 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info "EfficientZero-V1-Fast-Best-v2-${cpus}-cpus"
