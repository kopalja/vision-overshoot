#!/bin/bash
#SBATCH --account=p904-24-3
#SBATCH --mail-user=<jakub.kopal@kinit.sk>

## Nodes allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
set -xe

# Not using this one
##SBATCH --time=09:00:00 # Estimate to increase job priority

eval "$(conda shell.bash hook)"
conda activate overshoot

nvidia-smi
# Changes:
# 1) Not using --model-ema
# 2) Batch 512 and --nproc_per_node=2


# OVERSHOOT 5
torchrun --nproc_per_node=2 train.py --job-name overshoot --opt sgdo --overshoot 5 --data-path /projects/p904-24-3/imagenet --model resnet50 --batch-size 512 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --val-resize-size 232 --ra-sampler --ra-reps 4


# BASELINE
# torchrun --nproc_per_node=2 train.py --job-name baseline --opt sgd --overshoot 0 --data-path /projects/p904-24-3/imagenet --model resnet50 --batch-size 512 --lr 0.5 \
# --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
# --auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
# --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
# --train-crop-size 176 --val-resize-size 232 --ra-sampler --ra-reps 4


# Original configuration
# torchrun --nproc_per_node=2 train.py  --data-path /projects/p904-24-3/imagenet --model resnet50 --batch-size 512 --lr 0.5 \
# --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
# --auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
# --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
# --train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps 4