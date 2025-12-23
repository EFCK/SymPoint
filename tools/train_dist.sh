#!/usr/bin/env bash


export PYTHONPATH=./
GPUS=1

OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
	configs/svg/svg_pointT.yaml \
  --dist   \
	--exp_name sympointv1_deneme \
	--work_dir /temp/work_dir/ \
	--resume /content/drive/MyDrive/My_Computer/sympoint_data/svg/best.pth \
