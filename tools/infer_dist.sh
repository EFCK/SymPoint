#!/usr/bin/env bash

export PYTHONPATH=./
GPUS=1
workdir=.
OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/inference.py \
	$workdir/configs/svg/svg_pointT.yaml  $workdir/configs/lr00008_freeze5_fp2_dropout01_bh_r10/best.pth --out ./results/manuel_split/lr00008_freeze5_fp2_dropout01_bh_r10/ --datadir dataset/json/manuel_split_fp2_test/
