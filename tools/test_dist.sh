#!/usr/bin/env bash

export PYTHONPATH=./
GPUS=1
workdir=.
OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/test.py \
	$workdir/configs/svg/svg_pointT.yaml  $workdir/configs/svg/best.pth --dist
