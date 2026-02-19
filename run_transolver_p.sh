#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

python main.py \
    --exp_name "exp/Transolver_p/run0" \
    --seed 0 \
    --dataset_root "/local2/shared_data/us_crsp_nyse" \
    --data_save_path "" \
    --seq_length 20 \
    --split_interval 9 \
    --learning_rate 1e-4 \
    --end_learning_rate 1e-6 \
    --num_epochs 100 \
    --batch_size 128 \
    --save_freq 1 \
    --to_train \
    --model_path "" \
    --config "config/transolver_p.yaml"