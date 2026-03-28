#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

python main.py \
    --exp_name "exp/GRU_Probabilistic/run0" \
    --seed 0 \
    --dataset_root "/local2/shared_data/us_crsp_nyse" \
    --data_save_path "" \
    --seq_length 20 \
    --split_interval 9 \
    --batch_size 128 \
    --probabilistic \
    --predict_return \
    --model_path "exp/GRU_Probabilistic/run0/model.pth" \
    --config "exp/GRU_Probabilistic/run0/config.yaml"

