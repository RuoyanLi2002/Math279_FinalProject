#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"

PASSED=()
FAILED=()

for run in 0 1; do
  exp_dir="exp/RITTransformer/run${run}"
  log_file="${LOG_DIR}/RITTransformer_run${run}.log"

  echo "[RUN]  RITTransformer run${run} ..."
  /data1/hhzhang/miniconda3/envs/math279/bin/python main.py \
    --exp_name "$exp_dir" \
    --seed 0 \
    --dataset_root "/local2/shared_data/us_crsp_nyse" \
    --data_save_path "" \
    --seq_length 20 \
    --split_interval 9 \
    --batch_size 128 \
    --probabilistic \
    --predict_return \
    --model_path "${exp_dir}/model.pth" \
    --config "${exp_dir}/config.yaml" \
    > "$log_file" 2>&1

  if [ $? -eq 0 ]; then
    echo "[OK]   RITTransformer run${run} — log: $log_file"
    PASSED+=("run${run}")
  else
    echo "[FAIL] RITTransformer run${run} — see $log_file"
    FAILED+=("run${run}")
  fi
done

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Passed  (${#PASSED[@]}): ${PASSED[*]}"
echo "Failed  (${#FAILED[@]}): ${FAILED[*]}"

if [ ${#FAILED[@]} -gt 0 ]; then
  exit 1
fi
