#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"

PASSED=()
FAILED=()

for seqlen in 30 40 50 60 80 100; do
  exp_dir="exp/RITDiffusion/seqlen_${seqlen}"
  model_path="${exp_dir}/model.pth"
  log_file="${LOG_DIR}/RITDiffusion_seqlen_${seqlen}.log"

  echo "[RUN]  RITDiffusion seqlen=${seqlen} ..."
  python main.py \
    --exp_name "$exp_dir" \
    --seed 0 \
    --dataset_root "/local2/shared_data/us_crsp_nyse" \
    --data_save_path "" \
    --seq_length "$seqlen" \
    --split_interval 9 \
    --batch_size 128 \
    --probabilistic \
    --predict_return \
    --model_path "$model_path" \
    --config "${exp_dir}/config.yaml" \
    > "$log_file" 2>&1

  if [ $? -eq 0 ]; then
    echo "[OK]   RITDiffusion seqlen=${seqlen} — log: $log_file"
    PASSED+=("seqlen_${seqlen}")
  else
    echo "[FAIL] RITDiffusion seqlen=${seqlen} — see $log_file"
    FAILED+=("seqlen_${seqlen}")
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
