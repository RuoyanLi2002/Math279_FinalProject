#!/bin/bash

LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"

PASSED=()
FAILED=()
SKIPPED=()

run_eval() {
  local name="$1"
  local model_path="$2"
  local script="$3"
  local log_file="$LOG_DIR/${name}.log"

  if [ ! -f "$model_path" ]; then
    echo "[SKIP] $name — model.pth not found"
    SKIPPED+=("$name")
    return
  fi

  echo "[RUN]  $name ..."
  if bash "$script" > "$log_file" 2>&1; then
    echo "[OK]   $name — results saved to $log_file"
    PASSED+=("$name")
  else
    echo "[FAIL] $name — see $log_file for details"
    FAILED+=("$name")
  fi
}

run_eval "GRU_Probabilistic"       "exp/GRU_Probabilistic/run0/model.pth"       "run_eval_gru.sh"
run_eval "GRU_Deterministic"       "exp/GRU_Deterministic/run0/model.pth"       "run_eval_gru_deterministic.sh"
run_eval "LSTM_Probabilistic"      "exp/LSTM_Probabilistic/run0/model.pth"      "run_eval_lstm.sh"
run_eval "LSTM_Deterministic"      "exp/LSTM_Deterministic/run0/model.pth"      "run_eval_lstm_deterministic.sh"
run_eval "Transformer_Probabilistic"  "exp/Transformer_Probabilistic/run0/model.pth"  "run_eval_transformer.sh"
run_eval "Transformer_Deterministic"  "exp/Transformer_Deterministic/run0/model.pth"  "run_eval_transformer_deterministic.sh"
run_eval "Transolver_Probabilistic"   "exp/Transolver_Probabilistic/run0/model.pth"   "run_eval_transolver.sh"
run_eval "Transolver_Deterministic"   "exp/Transolver_Deterministic/run0/model.pth"   "run_eval_transolver_deterministic.sh"
run_eval "Transolver_p_Probabilistic" "exp/Transolver_p_Probabilistic/run0/model.pth" "run_eval_transolver_p.sh"
run_eval "Transolver_p_Deterministic" "exp/Transolver_p_Deterministic/run0/model.pth" "run_eval_transolver_p_deterministic.sh"
run_eval "Diffusion"               "exp/Diffusion/run0/model.pth"               "run_eval_diffusion.sh"
run_eval "RITDiffusion"            "exp/RITDiffusion/run0/model.pth"            "run_eval_ritdiffusion.sh"
run_eval "Ours"                    "exp/Ours/run0/model.pth"                    "run_eval_ours.sh"

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Passed  (${#PASSED[@]}): ${PASSED[*]}"
echo "Failed  (${#FAILED[@]}): ${FAILED[*]}"
echo "Skipped (${#SKIPPED[@]}): ${SKIPPED[*]}"

if [ ${#FAILED[@]} -gt 0 ]; then
  exit 1
fi