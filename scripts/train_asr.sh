#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
LAUNCHER="${LAUNCHER:-python}"

: "${DATASETS_CONFIG:?Set DATASETS_CONFIG to the datasets YAML path.}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-asr_gemma_connector}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/runs/${EXPERIMENT_NAME}}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Minimal default args; users can append extra flags when invoking the script
args=(
  --experiment_name "${EXPERIMENT_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
  --datasets_config "${DATASETS_CONFIG}"
  --encoder_model_name "${ENCODER_MODEL_NAME:-microsoft/wavlm-large}"
  --lm_model_name "${LM_MODEL_NAME:-google/gemma-3-1b-it}"
  --max_steps "${MAX_STEPS:-100000}"
)
# ensure N_GPUS has a reasonable default
N_GPUS=${N_GPUS:-1}
echo "N_GPUS=$N_GPUS"

# try to pick free GPUs; fall back to 0..N-1 if free_gpus.py fails or returns empty
GPU_IDS=$(python ~/free_gpus.py "$N_GPUS" 2>/dev/null || true)
if [ -z "$GPU_IDS" ]; then
  # default to using the first N_GPUS GPUs
  GPU_IDS=$(seq -s, 0 $((N_GPUS-1)) 2>/dev/null || echo 0)
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

HF_HOME=/path/to/huggingface \
  torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=${N_GPUS} "${ROOT_DIR}/src/train_asr.py" "${args[@]}" "$@"
