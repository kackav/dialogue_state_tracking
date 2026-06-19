#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${DATASETS_CONFIG:?Set DATASETS_CONFIG to the datasets YAML path.}"
: "${PRETRAINED_DIR:?Set PRETRAINED_DIR to the checkpoint directory, for example outputs/run/best.}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-dst_inference}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${EXPERIMENT_NAME}}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HOME

mkdir -p "${OUTPUT_DIR}"

args=(
  --output_dir "${OUTPUT_DIR}"
  --datasets_config "${DATASETS_CONFIG}"
  --pretrained_dir "${PRETRAINED_DIR}"
  --encoder_model_name "${ENCODER_MODEL_NAME:-microsoft/wavlm-large}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}"
  --dataloader_num_workers_val "${DATALOADER_NUM_WORKERS_VAL:-2}"
  --metric "${METRIC:-jga}"
)

if [[ -n "${LM_MODEL_NAME:-}" ]]; then
  args+=(--lm_model_name "${LM_MODEL_NAME}")
fi

if [[ -n "${ENCODER_DIR:-}" ]]; then
  args+=(--encoder_dir "${ENCODER_DIR}")
fi

if [[ -n "${BOS_TOKEN:-}" ]]; then
  args+=(--bos_token "${BOS_TOKEN}")
fi

if [[ "${USE_BOS_FROM_LM:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  args+=(--use_bos_from_lm)
fi

if [[ -n "${DOMAINS_TO_IGNORE:-}" ]]; then
  read -r -a domains <<< "${DOMAINS_TO_IGNORE}"
  args+=(--domains_to_ignore "${domains[@]}")
fi

if [[ -n "${SLOTS_TO_IGNORE:-}" ]]; then
  read -r -a slots <<< "${SLOTS_TO_IGNORE}"
  args+=(--slots_to_ignore "${slots[@]}")
fi

HF_HOME=/path/to/huggingface \
  torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=${N_GPUS} "${ROOT_DIR}/src/inference_generate.py" "${args[@]}" "$@"

esac
