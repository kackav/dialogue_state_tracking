#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

metric="${METRIC:-slot_k_f1}"
dataset="${DATASET:-multiwoz}"

"${PYTHON}" "${ROOT_DIR}/picker.py" --metric "${metric}" --dataset "${dataset}" "$@"
