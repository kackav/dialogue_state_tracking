# dialogue_state_tracking
DST training/fine-tuning and DST inference using a speech encoder, connector, and causal language model.
The ASR/encoder pretraining phase is intentionally not included here. 

## Contents

- `train_asr.py`: DST training entrypoint.
- `inference_generate.py`: multi-turn DST generation/evaluation entrypoint.
- `data_asr.py`: Hugging Face dataset wrappers and collators.
- `models_asr.py`: WavLM encoder wrapper, connector, text encoder, and LM composition.
- `compute_metrics.py`: DST training metrics.
- `picker.py`: helper to select a checkpoint from metric YAML files.
- `scripts/train_dst.sh`: configurable local or distributed training launcher.
- `scripts/infer_dst.sh`: configurable inference launcher.
- `scripts/submit_train_lumi.sh`: Slurm submission wrapper for LUMI.
- `scripts/submit_infer_lumi.sh`: Slurm submission wrapper for LUMI inference.
- `scripts/evaluate_multiwoz.sh`: optional wrapper for the external MultiWOZ evaluator.
- `configs/datasets.example.yaml`: dataset configuration template.

## Setup

Create or activate an environment with PyTorch, Hugging Face Transformers, Accelerate, PEFT, Datasets, and the small metric/parsing dependencies:

```bash
pip install -r requirements.txt
```

On LUMI, use the project environment that already provides ROCm PyTorch, FlashAttention support where available, and the correct Hugging Face cache.

Set the Hugging Face cache before running:

```bash
export HF_HOME=/path/to/huggingface
```

The loader first looks for prepared datasets under:

```text
$HF_HOME/modules/datasets_modules/datasets/prep_dial_<dataset_name>_<split>
```

If the prepared dataset is not present, it falls back to `datasets.load_dataset(...)` using the `name` and `split` fields in the YAML.

## Dataset Config

Start from:

```bash
cp configs/datasets.example.yaml configs/datasets.yaml
```

Then edit `configs/datasets.yaml` so the keys match your prepared dataset names. The training code expects at least:

- `train`
- `validation`

If `--text_input` is enabled, it also expects:

- `text_train`

## Training

Minimal run:

```bash
DATASETS_CONFIG=configs/datasets.yaml \
EXPERIMENT_NAME=dst_gemma_connector \
LM_MODEL_NAME=google/gemma-3-1b-it \
ENCODER_MODEL_NAME=microsoft/wavlm-large \
scripts/train_dst.sh
```

Resume a run:

```bash
DATASETS_CONFIG=configs/datasets.yaml \
OUTPUT_DIR=outputs/dst_gemma_connector \
RESUME=1 \
scripts/train_dst.sh
```

Continue from an ASR-pretrained encoder/connector:

```bash
DATASETS_CONFIG=configs/datasets.yaml \
PRETRAINED_DIR=/path/to/asr_pretrained_checkpoint/best \
scripts/train_dst.sh
```

The wrapper accepts extra `train_asr.py` arguments after the environment-driven defaults:

```bash
DATASETS_CONFIG=configs/datasets.yaml scripts/train_dst.sh --max_steps 20000 --validation_steps 1000
```

## LUMI Slurm

Submit a training job:

```bash
DATASETS_CONFIG=/users/$USER/scripts/datasets.yaml \
EXPERIMENT_NAME=dst_gemma_connector \
sbatch scripts/submit_train_lumi.sh
```

Most parameters can be overridden as environment variables, for example:

```bash
sbatch \
  --account=account_name \
  --time=48:00:00 \
  --nodes=2 \
  --ntasks-per-node=8 \
  scripts/submit_train_lumi.sh
```

## Inference

Generate DST predictions from a trained checkpoint:

```bash
DATASETS_CONFIG=configs/datasets.yaml \
PRETRAINED_DIR=outputs/dst_gemma_connector/best \
OUTPUT_DIR=outputs/dst_gemma_connector/inference_best \
scripts/infer_dst.sh
```

The script writes per-dataset prediction files such as:

```text
predictions_<dataset>_all.json
metrics.yaml
transcriptions.yaml
reference.yaml
```

## Checkpoint Selection

Use `picker.py` to select a checkpoint from one or more metric YAML files:

```bash
python picker.py --metric slot_k_f1 --dataset multiwoz outputs/*/metrics.yaml
```

Add `--minimize` for metrics where lower is better.

## Notes

- `train_lm` currently references `args.peak_lm_lr` in the training code, but that argument is not defined. Do not enable `--train_lm` until that is fixed.
- The launcher defaults mirror the previous LUMI experiment settings, but all paths are now configurable.
- `models_asr.py` still uses the `WavLMWrapper`; Parakeet or other encoders should be added as a separate code change.
