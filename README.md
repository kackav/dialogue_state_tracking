# dialogue_state_tracking
DST training/fine-tuning and DST inference using a speech encoder, connector, and causal language model.
There is an ASR pretraining phase and then the DST finetuning one.
These codes were used for the results in the paper: [Joint Speech And Text Training for LLM-based End-To-End Spoken Dialogue State Tracking](https://www.arxiv.org/pdf/2511.22503)

## Contents
The codes for training the model are in src/:
- `train_asr.py`: ASR pretraining code.
- `train_dst.py`: DST training code.
- `data.py`: Hugging Face dataset wrappers and collators.
- `models_asr.py`: WavLM encoder wrapper, connector, text encoder, and LM composition for ASR.
- `models_asr.py`: WavLM encoder wrapper, connector, text encoder, and LM composition for DST.
- `inference_generate.py`: multi-turn DST generation/evaluation entrypoint.
- `compute_metrics.py`: DST training metrics.
- `picker.py`: helper to select a checkpoint from metric YAML files.
The dataset configuration template is in configs/:
- `configs/datasets.example.yaml`: dataset configuration template.

## Setup

Create or activate an environment with PyTorch, Hugging Face Transformers, Accelerate, PEFT, Datasets, and the small metric/parsing dependencies:

```bash
conda env create -n name --file requirements.yaml

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

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

The data used for our training are available at: [huggingface.co/datasets/vendrkat/](https://huggingface.co/datasets/vendrkat/)

## Training

ASR pretraining:

```bash
DATASETS_CONFIG=configs/datasets.yaml \
EXPERIMENT_NAME=asr_gemma_connector \
LM_MODEL_NAME=google/gemma-3-1b-it \
ENCODER_MODEL_NAME=microsoft/wavlm-large \
scripts/train_asr.sh
```

Minimal DST run:

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

