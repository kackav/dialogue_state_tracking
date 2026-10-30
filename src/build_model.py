import yaml
import os
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import BitsAndBytesConfig
import transformers
import torch
import sys
import accelerate
import models_dst as models
import importlib

# Accelerator() shares its state with the one created in train_dst.py
accelerator = accelerate.Accelerator()


def load_encoder(wrapper_path):
    module_name, class_name = wrapper_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    wrapper_class = getattr(module, class_name)
    return wrapper_class


def make_text_encoder(model_config, encoder, tokenizer, lm_hidden_size):
    # use_llm_emb=True -> LM embeddings, False (old default) -> own embeddings
    if model_config.get('use_llm_emb', False):
        return models.TextEncoder(model_config['mask_rate'], vocab_size=None, dim_input=model_config['text_encoder_dim'], dim_output=encoder.hidden_size, num_heads=4, ff_size=4*(model_config['text_encoder_dim']), pad_token=tokenizer.pad_token_id, dim_lm_embeddings=lm_hidden_size, num_layers=model_config['text_encoder_layers'])
    return models.TextEncoder(model_config['mask_rate'], tokenizer.vocab_size, model_config['text_encoder_dim'], encoder.hidden_size, num_heads=4, ff_size=4*(model_config['text_encoder_dim']), pad_token=tokenizer.pad_token_id, num_layers=model_config['text_encoder_layers'])


def resume_from_checkpoint(model_config, pretrained_dir, output_dir, lm, device, text_input, tokenizer, lm_hidden_size):
    accelerator.print(f'Resuming from checkpoint {output_dir}')
    latest = os.path.join(output_dir, 'latest')
    connector = models.Connector.load_from_dir(latest, device)
    connector = connector.to(device)
    encoder_class = load_encoder(model_config["wrapper"])
    encoder_dir = latest if os.path.isfile(os.path.join(latest, 'encoder_model.pt')) else pretrained_dir
    encoder = encoder_class.load_from_dir(encoder_dir, device, deactivate_masked_spec_embed=True)

    if text_input:
        if os.path.isfile(os.path.join(latest, 'TextEncoder_model.pt')):
            text_encoder = models.TextEncoder.load_from_dir(latest, device)
        else:
            text_encoder = make_text_encoder(model_config, encoder, tokenizer, lm_hidden_size)
        model = models.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer, text_encoder)
    else:
        model = models.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer)

    with open(os.path.join(latest, 'metrics.yaml'), 'r') as f:
        all_metrics = yaml.safe_load(f)
        best_slot_k_f1 = max([m['slot_k_f1'] for m in all_metrics])

    model = model.to(device)
    return model, best_slot_k_f1, all_metrics


def load_from_checkpoint(model_config, pretrained_dir, lm, device, text_input, tokenizer, lm_hidden_size):
    accelerator.print(f'Loading encoder and connector from checkpoint {pretrained_dir}')
    connector = models.Connector.load_from_dir(pretrained_dir, device)
    connector = connector.to(device)
    encoder_class = load_encoder(model_config["wrapper"])
    encoder = encoder_class.load_from_dir(pretrained_dir, device, deactivate_masked_spec_embed=True)

    if text_input:
        if os.path.isfile(os.path.join(pretrained_dir, 'TextEncoder_model.pt')):
            accelerator.print(f'Loading pretrained text encoder from {pretrained_dir}')
            text_encoder = models.TextEncoder.load_from_dir(pretrained_dir, device)
        else:
            text_encoder = make_text_encoder(model_config, encoder, tokenizer, lm_hidden_size)
        model = models.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer, text_encoder)
    else:
        model = models.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer)
    model = model.to(device)
    best_slot_k_f1 = 0.0
    all_metrics = []
    return model, best_slot_k_f1, all_metrics


def load_from_scratch(model_config, lm, device, text_input, tokenizer, lm_hidden_size):
    accelerator.print('Loading model from scratch')
    encoder_class = load_encoder(model_config["wrapper"])
    encoder = encoder_class(model_config['encoder_model_name'])
    connector = models.Connector(encoder.hidden_size, lm_hidden_size, num_heads=4, ff_size=4*encoder.hidden_size, num_connector_layers=model_config['connector_layers'])

    if text_input:
        text_encoder = make_text_encoder(model_config, encoder, tokenizer, lm_hidden_size)
        model = models.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer, text_encoder)
    else:
        model = models.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer)
    model = model.to(device)
    best_slot_k_f1 = 0.0
    all_metrics = []
    return model, best_slot_k_f1, all_metrics


def main(model_config, pretrained_dir, output_dir, resume=False, device='cuda', text_input=False):
    # old checkpoints have no model_config.yaml -> keep the passed config in that case
    if pretrained_dir is not None and os.path.isfile(os.path.join(pretrained_dir, 'model_config.yaml')):
        model_config = yaml.load(open(os.path.join(pretrained_dir, 'model_config.yaml'), 'r'), Loader=yaml.FullLoader)
    #### LOAD LM:
    lm_model_name = model_config.get('lm_model_name')
    bos_token = model_config.get('bos_token')
    use_bos_from_lm = model_config.get('use_bos_from_lm')
    quantization_level = model_config.get('quantization_level', 'none')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_level == '4bit',
        load_in_8bit=quantization_level == '8bit',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    quantization_kwargs = {}
    if quantization_level != 'none':
        quantization_kwargs = {'quantization_config': bnb_config}

    if "flash_attn" in sys.modules:
        lm = transformers.AutoModelForCausalLM.from_pretrained(lm_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=accelerator.device, **quantization_kwargs, attn_implementation='flash_attention_2')
    else:
        lm = transformers.AutoModelForCausalLM.from_pretrained(lm_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=accelerator.device, **quantization_kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(lm_model_name, trust_remote_code=True)
    if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
        bos_token = tokenizer.bos_token_id
    else:
        if bos_token is None:
            bos_token = lm.config.eos_token_id
        else:
            bos_token = tokenizer.encode(bos_token)[0]

    lm_config = {"model_name": lm_model_name,
                 "use_bos_from_lm": use_bos_from_lm,
                 "bos_token": bos_token}
    accelerator.print("lm_config", lm_config)

    if hasattr(lm, "language_model"):
        lm = lm.language_model
    lm_hidden_size = lm.config.hidden_size

    lora_config = None
    if model_config.get('lora_rank') is not None:
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            target_modules='all-linear',
            r=model_config.get('lora_rank'),
            lora_alpha=model_config.get('lora_rank'),
            use_dora=model_config.get('lora_dora', False),
        )
        if resume:
            accelerator.print('Loading LoRA config from checkpoint')
            lm = PeftModel.from_pretrained(lm, os.path.join(output_dir, 'latest', 'lm'), is_trainable=True)
        elif pretrained_dir is not None and os.path.isfile(os.path.join(pretrained_dir, 'lm', 'adapter_config.json')):
            accelerator.print(f'Loading LoRA config from {pretrained_dir}')
            lm = PeftModel.from_pretrained(lm, os.path.join(pretrained_dir, 'lm'), is_trainable=True)
        else:
            lm = get_peft_model(lm, lora_config)

    if resume:
        model, best_slot_k_f1, all_metrics = resume_from_checkpoint(model_config, pretrained_dir, output_dir, lm, device, text_input, tokenizer, lm_hidden_size)
    elif pretrained_dir is not None:
        model, best_slot_k_f1, all_metrics = load_from_checkpoint(model_config, pretrained_dir, lm, device, text_input, tokenizer, lm_hidden_size)
    else:
        model, best_slot_k_f1, all_metrics = load_from_scratch(model_config, lm, device, text_input, tokenizer, lm_hidden_size)

    return model, tokenizer, lora_config, lm_config, best_slot_k_f1, all_metrics