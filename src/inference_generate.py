import json
import torch
import tqdm
import accelerate
import compute_metrics
import yaml
import os
import data
import transformers
import argparse
import models_dst as models
from peft import get_peft_model, LoraConfig, PeftModel
from copy import deepcopy
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import jiwer

hf_dataset_path = os.environ.get('HF_HOME', '')+ '/modules/datasets_modules/datasets'

print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', ''))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader_num_workers', type=int, default=2,
                        help='number of workers for dataloader')
    parser.add_argument('--dataloader_num_workers_val', type=int, default=2,
                            help='number of workers for dataloader')

    parser.add_argument('--per_device_eval_batch_size', type=int, default=2,
                        help='batch size per device for evaluation')
   
    parser.add_argument('--use_llm_emb', action='store_true',
                        help = "set to True if you don't want to initialize new embeddings in the Text Encoder")
    parser.add_argument('--use_bos_from_lm', action='store_true',
                        help = "set to True if you want to use lm.config.bos_token. If Lm doesn't have it, specify bos_token argument (default is eos token)")
    parser.add_argument('--bos_token', type=str, default=None, 
                        help="bos_token for lm, if None takes eos token from tokenizer")
    
    parser.add_argument('--domains_to_ignore', nargs='+', default=None,
                        help='domains that are ignored in inference')
    parser.add_argument('--slots_to_ignore', nargs='+', default=None,
                        help='slots that are ignored during inference')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='output directory to save model checkpoints and generated outputs')
    
    parser.add_argument('--lm_dir', type=str, default=None,
                        help='lm dir')
    parser.add_argument('--datasets_config', type=str, default=None,
                        help='datasets yaml file path')
    parser.add_argument('--encoder_model_name', type=str, default='microsoft/wavlm-large',
                        help='encoder model name')
    parser.add_argument('--lm_model_name', type=str, default=None,
                        help='lm name as string')

    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='pretrained connector and encoder dir')
    
    parser.add_argument('--encoder_dir', type=str, default=None,
                        help='encoder dir, if you want to load encoder from different dir than connector')
    parser.add_argument('--metric', type=str, default="jga", help='just for info in metrics file, so we know how was the chekpoint chosen')

    init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=1))
    accelerator = accelerate.Accelerator(kwargs_handlers=[init_process_kwargs])
    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    _x = torch.tensor([1.0], device=accelerator.device, dtype=torch.bfloat16)

    device = accelerator.device
    print(f'Using device: {device} - index: {accelerator.process_index}')
    n_gpus = accelerator.num_processes
    args = parser.parse_args()
    input_args = {}
    for arg in vars(args):
        accelerator.print(f'{arg}: {getattr(args, arg)}')
        input_args[f'{arg}'] = f'{getattr(args, arg)}'
    
    parent_pretrained_dir  = os.path.abspath(os.path.join(args.pretrained_dir, os.pardir))
    lm_config = yaml.load(open(os.path.join(parent_pretrained_dir, 'lm_config.yaml')), Loader=yaml.FullLoader)
    if args.lm_model_name is None:
        lm_model_name = lm_config['model_name']
    else:
        lm_model_name = args.lm_model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(lm_model_name)

    datasets_config = yaml.load(open(args.datasets_config), Loader=yaml.FullLoader)
    dset_valid = data.load_from_config('validation', datasets_config, hf_dataset_path, tokenizer,
                                                accelerator=accelerator, do_filter=False) # dictonary ["commonvoice": ..., "librispeech": ... etc]

    lm = transformers.AutoModelForCausalLM.from_pretrained(lm_model_name, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map=accelerator.device)
    if hasattr(lm, "language_model"):
        lm = lm.language_model

    if args.use_bos_from_lm or (hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None):
        bos_token = tokenizer.bos_token_id
    else:
        if args.bos_token is None:
            bos_token = tokenizer.eos_token_id
        else:
            bos_token = tokenizer.encode(args.bos_token)[0]
    val_subsets = {}
    for k, v in dset_valid.items():
        inner_dataset = data.DialogDatasetHF({k:v}, tokenizer, bos_token)
        val_subsets[k] = data.InferenceDialogDatasetHF(inner_dataset)

    collate_fn_dialog_inference = data.CollateFnDialogInference(tokenizer=tokenizer, non_inference_collate_fn=data.CollateFnDialog(tokenizer))
    
    validation_loaders = {}
    for name, ds in val_subsets.items():
        validation_loader = torch.utils.data.DataLoader(ds, batch_size=args.per_device_eval_batch_size,
                                                    collate_fn=collate_fn_dialog_inference,
                                                    num_workers=args.dataloader_num_workers, pin_memory= True,)
        validation_loaders[name] = validation_loader
    connector = models.Connector.load_from_dir(args.pretrained_dir, device)
    connector = connector.to(device)
    if os.path.isfile(os.path.join(args.pretrained_dir,'encoder_config.yaml')):
        accelerator.print(f'Loading Encoder config from {args.pretrained_dir}')
        encoder = models.WavLMWrapper.load_from_dir(args.pretrained_dir, device, deactivate_masked_spec_embed = True)
    elif args.encoder_dir is not None:
        encoder = models.WavLMWrapper.load_from_dir(args.encoder_dir, device, deactivate_masked_spec_embed = True)
    else:
        encoder = models.WavLMWrapper(args.encoder_model_name)
    if os.path.isfile(os.path.join(args.pretrained_dir, 'lm', 'adapter_config.json')):
        accelerator.print(f'Loading LoRA config from {args.pretrained_dir}')

        lm = PeftModel.from_pretrained(lm, os.path.join(args.pretrained_dir,'lm'),
                                       torch_device="cpu")
        lm = lm.merge_and_unload()
    else:
        print("lora is not being loaded")
    model = models.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer)

    model = model.to(device)
    for k, v in validation_loaders.items():
        validation_loaders[k] = accelerator.prepare(v)
    model = accelerator.prepare(model)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    
    for ds_name, val_dataset_loader in validation_loaders.items():
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            val_count = 0
            all_transcriptions = []
            all_references = []
            all_dst_metrics = {
                "domain_tp": 0,
                "domain_fp": 0,
                "domain_fn": 0,
                "slot_k_tp": 0,
                "slot_k_fp": 0,
                "slot_k_fn": 0,
                "slot_v_tp": 0,
                "slot_v_fp": 0,
                "slot_v_fn": 0,
                "num_erroneous_turns": 0,
                "num_turns": 0,
                }
            jga_full_dict = {}
            for f, (val_batch, val_pointers) in enumerate(tqdm.tqdm(val_dataset_loader, disable=not accelerator.is_main_process)):
                val_x = val_batch
                for vx in val_x:
                    for key, value in vx.items():
                        if isinstance(value, torch.Tensor):
                            vx[key] = vx[key].to(device)

                with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16):
                    hyp, jga_dict = accelerator.unwrap_model(model).generate_dialog_multi_turn(val_batch, val_pointers, 500, bos_token)
                jga_full_dict.update(jga_dict)
                ref = [v["text_trans"] for v in val_batch]
                ref = sum(ref, [])
                hyp = sum(hyp, [])
                batch_metrics = compute_metrics.compute_dst_training_metrics(ref, hyp,
                                                                             domains_to_ignore=args.domains_to_ignore,
                                                                             slots_to_ignore=args.slots_to_ignore,)
                all_transcriptions += deepcopy(batch_metrics.pop("hyp_labels"))
                all_references += deepcopy(batch_metrics.pop("ref_labels"))
                all_dst_metrics = {k: all_dst_metrics[k] + batch_metrics[k] for k in all_dst_metrics.keys()}
            keys = all_dst_metrics.keys()
            values = torch.tensor(list(all_dst_metrics.values()), device=device)
            values = accelerator.reduce(values).tolist()
            all_dst_metrics = {k: v for k, v in zip(keys, values)}
            dst_summary_metrics = compute_metrics.compute_dst_precision_recall_f1(all_dst_metrics)
            jga = (all_dst_metrics["num_turns"] - all_dst_metrics["num_erroneous_turns"]) / all_dst_metrics["num_turns"] if all_dst_metrics["num_turns"] > 0 else 0.0
            dst_summary_metrics["jga"] = jga
            with open(os.path.join(args.output_dir, f"predictions_{ds_name}_proc_{accelerator.process_index}.json"), "w") as f:
                    json.dump(jga_full_dict, f, indent=4)
            output_wer = jiwer.process_words(all_references, all_transcriptions)
            insertions = output_wer.insertions
            deletions = output_wer.deletions
            substitutions = output_wer.substitutions
            wer = output_wer.wer
            cer = jiwer.cer(all_references, all_transcriptions)
            num_words = sum(len(t.split()) for t in all_references)
            num_chars = sum(len(t) for t in all_references)

            metrics = torch.tensor([insertions, deletions, substitutions, wer * num_words, cer * num_chars, num_words, num_chars], device=device)
            metrics = accelerator.reduce(metrics, reduction='sum')
            insertions, deletions, substitutions, wer, cer, num_words, num_chars = metrics.tolist()
            wer = wer / num_words
            cer = cer / num_chars
            wer_metrics={'wer':wer, 'cer':cer}

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                with open(os.path.join(args.output_dir, f"predictions_{ds_name}_all.json"), "w") as f:
                    jga_full_dict = {}
                    for proc_id in range(accelerator.num_processes):
                        with open(os.path.join(args.output_dir, f"predictions_{ds_name}_proc_{proc_id}.json"), "r") as f_:
                            jga_full_dict.update(json.load(f_))
                    json.dump(jga_full_dict, f, indent=4)
                with open(os.path.join(args.output_dir, "transcriptions.yaml"), "w") as f:
                    yaml.dump(all_transcriptions, f, indent = 4)
                    
                with open(os.path.join(args.output_dir, "reference.yaml"), "w") as f:
                    yaml.dump(all_references, f, indent = 4)
                    
                print(f'Validation dst metrics for {ds_name}: {dst_summary_metrics}')
                with open(os.path.join(args.output_dir, "metrics.yaml"), "a") as f:
                    yaml.dump({f'metric': args.metric}, f)
                    yaml.dump({f'val_{ds_name}_dst_summary_metrics': dst_summary_metrics}, f)
                    yaml.dump({f'jiwer_{ds_name}_metrics':wer_metrics},f)
           

if __name__ == '__main__':
    main()
