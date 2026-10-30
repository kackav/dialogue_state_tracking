import os
import random
import requests
import json
import functools
import ast
import pickle


import torch
import torchaudio
import librosa
import soundfile
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
import datasets
import yaml
import copy
import augment_text

def train_bpe(data, vocab_size=1024, special_tokens=None, output_dir='tokenizer', num_sentences=100000):
    if special_tokens is None:
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"]
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_json = f'{output_dir}/bpe_tokenizer_{vocab_size}.json'

    if not os.path.isfile(tokenizer_json):
        data_list = [_['text'] for _ in data]
        data_list_strip = [_.strip() for _ in data_list if len(_) > 0]
        if num_sentences > len(data_list_strip):
            num_sentences = len(data_list_strip)
        data_list_sub = random.sample(data_list_strip, num_sentences)
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        tokenizer.train_from_iterator(data_list_sub, trainer)
        tokenizer.decoder = BPEDecoder()
        tokenizer.save(tokenizer_json)
    else:
        tokenizer = load_tokenizer(tokenizer_json)
    return tokenizer

def load_tokenizer(tokenizer_json):
    with open(tokenizer_json, 'r', encoding='utf-8') as f:
        tokenizer_str = f.read()
    tokenizer = Tokenizer.from_str(tokenizer_str)
    return tokenizer

def _as_hashable(v):
    return v if isinstance(v, (str, int, float, bool)) else json.dumps(v, sort_keys=True)

def _string_to_int(n_of_samples):
    if n_of_samples == 'all':
        int_n_of_samples = float('inf')
    else:
        int_n_of_samples = int(n_of_samples)
    return int_n_of_samples

def _sk(x):
    return (type(x).__name__, str(x))

def build_ontology(data, ontology_config=None):
    """data: a HF dataset or a dict {name: dataset}. Reads only the 'text' column (no audio decoding)."""
    domains_ontology = set()
    keys_ontology = {}
    values_ontology = {}
    default_cfg = {"slot_values": {"n_of_samples":10, "ground_truth":True, "additionals" : 3}, "slot_keys": {"n_of_samples":10, "ground_truth":True, "additionals" : 3},"domains": {"n_of_samples":0, "ground_truth":True, "additionals" : 3},"sampling":"fixed"}
    ontology_config = yaml.load(open(ontology_config, 'r'), Loader=yaml.FullLoader) if ontology_config else default_cfg
    int_n_of_domains_to_sample = _string_to_int(ontology_config["domains"]["n_of_samples"])
    int_n_of_keys_to_sample = _string_to_int(ontology_config["slot_keys"]["n_of_samples"])
    int_n_of_values_to_sample = _string_to_int(ontology_config["slot_values"]["n_of_samples"])
    for ds in (data.values() if isinstance(data, dict) else [data]):
        for raw in ds['text']:
            text_input = json.loads(raw)
            if "domains" in ontology_config and int_n_of_domains_to_sample > 0:
                domains_ontology.update(ast.literal_eval(text_input["domains"]))
            else:
                domains_ontology = None
            slot_dict = ast.literal_eval(text_input['slots'])
            for dom, slot_items in slot_dict.items():
                for slot_key, slot_value in slot_items.items():
                    if int_n_of_keys_to_sample > 0:
                        keys_ontology.setdefault(dom, set()).add(slot_key)

                    if int_n_of_values_to_sample > 0:    
                        values_ontology.setdefault(slot_key, set()).add(_as_hashable(slot_value))
                    
    return {'values': values_ontology, 'keys': keys_ontology, 'cfg': ontology_config}


def get_context_with_ontology(context, slots, val_ontology, slot_ontology, ontology_config, rng=None):
    rng = rng if rng is not None else random
    slots = slots or {}
    n_of_values_to_sample = ontology_config['slot_values']['n_of_samples'] # if 'slot_values' in ontology_config else 0
    
    n_of_keys_to_sample = ontology_config['slot_keys']['n_of_samples']# if 'slot_keys' in ontology_config else 0
    
    n_of_domains_to_sample = ontology_config['domains']['n_of_samples'] #if 'domains' in ontology_config else 0

    use_gt_slots = ontology_config['use_gt_slots']
    gt_slot_keys = set()
    gt_domains = set()
    values_for_context = set()
    slots_for_context = set()
    domains_for_context = set() 

    all_possible_vals = val_ontology
    all_possible_domains = set(list(slot_ontology.keys()))
    
    all_possible_slots = [sorted(slot_ontology[dom]) for dom in sorted(all_possible_domains)]
    all_possible_slots = [x for xs in all_possible_slots for x in xs]
    all_possible_slots = set(all_possible_slots)

    int_n_of_domains_to_sample = _string_to_int(n_of_domains_to_sample)
    int_n_of_values_to_sample = _string_to_int(n_of_values_to_sample)
    int_n_of_keys_to_sample = _string_to_int(n_of_keys_to_sample)

       
    for dom, slot_dict in slots.items():
        for slot_key, slot_value in slot_dict.items():
            slot_value = _as_hashable(slot_value)
            gt_val = set()
            if val_ontology:
                vals_subset = set(all_possible_vals.get(slot_key, ())) - {slot_value}
                if len(vals_subset) > int_n_of_values_to_sample:
                    sampled_vals = set()
                    values_for_context = set()
                    continue
                        
                if ontology_config['slot_values']['ground_truth']==True and int_n_of_values_to_sample>0:
                    gt_val.update({slot_value})
                    values_for_context.update(gt_val)
                
                sampled_vals = set(rng.sample(sorted(vals_subset, key=_sk), max(0, min(int_n_of_values_to_sample-len(gt_val), len(vals_subset)))))
                values_for_context.update(sampled_vals)

            if ontology_config['slot_keys']['ground_truth']==True and int_n_of_keys_to_sample >0:
                gt_slot_keys.update({slot_key})

        if ontology_config['domains']['ground_truth']==True and int_n_of_domains_to_sample >0:
            gt_domains.update({dom})

    if not(use_gt_slots):
        vals_subset = set([all_possible_vals[_k] for _k in all_possible_vals.keys()])
        gt_vals = set(slot_dict.values())
        values_for_context = vals_subset - gt_vals
        
    if slot_ontology:
        all_possible_domains = all_possible_domains - gt_domains #.discard(gt_domains)
        
        domains_for_context = set(rng.sample(sorted(all_possible_domains), max(0, min(int_n_of_domains_to_sample-len(gt_domains), len(all_possible_domains)))))
        if ontology_config['domains']['ground_truth']==True and int_n_of_domains_to_sample > 0:
            domains_for_context.update(gt_domains)
        slots_subset = all_possible_slots  - gt_slot_keys
        slots_for_context = set(rng.sample(sorted(slots_subset), max(0, min(int_n_of_keys_to_sample-len(gt_slot_keys), len(slots_subset)))))
        if ontology_config['slot_keys']['ground_truth']==True and int_n_of_keys_to_sample > 0:
            slots_for_context.update(gt_slot_keys)
    else:
        slots_for_context = set()
        domains_for_context = set()

    values_for_context = sorted(values_for_context, key=_sk) #sorted(values_for_context)
    slots_for_context = sorted(slots_for_context)
    domains_for_context = sorted(domains_for_context)
    str_dom = str(domains_for_context)+', ' if len(domains_for_context)>0 else ''
    str_ctx = str(slots_for_context)+', ' if len(slots_for_context)>0 else ''
    str_vals = str(values_for_context)+', ' if len(values_for_context)>0 else ''

    return f"{str_dom}{str_ctx}{str_vals}{context}"

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos_token):
        self.data = data
        self.tokenizer = tokenizer
        self.bos_token = bos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        inp =  text
        label = text
        return {"text_trans" : text,
                "input_ids" : [self.bos_token] + self.tokenizer.encode(inp),
                "input_len" : len(self.tokenizer.encode(inp)) + 1,
                "labels" : self.tokenizer.encode(label) + [self.tokenizer.eos_token_id],
                "labels_len" : len(self.tokenizer.encode(label)) + 1
                }
class AudioDatasetHF(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos_token):
        self.data = data
        self.is_fake = False
        self.fake_audio_len = 30
        self.fake_text_len = 20
        self.bos_token = bos_token
        
        self.data_lengths ={k: len(v) for k,v in data.items()}
        self.tokenizer = tokenizer

    def __len__(self):
        return sum(list(self.data_lengths.values()))
    
    def __getitem__(self, idx):
        if self.is_fake:
            text = "This is a fake text for resume."
            audio = torch.zeros(int(self.fake_audio_len * 16_000))  # shape (1, fake_audio_len * fs)
            audio = audio.unsqueeze(0)  # add channel dimension
            inp = [0] * self.fake_text_len
            label = [0] * self.fake_text_len
            return {"audio": audio,
                "audio_len": audio.shape[1],
                "text_trans": text,
                "input_ids": inp,
                "input_len": self.fake_text_len,
                "labels": label,
                "labels_len" : self.fake_text_len
                }
        
        total_length = 0
        for key, length in self.data_lengths.items():
            previous_length = total_length
            total_length += length
            if idx < total_length:
                idx -= previous_length
                break

        data_key = self.data[key][idx]
        text = data_key['normalized_text'] #if "normalized_text" in data_key else data_key["text"].lower()
        audio = torch.tensor(data_key['audio']['array'], dtype=torch.float32)
        sample_rate = data_key['audio']['sampling_rate']
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)        
        audio = audio.unsqueeze(0)  # add channel dimension
        inp = text
        label = text
        encoded_inp = self.tokenizer.encode(inp, add_special_tokens = False)
        encoded_lab = self.tokenizer.encode(label, add_special_tokens = False)
        
        return {"audio" : audio,
                "audio_len" : audio.shape[1],
                "text_trans" : text,
                "input_ids" : [self.bos_token] + encoded_inp,
                "input_len" : len(encoded_inp) + 1,
                "labels" : encoded_lab + [self.tokenizer.eos_token_id],
                "labels_len" : len(encoded_lab) + 1
                }

class DialogDatasetHF(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos_token, ontology=None):
        self.data = data
        self.min_audio_len = 8500
        self.max_pre_text_length = 500
        self.is_fake = False
        self.fake_audio_len = 30
        self.fake_text_len = 20
        self.bos_token = bos_token
        self.fake_context_len = 10
        self.data_lengths ={k: len(v) for k,v in data.items()}
        self.tokenizer = tokenizer
        if ontology:
            self.use_ontology = True
            self.val_ontology = ontology['values']
            self.slot_ontology = ontology['keys']
            # self.dom_ontology = ontology['domains']
            self.ontology_cfg = ontology['cfg']
        else:
            self.use_ontology=False

    def __len__(self):
        return sum(list(self.data_lengths.values()))
    
    def __getitem__(self, idx):
        if self.is_fake:
            text = "This is a fake text for resume."
            context = "This is a fake context for resume."
            audio = torch.zeros(int(self.fake_audio_len * 16_000))  # shape (1, fake_audio_len * fs)
            audio = audio.unsqueeze(0)  # add channel dimension
            inp = [0] * self.fake_text_len
            label = [0] * self.fake_text_len
            encoded_context = [0] * self.fake_context_len
            
            return {"audio" : audio,
                "audio_len" : audio.shape[1],
                "text_trans" : text,
                "context_trans" : context,
                "context" : [self.bos_token] + encoded_context,
                "context_len" : len(encoded_context) +1,
                "pre_text_trans" : text,
                "pre_text" : encoded_context,
                "pre_text_len" : len(encoded_context),
                "input_ids" : [self.bos_token] + encoded_context,
                "input_len" : len(encoded_context) + 1,
                "labels" : encoded_context + [self.tokenizer.eos_token_id],
                "labels_len" : len(encoded_context) + 1,
                "wav_id": text,
                "turn_index": text,
                "agent_text": text,
                "domains": text,
                "slots": text,
                # "audio" : audio,
            #     "audio_len" : audio.shape[1],
            #     "text_trans" : text,
            #     "context_trans" : context,
            #     "context" : encoded_context,
            #     "context_len" : len(encoded_context),
            #     "input_ids" : inp,
            #     "input_len" : len(inp),
            #     "labels" : label,
            #     "labels_len" : len(label),
                }
        
        total_length = 0
        for key, length in self.data_lengths.items():
            previous_length = total_length
            total_length += length
            if idx < total_length:
                idx -= previous_length
                break
 

        data_key = self.data[key][idx]
        text = data_key['text'] #if "normalized_text" in data_key else data_key["text"].lower()
        context = data_key['context']
        audio = torch.tensor(data_key['audio']['array'], dtype=torch.float32)
        if len(audio) < self.min_audio_len:
            padding = torch.zeros(self.min_audio_len - len(audio))
            audio = torch.cat((padding, audio))

        sample_rate = data_key['audio']['sampling_rate']
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)        
        audio = audio.unsqueeze(0)  # add channel dimension
        text_decoded = json.loads(text)

        domains = text_decoded["domains"]
        slots = text_decoded["slots"]
        pre_text = text_decoded['label']
        
        slots = ast.literal_eval(slots)
        domains = ast.literal_eval(domains)
        if self.use_ontology:
            rng = random.Random(f"{key}-{idx}") if self.ontology_cfg['sampling'] == 'fixed' else None
            context = get_context_with_ontology(context, slots, self.val_ontology, self.slot_ontology, self.ontology_cfg, rng=rng)
            text = {"label": pre_text,
                "domains": domains,
                "slots": slots,
                }
        text = json.dumps(text)
        
        inp = text
        label = text
        

        encoded_inp = self.tokenizer.encode(inp, add_special_tokens = False)
        encoded_lab = self.tokenizer.encode(label, add_special_tokens = False)
        encoded_context = self.tokenizer.encode(context, add_special_tokens = False)
        encoded_pre_text = self.tokenizer.encode(pre_text, add_special_tokens = False)
        if len(encoded_pre_text) > self.max_pre_text_length:
            encoded_pre_text = encoded_pre_text[:self.max_pre_text_length]
        if len(encoded_pre_text) == 0:
            encoded_pre_text = [self.bos_token]

        return {"audio" : audio,
                "audio_len" : audio.shape[1],
                "text_trans" : text,
                "context_trans" : context,
                "context" : [self.bos_token] + encoded_context,
                "context_len" : len(encoded_context) +1,
                "pre_text_trans" : pre_text,
                "pre_text" : encoded_pre_text,
                "pre_text_len" : len(encoded_pre_text),
                "input_ids" : [self.bos_token] + encoded_inp,
                "input_len" : len(encoded_inp) + 1,
                "labels" : encoded_lab + [self.tokenizer.eos_token_id],
                "labels_len" : len(encoded_lab) + 1,
                "wav_id": data_key["wav_id"],
                "turn_index": data_key["turn_index"],
                "agent_text": data_key["agent_text"],
                "domains": data_key["domains"],
                "slots": data_key["slots"],
                }


class InferenceDialogDatasetHF(torch.utils.data.Dataset):
    def __init__(self, non_inference_dialog_dataset: DialogDatasetHF):
        self.non_inference_dialog_dataset = non_inference_dialog_dataset      
        wavids = {k : list(v["wav_id"]) for k, v in self.non_inference_dialog_dataset.data.items()}
        wav_id_to_dataset = {}
        assert len(self.non_inference_dialog_dataset.data) == 1, "only one dataset supported for inference"
        for k, v in wavids.items():
            for w in v:
                if w in wav_id_to_dataset and wav_id_to_dataset[w] != k:
                    raise ValueError(f"wav_id {w} is present in multiple datasets, cannot do inference per wav_id")
                wav_id_to_dataset[w] = k
        self.wav_id_to_dataset = wav_id_to_dataset
        wavids = list(wavids.values())
        wavids = sum(wavids, [])
        wavids_to_inds = {}
        for i, w in enumerate(wavids):
            if w not in wavids_to_inds.keys():
                wavids_to_inds[w] = []
            wavids_to_inds[w].append(i)
        self.wavids_to_inds = wavids_to_inds
        self.wavids = list(self.wavids_to_inds.keys())

    def __len__(self):
        return len(self.wavids_to_inds)

    def __getitem__(self, ind):
        wavid = self.wavids[ind]
        dataset_of_dialog = self.wav_id_to_dataset[wavid]
        data_indices = self.wavids_to_inds[wavid]
        items = [self.non_inference_dialog_dataset[i] for i in data_indices]
        agent_text = [self.non_inference_dialog_dataset.data[dataset_of_dialog][i]["agent_text"] for i in data_indices]
        for i in range(len(items)):
            items[i]["agent_text"] = agent_text[i]
            items[i]["wav_id"] = wavid
        items = sorted(items, key=lambda x: x["turn_index"])
      
        return items


class TextDialogDatasetHF(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos_token, ontology=None, augment_config = None):
        self.data = data
        self.is_fake = False
        self.data_lengths = {key: len(v) for key,v in data.items()}
        self.tokenizer = tokenizer
        self.bos_token = bos_token
        self.fake_text_len = 30
        self.max_context_turns = 200
        if ontology:
            self.use_ontology = True
            self.val_ontology = ontology['values']
            self.slot_ontology = ontology['keys']
        
        if augment_config:
            default_cfg = {"names": True, "numbers": True, "categorical": False}
            augment_config = yaml.load(open(augment_config, 'r'), Loader=yaml.FullLoader) if os.path.isfile(augment_config) else default_cfg
            self.Augment = augment_text(augment_config, ontology)
            self.augment_func = self.Augment.augment_turn

            # self.dom_ontology = ontology['domains']
        else:
            self.use_ontology=False
    def __len__(self):
        return sum(list(self.data_lengths.values()))

    def __getitem__(self, idx):
        if self.is_fake:
            text = "This is a fake text for resume."
            inp = [0] * self.fake_text_len
            label = [0] * self.fake_text_len
            return {"text_trans": text,
                "input_ids": inp,
                "input_len": self.fake_text_len,
                "labels": label,
                "labels_len" : self.fake_text_len,
                "pre_text_trans" : text,
                "pre_text" : inp,
                "pre_text_len" : len(inp),
                "context_trans" : text,
                "context" : inp,
                "context_len" : len(inp) +1,
                "text_trans" : text,
                }

        total_length = 0
        for key, length in self.data_lengths.items():
            previous_length = total_length
            total_length += length
            if idx < total_length:
                idx -= previous_length
                break

        data_key = self.data[key][idx]
        text = data_key['text']

        text_decoded = json.loads(text)
        pre_text = text_decoded['label']
        slots = text_decoded['slots']
        domains = text_decoded['domains']

        slots = ast.literal_eval(slots)
        domains = ast.literal_eval(domains)
        text = {"label": pre_text,
                "domains": domains,
                "slots": slots,
                }
        text = json.dumps(text)
        
        inp = text
        label = text
        encoded_inp = self.tokenizer.encode(inp, add_special_tokens = False)
        encoded_inp = [self.bos_token] + encoded_inp
        if len(encoded_inp) > 512:
            encoded_inp = encoded_inp[:512]
        encoded_lab = self.tokenizer.encode(label, add_special_tokens = False)
        encoded_lab = encoded_lab + [self.tokenizer.eos_token_id]
        if len(encoded_lab) > 512:
            encoded_lab = encoded_lab[:512]
            
        context = data_key['context']
        context = "\n".join(context.split("\n")[-self.max_context_turns:])
        if self.use_ontology:
            context = get_context_with_ontology(context, slots, domains, self.val_ontology, self.slot_ontology)
                
        encoded_context = self.tokenizer.encode(context, add_special_tokens = False)
        encoded_pre_text = self.tokenizer.encode(pre_text, add_special_tokens = False)
    
        if len(encoded_inp) > 512:
            encoded_inp = encoded_inp[:512]
        if len(encoded_lab) > 512:
            encoded_lab = encoded_lab[:512]
        encoded_pre_text = self.tokenizer.encode(pre_text, add_special_tokens = False)
        if len(encoded_pre_text) > 512:
            encoded_pre_text = encoded_pre_text[:512]

        # if self.printing:
        #     print("TEXT TRANS", text_str_normalized)
        #     print("CONTEXT_TRANS", context)
        self.printing = False
        
        return {
                "context_trans" : context,
                "context" : [self.bos_token] + encoded_context,
                "context_len" : len(encoded_context) +1,
                "pre_text_trans" : pre_text,
                "pre_text" : encoded_pre_text,
                "pre_text_len" : len(encoded_pre_text),
                "text_trans" : str(text['label']),
                "input_ids" : encoded_inp,
                "input_len" : len(encoded_inp),
                "labels" : encoded_lab,
                "labels_len" : len(encoded_lab),
                }

def filter_long_context(example, tokenizer, max_text_len=800):
    encoded = tokenizer.encode(example["context"], add_special_tokens = False)
    return len(encoded) < max_text_len

def filter_short_text(example, tokenizer, min_text_len=10):
    try:
        raw = example.get("text") if isinstance(example, dict) else example
        if raw is None:
            return False
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode('utf-8', errors='ignore')
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                label = parsed.get("label", "")
            except Exception:
                label = raw
        elif isinstance(raw, dict):
            label = raw.get("label", "")
        else:
            label = str(raw)

        encoded = tokenizer.encode(label, add_special_tokens = False)
        return len(encoded) > min_text_len
    except Exception:
        return False

def filter_long_text(example, tokenizer, max_text_len=200):
    try:
        raw = example.get("text") if isinstance(example, dict) else example
        if raw is None:
            return False
        # handle case where text field is a JSON string containing a dict with 'label'
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode('utf-8', errors='ignore')
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                label = parsed.get("label", "")
            except Exception:
                # fallback: treat the string itself as label
                label = raw
        elif isinstance(raw, dict):
            label = raw.get("label", "")
        else:
            label = str(raw)

        encoded = tokenizer.encode(label, add_special_tokens = False)
        return len(encoded) < max_text_len
    except Exception:
        return False

def _apply_common_filters(ds, tokenizer=None, do_filter=True):
    """Same filters as the old load_dial_from_config."""
    if not do_filter:
        return ds
    if 'audio_len' in ds.features:
        ds = ds.filter(lambda item: item['audio_len'] < (30*16_000), num_proc=4)
        ds = ds.filter(lambda item: item['audio_len'] > (0.6*16_000), num_proc=4)
    elif 'audio' in ds.features:
        ds = ds.filter(lambda item: len(item['audio']["array"]) < (30*16_000), num_proc=4)
        ds = ds.filter(lambda item: len(item['audio']["array"]) > (0.6*16_000), num_proc=4)
    ds = ds.filter(functools.partial(filter_long_text, tokenizer=tokenizer, max_text_len=200), num_proc=4)
    ds = ds.filter(functools.partial(filter_long_context, tokenizer=tokenizer, max_text_len=800), num_proc=4)
    ds = ds.filter(functools.partial(filter_short_text, tokenizer=tokenizer), num_proc=4)
    return ds



def load_from_config(ds_type, config_datasets, hf_dataset_path=None, tokenizer=None, accelerator=None, do_filter=False):
    """
    Load dialogue datasets for a given split type (e.g., 'train', 'validation', 'text_train').

    Expected `config_datasets` format:
    {
      "train": {
         "multiwoz": {"path": "/abs/or/hf/path/or-id", "split": "train"},
         "spokenwoz": {"path": "/abs/path/to/local/dir"}
      },
      "validation": { ... }
    }

    The function will prefer local disk paths if `path` exists, otherwise it treats `path` as a Hugging Face dataset id.
    """
    loaded_datasets = {}
    entries = config_datasets.get(ds_type, {})
    for k, v in entries.items():
        path = v.get('path') or v.get('name')
        split = v.get('split', None)
        # treat missing, empty or blank `path` as not provided -> try hf_dataset_path fallback
        if path is None or (isinstance(path, str) and path.strip() == ''):
            if hf_dataset_path:
                # prefer convention-based prepared dataset directories under hf_dataset_path
                candidates = [
                    os.path.join(hf_dataset_path, f"{k}_{ds_type}"),
                    os.path.join(hf_dataset_path, f"{k}_{ds_type}"),
                    os.path.join(hf_dataset_path, f"{k}_{ds_type}"),
                ]
                used = None
                for c in candidates:
                    if os.path.exists(c):
                        used = c
                        break
                if used:
                    print(f"Warning: no `path` in config for '{k}'. Falling back to HF dataset path: {used}")
                    ds = datasets.load_from_disk(used)
                else:
                    print(f"Warning: no `path` for '{k}' and no candidate under HF dataset path '{hf_dataset_path}' found; skipping {k}.")
                    continue
            else:
                # nothing to load
                print(f"Warning: no `path` for '{k}' and no hf_dataset_path provided; skipping {k}.")
                continue
        else:
            # prefer explicit local path
            if os.path.exists(path):
                print(f"Loading dataset '{k}' from local path: {path}")
                ds = datasets.load_from_disk(path)
            else:
                # treat as HF dataset id; support "dataset:subset" notation
                if ':' in path and not path.startswith('/'):
                    ds_name, ds_subset = path.split(':', 1)
                    print(f"Loading dataset '{k}' from HF dataset id: {ds_name}, subset: {ds_subset}, split: {split}")
                    ds = datasets.load_dataset(ds_name, ds_subset, split=split, num_proc=2)
                else:
                    # path is HF id or name
                    print(f"Loading dataset '{k}' from HF dataset id: {path}, split: {split}")
                    if split:
                        ds = datasets.load_dataset(path, split=split, num_proc=2)
                    else:
                        ds = datasets.load_dataset(path, num_proc=2)
        ds = _apply_common_filters(ds, tokenizer=tokenizer, do_filter=do_filter)
        loaded_datasets[k] = ds
    return loaded_datasets

class CollateFnSpeech:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        audio = [_['audio'][0] for _ in batch]
        audio_len = torch.tensor([_['audio_len'] for _ in batch], dtype=torch.int)

        input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
        labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        max_audio_len = max(audio_len)
        nearest_thousand = int(max_audio_len / 1000) + 1 if max_audio_len % 1000 else int(max_audio_len / 1000)
        nearest_thousand *= 1000
        audio_batch = torch.zeros(len(audio), nearest_thousand)
        for i, a in enumerate(audio):
            audio_batch[i, :a.shape[0]] = a
        audio = audio_batch
        
        text_trans = [_['text_trans'] for _ in batch]
        input_len = torch.tensor([_['input_len'] for _ in batch], dtype=torch.int)
        labels_len = torch.tensor([_['labels_len'] for _ in batch], dtype=torch.int)

        return {"audio": audio,
                "audio_len": audio_len,
                "text_trans" : text_trans,
                "input_ids": input_ids,
                "input_len": input_len,
                "labels": labels,
                "labels_len": labels_len
                }
    
class CollateFnText:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
        labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        input_len = torch.tensor([_['input_len'] for _ in batch], dtype=torch.int)
        labels_len = torch.tensor([_['labels_len'] for _ in batch], dtype=torch.int)
        text_trans = [_['text_trans'] for _ in batch]
        
        return {"text_trans" : text_trans,
                "input_ids": input_ids,
                "input_len": input_len,
                "labels": labels,
                "labels_len": labels_len
                }
    
class CollateFnDialogInference(CollateFnSpeech):
    def __init__(self, non_inference_collate_fn, tokenizer=None):
        super().__init__(tokenizer)
        self.non_inference_collate_fn = non_inference_collate_fn
    
    def __call__(self, batch):#List[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[List[Optional[int]]]]:
        lengths = [len(x) for x in batch]
        max_length = max(lengths)
        out_batch = []
        pointers = []
        initial_context = ""
        initial_context = self.tokenizer.encode(initial_context, add_special_tokens=False)
        initial_context_length = len(initial_context)

        for i in range(max_length):
            out_batch.append([])
            pointers.append([])
            agent_text = []
            wav_ids = []
            for batch_idx, conversation in enumerate(batch):
                if i >= len(conversation):
                    pointers[-1].append(None)
                    continue
                turn = conversation[i]
                out_batch[-1].append(turn)
                pointers[-1].append(batch_idx)
                agent_text.append(turn["agent_text"])
                wav_ids.append(turn["wav_id"])
            
            out_batch[-1] = self.non_inference_collate_fn(out_batch[-1])
            out_batch[-1]["agent_text"] = agent_text
            out_batch[-1]["wav_id"] = wav_ids
        for i, p_out in enumerate(pointers):
            for j, p_in in enumerate(p_out):
                if p_in is None:
                    continue
                count_none = len([_ for _ in p_out[:j] if _ is None])
                pointers[i][j] -= count_none
                assert pointers[i][j] >= 0, ("all pointers should have non-negative values")
        return out_batch, pointers
    

def collate_all_text(batch_s, batch_t, tokenizer=None):
    input_ids_s = torch.nn.utils.rnn.unpad_sequence(
        batch_s['input_ids'], lengths = batch_s['input_len'], batch_first=True
    )
    input_ids_t = torch.nn.utils.rnn.unpad_sequence(
        batch_t['input_ids'], lengths = batch_t['input_len'], batch_first=True
    )
    input_ids = input_ids_s + input_ids_t

    labels_s = torch.nn.utils.rnn.unpad_sequence(batch_s['labels'], batch_first=True, lengths = batch_s['labels_len'])
    labels_t = torch.nn.utils.rnn.unpad_sequence(batch_t['labels'], batch_first=True, lengths = batch_t['labels_len'])
    labels = labels_s + labels_t
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    input_len = torch.tensor(batch_s['input_len'].tolist()+ batch_t['input_len'].tolist())
    labels_len = torch.tensor(batch_s['labels_len'].tolist() + batch_t['labels_len'].tolist())
    text_trans = batch_s['text_trans'] + batch_t['text_trans']
    
    
    return {"text_trans" : text_trans,
            "input_ids": input_ids,
            "input_len": input_len,
            "labels": labels,
            "labels_len": labels_len
            }
            
class CollateFnDialog:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        audio = [_['audio'][0] for _ in batch]
        audio_len = torch.tensor([_['audio_len'] for _ in batch], dtype=torch.int)

        input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
        labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
        context = [torch.tensor(_['context'], dtype = torch.long) for _ in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)
        context = torch.nn.utils.rnn.pad_sequence(context, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)

        text_trans = [_['text_trans'] for _ in batch]
        context_trans = [_['context_trans'] for _ in batch]

        input_len = torch.tensor([_['input_len'] for _ in batch], dtype=torch.int)
        labels_len = torch.tensor([_['labels_len'] for _ in batch], dtype=torch.int)
        context_len = torch.tensor([_['context_len'] for _ in batch], dtype=torch.int)

        pre_text = [torch.tensor(_['pre_text'], dtype=torch.long) for _ in batch]
        pre_text = torch.nn.utils.rnn.pad_sequence(pre_text, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)
        pre_text_len = torch.tensor([_['pre_text_len'] for _ in batch], dtype=torch.int)

        return {"audio": audio,
                "audio_len": audio_len,
                "text_trans" : text_trans,
                "context_trans" : context_trans,
                "context" : context,
                "context_len" : context_len,
                "input_ids": input_ids,
                "input_len": input_len,
                "labels": labels,
                "labels_len": labels_len,
                "pre_text": pre_text,
                "pre_text_len": pre_text_len,
                }

class CollateFnDialogText:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
        labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        input_len = torch.tensor([_['input_len'] for _ in batch], dtype=torch.int)
        labels_len = torch.tensor([_['labels_len'] for _ in batch], dtype=torch.int)
        text_trans = [_['text_trans'] for _ in batch]

        context = [torch.tensor(_['context'], dtype=torch.long) for _ in batch]
        pre_text = [torch.tensor(_['pre_text'], dtype=torch.long) for _ in batch]
        context = torch.nn.utils.rnn.pad_sequence(context, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)
        pre_text = torch.nn.utils.rnn.pad_sequence(pre_text, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)
        context_len = torch.tensor([_['context_len'] for _ in batch], dtype=torch.int)
        pre_text_len = torch.tensor([_['pre_text_len'] for _ in batch], dtype=torch.int)
        context_trans = [_['context_trans'] for _ in batch]

        return {"text_trans" : text_trans,
                "input_ids": input_ids,
                "input_len": input_len,
                "labels": labels,
                "labels_len": labels_len,
                "context": context,
                "context_trans": context_trans,
                "context_len": context_len,
                "pre_text": pre_text,
                "pre_text_len": pre_text_len,
                }

def collate_all_dialog_text(batch_s, batch_t, tokenizer=None):
    input_ids_s = torch.nn.utils.rnn.unpad_sequence(
        batch_s['input_ids'], lengths = batch_s['input_len'], batch_first=True
    )
    input_ids_t = torch.nn.utils.rnn.unpad_sequence(
        batch_t['input_ids'], lengths = batch_t['input_len'], batch_first=True
    )
    input_ids = input_ids_s + input_ids_t

    labels_s = torch.nn.utils.rnn.unpad_sequence(batch_s['labels'], batch_first=True, lengths = batch_s['labels_len'])
    labels_t = torch.nn.utils.rnn.unpad_sequence(batch_t['labels'], batch_first=True, lengths = batch_t['labels_len'])
    labels = labels_s + labels_t

    context_s = torch.nn.utils.rnn.unpad_sequence(batch_s['context'], batch_first=True, lengths = batch_s['context_len'])
    context_t = torch.nn.utils.rnn.unpad_sequence(batch_t['context'], batch_first=True, lengths = batch_t['context_len'])
    context = context_s + context_t

    pre_text_s = torch.nn.utils.rnn.unpad_sequence(batch_s['pre_text'], batch_first=True, lengths = batch_s['pre_text_len'])
    pre_text_t = torch.nn.utils.rnn.unpad_sequence(batch_t['pre_text'], batch_first=True, lengths = batch_t['pre_text_len'])
    pre_text = pre_text_s + pre_text_t


    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    context = torch.nn.utils.rnn.pad_sequence(context, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0)
    pre_text = torch.nn.utils.rnn.pad_sequence(pre_text, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0)

    input_len = torch.tensor(batch_s['input_len'].tolist()+ batch_t['input_len'].tolist())
    labels_len = torch.tensor(batch_s['labels_len'].tolist() + batch_t['labels_len'].tolist())
    pre_text_len = torch.tensor(batch_s['pre_text_len'].tolist() + batch_t['pre_text_len'].tolist())
    context_len = torch.tensor(batch_s['context_len'].tolist() + batch_t['context_len'].tolist())
    text_trans = batch_s['text_trans'] + batch_t['text_trans']
    context_trans = batch_s['context_trans'] + batch_t['context_trans']


    return {"text_trans" : text_trans,
            "context_trans": context_trans,
            "input_ids": input_ids,
            "input_len": input_len,
            "labels": labels,
            "labels_len": labels_len,
            "context": context,
            "context_len": context_len,
            "pre_text": pre_text,
            "pre_text_len": pre_text_len,
            }


def load_from_textfile(textfile_path='/mnt/matylda6/ivendrame/miniLM/data/librispeech_shuf.txt'):

    with open(textfile_path, 'r', encoding='utf-8') as f:
        data = [{'text': f'{_.strip().lower()}'} for _ in f.readlines() if _.strip() and len(_.split())< 231]

        return data
    

def load_from_transcript(trans_path, audio_dir):
    data_dict = []
    with open(trans_path, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
        
        for line in data:
            splt = line.split(maxsplit =  1)
            if len(splt)>1:
                audio_path = os.path.join(audio_dir, splt[0] + ".flac")
                trans = splt[1].lower()
                dict = {'track': f'{splt[0]}', 'text' : (trans, audio_path)}
                data_dict.append(dict)

        return data_dict
