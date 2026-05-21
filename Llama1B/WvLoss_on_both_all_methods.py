# =========================================================
# Compare BASE vs EMBED vs LORA vs SOFT(.pt) vs FINETUNE
# Llama-3.2-1B
# Real vs WikiText
# =========================================================

import os
import math
import json
import glob
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from peft import PeftModel



device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 16
max_length = 512

BASE_DIR  = "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
EMBED_DIR = "/scratch/yl258/kp759/llama3.2-1B/embed_tuning_no_valid_100_500/K_500/FINAL"
LORA_DIR  = "/scratch/yl258/kp759/llama3.2-1B/lora_only_no_valid_500/K_500/FINAL"
SOFT_DIR  = "/scratch/yl258/kp759/llama3.2-1B/soft_prompt_no_valid_500/K_500/FINAL"  


FINETUNE_DIR = "/scratch/yl258/kp759/llama3.2-1B/finetuning_no_valid_100_500extra/K_500/FINAL"

_ce = nn.CrossEntropyLoss(reduction="none")

def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def infer_target_vocab_size(adapter_dir, fallback=None):

    if adapter_dir is None or not os.path.isdir(adapter_dir):
        return fallback

    st_files = sorted(glob.glob(os.path.join(adapter_dir, "*.safetensors")))
    if not st_files:
        return fallback

    state = load_file(st_files[0], device="cpu")
    candidate_keys = [
        "base_model.model.model.embed_tokens.weight",
        "base_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "base_model.model.lm_head.weight",
        "lm_head.weight",
    ]
    for k in candidate_keys:
        if k in state and state[k].ndim == 2:
            return int(state[k].shape[0])

    return fallback


def load_sharded_safetensors_state_dict(model_dir):
  
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index file: {index_path}")

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))
    if not shard_files:
        raise ValueError(f"No shards found in weight_map of {index_path}")

    merged = {}
    for sf in shard_files:
        full_path = os.path.join(model_dir, sf)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing shard: {full_path}")
        merged.update(load_file(full_path, device="cpu"))
    return merged



with open("/scratch/yl258/kp759/llama3.2-1B/Sent/Food_real_test.txt") as f:
    real_sents = [l.strip() for l in f if l.strip()]

ds = load_dataset("wikitext", "wikitext-2-raw-v1")
wiki_sents = [t.strip() for t in ds["test"]["text"] if t.strip()]

print(f"Real: {len(real_sents)} | Wiki: {len(wiki_sents)}")



@torch.no_grad()
def lm_loss(model, tokenizer, sentences, prefix_len=0):
  
    model.eval()
    tot_loss, tot_tok = 0.0, 0

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        labels = input_ids.clone()
        labels[attn_mask == 0] = -100
        labels[:, 0] = -100

        if prefix_len > 0:
            bsz, T = labels.shape
            pad = torch.full((bsz, prefix_len), -100, device=labels.device, dtype=labels.dtype)
            labels_full = torch.cat([pad, labels], dim=1)
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels_full)
            valid = (labels != -100).sum().item()
        else:
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            valid = (labels != -100).sum().item()

        tot_loss += out.loss.item() * valid
        tot_tok += valid

    avg = tot_loss / max(1, tot_tok)
    return avg, math.exp(avg)



class FakeOnlyEmbedding(torch.nn.Module):
    def __init__(self, base_emb, fake_ids):
        super().__init__()
        self.base = base_emb
        self.base.weight.requires_grad = False

        vocab = base_emb.num_embeddings
        dim = base_emb.embedding_dim

        valid_ids = [t for t in fake_ids if t < vocab]
        print(f"[Embed] valid fake tokens: {len(valid_ids)} / {len(fake_ids)}")

        self.fake_emb = nn.Embedding(len(valid_ids), dim)

        lut = torch.full((vocab,), -1, dtype=torch.long)
        for i, tid in enumerate(valid_ids):
            lut[tid] = i
        self.register_buffer("lut", lut)

    def forward(self, input_ids):
        base_out = self.base(input_ids)
        idx = self.lut[input_ids]
        mask = idx >= 0
        if mask.any():
            fe = self.fake_emb(idx.clamp_min(0)).to(base_out.dtype)
            base_out = torch.where(mask.unsqueeze(-1), fe, base_out)
        return base_out


class SoftPromptCausalLM(nn.Module):
    def __init__(self, base_model, soft_prompt_embeds):
        super().__init__()
        self.base = base_model
        self.soft = nn.Parameter(soft_prompt_embeds)  # [P, D]
        self.prefix_len = int(soft_prompt_embeds.shape[0])

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        assert input_ids is not None, "SoftPromptCausalLM expects input_ids"
        bsz = input_ids.shape[0]

        tok_emb = self.base.get_input_embeddings()(input_ids)  # [B, T, D]
        soft = self.soft.unsqueeze(0).expand(bsz, -1, -1).to(tok_emb.dtype)  # [B, P, D]
        inputs_embeds = torch.cat([soft, tok_emb], dim=1)  # [B, P+T, D]

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.long)
        prefix_mask = torch.ones((bsz, self.prefix_len), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, P+T]

        return self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)


def load_soft_prompt_pt(soft_dir, base_model):
    pt_path = os.path.join(soft_dir, "soft_prompt.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing soft prompt file: {pt_path}")

    obj = torch.load(pt_path, map_location="cpu")

    if torch.is_tensor(obj):
        soft = obj
    elif isinstance(obj, dict):
        candidate_keys = ["soft_prompt", "soft_prompt_embeds", "prompt_embeddings", "soft_prompt.weight"]
        soft = None
        for k in candidate_keys:
            if k in obj and torch.is_tensor(obj[k]):
                soft = obj[k]
                break
        if soft is None:
            for v in obj.values():
                if torch.is_tensor(v):
                    soft = v
                    break
        if soft is None:
            raise ValueError(f"soft_prompt.pt dict has no tensor entries. Keys={list(obj.keys())}")
    else:
        raise ValueError(f"Unsupported soft_prompt.pt type: {type(obj)}")

    if soft.ndim == 3 and soft.shape[0] == 1:
        soft = soft.squeeze(0)
    if soft.ndim != 2:
        raise ValueError(f"Expected soft prompt shape [P, D], got {tuple(soft.shape)}")

    hidden = base_model.get_input_embeddings().weight.shape[1]
    if soft.shape[1] != hidden:
        raise ValueError(f"Soft prompt dim mismatch: soft={soft.shape[1]} vs hidden={hidden}")

    return SoftPromptCausalLM(base_model, soft)



def load_base():
    tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    model.config.use_cache = False
    return tok, model


def load_embed():
    tok, model = load_base()

    with open(os.path.join(EMBED_DIR, "fake_token_meta.json")) as f:
        meta = json.load(f)

    base_emb = model.get_input_embeddings()
    model.set_input_embeddings(FakeOnlyEmbedding(base_emb, meta["fake_token_ids"]).to(device))

    state_file = find_first_existing([
        os.path.join(EMBED_DIR, "model.safetensors"),
        os.path.join(EMBED_DIR, "pytorch_model.bin"),
        os.path.join(EMBED_DIR, "adapter_model.safetensors"),
    ])

    if state_file is None:
        print("[WARN] Embed checkpoint not found in:", EMBED_DIR)
        return tok, model

    if state_file.endswith(".safetensors"):
        state = load_file(state_file, device="cpu")
        model.load_state_dict(state, strict=False)
        print("[Embed] Loaded:", state_file)
    else:
        state = torch.load(state_file, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("[Embed] Loaded:", state_file)

    return tok, model


def load_lora():
    tok, base = load_base()

    target_vocab = infer_target_vocab_size(LORA_DIR, fallback=None)
    if target_vocab is not None and target_vocab != base.get_input_embeddings().weight.shape[0]:
        print(f"[LoRA] Resizing token embeddings: {base.get_input_embeddings().weight.shape[0]} -> {target_vocab}")
        base.resize_token_embeddings(target_vocab)

    if not os.path.exists(os.path.join(LORA_DIR, "adapter_config.json")):
        raise ValueError(f"LoRA folder '{LORA_DIR}' missing adapter_config.json")

    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.to(device)
    model.eval()
    return tok, model


def load_soft():
    tok, base = load_base()
    model = load_soft_prompt_pt(SOFT_DIR, base).to(device)
    model.eval()
    return tok, model


def load_finetune():
  
    tok = AutoTokenizer.from_pretrained(FINETUNE_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Most robust: let Transformers load weights (handles sharded safetensors via index.json). [web:45]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            FINETUNE_DIR,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        model.config.use_cache = False
        return tok, model
    except Exception as e:
        print(f"[Finetune][WARN] from_pretrained failed ({type(e).__name__}: {e}). Falling back to manual shard load.")

    # Fallback: instantiate base, resize to finetune tokenizer length, load shards by weight_map. [web:45]
    _, model = load_base()
    target_vocab = len(tok)
    cur_vocab = model.get_input_embeddings().weight.shape[0]
    if target_vocab != cur_vocab:
        print(f"[Finetune] Resizing token embeddings: {cur_vocab} -> {target_vocab}")
        model.resize_token_embeddings(target_vocab)

    state = load_sharded_safetensors_state_dict(FINETUNE_DIR)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Finetune] Loaded sharded weights | missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    model.config.use_cache = False
    return tok, model



models = {
    "Base": load_base(),
    "Embed": load_embed(),
    "LoRA": load_lora(),
    "Soft": load_soft(),
    "Finetune": load_finetune(),
}



datasets = {"Real": real_sents, "Wiki": wiki_sents}
results = {}

soft_prefix_len = models["Soft"][1].prefix_len  # used only for label alignment

for dname, data in datasets.items():
    print(f"\n===== {dname} =====")
    results[dname] = {}

    for mname, (tok, model) in models.items():
        prefix_len = soft_prefix_len if mname == "Soft" else 0
        L, P = lm_loss(model, tok, data, prefix_len=prefix_len)
        results[dname][mname] = (L, P)
        print(f"{mname:<8} | Loss {L:.4f} | PPL {P:.2f}")



print("\n===== FORGETTING (Δ vs BASE) =====")
for dname in datasets:
    print(f"\n--- {dname} ---")
    base_L, base_P = results[dname]["Base"]
    for m in ["Embed", "LoRA", "Soft", "Finetune"]:
        L, P = results[dname][m]
        print(f"{m:<8} | ΔLoss {L-base_L:+.4f} | ΔPPL {P-base_P:+.2f}")

print("\nDONE")
