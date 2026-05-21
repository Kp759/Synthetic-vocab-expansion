import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


torch.cuda.set_device(0)
device = torch.device("cuda:0")
print("Using primary device:", device)

model_name = "/scratch/yl258/kp759/hf/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"


train_path = "/scratch/yl258/kp759/llama3.2-1B/Sent/fake_sentences.txt"
test_path  = "/scratch/yl258/kp759/llama3.2-1B/Sent/Fake_test.txt"

max_length = 256
batch_size = 8             
lr = 2e-4                
epochs = 100


gradient_accumulation_steps = 4  

N_PROMPT = 20

save_root = "/scratch/yl258/kp759/llama3.1-8B/soft_prompt_no_valid_1000/"
os.makedirs(save_root, exist_ok=True)

K_CONFIGS = [10,30,100,300,500,1000,2000,3000]

# =================================================
# Tokenizer
# =================================================
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =================================================
# Data
# =================================================
with open(train_path) as f:
    full_train = [l.strip() for l in f if l.strip()]

with open(test_path) as f:
    test_sents = [l.strip() for l in f if l.strip()]

# =================================================
# Tokenization + Loss
# =================================================
def tokenize_batch(batch):
    enc = tokenizer(
        batch,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc

# --------- Evaluate with soft prompt ---------
@torch.no_grad()
def evaluate_soft(model, soft_prompt, sents):
    model.eval()
    tot, cnt = 0.0, 0

    for i in range(0, len(sents), batch_size):
        d = tokenize_batch(sents[i:i+batch_size])

        inputs_embeds = model.get_input_embeddings()(d["input_ids"])
        B = inputs_embeds.size(0)

        prompt = soft_prompt.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)

        attn = torch.cat([
            torch.ones(B, N_PROMPT, device=device),
            d["attention_mask"]
        ], dim=1)

        labels = torch.cat([
            torch.full((B, N_PROMPT), -100, device=device),
            d["labels"]
        ], dim=1)

        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels
        )

        valid = (labels != -100).sum().item()
        tot += out.loss.item() * valid
        cnt += valid

    model.train()
    return tot / cnt

# --------- Evaluate BASE model without soft prompt ---------
@torch.no_grad()
def evaluate_base(model, sents):
    model.eval()
    tot, cnt = 0.0, 0

    for i in range(0, len(sents), batch_size):
        d = tokenize_batch(sents[i:i+batch_size])
        out = model(**d)
        valid = (d["labels"] != -100).sum().item()
        tot += out.loss.item() * valid
        cnt += valid

    model.train()
    return tot / cnt

# =================================================
# Training (Soft Prompt ONLY)
# =================================================
for K in K_CONFIGS:
    run_dir = os.path.join(save_root, f"K_{K}")
    base_dir = os.path.join(run_dir, "BASE")
    final_dir = os.path.join(run_dir, "FINAL")
    os.makedirs(run_dir, exist_ok=True)

    log = open(os.path.join(run_dir, "log.txt"), "w")
    train_sents = full_train[:K]

    # ---- BASE MODEL ----
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # important for 8B if multi-gpu / memory limits
    )

    base.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    del base
    torch.cuda.empty_cache()

    # ---- LOAD MODEL ----
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # freeze model weights
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.get_input_embeddings().embedding_dim

    # ---- SOFT PROMPT ----
    soft_prompt = nn.Parameter(
        torch.randn(N_PROMPT, hidden_dim, dtype=torch.bfloat16, device=device)
    )

    total_params = sum(p.numel() for p in model.parameters()) + soft_prompt.numel()
    effective_params = soft_prompt.numel()

    print(f"[INFO] Hidden dim: {hidden_dim}")
    print(f"[INFO] Soft prompt tokens: {N_PROMPT}")
    print(f"[INFO] Effective trainable params: {effective_params:,}")
    print(f"[INFO] Effective percent: {100 * effective_params / total_params:.6f}%")

    optimizer = AdamW([soft_prompt], lr=lr)

    # ---- BASELINE ----
    base_train = evaluate_base(model, train_sents)
    base_test  = evaluate_base(model, test_sents)
    print(f"BASE | Train={base_train:.4f} | Test={base_test:.4f}")

    # ---- TRAIN SOFT PROMPT ----
    for epoch in range(epochs):
        random.shuffle(train_sents)

        optimizer.zero_grad(set_to_none=True)

        for step in range(0, len(train_sents), batch_size):
            d = tokenize_batch(train_sents[step:step+batch_size])

            inputs_embeds = model.get_input_embeddings()(d["input_ids"])
            B = inputs_embeds.size(0)

            prompt = soft_prompt.unsqueeze(0).expand(B, -1, -1)
            inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)

            attn = torch.cat([
                torch.ones(B, N_PROMPT, device=device),
                d["attention_mask"]
            ], dim=1)

            labels = torch.cat([
                torch.full((B, N_PROMPT), -100, device=device),
                d["labels"]
            ], dim=1)

            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                labels=labels
            )

            loss = out.loss / gradient_accumulation_steps
            loss.backward()

            if ((step // batch_size) + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # eval
        train_loss = evaluate_soft(model, soft_prompt, train_sents)
        test_loss  = evaluate_soft(model, soft_prompt, test_sents)

        msg = f"Epoch {epoch:03d} | Train={train_loss:.4f} | Test={test_loss:.4f}"
        print(msg)
        log.write(msg + "\n")
        log.flush()

    # ---- SAVE SOFT PROMPT ----
    os.makedirs(final_dir, exist_ok=True)
    torch.save(
        {"soft_prompt": soft_prompt.detach().cpu()},
        os.path.join(final_dir, "soft_prompt.pt")
    )
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump({
            "method": "soft_prompt",
            "model": "llama3.1-8B",
            "K": K,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum": gradient_accumulation_steps,
            "epochs": epochs,
            "num_prompt_tokens": N_PROMPT
        }, f, indent=2)

    log.close()
    del model
    torch.cuda.empty_cache()

print("DONE")
