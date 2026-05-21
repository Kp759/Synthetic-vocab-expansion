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

device = torch.device("cuda:0")
print("Using:", device)


model_name = "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

train_path = "/scratch/yl258/kp759/llama3.2-1B/Sent/fake_sentences.txt"
test_path  = "/scratch/yl258/kp759/llama3.2-1B/Sent/Fake_test.txt"

max_length = 256
batch_size = 32
lr = 3e-4
epochs = 100


N_PROMPT = 20

save_root = "/scratch/yl258/kp759/llama3.2-1B/soft_prompt_no_valid_500/"
os.makedirs(save_root, exist_ok=True)

K_CONFIGS = [10, 30, 100, 300, 500, 1000, 2000, 3000]


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


with open(train_path) as f:
    full_train = [l.strip() for l in f if l.strip()]

with open(test_path) as f:
    test_sents = [l.strip() for l in f if l.strip()]

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

_ce = nn.CrossEntropyLoss(reduction="none")

def tokenwise_loss_tensor(logits, labels):
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    B, T, V = logits.shape
    loss = _ce(logits.reshape(B*T, V), labels.reshape(B*T))
    mask = labels.reshape(-1) != -100
    return loss[mask].mean()


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


for K in K_CONFIGS:
    run_dir = os.path.join(save_root, f"K_{K}")
    base_dir = os.path.join(run_dir, "BASE")
    final_dir = os.path.join(run_dir, "FINAL")
    os.makedirs(run_dir, exist_ok=True)

    log = open(os.path.join(run_dir, "log.txt"), "w")
    train_sents = full_train[:K]


    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    ).to(device)

    base.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    del base
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16
    ).to(device)

    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.get_input_embeddings().embedding_dim


    soft_prompt = nn.Parameter(
        torch.randn(N_PROMPT, hidden_dim, device=device, dtype=torch.bfloat16)
    )

    total_params = sum(p.numel() for p in model.parameters()) + soft_prompt.numel()
    num_soft_tokens = N_PROMPT
    effective_params = soft_prompt.numel()

    print(f"[INFO] Hidden dim: {hidden_dim}")
    print(f"[INFO] Soft prompt tokens: {num_soft_tokens}")
    print(f"[INFO] Effective trainable params: {effective_params:,}")
    print(f"[INFO] Effective percent: {100 * effective_params / total_params:.6f}%")
    print(f"[INFO] Nominal trainable (PyTorch): {sum(p.numel() for p in [soft_prompt] if p.requires_grad):,}")
    optimizer = AdamW([soft_prompt], lr=lr)

    
    base_train = evaluate_base(model, train_sents)
    base_test  = evaluate_base(model, test_sents)
    print(f"BASE | Train={base_train:.4f} | Test={base_test:.4f}")

   
    for epoch in range(epochs):
        random.shuffle(train_sents)
        for i in range(0, len(train_sents), batch_size):
            d = tokenize_batch(train_sents[i:i+batch_size])

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

            optimizer.zero_grad(set_to_none=True)
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                labels=labels
            )
            out.loss.backward()
            optimizer.step()

        train_loss = evaluate_soft(model, soft_prompt, train_sents)
        test_loss  = evaluate_soft(model, soft_prompt, test_sents)

        msg = f"Epoch {epoch:03d} | Train={train_loss:.4f} | Test={test_loss:.4f}"
        print(msg)
        log.write(msg + "\n")
        log.flush()


    os.makedirs(final_dir, exist_ok=True)
    torch.save(
        {"soft_prompt": soft_prompt.detach().cpu()},
        os.path.join(final_dir, "soft_prompt.pt")
    )
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump({
            "method": "soft_prompt",
            "K": K,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "num_prompt_tokens": N_PROMPT
        }, f, indent=2)

    log.close()
    del model
    torch.cuda.empty_cache()

print("DONE")
