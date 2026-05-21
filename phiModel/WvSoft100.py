

import os
import json
import random
import numpy as np
import torch
from torch.optim import AdamW
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PromptTuningConfig, get_peft_model, TaskType

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using:", device)


model_name = "microsoft/Phi-3.5-mini-instruct"
train_path = "/scratch/yl258/kp759/llama3.2-1B/Sent/fake_sentences.txt"
test_path  = "/scratch/yl258/kp759/llama3.2-1B/Sent/Fake_test.txt"

max_length = 256
batch_size = 32
lr = 3e-4
epochs = 100
prompt_length = 20  

save_root = "/scratch/yl258/kp759/llPhi3.5-mini-Instruct/soft_prompt_Wv/"
os.makedirs(save_root, exist_ok=True)

K_CONFIGS = [10,30,100,300,500,1000,2000, 3000]

with open(train_path) as f:
    full_train = [l.strip() for l in f if l.strip()]

with open(test_path) as f:
    test_sents = [l.strip() for l in f if l.strip()]


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_batch(texts):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    labels[:, 0] = -100
    enc["labels"] = labels
    return enc

_ce = nn.CrossEntropyLoss(reduction="none")

def tokenwise_loss(logits, labels, prompt_len=0):
    logits = logits[:, prompt_len:-1]
    labels = labels[:, 1:]
    loss = _ce(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1)
    )
    mask = labels.reshape(-1) != -100
    return loss[mask].mean()

@torch.no_grad()
def evaluate(model, sents, prompt_len=0):
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(sents), batch_size):
        d = tokenize_batch(sents[i:i + batch_size])
        out = model(**d)
        logits = out.logits[:, prompt_len:-1]
        labels = d["labels"][:, 1:]
        mask = labels != -100
        tot += _ce(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )[mask.reshape(-1)].sum().item()
        cnt += mask.sum().item()
    model.train()
    return tot / max(cnt, 1)

for K in K_CONFIGS:

    run_dir   = os.path.join(save_root, f"K_{K}")
    final_dir = os.path.join(run_dir, "FINAL")
    os.makedirs(run_dir, exist_ok=True)


    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "K": K,
            "prompt_length": prompt_length
        }, f, indent=2)

    log_file = open(os.path.join(run_dir, "train.log"), "w")
    log_file.write("epoch,train_loss,test_loss\n")

    train_sents = full_train[:K]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.to(device)


    soft_prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=prompt_length
    )
    model = get_peft_model(model, soft_prompt_config)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {trainable_params:,} | Total: {total_params:,} | Fraction: {100*trainable_params/total_params:.6f}%")

    optimizer = AdamW(model.parameters(), lr=lr)

  
    base_train = evaluate(model, train_sents, prompt_len=prompt_length)
    base_test  = evaluate(model, test_sents, prompt_len=prompt_length)
    print(f"BASE | Train={base_train:.4f} | Test={base_test:.4f}")
    log_file.write(f"BASE | Train={base_train:.4f} | Test={base_test:.4f}\n")

 
    for epoch in range(epochs):
        random.shuffle(train_sents)
        for i in range(0, len(train_sents), batch_size):
            batch = tokenize_batch(train_sents[i:i + batch_size])
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = tokenwise_loss(out.logits, batch["labels"], prompt_len=prompt_length)
            loss.backward()
            optimizer.step()

        tr = evaluate(model, train_sents, prompt_len=prompt_length)
        te = evaluate(model, test_sents, prompt_len=prompt_length)

        print(f"Epoch {epoch:03d} | Train={tr:.4f} | Test={te:.4f}")
        log_file.write(f"Epoch {epoch:03d} | Train={tr:.4f} | Test={te:.4f}\n")
        log_file.flush()

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log_file.close()
    del model
    torch.cuda.empty_cache()

print("DONE")
