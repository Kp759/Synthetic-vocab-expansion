import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda:0")
print("Using:", device)

model_name = "/scratch/yl258/kp759/hf/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

train_path = "/scratch/yl258/kp759/llama3.2-1B/Sent/fake_sentences.txt"
test_path  = "/scratch/yl258/kp759/llama3.2-1B/Sent/Fake_test.txt"

max_length = 256
batch_size = 8       
lr = 3e-4
epochs = 100

save_root = "/scratch/yl258/kp759/llama3.1-8B/lora_only_no_valid_1000/"
os.makedirs(save_root, exist_ok=True)

K_CONFIGS = [10,30,100,300,500,1000,2000,3000]


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
def evaluate(model, sents):
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(sents), batch_size):
        d = tokenize_batch(sents[i:i+batch_size])
        out = model(**d)
        logits = out.logits[:, :-1]
        labels = d["labels"][:, 1:]
        loss = _ce(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        mask = labels.reshape(-1) != -100
        tot += loss[mask].sum().item()
        cnt += mask.sum().item()
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

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=lr)


    base_train = evaluate(model, train_sents)
    base_test  = evaluate(model, test_sents)
    print(f"BASE | Train={base_train:.4f} | Test={base_test:.4f}")


    for epoch in range(epochs):
        random.shuffle(train_sents)
        for i in range(0, len(train_sents), batch_size):
            d = tokenize_batch(train_sents[i:i+batch_size])
            optimizer.zero_grad(set_to_none=True)
            out = model(**d)
            loss = tokenwise_loss_tensor(out.logits, d["labels"])
            loss.backward()
            optimizer.step()

        train_loss = evaluate(model, train_sents)
        test_loss  = evaluate(model, test_sents)

        msg = f"Epoch {epoch:03d} | Train={train_loss:.4f} | Test={test_loss:.4f}"
        print(msg)
        log.write(msg + "\n")
        log.flush()


    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump({
            "method": "lora_only",
            "K": K,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }, f, indent=2)

    log.close()
    del model
    torch.cuda.empty_cache()

print("DONE")
