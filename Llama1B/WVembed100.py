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

save_root = "/scratch/yl258/kp759/llama3.2-1B/embed_tuning_no_valid_100_500/"
os.makedirs(save_root, exist_ok=True)

K_CONFIGS = [ 10, 30, 100, 300, 500, 1000, 2000, 3000]


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
fake_words = ['Quib', 'Flem', 'Snix', 'Glop', 'Whiz', 'Plix', 'Trog', 'Brin', 'Flob', 'Quex', 'Snef', 'Grim', 'Blip', 'Zink', 'Frop', 'Whup', 'Plim', 'Trix', 'Glim', 'Snop', 'Brem', 'Quix', 'Flep', 'Blix', 'Grop', 'Whim', 'Plek', 'Snib', 'Flib', 'Quop', 'Trem', 'Blox', 'Zeff', 'Glix', 'Whop', 'Prim', 'Flim', 'Queb', 'Brog', 'Glep', 'Whix', 'Plob', 'Trim', 'Flix', 'Zindlequo', 'Brimflux', 'Quorping', 'Snizzle', 'Tranglex', 'Florbenzy', 'Glimporix', 'Whamblu', 'Quizzlor', 'Bronglex', 'Splendor', 'Frimbola', 'Quibble', 'Snorfleg', 'Wizzlebo', 'Plinderg', 'Flabnort', 'Quimblejax', 'Grizzlew', 'Blomfuzz', 'Snickerd', 'Whimbler', 'Plinkbox', 'Fribbole', 'Quazzlen', 'Glorbent', 'Snifflex', 'Whomperl', 'Blimflur', 'Quorplet', 'Frimzigg', 'Snazzlep', 'Whizzlebot', 'Plumberox', 'Glorpenwix', 'Snufflebox', 'Quimbletop', 'Flibberjoy', 'Brazzlenip', 'Whamplebox', 'Quindlezop', 'Snorflequx', 'Blimpledor', 'Frazzelwin', 'Quopplebox', 'Glimmertux', 'Snibblerot', 'Whirbelux', 'Plonkertip', 'Frimblewig', 'Quazzlepox', 'Snorkelbox', 'Blimzerton', 'Whomblefiz', 'Glorpenflux', 'Quimzlebot']

assert len(fake_words) == 100

num_added = tokenizer.add_special_tokens({"additional_special_tokens": fake_words})
assert num_added == 100

fake_token_ids = tokenizer.convert_tokens_to_ids(fake_words)
assert all(tid != tokenizer.unk_token_id for tid in fake_token_ids)

old_vocab_size = tokenizer.vocab_size
new_vocab_size = len(tokenizer)
assert max(fake_token_ids) < new_vocab_size

with open(os.path.join(save_root, "fake_token_meta.json"), "w") as f:
    json.dump({
        "fake_words": fake_words,
        "fake_token_ids": fake_token_ids,
        "old_vocab": old_vocab_size,
        "new_vocab": new_vocab_size
    }, f, indent=2)


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


def mask_rows(param, rows):
    rows = torch.tensor(rows, device=param.device)
    def hook(grad):
        mask = torch.zeros_like(grad)
        mask.index_fill_(0, rows, 1.0)
        return grad * mask
    param.register_hook(hook)


for K in K_CONFIGS:
    run_dir = os.path.join(save_root, f"K_{K}")
    base_dir = os.path.join(run_dir, "BASE")
    final_dir = os.path.join(run_dir, "FINAL")
    os.makedirs(run_dir, exist_ok=True)

    log = open(os.path.join(run_dir, "log.txt"), "w")

    train_sents = full_train[:K]


    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    base.resize_token_embeddings(new_vocab_size, mean_resizing=False)
    base.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    del base
    torch.cuda.empty_cache()


    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.bfloat16
    ).to(device)

    for p in model.parameters():
        p.requires_grad = False

    emb = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    emb.weight.requires_grad = True
    lm_head.weight.requires_grad = True

    mask_rows(emb.weight, fake_token_ids)
    mask_rows(lm_head.weight, fake_token_ids)


    total_params = sum(p.numel() for p in model.parameters())
    hidden_dim = emb.weight.shape[1]
    num_fake = len(fake_token_ids)
    effective_params = num_fake * hidden_dim * 2  

    print(f"[INFO] Hidden dim: {hidden_dim}")
    print(f"[INFO] Fake tokens: {num_fake}")
    print(f"[INFO] Effective trainable params: {effective_params:,}")
    print(f"[INFO] Effective percent: {100 * effective_params / total_params:.6f}%")
    print(f"[INFO] Nominal trainable (PyTorch): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    params = list({id(p): p for p in [emb.weight, lm_head.weight]}.values())
    optimizer = AdamW([{"params": params, "lr": lr}])

   
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

        msg = (f"Epoch {epoch:03d} | Train={train_loss:.4f} | Test={test_loss:.4f}")
        print(msg)
        log.write(msg + "\n")
        log.flush()

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump({
            "method": "embedding_only",
            "K": K,
            "lr": lr,
            "batch_size": batch_size,
            "epochs_run": epochs,
            "old_vocab": old_vocab_size,
            "new_vocab": new_vocab_size,
            "num_fake_tokens": len(fake_token_ids)
        }, f, indent=2)

    log.close()
    del model
    torch.cuda.empty_cache()

print("DONE")
