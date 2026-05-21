import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------------
# HuggingFace cache dirs
# -------------------------
os.environ["HF_HOME"] = "/scratch/yl258/kp759/hf"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/yl258/kp759/hf"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yl258/kp759/hf"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model_name =  "/scratch/yl258/kp759/hf/models--Qwen--Qwen2.5-3B-instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

train_path = "/scratch/yl258/kp759/llama3.2-1B/Sent/fake_sentences.txt"
test_path  = "/scratch/yl258/kp759/llama3.2-1B/Sent/Fake_test.txt"

max_length = 256
batch_size = 32
lr = 2e-4
epochs = 100

save_root = "/scratch/yl258/kp759/llQwen2.5-3B-Instruct/WvEmbed_tuning_100/"
os.makedirs(save_root, exist_ok=True)

K_CONFIGS = [10,30,100,300,500,1000,2000,3000]


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


fake_words = ['Quib', 'Flem', 'Snix', 'Glop', 'Whiz', 'Plix', 'Trog', 'Brin', 'Flob', 'Quex', 
              'Snef', 'Grim', 'Blip', 'Zink', 'Frop', 'Whup', 'Plim', 'Trix', 'Glim', 'Snop',
              'Brem', 'Quix', 'Flep', 'Blix', 'Grop', 'Whim', 'Plek', 'Snib', 'Flib', 'Quop',
              'Trem', 'Blox', 'Zeff', 'Glix', 'Whop', 'Prim', 'Flim', 'Queb', 'Brog', 'Glep',
              'Whix', 'Plob', 'Trim', 'Flix', 'Zindlequo', 'Brimflux', 'Quorping', 'Snizzle',
              'Tranglex', 'Florbenzy', 'Glimporix', 'Whamblu', 'Quizzlor', 'Bronglex', 'Splendor',
              'Frimbola', 'Quibble', 'Snorfleg', 'Wizzlebo', 'Plinderg', 'Flabnort', 'Quimblejax',
              'Grizzlew', 'Blomfuzz', 'Snickerd', 'Whimbler', 'Plinkbox', 'Fribbole', 'Quazzlen',
              'Glorbent', 'Snifflex', 'Whomperl', 'Blimflur', 'Quorplet', 'Frimzigg', 'Snazzlep',
              'Whizzlebot', 'Plumberox', 'Glorpenwix', 'Snufflebox', 'Quimbletop', 'Flibberjoy',
              'Brazzlenip', 'Whamplebox', 'Quindlezop', 'Snorflequx', 'Blimpledor', 'Frazzelwin',
              'Quopplebox', 'Glimmertux', 'Snibblerot', 'Whirbelux', 'Plonkertip', 'Frimblewig',
              'Quazzlepox', 'Snorkelbox', 'Blimzerton', 'Whomblefiz', 'Glorpenflux', 'Quimzlebot']

assert len(fake_words) == 100

tokenizer.add_special_tokens({"additional_special_tokens": fake_words})
fake_token_ids = tokenizer.convert_tokens_to_ids(fake_words)

old_vocab_size = tokenizer.vocab_size
new_vocab_size = len(tokenizer)
print("Old vocab size:", old_vocab_size)
print("New vocab size:", new_vocab_size)


with open(train_path) as f:
    train_sents = [l.strip() for l in f if l.strip()]

with open(test_path) as f:
    test_sents = [l.strip() for l in f if l.strip()]


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

def tokenwise_loss(logits, labels):
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    loss = _ce(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    mask = labels.reshape(-1) != -100
    return loss[mask].mean()

@torch.no_grad()
def evaluate(model, sents):
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(sents), batch_size):
        d = tokenize_batch(sents[i:i + batch_size])
        out = model(**d)
        loss = _ce(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            d["labels"][:, 1:].reshape(-1)
        )
        mask = d["labels"][:, 1:].reshape(-1) != -100
        tot += loss[mask].sum().item()
        cnt += mask.sum().item()
    model.train()
    return tot / max(cnt, 1)


def mask_rows(param, rows):
    rows = torch.tensor(rows, device=param.device)
    def hook(grad):
        mask = torch.zeros_like(grad)
        mask.index_fill_(0, rows, 1.0)
        return grad * mask
    param.register_hook(hook)


for K in K_CONFIGS:

    run_dir   = os.path.join(save_root, f"K_{K}")
    base_dir  = os.path.join(run_dir, "BASE")
    final_dir = os.path.join(run_dir, "FINAL")
    os.makedirs(run_dir, exist_ok=True)


    log_path = os.path.join(run_dir, "train.log")
    log_f = open(log_path, "w")
    log_f.write("epoch,train_loss,test_loss\n")
    log_f.flush()


    with open(os.path.join(run_dir, "fake_token_meta.json"), "w") as f:
        json.dump(
            {
                "model": model_name,
                "K": K,
                "fake_words": fake_words,
                "fake_token_ids": fake_token_ids,
                "old_vocab": old_vocab_size,
                "new_vocab": new_vocab_size
            },
            f,
            indent=2
        )

    train_batch = train_sents[:K]

   
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
    model.config.use_cache = False

    if not os.path.exists(base_dir):
        model.save_pretrained(base_dir)
        tokenizer.save_pretrained(base_dir)


    for p in model.parameters():
        p.requires_grad = False

    emb = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    emb.weight.requires_grad = True
    lm_head.weight.requires_grad = True

    mask_rows(emb.weight, fake_token_ids)
    mask_rows(lm_head.weight, fake_token_ids)

    optimizer = AdamW([emb.weight, lm_head.weight], lr=lr)


    base_tr = evaluate(model, train_batch)
    base_te = evaluate(model, test_sents)

    print(f"K={K} | BASE | Train={base_tr:.4f} | Test={base_te:.4f}")
    log_f.write(f"K={K} | BASE | Train={base_tr:.4f} | Test={base_te:.4f}\n")
    log_f.flush()


    for epoch in range(epochs):
        random.shuffle(train_batch)
        for i in range(0, len(train_batch), batch_size):
            d = tokenize_batch(train_batch[i:i + batch_size])
            optimizer.zero_grad(set_to_none=True)
            out = model(**d)
            loss = tokenwise_loss(out.logits, d["labels"])
            loss.backward()
            optimizer.step()

        tr = evaluate(model, train_batch)
        te = evaluate(model, test_sents)

        print(f"K={K} | Epoch {epoch:03d} | Train={tr:.4f} | Test={te:.4f}")
        log_f.write(f"K={K} | Epoch {epoch:03d} | Train={tr:.4f} | Test={te:.4f}\n")
        log_f.flush()

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log_f.close()

print("DONE")
