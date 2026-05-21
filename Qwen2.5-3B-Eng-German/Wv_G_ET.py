import os
import json
import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch/yl258/kp759/hf"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/yl258/kp759/hf"
os.environ["HF_DATASETS_CACHE"] = "/scratch/yl258/kp759/hf"

SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model_name = "/scratch/yl258/kp759/hf/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

data_root = "/scratch/yl258/kp759/llQwen2.5-3B/data/eng_es_vocab_expansion"
save_root = "/scratch/yl258/kp759/llQwen2.5-3B/results_eng_de_et_lm_fast"
os.makedirs(data_root, exist_ok=True)
os.makedirs(save_root, exist_ok=True)

german_dir = os.path.join(data_root, "german")
english_dir = os.path.join(data_root, "english")
os.makedirs(german_dir, exist_ok=True)
os.makedirs(english_dir, exist_ok=True)

german_train_path = os.path.join(german_dir, "train.txt")
german_test_path = os.path.join(german_dir, "test.txt")
english_test_path = os.path.join(english_dir, "test.txt")

AUTO_PREPARE_DATA = True
NUM_DE_TRAIN_SAVE = 6000
NUM_DE_TEST_SAVE = 1000
NUM_EN_TEST_SAVE = 1000
MIN_CHAR_LEN = 200

K_CONFIGS = [1000]
NUM_NEW_TOKENS = 500
MIN_WORD_FREQ = 5
MIN_WORD_LEN = 3

max_length = 256
batch_size = 4
lr = 2e-4
epochs = 20

# Faster eval cadence
EVAL_DE_EVERY = 2   # German test every 2 epochs
EVAL_EN_EVERY = 5   # English test every 5 epochs

def clean_text(s: str) -> str:
    return " ".join(s.replace("\n", " ").strip().split())

def save_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_local_wiki_files():
    print("Preparing local German/English txt files from Hugging Face wikipedia...")

    ds_de = load_dataset("wikimedia/wikipedia", "20231101.de", split="train", streaming=True)
    de_texts = []
    for ex in ds_de:
        txt = clean_text(ex["text"])
        if len(txt) >= MIN_CHAR_LEN:
            de_texts.append(txt)
        if len(de_texts) >= (NUM_DE_TRAIN_SAVE + NUM_DE_TEST_SAVE):
            break

    ds_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    en_texts = []
    for ex in ds_en:
        txt = clean_text(ex["text"])
        if len(txt) >= MIN_CHAR_LEN:
            en_texts.append(txt)
        if len(en_texts) >= NUM_EN_TEST_SAVE:
            break

    if len(de_texts) < (NUM_DE_TRAIN_SAVE + NUM_DE_TEST_SAVE):
        raise RuntimeError(
            f"Not enough German examples collected: got {len(de_texts)}, "
            f"need at least {NUM_DE_TRAIN_SAVE + NUM_DE_TEST_SAVE}"
        )
    if len(en_texts) < NUM_EN_TEST_SAVE:
        raise RuntimeError(
            f"Not enough English examples collected: got {len(en_texts)}, "
            f"need at least {NUM_EN_TEST_SAVE}"
        )

    save_lines(german_train_path, de_texts[:NUM_DE_TRAIN_SAVE])
    save_lines(german_test_path, de_texts[NUM_DE_TRAIN_SAVE:NUM_DE_TRAIN_SAVE + NUM_DE_TEST_SAVE])
    save_lines(english_test_path, en_texts[:NUM_EN_TEST_SAVE])

    print("Saved:")
    print(" ", german_train_path, NUM_DE_TRAIN_SAVE)
    print(" ", german_test_path, NUM_DE_TEST_SAVE)
    print(" ", english_test_path, NUM_EN_TEST_SAVE)

if AUTO_PREPARE_DATA:
    if not (os.path.exists(german_train_path) and os.path.exists(german_test_path) and os.path.exists(english_test_path)):
        build_local_wiki_files()
    else:
        print("Local txt files already exist. Skipping download.")

# =========================================================
# LOAD BASE TOKENIZER
# =========================================================
base_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
    trust_remote_code=True
)

if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

base_vocab_size = len(base_tokenizer)
print("Base vocab size:", base_vocab_size)

# =========================================================
# LOAD LOCAL DATA
# =========================================================
german_train_sents_all = load_lines(german_train_path)
german_test_sents = load_lines(german_test_path)
english_test_sents = load_lines(english_test_path)

print(f"Loaded {len(german_train_sents_all)} German train examples")
print(f"Loaded {len(german_test_sents)} German test examples")
print(f"Loaded {len(english_test_sents)} English test examples")

# =========================================================
# MINE NEW GERMAN TOKENS
# =========================================================
_word_pattern = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)?", re.UNICODE)

def normalize_word(w: str) -> str:
    return w.strip().lower()

def get_candidate_new_tokens(texts, tokenizer, top_k=500, min_freq=5, min_len=3):
    """
    Choose frequent German words that are not currently atomic vocab items
    and are split into multiple tokenizer pieces.
    """
    counter = Counter()
    for line in texts:
        for w in _word_pattern.findall(line):
            w = normalize_word(w)
            if len(w) >= min_len:
                counter[w] += 1

    ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    existing_vocab = tokenizer.get_vocab()
    candidates = []

    for word, freq in ranked:
        if freq < min_freq:
            continue
        if word in existing_vocab:
            continue
        pieces = tokenizer.tokenize(word)
        if len(pieces) <= 1:
            continue
        candidates.append((word, freq, pieces))
        if len(candidates) >= top_k:
            break
    return candidates

candidate_tokens = get_candidate_new_tokens(
    german_train_sents_all,
    base_tokenizer,
    top_k=NUM_NEW_TOKENS,
    min_freq=MIN_WORD_FREQ,
    min_len=MIN_WORD_LEN,
)

new_tokens = [x[0] for x in candidate_tokens]
print(f"Selected {len(new_tokens)} new German tokens")
print("Sample mined tokens:", new_tokens[:30])

if len(new_tokens) == 0:
    raise RuntimeError("No new German tokens mined.")

_ce = nn.CrossEntropyLoss(reduction="none")

def build_tokenizer_with_new_tokens():
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Actually added {num_added} new tokens")
    return tokenizer

def tokenize_batch(tokenizer, texts):
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

def tokenwise_loss(logits, labels):
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    loss = _ce(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    mask = labels.reshape(-1) != -100
    return loss[mask].mean()

@torch.no_grad()
def evaluate(model, tokenizer, sents):
    model.eval()
    tot, cnt = 0.0, 0

    for i in range(0, len(sents), batch_size):
        batch_texts = sents[i:i + batch_size]
        d = tokenize_batch(tokenizer, batch_texts)
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

def tied_input_output_embeddings(model):
    in_emb = model.get_input_embeddings().weight
    out_emb = model.get_output_embeddings().weight
    return in_emb.data_ptr() == out_emb.data_ptr()

all_results = []

for K in K_CONFIGS:
    german_train_sents = german_train_sents_all[:K]

    run_dir = os.path.join(save_root, f"K_{K}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "train.log")
    result_path = os.path.join(run_dir, "result.json")
    token_meta_path = os.path.join(run_dir, "new_token_meta.json")

    tokenizer = build_tokenizer_with_new_tokens()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    new_vocab_size = model.get_input_embeddings().weight.shape[0]

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    new_token_ids = list(range(old_vocab_size, new_vocab_size))
    if len(new_token_ids) != len(new_tokens):
        print(f"WARNING: expected {len(new_tokens)} new ids, got {len(new_token_ids)}")

    with open(token_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "old_vocab_size": old_vocab_size,
                "new_vocab_size": new_vocab_size,
                "num_new_tokens": len(new_tokens),
                "new_tokens": new_tokens,
                "new_token_ids": new_token_ids,
                "K": K,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    tied = tied_input_output_embeddings(model)
    print("Tied input/output embeddings:", tied)
    if tied:
        print("ET+LM collapses to ET for this model.")

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    input_emb = model.get_input_embeddings().weight
    input_emb.requires_grad = True

    if tied:
        optimizer_params = [input_emb]
    else:
        output_emb = model.get_output_embeddings().weight
        output_emb.requires_grad = True
        optimizer_params = [input_emb, output_emb]

    # Gradient mask: only update new rows
    input_mask = torch.zeros_like(input_emb, dtype=torch.bool)
    input_mask[new_token_ids, :] = True

    if not tied:
        output_mask = torch.zeros_like(output_emb, dtype=torch.bool)
        output_mask[new_token_ids, :] = True

    def grad_mask_hook_input(grad):
        masked = torch.zeros_like(grad)
        masked[input_mask] = grad[input_mask]
        return masked

    input_emb.register_hook(grad_mask_hook_input)

    if not tied:
        def grad_mask_hook_output(grad):
            masked = torch.zeros_like(grad)
            masked[output_mask] = grad[output_mask]
            return masked
        output_emb.register_hook(grad_mask_hook_output)

    optimizer = AdamW(optimizer_params, lr=lr)

    # Baseline eval once
    base_de_test = evaluate(model, tokenizer, german_test_sents)
    base_en_test = evaluate(model, tokenizer, english_test_sents)

    latest_de_test = base_de_test
    latest_en_test = base_en_test

    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("epoch,de_test_loss,en_test_loss\n")
        log_f.write(f"BASE,{base_de_test:.6f},{base_en_test:.6f}\n")
        log_f.flush()

        for epoch in range(epochs):
            random.shuffle(german_train_sents)
            model.train()

            for i in range(0, len(german_train_sents), batch_size):
                batch_texts = german_train_sents[i:i + batch_size]
                d = tokenize_batch(tokenizer, batch_texts)

                optimizer.zero_grad(set_to_none=True)
                out = model(**d)
                loss = tokenwise_loss(out.logits, d["labels"])
                loss.backward()
                optimizer.step()

            do_de = ((epoch + 1) % EVAL_DE_EVERY == 0) or (epoch == epochs - 1)
            do_en = ((epoch + 1) % EVAL_EN_EVERY == 0) or (epoch == epochs - 1)

            de_msg = "SKIP"
            en_msg = "SKIP"

            if do_de:
                latest_de_test = evaluate(model, tokenizer, german_test_sents)
                de_msg = f"{latest_de_test:.4f}"

            if do_en:
                latest_en_test = evaluate(model, tokenizer, english_test_sents)
                en_msg = f"{latest_en_test:.4f}"

            print(f"[EPOCH {epoch:03d}] de_test={de_msg} | en_test={en_msg}")
            log_f.write(f"{epoch},{latest_de_test:.6f},{latest_en_test:.6f}\n")
            log_f.flush()

    # Final eval once
    final_de_test = evaluate(model, tokenizer, german_test_sents)
    final_en_test = evaluate(model, tokenizer, english_test_sents)

    result = {
        "method": "ET+LM_FAST_DE",
        "seed": SEED,
        "model": model_name,
        "K": K,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "num_new_tokens": len(new_tokens),
        "tied_embeddings": tied,
        "eval_de_every": EVAL_DE_EVERY,
        "eval_en_every": EVAL_EN_EVERY,
        "base_de_test_loss": base_de_test,
        "base_en_test_loss": base_en_test,
        "final_de_test_loss": final_de_test,
        "final_en_test_loss": final_en_test,
        "forgetting_en_delta_loss": final_en_test - base_en_test,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    all_results.append(result)

with open(os.path.join(save_root, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("DONE")