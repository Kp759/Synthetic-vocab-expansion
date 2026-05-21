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
from peft import PromptTuningConfig, get_peft_model, TaskType

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

model_name = "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"

data_root = "/scratch/yl258/kp759/llQwen2.5-3B/data/eng_ar_vocab_expansion"
save_root = "/scratch/yl258/kp759/llama3.2-3B/results_eng_ar_prompt_tuning_fast"
os.makedirs(data_root, exist_ok=True)
os.makedirs(save_root, exist_ok=True)

arabic_dir = os.path.join(data_root, "arabic")
english_dir = os.path.join(data_root, "english")
os.makedirs(arabic_dir, exist_ok=True)
os.makedirs(english_dir, exist_ok=True)

arabic_train_path = os.path.join(arabic_dir, "train.txt")
arabic_test_path = os.path.join(arabic_dir, "test.txt")
english_test_path = os.path.join(english_dir, "test.txt")

AUTO_PREPARE_DATA = True
NUM_AR_TRAIN_SAVE = 6000
NUM_AR_TEST_SAVE = 1000
NUM_EN_TEST_SAVE = 1000
MIN_CHAR_LEN = 200

K_CONFIGS = [1000]
NUM_NEW_TOKENS = 500
MIN_WORD_FREQ = 5
MIN_WORD_LEN = 3

max_length = 256
batch_size = 4
lr = 1e-5
epochs = 10
prompt_length = 10

# Faster eval cadence
EVAL_AR_EVERY = 2   # Arabic test every 2 epochs
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
    ds_ar = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
    ar_texts = []
    for ex in ds_ar:
        txt = clean_text(ex["text"])
        if len(txt) >= MIN_CHAR_LEN:
            ar_texts.append(txt)
        if len(ar_texts) >= (NUM_AR_TRAIN_SAVE + NUM_AR_TEST_SAVE):
            break

    ds_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    en_texts = []
    for ex in ds_en:
        txt = clean_text(ex["text"])
        if len(txt) >= MIN_CHAR_LEN:
            en_texts.append(txt)
        if len(en_texts) >= NUM_EN_TEST_SAVE:
            break

    save_lines(arabic_train_path, ar_texts[:NUM_AR_TRAIN_SAVE])
    save_lines(arabic_test_path, ar_texts[NUM_AR_TRAIN_SAVE:NUM_AR_TRAIN_SAVE + NUM_AR_TEST_SAVE])
    save_lines(english_test_path, en_texts[:NUM_EN_TEST_SAVE])

if AUTO_PREPARE_DATA:
    if not (os.path.exists(arabic_train_path) and os.path.exists(arabic_test_path) and os.path.exists(english_test_path)):
        build_local_wiki_files()
    else:
        print("Local txt files already exist. Skipping download.")

base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

arabic_train_sents_all = load_lines(arabic_train_path)
arabic_test_sents = load_lines(arabic_test_path)
english_test_sents = load_lines(english_test_path)

_word_pattern = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")

def normalize_word(w: str) -> str:
    return w.strip().lower()

def get_candidate_new_tokens(texts, tokenizer, top_k=500, min_freq=5, min_len=3):
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
    arabic_train_sents_all, base_tokenizer, top_k=NUM_NEW_TOKENS, min_freq=MIN_WORD_FREQ, min_len=MIN_WORD_LEN
)
new_tokens = [x[0] for x in candidate_tokens]
print(f"Selected {len(new_tokens)} new Arabic tokens")

_ce = nn.CrossEntropyLoss(reduction="none")

def build_tokenizer_with_new_tokens():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(new_tokens)
    return tokenizer

def tokenize_batch(tokenizer, texts):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    labels[:, 0] = -100
    enc["labels"] = labels
    return enc

def tokenwise_loss(logits, labels, prompt_len):
    logits = logits[:, prompt_len:-1]
    labels = labels[:, 1:]
    loss = _ce(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    mask = labels.reshape(-1) != -100
    return loss[mask].mean()

@torch.no_grad()
def evaluate(model, tokenizer, sents, prompt_len):
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(sents), batch_size):
        d = tokenize_batch(tokenizer, sents[i:i + batch_size])
        out = model(**d)
        logits = out.logits[:, prompt_len:-1]
        labels = d["labels"][:, 1:]
        loss = _ce(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        mask = labels.reshape(-1) != -100
        tot += loss[mask].sum().item()
        cnt += mask.sum().item()
    model.train()
    return tot / max(cnt, 1)

all_results = []

for K in K_CONFIGS:
    train_sents = arabic_train_sents_all[:K]
    run_dir = os.path.join(save_root, f"K_{K}")
    os.makedirs(run_dir, exist_ok=True)

    tokenizer = build_tokenizer_with_new_tokens()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=prompt_length
    )

    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=lr)

    # Baseline eval once
    base_ar_test = evaluate(model, tokenizer, arabic_test_sents, prompt_length)
    base_en_test = evaluate(model, tokenizer, english_test_sents, prompt_length)

    latest_ar_test = base_ar_test
    latest_en_test = base_en_test

    with open(os.path.join(run_dir, "train.log"), "w", encoding="utf-8") as log_f:
        log_f.write("epoch,ar_test_loss,en_test_loss\n")
        log_f.write(f"BASE,{base_ar_test:.6f},{base_en_test:.6f}\n")

        for epoch in range(epochs):
            random.shuffle(train_sents)

            for i in range(0, len(train_sents), batch_size):
                batch = tokenize_batch(tokenizer, train_sents[i:i + batch_size])
                optimizer.zero_grad(set_to_none=True)
                out = model(**batch)
                loss = tokenwise_loss(out.logits, batch["labels"], prompt_length)
                loss.backward()
                optimizer.step()

            do_ar = ((epoch + 1) % EVAL_AR_EVERY == 0) or (epoch == epochs - 1)
            do_en = ((epoch + 1) % EVAL_EN_EVERY == 0) or (epoch == epochs - 1)

            ar_msg = "SKIP"
            en_msg = "SKIP"

            if do_ar:
                latest_ar_test = evaluate(model, tokenizer, arabic_test_sents, prompt_length)
                ar_msg = f"{latest_ar_test:.4f}"

            if do_en:
                latest_en_test = evaluate(model, tokenizer, english_test_sents, prompt_length)
                en_msg = f"{latest_en_test:.4f}"

            print(f"[EPOCH {epoch:03d}] ar_test={ar_msg} | en_test={en_msg}")
            log_f.write(f"{epoch},{latest_ar_test:.6f},{latest_en_test:.6f}\n")
            log_f.flush()

    # Final eval once
    final_ar_test = evaluate(model, tokenizer, arabic_test_sents, prompt_length)
    final_en_test = evaluate(model, tokenizer, english_test_sents, prompt_length)

    result = {
        "method": "PromptTuning_FAST",
        "seed": SEED,
        "model": model_name,
        "K": K,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "num_new_tokens": len(new_tokens),
        "prompt_length": prompt_length,
        "eval_ar_every": EVAL_AR_EVERY,
        "eval_en_every": EVAL_EN_EVERY,
        "base_ar_test_loss": base_ar_test,
        "base_en_test_loss": base_en_test,
        "final_ar_test_loss": final_ar_test,
        "final_en_test_loss": final_en_test,
        "forgetting_en_delta_loss": final_en_test - base_en_test,
    }

    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    all_results.append(result)

with open(os.path.join(save_root, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("DONE")