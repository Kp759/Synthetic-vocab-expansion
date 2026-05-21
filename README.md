# Synthetic Vocabulary Expansion

Official code release for the ICML 2026 paper:

**Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping**  
Kaustubh Pethkar, Ziyang Xiong, Zuofeng Shang, Yingcong Li

This repository contains experiments for studying **post-training vocabulary expansion** in large language models. The central question is:

> Can a pretrained language model learn newly introduced tokens from limited data while preserving its original behavior?

The paper models autoregressive language generation as a **Markov process over tokens**, where model memory can be interpreted through token-to-token transition dynamics. Under this view, adding new tokens corresponds to expanding the state space. The proposed **token-to-dictionary mapping** strategy learns representations for new tokens by reusing the existing embedding dictionary, and is implemented in practice through **embedding tuning (ET)**.

---

## Overview

This repository focuses on the synthetic vocabulary expansion and cross-lingual vocabulary expansion experiments from the paper. The synthetic benchmark introduces artificial words that have no prior semantic meaning in the pretrained model and evaluates whether different adaptation methods can learn these new tokens without catastrophic forgetting.

The main methods compared are:

| Method | Description |
|---|---|
| **Embedding Tuning (ET)** | Updates only the newly introduced token embeddings, and when applicable the corresponding LM-head rows. |
| **Full Fine-Tuning (FFT)** | Updates all model parameters on the synthetic vocabulary task. |
| **LoRA** | Trains low-rank adapter weights while keeping the backbone mostly frozen. |
| **Prompt Tuning (PT)** | Optimizes continuous soft prompt embeddings prepended to the input. |

The key result is that **embedding tuning learns the new vocabulary competitively while inducing near-zero or zero forgetting on held-out original-language data**, whereas full fine-tuning, LoRA, and prompt tuning can cause measurable forgetting.

---

## Paper Motivation

Large language models often need to incorporate new words after pretraining, including new entities, domain-specific terminology, and newly coined expressions. Standard fine-tuning can adapt the model to new data, but it may also overwrite previous knowledge.

This project studies a controlled version of that problem using synthetic tokens. Since the fake words have no prior meaning, the benchmark provides a clean way to measure:

1. how efficiently a model learns new vocabulary;
2. how much the model forgets after adaptation;
3. whether embedding-only updates preserve original model behavior better than broader parameter updates.

---

## Synthetic Vocabulary Task

The synthetic vocabulary experiment constructs a benchmark with **100 newly introduced synthetic words/tokens**. These tokens are inserted into otherwise natural sentences so that the surrounding context remains realistic.

Example:

```text
Real:      The fiber in broccoli helps regulate blood sugar levels.
Synthetic: The fiber in glor helps regulate blood zorp levels.
```

Here, `glor` and `zorp` are artificial tokens introduced during adaptation. The model is trained to use these new tokens correctly while preserving its behavior on general text.

### Evaluation

The experiment measures two quantities:

| Metric | Meaning |
|---|---|
| **Synthetic test loss** | Token-level cross-entropy loss on held-out synthetic sentences. Lower is better. |
| **Forgetting** | Increase in WikiText loss after adaptation compared with the base model. Lower is better. |

Formally:

```text
forgetting = WikiTextLoss(after adaptation) - WikiTextLoss(base model)
```

A forgetting value close to `0.00` indicates that the method preserved the original model behavior on the held-out original-language evaluation set.

---

## Repository Structure

```text
Synthetic-vocab-expansion/
├── Llama1B/                         # Synthetic vocabulary experiments on Llama 3.2 1B
│   ├── WVembed100.py                # Embedding tuning
│   ├── Wvfinetune100.py             # Full fine-tuning
│   ├── WvLoRA.py                    # LoRA adaptation
│   ├── WvSoft100.py                 # Soft prompt / prompt tuning
│   ├── WvLoss_on_both_all_methods.py# Evaluation on synthetic data and WikiText
│   └── Sentence/                    # Synthetic sentence data
│
├── Llama8B/                         # Synthetic vocabulary experiments on Llama 8B
├── phiModel/                        # Synthetic vocabulary experiments on Phi-3.5 Mini Instruct
├── Qwen model/                      # Synthetic vocabulary experiments on Qwen models
│
├── Qwen2.5-3B Eng-spanish/          # English-to-Spanish vocabulary expansion
├── Qwen2.5-3B-Eng-German/           # English-to-German vocabulary expansion
├── Qwen2.5-3B- Eng-Arabic/          # English-to-Arabic vocabulary expansion with Qwen2.5-3B
├── Llama3.2-3B Eng-Arabic/          # English-to-Arabic vocabulary expansion with Llama3.2-3B
│
├── embed_map-main/                  # Data/utilities for embedding-map experiments
│   ├── data/
│   └── utils/
│
└── README.md
```

---

## Installation

Create a clean Python environment:

```bash
git clone https://github.com/Kp759/Synthetic-vocab-expansion.git
cd Synthetic-vocab-expansion

conda create -n vocab-expansion python=3.10 -y
conda activate vocab-expansion
```

Install dependencies:

```bash
pip install torch transformers datasets accelerate peft sentencepiece protobuf
pip install numpy pandas tqdm scikit-learn matplotlib
```

For gated Hugging Face models such as Llama, log in before running experiments:

```bash
huggingface-cli login
```

Depending on your CUDA version and cluster setup, you may need to install the PyTorch build that matches your GPU driver. See the official PyTorch installation selector if the default `pip install torch` does not match your system.

---

## Running the Synthetic Vocabulary Experiments

The following example runs the Llama 3.2 1B synthetic vocabulary experiments.

```bash
cd Llama1B
```

### 1. Embedding Tuning

```bash
python WVembed100.py
```

This trains only the parameters associated with the newly introduced synthetic tokens.

### 2. Full Fine-Tuning

```bash
python Wvfinetune100.py
```

This updates all model parameters on the synthetic vocabulary training set.

### 3. LoRA

```bash
python WvLoRA.py
```

The LoRA baseline freezes the backbone model and trains low-rank adapter parameters.

### 4. Prompt Tuning

```bash
python WvSoft100.py
```

The prompt-tuning baseline optimizes continuous prompt embeddings while keeping the backbone frozen.

### 5. Evaluate Adaptation and Forgetting

```bash
python WvLoss_on_both_all_methods.py
```

This evaluates:

- synthetic vocabulary test loss;
- base WikiText loss;
- post-adaptation WikiText loss;
- forgetting, defined as the WikiText loss increase after tuning.

---

## Expected Results

The paper reports that embedding tuning provides the best trade-off between adaptation and retention. With `N = 1000` synthetic training sentences, embedding tuning obtains competitive synthetic test loss and `0.00` forgetting across the evaluated models.

| Model | FFT Loss | LoRA Loss | PT Loss | ET Loss | Base Wiki Loss | FFT Forget | LoRA Forget | PT Forget | ET Forget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phi-3.5 Mini Instruct | 3.28 | 6.01 | 1.87 | 2.75 | 2.43 | +9.10 | +8.80 | +0.39 | 0.00 |
| Llama3.2-1B | 4.57 | 3.55 | 6.37 | 3.50 | 2.86 | +0.70 | +2.34 | +1.56 | 0.00 |
| Llama3.2-3B | 4.54 | 4.08 | 5.76 | 3.47 | 2.62 | +0.51 | +1.35 | +0.62 | 0.00 |
| Llama3.1-8B | 4.40 | 7.63 | 5.86 | 3.79 | 2.42 | +1.24 | +8.36 | +0.69 | 0.00 |
| Qwen2.5-1.5B | 4.81 | 3.72 | 4.03 | 3.46 | 2.73 | +11.59 | +1.60 | +2.19 | 0.00 |
| Qwen2.5-3B | 5.14 | 3.45 | 4.71 | 3.39 | 2.59 | +4.71 | +1.13 | +0.06 | 0.00 |
| Qwen2.5-7B | 5.22 | 3.86 | 4.62 | 3.37 | 2.43 | +7.37 | +8.60 | +0.01 | 0.00 |

---

## Cross-Lingual Vocabulary Expansion

The repository also includes cross-lingual vocabulary expansion experiments for:

- English-to-Spanish;
- English-to-German;
- English-to-Arabic.

These experiments evaluate whether the same embedding-tuning principle extends beyond synthetic fake words to real target-language vocabulary. The paper reports that ET obtains strong target-language adaptation while maintaining near-zero English forgetting.

Example Qwen2.5-3B results:

| Target Language | FFT Loss | LoRA Loss | PT Loss | ET Loss | Base English Loss | FFT Forget | LoRA Forget | PT Forget | ET Forget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spanish | 5.56 | 3.17 | 3.45 | 2.30 | 2.01 | +9.83 | +0.66 | -0.03 | -0.04 |
| German | 6.80 | 3.29 | 3.63 | 3.27 | 2.01 | +9.81 | +0.49 | -0.03 | -0.08 |
| Arabic | 7.95 | 3.95 | 4.34 | 2.82 | 2.01 | +11.95 | +0.46 | +0.06 | 0.00 |

---

## Method Details

### Embedding Tuning

Embedding tuning extends the tokenizer with new synthetic tokens and resizes the embedding matrix accordingly. During training, only the newly added token rows are updated. When the model uses separate input embeddings and output LM-head weights, the corresponding new output rows may also be updated.

This gives a parameter count proportional to:

```text
number_of_new_tokens × hidden_dimension
```

or, when both input embedding and LM head are updated:

```text
2 × number_of_new_tokens × hidden_dimension
```

This is substantially smaller than full fine-tuning and often smaller than or comparable to LoRA, depending on model size and LoRA rank.

### LoRA

LoRA is used as a parameter-efficient fine-tuning baseline. In the paper setup, adapters are inserted into attention projection layers with:

```text
rank r = 8
alpha = 32
dropout = 0.05
bias = none
```

### Prompt Tuning

Prompt tuning optimizes a fixed number of continuous soft prompt embeddings prepended to the model input while leaving the backbone frozen.

---

## Reproducibility Notes

Large model experiments can be sensitive to hardware, CUDA version, model access, and tokenizer behavior. For reproducible runs, record:

- model checkpoint name;
- random seed;
- number of synthetic training sentences;
- number of synthetic tokens;
- max sequence length;
- learning rate;
- batch size;
- number of epochs;
- whether input embeddings only or both input embeddings and LM-head rows are updated;
- WikiText split used for forgetting evaluation.

Recommended output format:

```text
outputs/
├── METHOD_NAME/
│   ├── config.json
│   ├── train_log.txt
│   ├── synthetic_test_loss.json
│   ├── wikitext_before_after.json
│   └── checkpoint/
```

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{pethkar2026memory,
  title     = {Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping},
  author    = {Pethkar, Kaustubh and Xiong, Ziyang and Shang, Zuofeng and Li, Yingcong},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```
---

## Acknowledgements

This repository accompanies the ICML 2026 paper **Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping**.
