import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import random

def get_token_id(tokenizer, token_str: str) -> int:
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id != tokenizer.unk_token_id and token_id is not None:
            return token_id
    encoded = tokenizer.encode(token_str, add_special_tokens=False)
    if len(encoded) == 1:
        return encoded[0]
    elif len(encoded) > 1:
        return encoded[0]
    else:
        raise ValueError(f"Token '{token_str}' could not be encoded")


def extract_token_embedding(model, tokenizer, token_str: str) -> Tuple[torch.Tensor, str]:
    """
    Extract token embedding and return both the embedding and source information.
    Returns: (embedding_tensor, source_description)
    """
    token_id = get_token_id(tokenizer, token_str)
    
    # Check if there's a separate trainable parameter for this token (special_token mode)
    # In special_token finetune mode, the trained embedding is stored in a separate parameter
    param_name = f'special_token_{token_id}_embedding'
    
    # Check in model attributes
    if hasattr(model, param_name):
        param = getattr(model, param_name)
        with torch.no_grad():
            return param.clone().cpu().float(), f"special_token parameter ({param_name})"
    
    # Also check in state_dict (in case it was saved but not as an attribute)
    state_dict = model.state_dict()
    if param_name in state_dict:
        with torch.no_grad():
            return state_dict[param_name].clone().cpu().float(), f"state_dict ({param_name})"
    
    # Otherwise, extract from the standard embedding layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and "embed_tokens" in name:
            with torch.no_grad():
                return module.weight[token_id].clone().cpu().float(), f"embed_tokens.weight[{token_id}]"
    raise ValueError("Could not find embed_tokens layer")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = F.normalize(a, p=2, dim=0)
    b_norm = F.normalize(b, p=2, dim=0)
    return torch.dot(a, b).item()


def pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient between two vectors."""
    a_mean = a.mean()
    b_mean = b.mean()
    a_centered = a - a_mean
    b_centered = b - b_mean
    numerator = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())
    if denominator == 0:
        return 0.0
    return (numerator / denominator).item()


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """
    KL(p || q) for discrete distributions.
    p, q are assumed to be probability vectors (non-negative); we renormalize and clamp for stability.
    """
    p = p.float()
    q = q.float()
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return (p * (p.log() - q.log())).sum().item()


def extract_token_output_embedding(model, tokenizer, token_str: str) -> Tuple[torch.Tensor, str]:
    """
    Extract output embedding (hidden state) for a token by running forward pass.
    Each token is processed independently to avoid context effects.
    Returns: (output_embedding_tensor, source_description)
    """
    token_id = get_token_id(tokenizer, token_str)
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        # Create input with just this single token
        input_ids = torch.tensor([[token_id]], device=device)
        
        # Forward pass to get hidden states
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        
        # Get the last hidden state (output of the final layer)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
        else:
            hidden_states = outputs.last_hidden_state
        
        # Extract the output embedding for this token (at position 0)
        output_embedding = hidden_states[0, 0, :].clone().cpu().float()
        
        return output_embedding, f"output_embedding (hidden_state[-1])"


def extract_prob_distribution_after_token(model, tokenizer, messages: List[Dict], ans: int, target_token: str) -> Tuple[torch.Tensor, str]:
    device = next(model.parameters()).device
    model.eval()
    
    # Format messages using chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        formatted_text = ""
        for msg in messages:
            if msg['role'] == 'system':
                formatted_text += f"System: {msg['content']}\n"
            elif msg['role'] == 'user':
                formatted_text += f"User: {msg['content']}\n"
    
    # Tokenize the formatted text
    encoded = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Display tokenized input sequence
    # print(f"\n{'='*80}")
    # print(f"Input sequence (tokenized):")
    # print(f"Formatted text: {formatted_text}")
    # print(f"\nToken breakdown (length: {input_ids.shape[1]}):")
    # print(f"{'='*80}\n")

    # Use model.generate to get full output sequence
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Explicitly pass attention mask
            max_new_tokens=100,  # Generate up to 100 new tokens
            do_sample=False,  # Use greedy decoding
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        
        # Decode the full sequence
        full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        # print(f"Full output (input + generated):")
        # print(f"  {full_output}")
        
        # Find the position of 'assistant<|end_header_id|>\n\n' in the generated sequence
        search_pattern = 'assistant<|end_header_id|>\n\n'
        double_newline_pos = full_output.find(search_pattern)

        if double_newline_pos == -1:
            search_pattern = 'assistant='
            double_newline_pos = full_output.find(search_pattern)
        
            
        if double_newline_pos == -1:
            search_pattern = 'assistant<|start_header_id|>'
            double_newline_pos = full_output.find(search_pattern)
        
        if double_newline_pos == -1:
            search_pattern = 'assistant<|end_header_id|>'
            double_newline_pos = full_output.find(search_pattern)

        if double_newline_pos == -1:
            search_pattern = 'assistant'
            double_newline_pos = full_output.find(search_pattern)

        if double_newline_pos == -1:
            raise ValueError(f"'{search_pattern}' not found in the generated sequence")
        
        # Find the character position after the search pattern
        char_pos_after_nn = double_newline_pos + len(search_pattern)
        
        
        # print(f"\n{'='*80}")
        # print(f"Found '\\n\\n' at character position {double_newline_pos}")
        # print(f"Looking for token after position {char_pos_after_nn}")
        print(f"Text after '\\n\\n': '{full_output[char_pos_after_nn:char_pos_after_nn+20]}...'")
        # print(f"{'='*80}\n")
        # breakpoint()
        
        token_pos_after_nn = None
        
        for i in range(generated_ids.shape[1]):
            # Decode up to this token
            tokens_up_to_i = generated_ids[0, :i+1]
            text_up_to_i = tokenizer.decode(tokens_up_to_i, skip_special_tokens=False)
            
            if len(text_up_to_i) > char_pos_after_nn:
                # This token contains or is after the target character position
                token_pos_after_nn = i
                break
        
        if token_pos_after_nn is None:
            raise ValueError(f"Could not find token position for character position {char_pos_after_nn}")
        
        # print(f"Token position after '\\n\\n': {token_pos_after_nn}")
        token_id_after_nn = generated_ids[0, token_pos_after_nn].item()
        token_str_after_nn = tokenizer.decode([token_id_after_nn])
        # print(f"Token at position {token_pos_after_nn}: ID {token_id_after_nn} -> '{token_str_after_nn}'")
        
        # Now run forward pass on the sequence up to (but not including) the token after '\n\n'
        # to get the logits for predicting that token
        sequence_up_to_token = generated_ids[0, :token_pos_after_nn]
        sequence_up_to_token = sequence_up_to_token.unsqueeze(0).to(device)
        
        # Forward pass to get logits at the position before the target token
        outputs = model(input_ids=sequence_up_to_token)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Get logits for predicting the token after '\n\n'
        # logits[0, -1, :] gives logits for predicting the next token after the last token in sequence_up_to_token
        target_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
        
        # Apply softmax to get probability distribution
        prob_dist = F.softmax(target_token_logits, dim=0).cpu().float()
        
        return prob_dist, f"prob_distribution for token after '\\n\\n' (position {token_pos_after_nn}, token: '{token_str_after_nn}')"


def extract_token_from_messages(model, tokenizer, messages: List[Dict], target_token: str, use_output_embedding: bool = True) -> Tuple[torch.Tensor, str]:
    """
    Extract embedding for a specific token from messages format.
    Uses chat template to format messages, then finds the token position and extracts its embedding.
    
    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        messages: List of message dicts with 'role' and 'content'
        target_token: The token string to extract (e.g., "<|reserved_special_token_0|>", "=", "*")
        use_output_embedding: If True, extract output embedding (hidden state), else input embedding
    
    Returns:
        (embedding_tensor, source_description)
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Format messages using chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        # Fallback: manually format
        formatted_text = ""
        for msg in messages:
            if msg['role'] == 'system':
                formatted_text += f"System: {msg['content']}\n"
            elif msg['role'] == 'user':
                formatted_text += f"User: {msg['content']}\n"
    
    # Tokenize the formatted text
    encoded = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded['input_ids'].to(device)
    
    # Find the position of the target token in the tokenized sequence
    target_token_id = get_token_id(tokenizer, target_token)
    token_positions = (input_ids[0] == target_token_id).nonzero(as_tuple=True)[0]
    
    if len(token_positions) == 0:
        raise ValueError(f"Token '{target_token}' (id={target_token_id}) not found in the tokenized messages")
    
    # Use the first occurrence
    token_pos = token_positions[0].item()
    
    if use_output_embedding:
        # Forward pass to get hidden states (output embeddings)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            
            # Get the last hidden state (output of the final transformer layer)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
            else:
                hidden_states = outputs.last_hidden_state
            
            # Extract output embedding at the token position (this is the contextualized embedding)
            embedding = hidden_states[0, token_pos, :].clone().cpu().float()
            return embedding, f"output_embedding (hidden_state[-1]) from messages (position {token_pos})"
    else:
        # Extract from input embedding layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding) and "embed_tokens" in name:
                with torch.no_grad():
                    embedding = module.weight[target_token_id].clone().cpu().float()
                    return embedding, f"input_embedding from messages (token_id {target_token_id})"
        raise ValueError("Could not find embed_tokens layer")


def load_model_from_checkpoint(checkpoint_path: str, base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """
    Load model and tokenizer from checkpoint path or HuggingFace model name.
    If checkpoint_path is a local path that exists, load from there.
    Otherwise, treat it as a HuggingFace model name.
    """
    # Check if it's a local path
    is_local_path = os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path)
    
    if is_local_path:
        tokenizer_path = checkpoint_path if os.path.exists(os.path.join(checkpoint_path, "tokenizer.json")) else base_model_name
    else:
        # Treat as HuggingFace model name
        tokenizer_path = checkpoint_path
        checkpoint_path = checkpoint_path  # Use the model name directly
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def compare_two_checkpoints_weights(
    checkpoint1_path: str,
    checkpoint2_path: str,
    tokens: List[str],
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    use_output_embeddings: bool = False
) -> Dict:
    """Directly compare embedding weights (input or output) from two checkpoints."""
    embed_type = "output" if use_output_embeddings else "input"
    results = {
        "checkpoint1": os.path.basename(checkpoint1_path.rstrip('/')),
        "checkpoint2": os.path.basename(checkpoint2_path.rstrip('/')),
        "embedding_type": embed_type,
        "comparisons": {}
    }
    
    print(f"\n{'='*80}")
    print(f"Comparing {embed_type} embeddings: {results['checkpoint1']} vs {results['checkpoint2']}")
    print(f"{'='*80}\n")
    
    try:
        model1, tokenizer1 = load_model_from_checkpoint(checkpoint1_path, base_model_name)
        model2, tokenizer2 = load_model_from_checkpoint(checkpoint2_path, base_model_name)
        
        extract_func = extract_token_output_embedding if use_output_embeddings else extract_token_embedding
        
        for token_str in tokens:
            token_id1 = get_token_id(tokenizer1, token_str)
            token_id2 = get_token_id(tokenizer2, token_str)
            
            if token_id1 != token_id2:
                print(f"Warning: Token '{token_str}' has different IDs: {token_id1} vs {token_id2}")
            
            emb1, source1 = extract_func(model1, tokenizer1, token_str)
            emb2, source2 = extract_func(model2, tokenizer2, token_str)
            
            # Direct weight comparison
            diff = (emb1 - emb2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            std_diff = diff.std().item()
            cosine_sim = cosine_similarity(emb1, emb2)
            
            # Check if they're exactly the same
            are_identical = torch.allclose(emb1, emb2, atol=1e-6)
            are_very_similar = torch.allclose(emb1, emb2, atol=1e-3)
            
            # Get first few values for inspection
            first_5_1 = emb1[:5].tolist()
            first_5_2 = emb2[:5].tolist()
            first_5_diff = (emb1[:5] - emb2[:5]).abs().tolist()
            
            comparison = {
                "token_id": token_id1,
                "checkpoint1_source": source1,
                "checkpoint2_source": source2,
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "std_difference": std_diff,
                "cosine_similarity": cosine_sim,
                "are_identical_atol_1e6": are_identical,
                "are_very_similar_atol_1e3": are_very_similar,
                "checkpoint1_first_5": first_5_1,
                "checkpoint2_first_5": first_5_2,
                "first_5_difference": first_5_diff,
                "checkpoint1_mean": float(emb1.mean()),
                "checkpoint1_std": float(emb1.std()),
                "checkpoint2_mean": float(emb2.mean()),
                "checkpoint2_std": float(emb2.std()),
            }
            
            results["comparisons"][token_str] = comparison
            
            print(f"\nToken: {token_str} (id={token_id1})")
            print(f"  Checkpoint1 source: {source1}")
            print(f"  Checkpoint2 source: {source2}")
            print(f"  Max difference: {max_diff:.8f}")
            print(f"  Mean difference: {mean_diff:.8f}")
            print(f"  Std difference: {std_diff:.8f}")
            print(f"  Cosine similarity: {cosine_sim:.8f}")
            print(f"  Are identical (atol=1e-6): {are_identical}")
            print(f"  Are very similar (atol=1e-3): {are_very_similar}")
            if are_identical:
                print(f"  ⚠️  WARNING: Embeddings are IDENTICAL!")
            elif are_very_similar:
                print(f"  ⚠️  WARNING: Embeddings are VERY SIMILAR (difference < 1e-3)!")
            print(f"  Checkpoint1 first 5: {[f'{x:.6f}' for x in first_5_1]}")
            print(f"  Checkpoint2 first 5: {[f'{x:.6f}' for x in first_5_2]}")
            print(f"  First 5 diff: {[f'{x:.8f}' for x in first_5_diff]}")
        
        del model1, model2, tokenizer1, tokenizer2
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        return {"error": str(e)}
    
    return results


def compare_token_embeddings(
    checkpoint_path: str,
    tokens: List[str],
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    use_output_embeddings: bool = False
) -> Dict:
    """Compare token embeddings (input or output) within a single checkpoint."""
    embed_type = "output" if use_output_embeddings else "input"
    results = {
        "checkpoint": os.path.basename(checkpoint_path.rstrip('/')),
        "embedding_type": embed_type,
        "similarities": {}
    }
    
    try:
        model, tokenizer = load_model_from_checkpoint(checkpoint_path, base_model_name)
        embeddings = {}
        extract_func = extract_token_output_embedding if use_output_embeddings else extract_token_embedding
        
        for token_str in tokens:
            try:
                embedding, source = extract_func(model, tokenizer, token_str)
                embeddings[token_str] = embedding
                print(f"  Extracted '{token_str}' ({embed_type} embedding) from: {source}")
            except Exception as e:
                print(f"  Failed to extract '{token_str}': {e}")
                embeddings[token_str] = None
        
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        return {"error": str(e)}
    
    # Compute similarities between token pairs
    for i, token1 in enumerate(tokens):
        for token2 in tokens[i+1:]:
            if embeddings.get(token1) is not None and embeddings.get(token2) is not None:
                key = f"{token1} vs {token2}"
                results["similarities"][key] = cosine_similarity(embeddings[token1], embeddings[token2])
    
    return results


def compare_tokens_from_messages(
    checkpoint_path: str,
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    use_output_embeddings: bool = True
) -> Dict:
    """
    Extract tokens from hardcoded messages and compare their similarities.
    Messages are hardcoded as specified:
    1. messages1: extracts <|reserved_special_token_0|> and =
    2. messages2: extracts * and =
    Then compares similarity between all three tokens.
    """
    embed_type = "output" if use_output_embeddings else "input"
    
    # Hardcoded messages
    messages1 = [
        {"role": "system", "content": "Please answer with only the number, no other text."},
        {"role": "user", "content": "<|reserved_special_token_0|>(38,11)="}
    ]
    
    messages2 = [
        {"role": "system", "content": "Please answer with only the number, no other text."},
        {"role": "user", "content": "38*11="}
    ]
    
    
    results = {
        "checkpoint": os.path.basename(checkpoint_path.rstrip('/')),
        "embedding_type": embed_type,
        "messages1": messages1,
        "messages2": messages2,
        "embeddings": {},
        "similarities": {}
    }
    
    print(f"\n{'='*80}")
    print(f"Extracting tokens from messages format")
    print(f"Embedding type: {embed_type} embeddings (output embeddings from hidden states)")
    print(f"Messages1: {messages1}")
    print(f"Messages2: {messages2}")
    print(f"{'='*80}\n")
    print(f"use_output_embeddings: {use_output_embeddings}")

    try:
        model, tokenizer = load_model_from_checkpoint(checkpoint_path, base_model_name)
        embeddings = {}
        
        # Extract <|reserved_special_token_0|> from messages1
        try:
            emb, source = extract_token_from_messages(model, tokenizer, messages1, "<|reserved_special_token_0|>", use_output_embeddings)
            embeddings["<|reserved_special_token_0|>"] = emb
            results["embeddings"]["<|reserved_special_token_0|>"] = {
                "source": source
            }
            print(f"  Extracted '<|reserved_special_token_0|>' from messages1: {source}")
        except Exception as e:
            print(f"  Failed to extract '<|reserved_special_token_0|>': {e}")
            embeddings["<|reserved_special_token_0|>"] = None
        
        # Extract * from messages2
        try:
            emb, source = extract_token_from_messages(model, tokenizer, messages2, "*", use_output_embeddings)
            embeddings["*"] = emb
            results["embeddings"]["*"] = {
                "source": source
            }
            print(f"  Extracted '*' from messages2: {source}")
        except Exception as e:
            print(f"  Failed to extract '*': {e}")
            embeddings["*"] = None
        
        # Extract = from messages2
        try:
            emb, source = extract_token_from_messages(model, tokenizer, messages2, "=", use_output_embeddings)
            embeddings["="] = emb
            results["embeddings"]["="] = {
                "source": source
            }
            print(f"  Extracted '=' from messages2: {source}")
        except Exception as e:
            print(f"  Failed to extract '=' from messages2: {e}")
            embeddings["="] = None
        
        # Compute similarities: * vs = and * vs <|reserved_special_token_0|>
        print(f"\n{'='*80}")
        print(f"Computing similarities between tokens")
        print(f"{'='*80}\n")
        
        # * vs =
        if embeddings.get("*") is not None and embeddings.get("=") is not None:
            key = "* vs ="
            sim = cosine_similarity(embeddings["*"], embeddings["="])
            results["similarities"][key] = sim
            print(f"  {key}: {sim:.8f}")
        else:
            print(f"  Warning: Cannot compute '* vs =' similarity (missing embeddings)")
        
        # * vs <|reserved_special_token_0|>
        if embeddings.get("*") is not None and embeddings.get("<|reserved_special_token_0|>") is not None:
            key = "* vs <|reserved_special_token_0|>"
            sim = cosine_similarity(embeddings["*"], embeddings["<|reserved_special_token_0|>"])
            results["similarities"][key] = sim
            print(f"  {key}: {sim:.8f}")
        else:
            print(f"  Warning: Cannot compute '* vs <|reserved_special_token_0|>' similarity (missing embeddings)")
        
        # <|reserved_special_token_0|> vs =
        if embeddings.get("<|reserved_special_token_0|>") is not None and embeddings.get("=") is not None:
            key = "<|reserved_special_token_0|> vs ="
            sim = cosine_similarity(embeddings["<|reserved_special_token_0|>"], embeddings["="])
            results["similarities"][key] = sim
            print(f"  {key}: {sim:.8f}")
        else:
            print(f"  Warning: Cannot compute '<|reserved_special_token_0|> vs =' similarity (missing embeddings)")
        
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        return {"error": str(e)}
    
    return results


def compare_prob_distributions_after_equals(
    checkpoint_path: str,
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
) -> Dict:
    messages1 = [
        {"role": "system", "content": "Please answer with only the number, no other text."},
        {"role": "user", "content": "<|reserved_special_token_0|>(38,11)="}
    ]
    
    messages2 = [
        {"role": "system", "content": "Please answer with only the number, no other text."},
        {"role": "user", "content": "38*11="}
    ]
    
    results = {
        "checkpoint": os.path.basename(checkpoint_path.rstrip('/')),
        "messages1": messages1,
        "messages2": messages2,
        "correlation": None,
        "cosine_similarity": None
    }
    
    print(f"\n{'='*80}")
    print(f"Comparing probability distributions after '=' token")
    print(f"Messages1: {messages1}")
    print(f"Messages2: {messages2}")
    print(f"{'='*80}\n")
    
    try:
        model, tokenizer = load_model_from_checkpoint(checkpoint_path, base_model_name)
        
        # Extract probability distributions after '=' from both messages
        try:
            prob_dist1, source1 = extract_prob_distribution_after_token(model, tokenizer, messages1, "=")
            print(f"  Extracted prob distribution from messages1: {source1}")
        except Exception as e:
            print(f"  Failed to extract prob distribution from messages1: {e}")
            return {"error": f"Failed to extract from messages1: {str(e)}"}
        
        try:
            prob_dist2, source2 = extract_prob_distribution_after_token(model, tokenizer, messages2, "=")
            print(f"  Extracted prob distribution from messages2: {source2}")
        except Exception as e:
            print(f"  Failed to extract prob distribution from messages2: {e}")
            return {"error": f"Failed to extract from messages2: {str(e)}"}
        
        # Compute correlation and cosine similarity
        correlation = pearson_correlation(prob_dist1, prob_dist2)
        cosine_sim = cosine_similarity(prob_dist1, prob_dist2)
        
        # Debug: check differences
        results["correlation"] = correlation
        results["cosine_similarity"] = cosine_sim
        

        print(f"\n{'='*80}")
        print(f"Pearson correlation between probability distributions: {correlation:.8f}")
        print(f"Cosine similarity between probability distributions: {cosine_sim:.8f}")
        print(f"{'='*80}\n")
        
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        return {"error": str(e)}
    
    return results


def compare_two_checkpoints_prob_distributions_after_equals(
    checkpoint1_path: str,
    checkpoint2_path: str,
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    num_samples: int = 40
) -> Dict:
    """
    Compare probability distributions after '=' token for two checkpoints.
    Randomly samples num_samples pairs of numbers and computes average metrics.
    """
    results = {
        "checkpoint1": os.path.basename(checkpoint1_path.rstrip('/')),
        "checkpoint2": os.path.basename(checkpoint2_path.rstrip('/')),
        "num_samples": num_samples,
        "correlation_1": None,
        "cosine_similarity_1": None,
        "kl_divergence_p_to_q_1": None,
        "individual_results": []
    }
    
    try:
        model1, tokenizer1 = load_model_from_checkpoint(checkpoint1_path, base_model_name)
        model2, tokenizer2 = load_model_from_checkpoint(checkpoint2_path, base_model_name)
        
        correlations = []
        cosine_sims = []
        kl_divs = []
        
        print(f"\n{'='*80}")
        print(f"Comparing {num_samples} random samples")
        print(f"{'='*80}\n")
        
        # Generate random pairs of numbers
        random.seed(42)  # For reproducibility
        number_pairs = []
        for _ in range(num_samples):
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            number_pairs.append((a, b))
        
        successful_samples = 0
        
        for idx, (a, b) in enumerate(number_pairs, 1):
            messages1 = [
                {"role": "system", "content": "Please answer with only the number, no other text."},
                {"role": "user", "content": f"<|reserved_special_token_0|>({a},{b})="}
            ]
            
            messages2 = [
                {"role": "system", "content": "Please answer with only the number, no other text."},
                {"role": "user", "content": f"{a}*{b}="}
            ]
            
            ans = a*b
            
            try:
                # Extract probability distributions after '=' from both messages
                prob_dist1_1, source1 = extract_prob_distribution_after_token(model1, tokenizer1, messages1, ans, "=")
                prob_dist1_2, source2 = extract_prob_distribution_after_token(model2, tokenizer2, messages2, ans,"=")
                
                # Compute metrics
                correlation = pearson_correlation(prob_dist1_1, prob_dist1_2)
                cosine_sim = cosine_similarity(prob_dist1_1, prob_dist1_2)
                kl_pq = kl_divergence(prob_dist1_1, prob_dist1_2)
                
                correlations.append(correlation)
                cosine_sims.append(cosine_sim)
                kl_divs.append(kl_pq)
                successful_samples += 1
                
                results["individual_results"].append({
                    "numbers": (a, b),
                    "correlation": correlation,
                    "cosine_similarity": cosine_sim,
                    "kl_divergence": kl_pq
                })
                
                if idx % 5 == 0:
                    print(f"  Processed {idx}/{num_samples} samples...")
                    
            except Exception as e:
                print(f"  Warning: Failed sample {idx} ({a},{b}): {e}")
                continue
        
        if successful_samples == 0:
            return {"error": "All samples failed"}
        
        # Compute averages
        avg_correlation = sum(correlations) / len(correlations)
        avg_cosine_sim = sum(cosine_sims) / len(cosine_sims)
        avg_kl_div = sum(kl_divs) / len(kl_divs)
        
        results["correlation_1"] = avg_correlation
        results["cosine_similarity_1"] = avg_cosine_sim
        results["kl_divergence_p_to_q_1"] = avg_kl_div
        results["successful_samples"] = successful_samples
        
        print(f"\n{'='*80}")
        print(f"Average metrics over {successful_samples} successful samples:")
        print(f"  Average Pearson correlation: {avg_correlation:.8f}")
        print(f"  Average Cosine similarity: {avg_cosine_sim:.8f}")
        print(f"  Average KL(p||q): {avg_kl_div:.8f}")
        print(f"{'='*80}\n")
        
        del model1, model2, tokenizer1, tokenizer2
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        return {"error": str(e)}
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare token embeddings in checkpoint(s)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory (single checkpoint mode)")
    parser.add_argument("--checkpoint1", type=str, default=None, help="First checkpoint path (comparison mode)")
    parser.add_argument("--checkpoint2", type=str, default=None, help="Second checkpoint path (comparison mode)")
    parser.add_argument("--tokens", type=str, nargs="+", default=["<|reserved_special_token_0|>", "=", "*"],
                       help="Token strings to compare")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model name")
    parser.add_argument("--output", type=str, default="embed_compare/_100_out_distribution.json", help="Output JSON file to save results")
    parser.add_argument("--use_output_embeddings", action="store_true",
                       help="Compare output embeddings (hidden states) instead of input embeddings (embedding layer weights)")
    parser.add_argument("--from_messages", action="store_true",
                       help="Extract tokens from hardcoded messages format and compare similarities")
    parser.add_argument("--compare_prob_dist", action="store_true",
                       help="Compare probability distributions after '=' token in two hardcoded messages")
    
    args = parser.parse_args()

    kl_divergences = []
    
    for checkpoint in ["meta-llama/Llama-3.2-3B-Instruct", "outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-10","outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-20","outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-30","outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-40","outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-50", "outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-60", "outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-70", "outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-80", "outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-90", "outputs/full_ft_reserved_assistant_only_3B_1000_2e-5/checkpoint-100"]:
        args.checkpoint1 = checkpoint
        args.checkpoint2 = "meta-llama/Llama-3.2-3B-Instruct"
        print(f"Comparing {args.checkpoint1} and {args.checkpoint2}")
        results = compare_two_checkpoints_prob_distributions_after_equals(
            args.checkpoint1, args.checkpoint2, args.base_model
        )
        kl_divergences.append(results["kl_divergence_p_to_q_1"])

        # if args.output:
        #     with open(args.output, 'w') as f:
        #         json.dump(results, f, indent=2)
        #     print(f"\nResults saved to: {args.output}")
        # else:
        #     print(json.dumps(results, indent=2))
    x = [i*10 for i in range(len(kl_divergences))]
    plt.plot(x, kl_divergences)
    plt.xlabel("Checkpoint")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence of Probability Distributions of outputs")
    plt.savefig("kl_divergences.png")
    return

    results = compare_two_checkpoints_prob_distributions_after_equals(
        args.checkpoint1, args.checkpoint2, args.base_model
    )
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print(json.dumps(results, indent=2))
    return
        

    # Probability distribution comparison mode
    if args.compare_prob_dist:
        if args.checkpoint:
            results = compare_prob_distributions_after_equals(
                args.checkpoint, args.base_model
            )
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {args.output}")
            else:
                print(json.dumps(results, indent=2))
            return
        else:
            results = compare_two_checkpoints_prob_distributions_after_equals(
                args.checkpoint1, args.checkpoint2, args.base_model
            )
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {args.output}")
            else:
                print(json.dumps(results, indent=2))
            return
        
        
    if args.from_messages:
        if not args.checkpoint:
            print("Error: --checkpoint must be provided when using --from_messages")
            return
        
        results = compare_tokens_from_messages(
            args.checkpoint, args.base_model, args.use_output_embeddings
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        else:
            print(json.dumps(results, indent=2))
        return
    
    # Comparison mode: compare two checkpoints
    if args.checkpoint1 and args.checkpoint2:
        # Allow HuggingFace model names, so don't check path existence
        results = compare_two_checkpoints_weights(
            args.checkpoint1, args.checkpoint2, args.tokens, args.base_model, args.use_output_embeddings
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nComparison results saved to: {args.output}")
        else:
            print(json.dumps(results, indent=2))
        return
    
    # Single checkpoint mode
    if not args.checkpoint:
        print("Error: Either --checkpoint or both --checkpoint1 and --checkpoint2 must be provided")
        return
    
    # Allow HuggingFace model names, so don't check path existence
    results = compare_token_embeddings(args.checkpoint, args.tokens, args.base_model, args.use_output_embeddings)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

