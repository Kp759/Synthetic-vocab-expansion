import os
import json
import torch
import re
import gc
import shutil
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import Dataset
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import matplotlib
from transformers.data.data_collator import DataCollatorMixin
matplotlib.use('Agg') 

# Add parent directory to path to import calculate_acc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.calculate_acc import extract_number, load_model as load_model_for_eval


def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    print(f"Loading dataset from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data

class DataCollatorForAssistantOnlyLM(DataCollatorMixin):
    def __init__(self, tokenizer, mlm=False, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, examples):
        base_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=self.mlm,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch = base_collator(examples)
        
        labels = batch["labels"].clone()
        
        for i, example in enumerate(examples):
            input_ids = batch["input_ids"][i]
            
            assistant_header_text = "<|start_header_id|>assistant<|end_header_id|>"
            assistant_header_ids = self.tokenizer.encode(assistant_header_text, add_special_tokens=False)
            
            assistant_start_idx = None
            input_ids_list = input_ids.tolist()
            
            # Search for assistant header in the sequence
            for idx in range(len(input_ids_list) - len(assistant_header_ids) + 1):
                if input_ids_list[idx:idx+len(assistant_header_ids)] == assistant_header_ids:
                    assistant_start_idx = idx + len(assistant_header_ids)
                    while assistant_start_idx < len(input_ids_list) and input_ids_list[assistant_start_idx] in [13, 198, 271]:  # common newline tokens
                        assistant_start_idx += 1
                    break
            labels[i, :assistant_start_idx] = -100
        
        batch["labels"] = labels
        return batch

def prepare_dataset_for_training(dataset: List[Dict], tokenizer, max_length: int = 128):
    """
    Prepare dataset for training by formatting inputs and targets.
    Format matches calculate_acc.py: system message + user message (input_text) + assistant response (target)
    """
    def format_example(example):
        input_text = example["input"]
        target = example["target"]

        # Use the same message format as evaluation (calculate_acc.py)
        messages = [
            {"role": "system", "content": "Please answer with only the number, no other text."},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": str(target)}
        ]
        
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            formatted_text = f"System: Please answer with only the number, no other text.\nUser: {input_text}\nAssistant: {target}"
        
        return {"text": formatted_text, "input": input_text, "target": target}
    
    formatted_data = [format_example(ex) for ex in dataset]
    
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        return tokenized
    
    dataset_obj = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset_obj.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "input", "target"] 
    )
    
    return tokenized_dataset


def freeze_all_except_embeddings(model):
    """
    Freeze all layers except embedding layers.
    """
    print("\n🔧 Freezing all layers except embeddings...")
    
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "embed_positions" in name or "lm_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return model


def freeze_all_except_special_token_embedding(model, tokenizer, special_token: str = "<|reserved_special_token_0|>"):
    """
    Freeze all layers except the embedding of a specific special token.
    Only the embedding vector for the specified special token will be trainable.
    
    This implementation creates a separate trainable parameter for the special token
    and uses a forward hook to replace it during forward pass.
    
    Args:
        model: The model to modify
        tokenizer: Tokenizer to get the token ID
        special_token: The special token string (default: "<|reserved_special_token_0|>")
    
    Returns:
        Modified model with only special token embedding trainable
    """
    print(f"\n🔧 Freezing all layers except special token embedding: {special_token}")
    
    # Get the token ID for the special token
    try:
        # Try convert_tokens_to_ids first
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            token_id = tokenizer.convert_tokens_to_ids(special_token)
            if token_id == tokenizer.unk_token_id or token_id is None:
                # Try encoding instead
                encoded = tokenizer.encode(special_token, add_special_tokens=False)
                if len(encoded) == 1:
                    token_id = encoded[0]
                else:
                    raise ValueError(f"Special token '{special_token}' encodes to multiple tokens: {encoded}")
        else:
            # Fallback to encoding
            encoded = tokenizer.encode(special_token, add_special_tokens=False)
            if len(encoded) == 1:
                token_id = encoded[0]
            else:
                raise ValueError(f"Special token '{special_token}' encodes to multiple tokens: {encoded}")
    except Exception as e:
        raise ValueError(f"Failed to get token ID for '{special_token}': {e}")
    
    print(f"   Special token ID: {token_id}")
    
    # First, freeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Find the embedding layer
    embed_layer = None
    embed_layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and "embed_tokens" in name:
            embed_layer = module
            embed_layer_name = name
            break
    
    if embed_layer is None:
        raise ValueError("Could not find embed_tokens layer in the model")
    
    print(f"   Found embedding layer: {embed_layer_name}")
    print(f"   Embedding dimension: {embed_layer.embedding_dim}")
    print(f"   Vocabulary size: {embed_layer.num_embeddings}")
    
    # Verify token_id is valid
    if token_id >= embed_layer.num_embeddings:
        raise ValueError(f"Token ID {token_id} is out of range for vocabulary size {embed_layer.num_embeddings}")
    
    # SIMPLEST APPROACH: Make embed_tokens.weight trainable, but use gradient hook
    # to zero out gradients for all tokens except the special token
    # This avoids forward hook gradient issues entirely
    
    # Make the entire embedding weight trainable
    embed_layer.weight.requires_grad = True
    
    # Register a backward hook to zero out gradients for all tokens except special_token
    def selective_grad_hook(grad):
        """Zero out gradients for all tokens except the special token."""
        if grad is not None:
            # Zero out all gradients except for token_id
            grad_clone = grad.clone()
            grad_clone[token_id] = grad[token_id]  # Keep gradient for special token
            # Zero out all other gradients
            grad_clone[:token_id] = 0
            grad_clone[token_id+1:] = 0
            return grad_clone
        return grad
    
    # Register the hook on the weight parameter
    embed_layer.weight.register_hook(selective_grad_hook)
    
    # Store token_id for reference
    embed_layer._special_token_id = token_id
    
    print(f"   ✅ Only token {token_id} ('{special_token}') embedding will be updated during training")
    print(f"   📊 Trainable parameters: {embed_layer.embedding_dim} (just the embedding vector for this token)")
    
    return model

def freeze_only_embeddings(model, tokenizer):
    """
    Freeze all layers except embedding layers.
    """
    print("\n🔧 Freezing only embeddings...")
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "embed_positions" in name or "lm_head" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model

def freeze_only_special_token_embedding(model, tokenizer, special_token: str = "<|reserved_special_token_0|>"):
    """
    Freeze only the embedding of a specific special token.
    All other layers remain trainable, only the special token embedding will be frozen.
    
    This implementation uses a backward hook to zero out gradients for the special token
    embedding during backpropagation, effectively freezing it while keeping other parameters trainable.
    
    Args:
        model: The model to modify
        tokenizer: Tokenizer to get the token ID
        special_token: The special token string (default: "<|reserved_special_token_0|>")
    
    Returns:
        Modified model with only special token embedding frozen
    """
    print(f"\n🔧 Freezing only special token embedding: {special_token}")
    
    # Get the token ID for the special token
    try:
        # Try convert_tokens_to_ids first
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            token_id = tokenizer.convert_tokens_to_ids(special_token)
            if token_id == tokenizer.unk_token_id or token_id is None:
                # Try encoding instead
                encoded = tokenizer.encode(special_token, add_special_tokens=False)
                if len(encoded) == 1:
                    token_id = encoded[0]
                else:
                    raise ValueError(f"Special token '{special_token}' encodes to multiple tokens: {encoded}")
        else:
            # Fallback to encoding
            encoded = tokenizer.encode(special_token, add_special_tokens=False)
            if len(encoded) == 1:
                token_id = encoded[0]
            else:
                raise ValueError(f"Special token '{special_token}' encodes to multiple tokens: {encoded}")
    except Exception as e:
        raise ValueError(f"Failed to get token ID for '{special_token}': {e}")
    
    print(f"   Special token ID: {token_id}")
    
    # Ensure all parameters are trainable by default
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    # Find the embedding layer
    embed_layer = None
    embed_layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and "embed_tokens" in name:
            embed_layer = module
            embed_layer_name = name
            break
    
    if embed_layer is None:
        raise ValueError("Could not find embed_tokens layer in the model")
    
    print(f"   Found embedding layer: {embed_layer_name}")
    print(f"   Embedding dimension: {embed_layer.embedding_dim}")
    print(f"   Vocabulary size: {embed_layer.num_embeddings}")
    
    # Verify token_id is valid
    if token_id >= embed_layer.num_embeddings:
        raise ValueError(f"Token ID {token_id} is out of range for vocabulary size {embed_layer.num_embeddings}")
    
    # Save the original embedding value for the special token (detached clone)
    original_embedding = embed_layer.weight[token_id].clone().detach()
    
    # Store token_id and original embedding in the embedding layer
    embed_layer._frozen_token_id = token_id
    embed_layer._frozen_token_original = original_embedding
    
    # Register a hook on the weight parameter to zero out gradients for the special token
    # This hook is called when the gradient is computed for the weight parameter
    def zero_grad_hook(grad):
        """Hook to zero out gradient for the frozen special token embedding."""
        if grad is not None:
            # Zero out the gradient for the special token (modify in-place)
            grad[token_id].zero_()
        return grad
    
    # Register the hook on the weight parameter
    embed_layer.weight.register_hook(zero_grad_hook)
    
    def restore_frozen_embedding():
        """Restore the frozen token embedding to its original value."""
        with torch.no_grad():
            embed_layer.weight[token_id].copy_(original_embedding)
    
    # Store the restore function in the model for easy access
    model._restore_frozen_embedding = restore_frozen_embedding
    
    print(f"   ✅ Only token {token_id} ('{special_token}') embedding will be frozen during training")
    print(f"   📊 All other parameters remain trainable")
    
    return model

def get_trainable_parameters_info(model):
    """
    Get detailed information about trainable vs frozen parameters.
    Returns a dict with total, trainable, frozen counts and percentages.
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    trainable_param_names = []
    frozen_param_names = []
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            trainable_param_names.append(name)
        else:
            frozen_params += num_params
            frozen_param_names.append(name)
    
    trainable_ratio = (trainable_params / total_params * 100) if total_params > 0 else 0.0
    frozen_ratio = (frozen_params / total_params * 100) if total_params > 0 else 0.0
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_ratio": trainable_ratio,
        "frozen_ratio": frozen_ratio,
        "trainable_param_names": trainable_param_names,
        "frozen_param_names": frozen_param_names
    }


def print_trainable_parameters_summary(model, finetune_mode: str, output_dir: str = None):
    """
    Print a detailed summary of trainable parameters.
    Also saves to file and logs to W&B if available.
    """
    info = get_trainable_parameters_info(model)
    
    print(f"\n{'='*70}")
    print(f"📊 PARAMETER SUMMARY ({finetune_mode.upper()} FINETUNE)")
    print(f"{'='*70}")
    print(f"  Total parameters:        {info['total_params']:>20,} ({info['total_params']/1e9:.4f}B)")
    print(f"  Trainable parameters:   {info['trainable_params']:>20,} ({info['trainable_params']/1e6:.4f}M) - {info['trainable_ratio']:.2f}%")
    print(f"  Frozen parameters:      {info['frozen_params']:>20,} ({info['frozen_params']/1e9:.4f}B) - {info['frozen_ratio']:.2f}%")
    print(f"{'='*70}")
    
    return info


def prepare_lora_model(model, args):
    """
    Inject LoRA adapters for low-rank finetuning.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA finetune requested but `peft` is not installed. "
            "Install with `pip install peft`."
        ) from exc
    
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("lora_target_modules resolved to an empty list. Provide at least one module name.")
    
    print(f"\n🔧 Configuring LoRA adapters:")
    print(f"   Target modules: {target_modules}")
    print(f"   LoRA rank (r): {args.lora_r}")
    print(f"   LoRA alpha: {args.lora_alpha}")
    print(f"   LoRA dropout: {args.lora_dropout}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print PEFT's summary (it's already good)
    print("\n📋 PEFT Library Summary:")
    model.print_trainable_parameters()
    
    return model


def plot_accuracy_history(accuracy_history: List[Dict], output_dir: str):
    """
    Plot accuracy history over training steps.
    
    Args:
        accuracy_history: List of dicts with 'step', 'accuracy', 'correct', 'total'
        output_dir: Directory to save the plot
    """
    if not accuracy_history:
        print("No accuracy history to plot")
        return
    
    steps = [entry["step"] for entry in accuracy_history]
    accuracies = [entry["accuracy"] for entry in accuracy_history]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, accuracies, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Training Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.0])
    
    # Add value labels on points
    for step, acc in zip(steps, accuracies):
        plt.annotate(f'{acc:.3f}', (step, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Save plot
    plot_file = os.path.join(output_dir, "accuracy_history.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy plot saved to {plot_file}")


class RestoreFrozenEmbeddingCallback(TrainerCallback):
    """Callback to restore frozen token embedding after each optimizer step.
    
    This callback ensures the frozen embedding doesn't change during training by
    restoring it to its original value after each optimizer.step() call.
    
    Why this is needed:
    - Even with zero gradients, adaptive optimizers (Adam, AdamW) may apply tiny
      updates due to their internal momentum/state buffers
    - Floating-point precision can cause small numerical drift
    - This provides a safety guarantee that the embedding stays exactly frozen
    """
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Restore frozen embedding after each training step (after optimizer.step())."""
        if hasattr(model, '_restore_frozen_embedding'):
            model._restore_frozen_embedding()


class SaveSpecialTokenEmbeddingCallback(TrainerCallback):
    """Callback to write trained special_token embedding back to embed_tokens.weight before saving.
    
    This ensures that when checkpoints are saved, the trained embedding is in the standard
    embed_tokens.weight location, not just in the separate parameter.
    """
    
    def __init__(self, tokenizer, special_token: str):
        self.tokenizer = tokenizer
        self.special_token = special_token
        self.token_id = None
        # Get token_id once
        try:
            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                self.token_id = tokenizer.convert_tokens_to_ids(special_token)
                if self.token_id == tokenizer.unk_token_id or self.token_id is None:
                    encoded = tokenizer.encode(special_token, add_special_tokens=False)
                    if len(encoded) == 1:
                        self.token_id = encoded[0]
            else:
                encoded = tokenizer.encode(special_token, add_special_tokens=False)
                if len(encoded) == 1:
                    self.token_id = encoded[0]
        except:
            pass
    
    def _write_embedding_back(self, model):
        """Write trained special_token embedding back to embed_tokens.weight."""
        if self.token_id is None:
            return
        
        param_name = f'special_token_{self.token_id}_embedding'
        if hasattr(model, param_name):
            trained_embedding = getattr(model, param_name)
            # Find embed_tokens layer and write the trained embedding back
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Embedding) and "embed_tokens" in name:
                    with torch.no_grad():
                        module.weight[self.token_id].copy_(trained_embedding)
                    break
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Write embedding back before saving checkpoint."""
        self._write_embedding_back(model)


class TokenSimilarityCallback(TrainerCallback):
    """Callback to monitor cosine similarity between token embeddings during training.
    
    Tracks similarity between specified tokens and logs to wandb.
    Works with all finetune modes.
    """
    
    def __init__(self, tokenizer, tokens: List[str] = None, log_steps: int = 10):
        """
        Args:
            tokenizer: Tokenizer to get token IDs
            tokens: List of token strings to monitor (default: ["<|reserved_special_token_0|>", "=", "*"])
            log_steps: Log similarity every N steps
        """
        self.tokenizer = tokenizer
        self.tokens = tokens if tokens else ["<|reserved_special_token_0|>", "=", "*"]
        self.log_steps = log_steps
        self.token_ids = {}
        self.similarity_history = []
        
        # Get token IDs for all tokens
        for token_str in self.tokens:
            try:
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    token_id = tokenizer.convert_tokens_to_ids(token_str)
                    if token_id == tokenizer.unk_token_id or token_id is None:
                        encoded = tokenizer.encode(token_str, add_special_tokens=False)
                        if len(encoded) == 1:
                            token_id = encoded[0]
                else:
                    encoded = tokenizer.encode(token_str, add_special_tokens=False)
                    if len(encoded) == 1:
                        token_id = encoded[0]
                    else:
                        token_id = None
                
                if token_id is not None:
                    self.token_ids[token_str] = token_id
            except Exception as e:
                print(f"Warning: Could not get token ID for '{token_str}': {e}")
    
    def _get_token_embedding(self, model, token_str: str, token_id: int) -> torch.Tensor:
        """Extract token embedding, handling special_token mode."""
        # Check if there's a separate trainable parameter (special_token mode)
        param_name = f'special_token_{token_id}_embedding'
        
        # Check in model attributes
        if hasattr(model, param_name):
            param = getattr(model, param_name)
            with torch.no_grad():
                return param.clone().cpu().float()
        
        # Check in state_dict
        state_dict = model.state_dict()
        if param_name in state_dict:
            with torch.no_grad():
                return state_dict[param_name].clone().cpu().float()
        
        # Otherwise, extract from standard embedding layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding) and "embed_tokens" in name:
                with torch.no_grad():
                    return module.weight[token_id].clone().cpu().float()
        
        raise ValueError(f"Could not find embedding for token '{token_str}' (id={token_id})")
    
    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = torch.nn.functional.normalize(a, p=2, dim=0)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=0)
        return torch.dot(a_norm, b_norm).item()
    def _similarity_without_norm(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors without normalization."""
        return torch.dot(a, b).item()
        
    def _compute_similarities(self, model):
        """Compute pairwise similarities between all monitored tokens."""
        embeddings = {}
        
        # Extract embeddings for all tokens
        for token_str, token_id in self.token_ids.items():
            try:
                embeddings[token_str] = self._get_token_embedding(model, token_str, token_id)
            except Exception as e:
                print(f"Warning: Could not extract embedding for '{token_str}': {e}")
                return None
        
        # Compute pairwise similarities
        similarities = {}
        token_list = list(embeddings.keys())
        for i, token1 in enumerate(token_list):
            for token2 in token_list[i+1:]:
                key = f"{token1}_vs_{token2}"
                sim = self._similarity_without_norm(embeddings[token1], embeddings[token2])
                similarities[key] = sim
        
        return similarities
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log similarities to wandb and console."""
        if state.global_step % self.log_steps == 0:
            similarities = self._compute_similarities(model)
            if similarities is None:
                return
            
            # Store in history
            entry = {
                "step": state.global_step,
                **similarities
            }
            self.similarity_history.append(entry)
            
            # Log to wandb
            try:
                import wandb
                if wandb.run:
                    log_dict = {f"embed_sim/{k}": v for k, v in similarities.items()}
                    log_dict["step"] = state.global_step
                    wandb.log(log_dict)
            except (ImportError, Exception):
                pass  # W&B not available, continue without logging
            
            # Print to console
            print(f"\n[Step {state.global_step}] Token Embedding Similarities:")
            for key, sim in similarities.items():
                print(f"  {key}: {sim:.6f}")
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Save similarity history to file at end of training."""
        if self.similarity_history and hasattr(args, 'output_dir') and args.output_dir:
            similarity_file = os.path.join(args.output_dir, "token_similarity_history.json")
            with open(similarity_file, 'w') as f:
                json.dump(self.similarity_history, f, indent=2)
            print(f"\nToken similarity history saved to: {similarity_file}")


class AccuracyCallback(TrainerCallback):
    """Callback to evaluate accuracy on test set every N steps.
    Uses the same approach as calculate_acc.py: individual inference without batching.
    """
    
    def __init__(self, test_dataset: List[Dict], tokenizer, model_name: str, eval_steps: int = 10):
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.eval_steps = eval_steps
        self.accuracy_history = []
    
    def generate_single_response(self, model, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate response for a single prompt, matching calculate_acc.py and training format."""
        try:
            # Format messages (same format as training: system + user)
            messages = [
                {"role": "system", "content": "Please answer with only the number, no other text."},
                {"role": "user", "content": prompt}
            ]
            
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # True for inference (model needs to generate)
                )
            else:
                text = f"System: Please answer with only the number, no other text.\nUser: {prompt}\nAssistant:"
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0, input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def evaluate_accuracy(self, model, step: int, args):
        """Evaluate accuracy and save to history. Shared by on_train_begin and on_step_end."""
        print(f"\n{'='*60}")
        print(f"Evaluating accuracy at step {step}")
        print(f"{'='*60}")
        
        # Set model to eval mode
        model.eval()
        
        # Evaluate on test set (same as calculate_acc.py: individual inference)
        correct = 0
        total = len(self.test_dataset)
        
        for item in tqdm(self.test_dataset, desc="Evaluating"):
            input_text = item["input"]
            target = item["target"]
            
            response = self.generate_single_response(model, input_text, max_new_tokens=50)
            predicted = extract_number(response)
            
            if predicted == target:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        self.accuracy_history.append({
            "step": step,
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        })
        
        # Log to W&B if available
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "eval/accuracy": accuracy,
                    "eval/correct": correct,
                    "eval/total": total,
                    "step": step
                })
        except (ImportError, Exception):
            pass  # W&B not available or failed, continue without logging
        
        print(f"\nStep {step} - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Correct: {correct}/{total}")
        print(f"{'='*60}\n")
        
        # Set model back to train mode
        model.train()
        
        if hasattr(args, 'output_dir') and args.output_dir:
            log_file = os.path.join(args.output_dir, "accuracy_history.json")
            with open(log_file, 'w') as f:
                json.dump(self.accuracy_history, f, indent=2)
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Evaluate accuracy at step 0 before training starts."""
        self.evaluate_accuracy(model, step=0, args=args)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Evaluate accuracy every eval_steps. Uses same approach as calculate_acc.py."""
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            self.evaluate_accuracy(model, step=state.global_step, args=args)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune llama-3-8b-instruct with embedding-only or LoRA adapters")
    parser.add_argument("--train_dataset", type=str, default="data/special_multiply_train_dataset_1_reserved.json",
                       help="Path to dataset JSON file")
    parser.add_argument("--test_dataset", type=str, default="data/special_multiply_test_dataset_1_reserved.json",
                       help="Path to dataset JSON file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                       help="Model name from HuggingFace")
# meta-llama/Llama-3.2-3B-Instruct
    parser.add_argument("--output_dir", type=str, default="outputs/full_ft_reserved_assistant_only_3B_7000_2e-5",
                       help="Output directory for saved model")
    parser.add_argument("--train_size", type=int, default=7000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=1000,
                       help="Number of test samples")
    parser.add_argument("--val_split_ratio", type=float, default=0,
                       help="Ratio of training data to use for validation (default: 0.2 = 20%%)")
    parser.add_argument("--val_split_seed", type=int, default=42,
                       help="Random seed for train/val split (default: 42)")
    parser.add_argument("--eval_steps", type=int, default=10,
                       help="Evaluate accuracy every N steps (default: 10)")
    parser.add_argument("--max_steps", type=int, default=200,
                       help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio (default: 0.1 = 10%% of max_steps)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping (critical for full finetune)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for regularization (helps prevent overfitting in full finetune)")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
                       help="Learning rate scheduler type")
    parser.add_argument("--finetune_mode", type=str, default="embedding", 
                       choices=["embedding", "lora", "full", "special_token", "freeze_embeddings", "freeze_special_token"],
                       help="Choose finetune strategy: embedding (freeze all but embeddings), lora (inject adapters), full (all trainable), or special_token (only special token embedding)")
    parser.add_argument("--special_token", type=str, default="<|reserved_special_token_0|>",
                       help="Special token to finetune (only used when finetune_mode=special_token)")
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank (only used when finetune_mode=lora)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                       help="LoRA alpha scaling (only used when finetune_mode=lora)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout probability (only used when finetune_mode=lora)")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                       help="Comma separated module names to apply LoRA on (only used when finetune_mode=lora)")
    parser.add_argument("--wandb_project", type=str, default="embed-finetune",
                       help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="W&B run name (default: auto-generated from output_dir)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="W&B entity/team name (optional)")
    
    args = parser.parse_args()
    
    # Load dataset
    train_dataset = load_dataset(args.train_dataset)
    test_dataset = load_dataset(args.test_dataset)
    eval_dataset = load_dataset(args.test_dataset)
    
    # Split training data into train and validation sets
    import random
    random.seed(args.val_split_seed)
    random.shuffle(train_dataset)
    train_dataset_raw = train_dataset[:args.train_size]
    test_dataset_raw = test_dataset[:args.test_size]
    eval_dataset_raw = eval_dataset
    
    # Shuffle and split training data
    shuffled_train = train_dataset_raw.copy()
    random.shuffle(shuffled_train)
    
    val_size = int(len(shuffled_train) * args.val_split_ratio)
    # if val_size == 0 and len(shuffled_train) > 0:
    #     val_size = 1  # Ensure at least 1 sample for validation if possible
    #     print(f"⚠️  Warning: val_split_ratio too small, using 1 sample for validation")
    
    val_dataset_raw = shuffled_train[:val_size]
    train_dataset_raw = shuffled_train[val_size:]
    
    print(f"Train samples: {len(train_dataset_raw)}")
    print(f"Validation samples: {len(val_dataset_raw)} ({args.val_split_ratio*100:.1f}% of training data)")
    print(f"Test samples: {len(test_dataset_raw)}")
    print(f"Eval samples: {len(eval_dataset_raw)}")
    
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if args.finetune_mode == "embedding":
        model = freeze_all_except_embeddings(model)
        print("\n✅ Finetune mode: EMBEDDING-ONLY (only embedding + LM head are trainable)")
    elif args.finetune_mode == "lora":
        model = prepare_lora_model(model, args)
        print("\n✅ Finetune mode: LoRA (adapters inserted on selected modules)")
        if args.learning_rate == 5e-5:
            print("ℹ️  Consider increasing --learning_rate (e.g., 2e-4) for LoRA if training converges slowly.")
    elif args.finetune_mode == "full":
        print("\n✅ Finetune mode: FULL (all layers are trainable)")
        # Adjust learning rate for full finetune if using default
        if args.learning_rate == 5e-5:
            print("⚠️  Warning: Full finetune with default LR (5e-5) may cause overfitting.")
            print("   Consider using lower LR (e.g., 2e-5) or adding regularization.")
        # Suggest gradient checkpointing for memory efficiency
        print("ℹ️  Full finetune uses all parameters - ensure sufficient regularization to prevent overfitting.")
    elif args.finetune_mode == "special_token":
        model = freeze_all_except_special_token_embedding(model, tokenizer, args.special_token)
        print(f"\n✅ Finetune mode: SPECIAL-TOKEN-ONLY (only '{args.special_token}' embedding is trainable)")
    elif args.finetune_mode == "freeze_embeddings":
        model = freeze_only_embeddings(model, tokenizer)
        print("\n✅ Finetune mode: FREEZE-EMBEDDINGS (only embeddings are trainable)")
    elif args.finetune_mode == "freeze_special_token":
        model = freeze_only_special_token_embedding(model, tokenizer, args.special_token)
        print(f"\n✅ Finetune mode: FREEZE-SPECIAL-TOKEN (only '{args.special_token}' embedding is trainable)")
    
    # Print detailed parameter summary
    print_trainable_parameters_summary(model, args.finetune_mode, args.output_dir)
    
    print("\nPreparing training dataset...")
    train_dataset = prepare_dataset_for_training(train_dataset_raw, tokenizer)
    
    if len(val_dataset_raw) > 0:
        print("Preparing validation dataset...")
        val_dataset = prepare_dataset_for_training(val_dataset_raw, tokenizer)
    else:
        val_dataset = None

    print("Preparing test dataset...")
    test_dataset = prepare_dataset_for_training(test_dataset_raw, tokenizer)
    eval_dataset = prepare_dataset_for_training(eval_dataset_raw, tokenizer)


    data_collator = DataCollatorForAssistantOnlyLM(
        tokenizer=tokenizer,
        mlm=False
    )
    
    wandb_run_name = args.wandb_run_name
    if wandb_run_name is None:
        wandb_run_name = os.path.basename(args.output_dir.rstrip('/'))
    
    # Calculate warmup steps
    warmup_steps = int(args.max_steps * args.warmup_ratio) if args.warmup_ratio > 0 else 0
    
    # Adjust learning rate for full finetune to prevent overfitting
    effective_lr = args.learning_rate
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=effective_lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        max_grad_norm=args.max_grad_norm,  # Critical for full finetune stability
        weight_decay=args.weight_decay,  # Regularization to prevent overfitting
        fp16=False,
        bf16=True,
        gradient_checkpointing=(args.finetune_mode == "full"),  # Enable for full finetune to save memory
        logging_steps=5,
        save_steps=50,
        save_total_limit=10,
        eval_strategy="no",  # Enable evaluation if validation set exists
        eval_steps=args.eval_steps,  # Evaluate every N steps (same as accuracy callback)
        load_best_model_at_end=False,  # Set to True if you want to load best model based on validation loss
        report_to="wandb",
        run_name=wandb_run_name,
        remove_unused_columns=False,
    )
    
    try:
        import wandb
        if not wandb.run:
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                entity=args.wandb_entity,
                config=vars(args),
                tags=["embedding-finetune", "llama-3-8b"]
            )
            print(f"✅ W&B initialized: project={args.wandb_project}, run={wandb_run_name}")
    except Exception as e:
        print(f"Error initializing W&B: {e}")
        
    
    accuracy_callback = AccuracyCallback(
        test_dataset=test_dataset_raw,  # Raw dataset with input/target for evaluation
        tokenizer=tokenizer,
        model_name=args.model,
        eval_steps=args.eval_steps
    )
    
    # Add callback to restore frozen embedding if using freeze_special_token mode
    callbacks = [accuracy_callback]
    if args.finetune_mode == "freeze_special_token":
        restore_callback = RestoreFrozenEmbeddingCallback()
        callbacks.append(restore_callback)
    
    # Add callback to write special_token embedding back to embed_tokens.weight before saving
    if args.finetune_mode == "special_token":
        save_embedding_callback = SaveSpecialTokenEmbeddingCallback(tokenizer, args.special_token)
        callbacks.append(save_embedding_callback)
    
    # Add callback to monitor token embedding similarities (works for all finetune modes)
    similarity_callback = TokenSimilarityCallback(
        tokenizer=tokenizer,
        tokens=["<|reserved_special_token_0|>", "=", "*"],
        log_steps=args.eval_steps  # Log at same frequency as accuracy evaluation
    )
    callbacks.append(similarity_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset_raw) > 0 else None,  # Add validation dataset
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    trainer.train()
    
    print(f"\nSaving final model to {args.output_dir}")
    # Use safe_serialization=False to avoid tensor sharing issues with special token parameter
    # The SaveSpecialTokenEmbeddingCallback will handle writing the embedding back before save
    trainer.save_model(safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save accuracy history
    accuracy_file = os.path.join(args.output_dir, "accuracy_history.json")
    with open(accuracy_file, 'w') as f:
        json.dump(accuracy_callback.accuracy_history, f, indent=2)
    print(f"Accuracy history saved to {accuracy_file}")
    
    if accuracy_callback.accuracy_history:
        plot_accuracy_history(accuracy_callback.accuracy_history, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Final evaluation on test set...")
    print(f"{'='*60}")
    
    # Use the trained model directly (no need to reload) - same method as training evaluation
    model.eval()
    
    correct = 0
    total = len(eval_dataset_raw)
    
    # Use the same generation method as AccuracyCallback for consistency
    for item in tqdm(eval_dataset_raw, desc="Final evaluation"):
        input_text = item["input"]
        target = item["target"]
        
        # Use same method as training evaluation (direct model.generate, no pipeline)
        response = accuracy_callback.generate_single_response(model, input_text, max_new_tokens=50)
        predicted = extract_number(response)
        
        if predicted == target:
            correct += 1
    
    final_accuracy = correct / total if total > 0 else 0.0
    print(f"\nFinal Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Correct: {correct}/{total}")
    
    print(f"\nTraining completed! Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()