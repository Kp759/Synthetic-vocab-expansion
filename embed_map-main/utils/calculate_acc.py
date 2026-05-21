import json
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


def load_model(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", checkpoint_path: str = None):
    """
    Load model and tokenizer for inference, either from HuggingFace or from checkpoint.
    
    Args:
        model_name (str): Name of the base model to load (HuggingFace id)
        checkpoint_path (str): Path to checkpoint directory (optional)
        
    Returns:
        (model, tokenizer): Loaded model and tokenizer
    """
    load_path = checkpoint_path if checkpoint_path and os.path.exists(checkpoint_path) else model_name
    print(f"Loading model from: {load_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    return model, tokenizer


def generate_single_response(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """
    Generate response for a single prompt, using the same format as embed_finetune.py.
    System: Please answer with only the number, no other text.
    User:   prompt (e.g., @(a,b)=)
    """
    try:
        messages = [
            {"role": "system", "content": "Please answer with only the number, no other text."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template – same logic as embed_finetune.py
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # for inference
            )
        else:
            text = (
                "System: Please answer with only the number, no other text.\n"
                f"User: {prompt}\nAssistant:"
            )
        
        inputs = tokenizer(
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
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0, input_len:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""


def extract_number(text: str) -> int:
    """
    Extract the first integer number from the text.
    
    Args:
        text (str): Text to extract number from
        
    Returns:
        int: Extracted number, or None if no number found
    """
    # Try to find the first integer in the text
    # Look for numbers that might be at the start or after some text
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            return None
    return None


def load_dataset(file_path: str):
    """
    Load dataset from JSON file.
    
    Args:
        file_path (str): Path to the dataset JSON file
        
    Returns:
        list: List of dataset items with 'input' and 'target' fields
    """
    print(f"Loading dataset from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")
    return data


def calculate_accuracy(dataset_path: str, 
                      model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                      checkpoint_path: str = None,
                      max_samples: int = None,
                      max_new_tokens: int = 50):
    """
    Calculate accuracy of a model on a dataset.
    
    Args:
        dataset_path (str): Path to the dataset JSON file
        model_name (str): Name of the model to evaluate
        checkpoint_path (str): Optional path to model checkpoint
        max_samples (int): Optional limit on number of samples to evaluate
        max_new_tokens (int): Maximum tokens to generate
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    dataset = load_dataset(dataset_path)
    output_path = dataset_path.replace(".json", "_results.json")
    rst = {"accuracy": [], "input": []}

    if max_samples:
        dataset = dataset[:max_samples]
        print(f"Evaluating on first {max_samples} samples")
    
    # Load model & tokenizer in the same way as embed_finetune.py
    model, tokenizer = load_model(model_name, checkpoint_path)
    
    # Evaluate
    correct = 0
    total = len(dataset)
    results = []
    
    print(f"Evaluating model on {total} samples...")
    for i, item in tqdm(enumerate(dataset), desc="Evaluating"):
        input_text = item["input"]
        target = item["target"]
        # Use the same generation format as finetune eval
        response = generate_single_response(model, tokenizer, input_text, max_new_tokens=max_new_tokens)
        
        predicted = extract_number(response)
        
        # Check if correct
        is_correct = (predicted == target)
        if is_correct:
            correct += 1
        
        results.append({
            "input": input_text,
            "target": target,
            "predicted": predicted,
            "response": response,
            "correct": is_correct
        })
        rst["accuracy"].append(is_correct)
        rst["input"].append(input_text)
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\nResults:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    with open(output_path, "w") as f:
        json.dump(rst, f, indent=2)
    print(f"Results saved to: {output_path}")
    return accuracy, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate model accuracy on arithmetic dataset")
    parser.add_argument("--dataset", type=str, default="data/add_mul_dataset.json",
                       help="Path to dataset JSON file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                       help="Model name from HuggingFace")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (optional)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")
    parser.add_argument("--max_new_tokens", type=int, default=5,
                       help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    
    accuracy, results = calculate_accuracy(
        dataset_path=args.dataset,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens
    )

