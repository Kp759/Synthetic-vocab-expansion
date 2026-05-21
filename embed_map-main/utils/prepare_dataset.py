import json
import random
import re


def generate_normal_dataset(num_samples: int = 1000, output_path: str = "data/original_dataset.json"):
    dataset = []
    
    for _ in range(num_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        c = random.randint(1, 100)
        
        input_str = f"{a}*{b}+{c}="
        target = a * b + c
        
        dataset.append({
            "input": input_str,
            "target": target
        })

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {num_samples} samples and saved to {output_path}")
    return dataset

def generate_special_dataset(num_samples: int = 1000, output_path: str = "data/special_dataset.json"):
    dataset = []
    
    for _ in range(num_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        c = random.randint(1, 100)
        
        input_str = f"@({a},{b},{c})="
        target = a * b + c
        
        dataset.append({
            "input": input_str,
            "target": target
        })
    
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Generated {num_samples} samples and saved to {output_path}")
    return dataset

def generate_multiply_dataset(num_samples: int = 1000, output_path: str = "data/multiply_dataset.json"):
    dataset = []
    
    for _ in range(num_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        
        input_str = f"{a}*{b}="
        target = a * b

        dataset.append({
            "input": input_str,
            "target": target
        })
    
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Generated {num_samples} samples and saved to {output_path}")
    return dataset

def generate_add_dataset(num_samples: int = 1000, output_path: str = "data/add_mul_dataset.json"):
    dataset = []
    
    for _ in range(num_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        
        input_str = f"{a}+{b}="
        target = a * b

        dataset.append({
            "input": input_str,
            "target": target
        })
    
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Generated {num_samples} samples and saved to {output_path}")
    return dataset

def generate_special_multiply_dataset(num_samples: int = 1000, output_path: list[str] = ["data/special_multiply_train_dataset_1.json", "data/special_multiply_test_dataset_1.json"]):
    all_pairs = [(a, b) for a in range(1, 101) for b in range(1, 101)]
    
    random.shuffle(all_pairs)
    selected_pairs = all_pairs[:num_samples*9]
    
    dataset = []
    for a, b in selected_pairs[:num_samples*8]:
        input_str = f"@({a},{b})="
        target = a * b
        
        dataset.append({
            "input": input_str,
            "target": target
        })
    
    with open(output_path[0], "w") as f:
        json.dump(dataset, f, indent=2)
    
    dataset = []
    for a, b in selected_pairs[num_samples*8:]:
        input_str = f"@({a},{b})="
        target = a * b
        
        dataset.append({
            "input": input_str,
            "target": target
        })
    
    with open(output_path[1], "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} unique samples and saved to {output_path}")
    return dataset

def convert_special_to_normal_format(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    converted_data = []
    for item in data:
        input_str = item["input"]
        target = item["target"]
        
        # Extract numbers from @(a,b)= format using regex
        match = re.match(r'@\((\d+),(\d+)\)=', input_str)
        if match:
            a = int(match.group(1))
            b = int(match.group(2))
            
            # Convert to a*b= format
            new_input = f"{a}*{b}="
            
            converted_data.append({
                "input": new_input,
                "target": target
            })
        else:
            print(f"Warning: Could not parse input: {input_str}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples and saved to {output_file}")
    return converted_data

if __name__ == "__main__":
    normal_dataset = generate_normal_dataset(num_samples=1000, output_path="data/original_dataset.json")
    special_dataset = generate_special_dataset(num_samples=1000, output_path="data/special_dataset.json")
    multiply_dataset = generate_multiply_dataset(num_samples=1000, output_path="data/multiply_dataset_2.json")
    special_multiply_dataset = generate_special_multiply_dataset(num_samples=1000, output_path=["data/special_multiply_train_dataset_1.json", "data/special_multiply_test_dataset_1.json"])
    add_dataset = generate_add_dataset(num_samples=1000, output_path="data/add_mul_dataset.json")
    Convert special format to normal format
    convert_special_to_normal_format(
        input_file="data/special_multiply_test_dataset_1.json",
        output_file="data/mul_test_dataset_1.json"
    )