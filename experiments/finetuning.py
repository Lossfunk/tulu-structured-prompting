"""
Fine-tuning Experiment for Llama 3.2 3B with Unsloth

This script fine-tunes Llama 3.2 3B on Tulu question-answer pairs using Unsloth
for efficient training, then evaluates performance against prompt-based baseline.
"""

import os
import json
import torch
from typing import List, Dict, Optional
from pathlib import Path
import time

# Unsloth imports (will fail gracefully if not installed)
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: Unsloth not installed. Install with:")
    print('  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    print("  pip install --no-deps xformers trl peft accelerate bitsandbytes")


def load_tulu_dataset(file_path: str) -> List[Dict]:
    """
    Load Tulu dataset from JSON file.
    
    Handles both formats:
    - Direct list: [{"english": "...", "tulu": "..."}, ...]
    - With metadata: {"sentences": [{"english": "...", "tulu": "..."}, ...]}
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of dicts with 'english' and 'tulu' keys
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        examples = data
    elif "sentences" in data:
        examples = data["sentences"]
    elif "examples" in data:
        examples = data["examples"]
    else:
        raise ValueError(f"Unknown data format in {file_path}")
    
    # Ensure required fields
    formatted = []
    for ex in examples:
        if "english" in ex and "tulu" in ex:
            formatted.append({
                "question": ex["english"],
                "tulu_response": ex["tulu"]
            })
        elif "question" in ex and "tulu_response" in ex:
            formatted.append(ex)
        else:
            print(f"Warning: Skipping example missing required fields: {ex}")
    
    return formatted


def format_llama3_prompt(example: Dict) -> Dict:
    """
    Format example as Llama 3 instruction-following prompt.
    
    Args:
        example: Dict with 'question' and 'tulu_response' keys
        
    Returns:
        Dict with 'text' key containing formatted prompt
    """
    question = example.get("question", "")
    tulu_response = example.get("tulu_response", "")
    
    text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{tulu_response}<|eot_id|>"""
    
    return {"text": text}


def load_model_with_unsloth(
    model_name: str = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length: int = 512,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    random_state: int = 42
):
    """
    Load model with Unsloth optimizations.
    
    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        load_in_4bit: Use 4-bit quantization
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        random_state: Random seed
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is not available. Please install it first.")
    
    print(f"Loading model: {model_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  4-bit quantization: {load_in_4bit}")
    print(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
    )
    
    print("Model loaded successfully!")
    return model, tokenizer


def train_model(
    model,
    tokenizer,
    train_data: List[Dict],
    val_data: List[Dict],
    output_dir: str = "outputs/finetuned_llama3.2_3b",
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    max_seq_length: int = 512,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    eval_strategy: str = "epoch",
    weight_decay: float = 0.01,
    seed: int = 42,
):
    """
    Train the model using SFTTrainer.
    
    Args:
        model: Unsloth model
        tokenizer: Tokenizer
        train_data: Training examples
        val_data: Validation examples
        output_dir: Output directory for checkpoints
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio
        max_seq_length: Maximum sequence length
        logging_steps: Logging frequency
        save_strategy: When to save checkpoints
        eval_strategy: When to evaluate
        weight_decay: Weight decay
        seed: Random seed
        
    Returns:
        Trainer object
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is not available. Please install it first.")
    
    print(f"\nPreparing datasets...")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(val_data)}")
    
    # Format prompts
    train_dataset = Dataset.from_list(train_data).map(format_llama3_prompt)
    val_dataset = Dataset.from_list(val_data).map(format_llama3_prompt)
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        optim="adamw_8bit",
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        seed=seed,
        output_dir=output_dir,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,  # Changed from evaluation_strategy
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )
    
    print(f"\nStarting training...")
    print(f"  Effective batch size: {per_device_batch_size * gradient_accumulation_steps}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output directory: {output_dir}")
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time/60:.2f} minutes!")
    
    # Save final model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer


def generate_response(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate Tulu response for a question.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        question: Input question
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Generated Tulu response
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is not available. Please install it first.")
    
    # Format prompt
    prompt_dict = format_llama3_prompt({
        "question": question,
        "tulu_response": ""
    })
    prompt = prompt_dict["text"]
    
    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract Tulu response (after assistant header)
    if "assistant" in response:
        tulu_response = response.split("assistant")[-1].strip()
        # Remove any remaining special tokens
        tulu_response = tulu_response.replace("<|eot_id|>", "").strip()
    else:
        # Fallback: return everything after the prompt
        prompt_len = len(prompt)
        tulu_response = response[prompt_len:].strip()
    
    return tulu_response


def main():
    """Main training function."""
    if not UNSLOTH_AVAILABLE:
        print("ERROR: Unsloth is not installed.")
        print("\nTo install Unsloth, run:")
        print('  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        print("  pip install --no-deps xformers trl peft accelerate bitsandbytes")
        return
    
    # Configuration
    TRAIN_DATA_PATH = "data/tulu_train.json"
    TEST_DATA_PATH = "data/tulu_test.json"
    DEV_DATA_PATH = "data/tulu_dev.json"
    OUTPUT_DIR = "outputs/finetuned_llama3.2_3b"
    
    # Hyperparameters
    config = {
        "model_name": "unsloth/Llama-3.2-3B-Instruct",
        "max_seq_length": 512,
        "load_in_4bit": True,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "random_state": 42,
    }
    
    print("=" * 70)
    print("Fine-tuning Llama 3.2 3B with Unsloth")
    print("=" * 70)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load data
    print(f"\nLoading datasets...")
    try:
        train_data = load_tulu_dataset(TRAIN_DATA_PATH)
        test_data = load_tulu_dataset(TEST_DATA_PATH)
        
        # Use dev set for validation, or split from train if not available
        if os.path.exists(DEV_DATA_PATH):
            val_data = load_tulu_dataset(DEV_DATA_PATH)
        else:
            # Use 15% of training data for validation
            val_size = int(len(train_data) * 0.15)
            val_data = train_data[-val_size:]
            train_data = train_data[:-val_size]
        
        print(f"  Training examples: {len(train_data)}")
        print(f"  Validation examples: {len(val_data)}")
        print(f"  Test examples: {len(test_data)}")
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    
    # Load model
    try:
        model, tokenizer = load_model_with_unsloth(
            model_name=config["model_name"],
            max_seq_length=config["max_seq_length"],
            load_in_4bit=config["load_in_4bit"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            random_state=config["random_state"],
        )
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Train
    try:
        trainer = train_model(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            val_data=val_data,
            output_dir=OUTPUT_DIR,
            per_device_batch_size=config["per_device_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_epochs=config["num_epochs"],
            learning_rate=config["learning_rate"],
            warmup_ratio=config["warmup_ratio"],
            max_seq_length=config["max_seq_length"],
            seed=config["random_state"],
        )
        
        # Enable fast inference
        FastLanguageModel.for_inference(model)
        
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
        print(f"\nModel saved to: {OUTPUT_DIR}")
        print(f"\nNext steps:")
        print("  1. Run evaluation: python experiments/evaluate_finetuned.py")
        print("  2. Compare with baseline: python experiments/compare_finetuning.py")
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

