#!/usr/bin/env python3
"""
Fine-tune a small LLM on Yahoo Answers data to make it talk like Yahoo Answers users.
Uses SFT (Supervised Fine-Tuning) with LoRA for efficient training.
"""

import sys
import platform
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import random

# Check if bitsandbytes is available (not on macOS)
USE_QUANTIZATION = False
try:
    import bitsandbytes
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    USE_QUANTIZATION = True
except (ImportError, ModuleNotFoundError):
    print("Note: Running without quantization (bitsandbytes not available)")
    USE_QUANTIZATION = False


def load_and_prepare_dataset(dataset_name="sentence-transformers/yahoo-answers", config="title-question-answer-pair", sample_size=None):
    """Load the Yahoo Answers dataset and explore it."""
    print(f"Loading dataset: {dataset_name} ({config})")
    dataset = load_dataset(dataset_name, config)

    print(f"\nDataset structure: {dataset}")
    print(f"\nTrain split size: {len(dataset['train'])}")

    # Show a sample
    sample = dataset['train'][0]
    print(f"\nSample entry:")
    for key, value in sample.items():
        print(f"  {key}: {value[:200] if isinstance(value, str) else value}")

    # Use a subset for faster training if specified
    if sample_size and sample_size < len(dataset['train']):
        print(f"\nUsing subset of {sample_size} examples for training")
        indices = random.sample(range(len(dataset['train'])), sample_size)
        dataset['train'] = dataset['train'].select(indices)

    return dataset


def format_yahoo_answer(example):
    """
    Format a Yahoo Answers example into a conversational prompt.
    The model learns to respond like a Yahoo Answers user.
    """
    question = example['question']
    answer = example['answer']

    # Format: Question -> Answer (Yahoo Answers style)
    # We'll train the model to generate the answer given the question
    text = f"""<|user|>
{question}
<|assistant|>
{answer}"""

    return {"text": text}


def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B parameters, good for Yahoo Answers style
    OUTPUT_DIR = "./yahoo-answers-model"
    DATASET_NAME = "sentence-transformers/yahoo-answers"

    # Training hyperparameters
    SAMPLE_SIZE = 10  # Tiny sample for quick testing
    MAX_SEQ_LENGTH = 256
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 1
    MAX_STEPS = 3  # Only 3 steps for very quick test

    print("=" * 60)
    print("Yahoo Answers LLM Fine-tuning")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU/MPS'}")
    print(f"Quantization: {'Enabled (4-bit)' if USE_QUANTIZATION else 'Disabled (full precision)'}")

    # Load dataset
    dataset = load_and_prepare_dataset(DATASET_NAME, sample_size=SAMPLE_SIZE)

    # Format dataset
    print("\nFormatting dataset for SFT...")
    formatted_dataset = dataset['train'].map(
        format_yahoo_answer,
        remove_columns=dataset['train'].column_names
    )

    # Show formatted sample
    print("\nFormatted sample:")
    print(formatted_dataset[0]['text'][:500])

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")

    if USE_QUANTIZATION:
        # QLoRA configuration for efficient training (Linux/CUDA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        # Full precision training (macOS)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=2,
        save_strategy="steps",
        save_steps=MAX_STEPS,
        optim="paged_adamw_8bit" if USE_QUANTIZATION else "adamw_torch",
        fp16=not USE_QUANTIZATION,  # Use fp16 on macOS, bf16 with quantization
        bf16=USE_QUANTIZATION,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Save model
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✓ Training complete! Model saved to: {OUTPUT_DIR}")
    print("\nTo use the model, run: python inference.py")


if __name__ == "__main__":
    main()