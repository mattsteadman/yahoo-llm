#!/usr/bin/env python3
"""
Fine-tune a small LLM to ASK QUESTIONS like Yahoo Answers users (not answer them).
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


def load_and_prepare_dataset(dataset_name="sentence-transformers/yahoo-answers", sample_size=None):
    """Load the Yahoo Answers dataset and explore it."""
    # Load both configs to get more variety
    print(f"Loading dataset: {dataset_name}")

    # Use title-question-pair to get title -> question body mappings
    dataset = load_dataset(dataset_name, "title-question-pair")

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


def format_question_style_v1(example):
    """
    Option 1: Generate full questions from titles.
    Model learns to expand a title into a full Yahoo Answers question.
    """
    title = example['title']
    question_body = example['questions']

    # Train model to expand title into full question
    text = f"""<|user|>
Expand this into a detailed question: {title}
<|assistant|>
{title} {question_body}"""

    return {"text": text}


def format_question_style_v2(example):
    """
    Option 2: Generate question bodies from titles.
    Model learns to write the question elaboration given just a title.
    """
    title = example['title']
    question_body = example['questions']

    text = f"""<|user|>
Write a detailed question about: {title}
<|assistant|>
{question_body}"""

    return {"text": text}


def format_question_style_v3(example):
    """
    Option 3: Just learn to talk like a Yahoo Answers questioner.
    Model learns the rambling, casual style of Yahoo Answers questions.
    """
    title = example['title']
    question_body = example['questions']

    # Simple: given a topic, generate a question in Yahoo Answers style
    # Extract a keyword from title for variety
    text = f"""<|user|>
Ask a question
<|assistant|>
{title} {question_body}"""

    return {"text": text}


def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    OUTPUT_DIR = "./yahoo-questions-model"
    DATASET_NAME = "sentence-transformers/yahoo-answers"

    # Training hyperparameters
    SAMPLE_SIZE = 10  # Tiny sample for quick testing
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 1
    MAX_STEPS = 3  # Only 3 steps for very quick test

    # Choose formatting style (1, 2, or 3)
    FORMAT_STYLE = 3  # Option 3: Just talk like a Yahoo Answers questioner

    format_functions = {
        1: format_question_style_v1,
        2: format_question_style_v2,
        3: format_question_style_v3,
    }

    format_func = format_functions[FORMAT_STYLE]

    print("=" * 60)
    print("Yahoo Answers QUESTION Style LLM Fine-tuning")
    print("=" * 60)
    print(f"Format Style: {FORMAT_STYLE}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU/MPS'}")
    print(f"Quantization: {'Enabled (4-bit)' if USE_QUANTIZATION else 'Disabled (full precision)'}")

    # Load dataset
    dataset = load_and_prepare_dataset(DATASET_NAME, sample_size=SAMPLE_SIZE)

    # Format dataset
    print("\nFormatting dataset for SFT...")
    formatted_dataset = dataset['train'].map(
        format_func,
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
        model = prepare_model_for_kbit_training(model)
    else:
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
        fp16=not USE_QUANTIZATION,
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
    print("\nThe model now talks like Yahoo Answers QUESTIONERS!")
    print(f"Format style used: {FORMAT_STYLE}")


if __name__ == "__main__":
    main()
