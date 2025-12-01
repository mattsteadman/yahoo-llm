#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-1.5B-Instruct to ASK QUESTIONS like Yahoo Answers users.

Key changes vs previous version:
- No manual ChatML formatting
- Use tokenizer.apply_chat_template for conversation formatting
- Explicit PAD/EOS + model config alignment
"""

import platform
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb


# =====================================================================
# DATASET
# =====================================================================


def load_and_prepare_dataset(
    dataset_name="sentence-transformers/yahoo-answers",
    sample_size=None,
    val_split=0.1,
):
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, "title-question-pair")

    print(f"\nDataset structure: {dataset}")
    print(f"Train split size: {len(dataset['train']):,}")

    sample = dataset["train"][0]
    print(f"\nSample entry:")
    for key, value in sample.items():
        print(f"  {key}: {value[:200] if isinstance(value, str) else value}")

    # Optional subsample for faster runs
    if sample_size and sample_size < len(dataset["train"]):
        print(f"\nUsing subset of {sample_size:,} examples")
        indices = random.sample(range(len(dataset["train"])), sample_size)
        dataset["train"] = dataset["train"].select(indices)

    split_dataset = dataset["train"].train_test_split(test_size=val_split, seed=42)

    train_size = len(split_dataset["train"])
    val_size = len(split_dataset["test"])

    print(f"\nSplit into:")
    print(f"  Training: {train_size:,} examples ({(1 - val_split) * 100:.0f}%)")
    print(f"  Validation: {val_size:,} examples ({val_split * 100:.0f}%)")

    return split_dataset["train"], split_dataset["test"]


# Step 1: keep structured messages, no manual ChatML
def to_conversation(example):
    title = example["title"]
    question_body = example["questions"]

    messages = [
        {"role": "user", "content": "Ask a question"},
        {"role": "assistant", "content": f"{title} {question_body}"},
    ]
    return {"messages": messages}


# Step 2: use tokenizer.apply_chat_template to get a single text field
def make_apply_template(tokenizer):
    def apply_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return apply_template


# =====================================================================
# SEED
# =====================================================================


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# =====================================================================
# MAIN
# =====================================================================


def main():
    # --------------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------------
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    OUTPUT_DIR = "/workspace/yahoo-questions-model"
    DATASET_NAME = "sentence-transformers/yahoo-answers"

    SAMPLE_SIZE = 100_000
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    SAVE_STEPS = 500
    LOGGING_STEPS = 10
    WANDB_PROJECT = "yahoo-llm"
    RANDOM_SEED = 42

    # --------------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------------
    set_random_seed(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED}")

    # --------------------------------------------------------------
    # WANDB
    # --------------------------------------------------------------
    if WANDB_PROJECT:
        wandb.init(
            entity="stead",
            project=WANDB_PROJECT,
            name=f"qwen-1.5b-{SAMPLE_SIZE // 1000}k-samples",
            config={
                "model": MODEL_NAME,
                "sample_size": SAMPLE_SIZE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "max_length": 256,
                "lora_r": 16,
                "lora_alpha": 32,
                "random_seed": RANDOM_SEED,
            },
        )
        print("\n✓ Weights & Biases initialized")
        print(
            f"View training at: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}"
        )

    print("\n" + "=" * 60)
    print("Yahoo Answers QUESTION Style LLM - RunPod Training")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    print(f"Training samples: {SAMPLE_SIZE:,}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    # --------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------
    train_dataset, val_dataset = load_and_prepare_dataset(
        DATASET_NAME, sample_size=SAMPLE_SIZE
    )

    print("\nConverting to conversations...")
    train_with_msgs = train_dataset.map(
        to_conversation,
        remove_columns=train_dataset.column_names,
        desc="Building chat messages (train)",
    )
    val_with_msgs = val_dataset.map(
        to_conversation,
        remove_columns=val_dataset.column_names,
        desc="Building chat messages (val)",
    )

    # --------------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------------
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # PAD/EOS alignment
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Now turn messages -> text using the chat template
    apply_template = make_apply_template(tokenizer)

    print("\nApplying chat template to train dataset...")
    formatted_train = train_with_msgs.map(
        apply_template,
        remove_columns=["messages"],
        desc="Applying chat template (train)",
    )

    print("Applying chat template to validation dataset...")
    formatted_val = val_with_msgs.map(
        apply_template,
        remove_columns=["messages"],
        desc="Applying chat template (val)",
    )

    print("\nFormatted sample:")
    print(formatted_train[0]["text"][:500])

    # --------------------------------------------------------------
    # QUANTIZATION (QLoRA)
    # --------------------------------------------------------------
    print("\nConfiguring 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Align model config with tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # --------------------------------------------------------------
    # LoRA
    # --------------------------------------------------------------
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --------------------------------------------------------------
    # TRAINING ARGS
    # --------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        bf16=True,  # good on RTX 3090
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb" if WANDB_PROJECT else "none",
        gradient_checkpointing=True,
        # SFT-specific
        max_length=256,
        dataset_text_field="text",
        eos_token=tokenizer.eos_token,
    )

    # --------------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        processing_class=tokenizer,
    )

    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------
    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✓ Training complete! Model saved to: {OUTPUT_DIR}")
    if WANDB_PROJECT:
        wandb.finish()
        print("\n✓ Wandb run finished")


if __name__ == "__main__":
    main()
