#!/usr/bin/env python3
"""
Fine-tune a small LLM to ASK QUESTIONS like Yahoo Answers users.
Optimized for RunPod GPU training.

TRAINING STRATEGY:
This script uses Supervised Fine-Tuning (SFT) to teach a language model to generate
questions in the style of Yahoo Answers users. Instead of training it to answer questions,
we train it to ASK questions with the characteristic casual, rambling style of Yahoo Answers.

EFFICIENCY TECHNIQUES:
- QLoRA (Quantized Low-Rank Adaptation): Loads model in 4-bit precision to reduce VRAM usage
- LoRA: Only trains ~1% of parameters (18M out of 1.5B) via low-rank adapter matrices
- Gradient checkpointing: Trades compute for memory by recomputing activations during backward pass
"""

import platform
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
from trl import SFTTrainer
import random
import wandb


def load_and_prepare_dataset(
    dataset_name="sentence-transformers/yahoo-answers", sample_size=None, val_split=0.1
):
    """
    Load the Yahoo Answers dataset from HuggingFace and split into train/val.

    DATASET STRUCTURE:
    We use the 'title-question-pair' configuration which contains:
    - 'title': The question title (e.g., "How do I clean my keyboard?")
    - 'questions': The question body/elaboration (e.g., "I have stuff stuck under the keys...")

    This gives us the full question context including both the title and the rambling
    elaboration that Yahoo Answers users typically add.

    TRAIN/VAL SPLIT:
    We split the data into training and validation sets. The validation set helps us
    monitor for overfitting - if training loss drops but validation loss stays high
    or increases, the model is memorizing rather than learning generalizable patterns.
    """
    print(f"Loading dataset: {dataset_name}")

    # Load the title-question-pair config to get full questions with both title and body.
    # This is preferred over question-answer-pair because we want to learn the QUESTION
    # writing style, not the answer style.
    dataset = load_dataset(dataset_name, "title-question-pair")

    print(f"\nDataset structure: {dataset}")
    print(f"Train split size: {len(dataset['train']):,}")

    # Show a sample to verify data quality
    sample = dataset["train"][0]
    print(f"\nSample entry:")
    for key, value in sample.items():
        print(f"  {key}: {value[:200] if isinstance(value, str) else value}")

    # For faster iteration or memory constraints, we can use a subset of the data.
    # Random sampling ensures we get a diverse set of question styles.
    if sample_size and sample_size < len(dataset["train"]):
        print(f"\nUsing subset of {sample_size:,} examples")
        indices = random.sample(range(len(dataset["train"])), sample_size)
        dataset["train"] = dataset["train"].select(indices)

    # Split into train and validation sets
    # seed=42 ensures reproducibility - same split every time
    split_dataset = dataset["train"].train_test_split(test_size=val_split, seed=42)

    train_size = len(split_dataset["train"])
    val_size = len(split_dataset["test"])

    print(f"\nSplit into:")
    print(f"  Training: {train_size:,} examples ({(1 - val_split) * 100:.0f}%)")
    print(f"  Validation: {val_size:,} examples ({val_split * 100:.0f}%)")

    return split_dataset["train"], split_dataset["test"]


def format_question_style(example):
    """
    Format training examples for SFT (Supervised Fine-Tuning).

    FORMATTING STRATEGY:
    We create a simple prompt-completion pair where the model learns to generate
    Yahoo Answers-style questions when prompted with "Ask a question".

    The format uses the ChatML template (Qwen 2.5's native format):
    - <|im_start|>user / <|im_end|>: Marks the user's input/prompt
    - <|im_start|>assistant / <|im_end|>: Marks the assistant's response (what we're training it to generate)

    By combining title + question_body, we teach the model to generate complete
    Yahoo Answers questions with both a title and elaboration in one output.
    """
    title = example["title"]
    question_body = example["questions"]

    # Combine title and body to create the full Yahoo Answers-style question.
    # This teaches the model to generate both parts in sequence, which captures
    # the typical Yahoo Answers pattern of: "Title? Rambling elaboration..."
    # Using ChatML format which is Qwen 2.5's native chat template.
    text = f"""<|im_start|>user
Ask a question<|im_end|>
<|im_start|>assistant
{title} {question_body}<|im_end|>"""

    return {"text": text}


def set_random_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.

    REPRODUCIBILITY:
    Setting seeds ensures that:
    - Dataset sampling is the same every run
    - Model weight initialization is identical
    - Dropout patterns are reproducible
    - Data shuffling order is consistent

    This is crucial for:
    - Comparing different hyperparameters fairly
    - Debugging (can reproduce exact same behavior)
    - Scientific reproducibility of results
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # Sets seed for transformers, datasets, etc.


def main():
    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    # MODEL SELECTION:
    # Qwen 2.5-1.5B-Instruct is chosen because:
    # - Small enough to train quickly (~2-3 hours on RTX 4090)
    # - Large enough to learn stylistic patterns
    # - Already instruction-tuned, so it understands chat format
    # - Good quality-to-size ratio for this task
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    # RunPod stores data in /workspace which persists across sessions
    OUTPUT_DIR = "/workspace/yahoo-questions-model"
    DATASET_NAME = "sentence-transformers/yahoo-answers"

    # TRAINING HYPERPARAMETERS:
    # These control the training process and quality-speed tradeoff

    # SAMPLE_SIZE: Number of examples to train on (max: 660k in dataset)
    # 100k is a sweet spot: enough diversity for good style learning, fast enough
    # to train in a few hours. Increase for better quality, decrease for speed.
    SAMPLE_SIZE = 100000

    # BATCH_SIZE: Number of examples processed in parallel per GPU
    # Limited by VRAM - 8 works well for 24GB GPUs with this model size.
    # If you get OOM errors, reduce this. Larger batches = faster training.
    BATCH_SIZE = 8

    # GRADIENT_ACCUMULATION_STEPS: Accumulates gradients over N steps before updating
    # Effective batch size = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS = 32
    # This lets us use a large effective batch size without running out of VRAM.
    # Larger effective batches = more stable training.
    GRADIENT_ACCUMULATION_STEPS = 4

    # LEARNING_RATE: How much to update model weights each step
    # 2e-4 is standard for LoRA fine-tuning - high enough to learn quickly,
    # low enough to avoid catastrophic forgetting of the base model's capabilities.
    LEARNING_RATE = 2e-4

    # NUM_EPOCHS: How many times to iterate through the entire dataset
    # 3 is typical for fine-tuning - more risks overfitting, fewer may underfit
    NUM_EPOCHS = 3

    # SAVE_STEPS: Save a checkpoint every N training steps
    # 500 steps = ~30-45 min intervals, allows recovery if training crashes
    SAVE_STEPS = 500

    # LOGGING_STEPS: Print training metrics every N steps
    # Helps monitor if training is progressing normally
    LOGGING_STEPS = 10

    # WANDB_PROJECT: Weights & Biases project name for experiment tracking
    # Set to None to disable wandb logging
    WANDB_PROJECT = "yahoo-llm"

    # RANDOM_SEED: Seed for reproducibility
    # Using the same seed ensures identical results across runs
    RANDOM_SEED = 42

    # ============================================================================
    # REPRODUCIBILITY
    # ============================================================================

    # Set random seeds for all libraries to ensure reproducible results
    # This makes experiments comparable and results debuggable
    set_random_seed(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED}")

    # ============================================================================
    # WEIGHTS & BIASES INITIALIZATION
    # ============================================================================

    # Weights & Biases (wandb) is a powerful experiment tracking tool that:
    # - Logs all training metrics (loss, learning rate, etc.) in real-time
    # - Creates interactive plots and dashboards viewable from anywhere
    # - Tracks hyperparameters and system metrics (GPU usage, etc.)
    # - Allows comparison of multiple training runs
    # - Stores model checkpoints and artifacts
    #
    # To use wandb on RunPod:
    # 1. Get your API key from https://wandb.ai/settings
    # 2. On RunPod, run: wandb login <your-api-key>
    # 3. Training metrics will automatically sync to wandb.ai
    #
    # You can view your training dashboard at: https://wandb.ai/<username>/<project>

    if WANDB_PROJECT:
        wandb.init(
            entity="stead",
            project=WANDB_PROJECT,
            name=f"qwen-1.5b-{SAMPLE_SIZE // 1000}k-samples",  # Run name in wandb
            config={
                "model": MODEL_NAME,
                "sample_size": SAMPLE_SIZE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "max_seq_length": 256,
                "lora_r": 16,
                "lora_alpha": 32,
                "random_seed": RANDOM_SEED,
            },
        )
        print("\n✓ Weights & Biases initialized")
        print(
            f"View training at: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}"
        )

    # Print configuration for verification
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

    # ============================================================================
    # DATA PREPARATION
    # ============================================================================

    # Load and sample the dataset, returns both train and validation sets
    train_dataset, val_dataset = load_and_prepare_dataset(
        DATASET_NAME, sample_size=SAMPLE_SIZE
    )

    # Format each example into the chat template format expected by the model.
    # remove_columns ensures we only keep the 'text' field in the final dataset,
    # removing the original 'title' and 'questions' columns which are no longer needed.
    print("\nFormatting training dataset...")
    formatted_train = train_dataset.map(
        format_question_style,
        remove_columns=train_dataset.column_names,
        desc="Formatting train questions",
    )

    print("Formatting validation dataset...")
    formatted_val = val_dataset.map(
        format_question_style,
        remove_columns=val_dataset.column_names,
        desc="Formatting val questions",
    )

    # Verify the formatting looks correct
    print("\nFormatted sample:")
    print(formatted_train[0]["text"][:500])

    # ============================================================================
    # TOKENIZER SETUP
    # ============================================================================

    # Load the tokenizer that converts text to token IDs the model understands.
    # trust_remote_code=True allows loading custom tokenizer code from HuggingFace.
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokenizer.pad_token = "<|im_end|>"
    tokenizer.eos_token = "<|im_end|>"

    # Ensure the tokenizer has a padding token (needed for batching variable-length sequences).
    # Some models don't define this by default, so we use the EOS token as padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================================================
    # MODEL QUANTIZATION (QLoRA)
    # ============================================================================

    # QLoRA = Quantized LoRA, allows training large models on consumer GPUs.
    # This reduces the model from 16-bit to 4-bit precision, cutting memory usage by ~75%.
    # Despite the reduced precision, QLoRA maintains nearly the same training quality.
    print("\nConfiguring 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        # load_in_4bit: Loads model weights in 4-bit instead of 16-bit
        # Reduces VRAM from ~3GB to ~800MB for this 1.5B model
        load_in_4bit=True,
        # bnb_4bit_quant_type="nf4": Uses NormalFloat4, optimized for normally distributed weights
        # Better than standard 4-bit for neural network weights
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype: Actual computations still done in bfloat16 for stability
        # Only the stored weights are 4-bit, computation is upcasted
        bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_use_double_quant: Quantizes the quantization constants themselves
        # Saves an additional ~0.4 bits per parameter (minor memory savings)
        bnb_4bit_use_double_quant=True,
    )

    # Load the base model with quantization applied.
    # device_map="auto" automatically distributes the model across available GPUs.
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Prepare the quantized model for training by freezing base weights and
    # preparing LoRA layers for gradient computation.
    model = prepare_model_for_kbit_training(model)

    # ============================================================================
    # LoRA CONFIGURATION
    # ============================================================================

    # LoRA (Low-Rank Adaptation) trains small "adapter" matrices instead of all model weights.
    # This dramatically reduces trainable parameters from 1.5B to ~18M (1.2%).
    #
    # HOW IT WORKS:
    # Instead of updating weight matrix W, LoRA learns two small matrices A and B where:
    # W_new = W_frozen + (A × B)
    # A is (hidden_dim × r), B is (r × hidden_dim), where r << hidden_dim
    peft_config = LoraConfig(
        # r (rank): Dimension of the low-rank matrices
        # Higher r = more expressive but more parameters. 16 is a good balance.
        # r=16 means each adapter is hidden_dim × 16 instead of hidden_dim × hidden_dim
        r=16,
        # lora_alpha: Scaling factor for LoRA updates
        # The update is scaled by (alpha / r). With alpha=32, r=16, scaling = 2.0
        # Higher alpha makes LoRA updates more pronounced relative to frozen weights
        lora_alpha=32,
        # lora_dropout: Dropout probability in LoRA layers
        # 0.05 = 5% dropout, helps prevent overfitting in the adapter layers
        lora_dropout=0.05,
        # bias: Whether to train bias parameters
        # "none" means bias stays frozen, reducing parameters and typically works well
        bias="none",
        # task_type: Type of task we're training for
        # CAUSAL_LM = causal language modeling (predict next token)
        task_type="CAUSAL_LM",
        # target_modules: Which weight matrices to apply LoRA to
        # We target all attention and MLP projection layers in the transformer.
        # q_proj, k_proj, v_proj, o_proj = attention matrices (query, key, value, output)
        # gate_proj, up_proj, down_proj = MLP layers in transformer blocks
        # Targeting all of these ensures comprehensive adaptation while staying parameter-efficient
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

    # Apply LoRA to the model - this wraps the target modules with LoRA adapters
    # and freezes all other parameters. Only the adapter matrices will be trained.
    model = get_peft_model(model, peft_config)

    # Print how many parameters are actually being trained vs frozen
    # Should show ~18M trainable out of 1.5B total (~1.2%)
    model.print_trainable_parameters()

    # ============================================================================
    # TRAINING ARGUMENTS
    # ============================================================================

    # Configuration for the Hugging Face Trainer
    # ============================================================================
    # TRAINING ARGUMENTS (TRL SFTConfig)
    # ============================================================================

    # SFTConfig extends HF TrainingArguments with SFT-specific options like max_length,
    # dataset_text_field, eos_token, packing, etc.
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
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb" if WANDB_PROJECT else "none",
        gradient_checkpointing=True,
        # --- SFT-specific bits ---
        # Your Yahoo questions are short; 256 is a nice speed/coverage trade-off.
        max_seq_length=256,
        dataset_text_field="text",  # your formatted dataset column
        # For Qwen2.5 Instruct with ChatML, TRL docs recommend aligning eos_token
        # with the chat template end token:
        # https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer#instruction-tuning-example
        eos_token="<|im_end|>",
    )

    # ============================================================================
    # TRAINER SETUP
    # ============================================================================

    # SFTTrainer (Supervised Fine-Tuning Trainer) from the TRL library
    # Handles the training loop, including:
    # - Batching and data loading
    # - Forward/backward passes
    # - Optimizer steps
    # - Checkpoint saving
    # - Logging
    # - Validation evaluation (when eval_dataset is provided)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,  # Validation set for monitoring overfitting
        # processing_class: The new parameter name for tokenizer in recent TRL versions
        # Used for tokenizing the text data on-the-fly during training
        processing_class=tokenizer,
        # max_length: Maximum sequence length for training
        # 256 captures 99.44% of questions (avg: 66 tokens, 95th percentile: 147 tokens)
        # Only extreme outliers get truncated, while saving ~40-50% training time vs 512
    )

    # ============================================================================
    # TRAINING
    # ============================================================================

    # Start the training process
    # This will run for NUM_EPOCHS epochs over the SAMPLE_SIZE examples
    # Expected time: ~2-3 hours for 100k samples on RTX 4090
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # ============================================================================
    # SAVE MODEL
    # ============================================================================

    # Save the trained LoRA adapter weights
    # Only the adapter weights are saved (~70MB), not the full model (3GB)
    # To use the model later, you load the base model + these adapters
    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✓ Training complete! Model saved to: {OUTPUT_DIR}")
    print("\nThe model now talks like Yahoo Answers QUESTIONERS!")
    print(f"\nTo download from RunPod, use:")
    print(f"  runpodctl receive {OUTPUT_DIR}")

    # Finish wandb run
    if WANDB_PROJECT:
        wandb.finish()
        print("\n✓ Wandb run finished")


if __name__ == "__main__":
    main()
