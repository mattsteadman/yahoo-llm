# Yahoo Answers LLM Fine-tuning

Fine-tune a small language model (<=3B parameters) to talk like Yahoo Answers users using Supervised Fine-Tuning (SFT).

## Overview

This project uses:
- **Dataset**: `sentence-transformers/yahoo-answers` from Hugging Face
- **Base Model**: Qwen2.5-1.5B-Instruct (1.5B parameters)
- **Method**: SFT with QLoRA for efficient training
- **Framework**: Hugging Face Transformers + TRL + PEFT

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Explore the Dataset (Optional)

```bash
python explore_dataset.py
```

### 2. Train the Model

```bash
python train_yahoo_llm.py
```

This will:
- Load 50,000 examples from the Yahoo Answers dataset
- Format them for conversational SFT
- Fine-tune Qwen2.5-1.5B-Instruct using QLoRA
- Save the model to `./yahoo-answers-model/`

Training configuration:
- 3 epochs
- Batch size: 4 (with gradient accumulation)
- Learning rate: 2e-4
- LoRA rank: 16
- 4-bit quantization for efficient training

### 3. Test the Model

```bash
python inference.py
```

Ask questions and get Yahoo Answers style responses!

## Example

```
Question: How do I make my computer faster?

Answer: [Yahoo Answers style response based on the training data]
```

## Customization

Edit `train_yahoo_llm.py` to:
- Change the base model (must be <=3B params)
- Adjust training hyperparameters
- Modify the dataset size
- Change the prompt format

## Hardware Requirements

- GPU with at least 8GB VRAM (4-bit quantization used)
- ~16GB RAM recommended
- Training time: ~1-3 hours depending on GPU
