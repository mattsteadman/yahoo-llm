# Yahoo Answers LLM Fine-tuning

Fine-tune a small language model to ASK QUESTIONS like Yahoo Answers users using Supervised Fine-Tuning (SFT).

## Overview

This project trains a language model to generate questions in the characteristic casual, rambling style of Yahoo Answers users. Instead of answering questions, the model learns to ask them.

**Key Features:**
- **Dataset**: `sentence-transformers/yahoo-answers` (title-question-pair config)
- **Base Model**: Qwen2.5-1.5B-Instruct (1.5B parameters)
- **Method**: QLoRA (4-bit quantization) + LoRA adapters (~1.2% parameters trained)
- **Platform**: Optimized for RunPod GPU training
- **Tracking**: Weights & Biases integration for experiment tracking
- **Validation**: Train/val split (90/10) for overfitting detection
- **Reproducibility**: Random seed support across all libraries

## Setup

1. Install dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```

Or with pip:
```bash
pip install torch transformers datasets trl peft accelerate sentencepiece protobuf wandb
```

## Project Structure

```
src/
├── train_question_style_runpod.py  # Main production training script (GPU)
├── train_question_style.py         # Local testing script (CPU/MPS)
├── test_question_model.py          # Test the trained model
├── analyze_sequence_lengths.py     # Analyze token lengths in dataset
├── explore_dataset.py              # Explore dataset structure
└── explore_all_configs.py          # Compare all dataset configs
```

## Usage

### 1. Explore the Dataset (Optional)

```bash
uv run src/explore_dataset.py
```

View token length distribution:
```bash
uv run src/analyze_sequence_lengths.py
```

### 2. Local Testing (Optional)

Test the training pipeline locally with a tiny sample:

```bash
uv run src/train_question_style.py
```

This runs 3 training steps on 10 samples to verify the code works.

### 3. Train on RunPod

The main training script is optimized for RunPod GPU instances:

```bash
# On RunPod, first login to wandb (optional but recommended)
wandb login YOUR_API_KEY

# Run training
python src/train_question_style_runpod.py
```

**Training Configuration:**
- 100,000 samples (90k train / 10k validation)
- 3 epochs
- Batch size: 8 (effective: 32 with gradient accumulation)
- Learning rate: 2e-4
- LoRA rank: 16
- max_seq_length: 256 tokens (captures 99.44% of questions)
- 4-bit quantization (QLoRA)
- Random seed: 42

**Expected Training Time:**
- RTX 4090: ~2-3 hours
- RTX 3090: ~3-4 hours
- A100: ~1-2 hours

**Cost Estimate:**
- ~$1-2 on RunPod (depending on GPU tier)

### 4. Test the Model

```bash
uv run src/test_question_model.py
```

Generates 5 Yahoo Answers-style questions to verify the model learned the style.

## Example Output

Input prompt: `Ask a question`

Model output:
```
How do I make my computer faster? Like, it's been running super slow lately
and I don't know what to do. I tried deleting some files but it didn't help.
Is there some program I need to download or something?
```

## Technical Details

### Why QLoRA?
- Reduces VRAM usage by ~75% (3GB → 800MB for model weights)
- Maintains training quality despite 4-bit quantization
- Enables training on consumer GPUs

### Why LoRA?
- Trains only ~18M parameters instead of 1.5B (~1.2%)
- Much faster training and smaller checkpoint files (~70MB vs 3GB)
- Reduces risk of catastrophic forgetting

### ChatML Format
Qwen 2.5 uses ChatML template:
```
<|im_start|>user
Ask a question<|im_end|>
<|im_start|>assistant
{title} {question_body}<|im_end|>
```

### Validation Split
- 10% of data reserved for validation
- Monitors overfitting during training
- Best checkpoint selected based on validation loss

### Weights & Biases Integration
- Real-time training metrics and plots
- Hyperparameter tracking
- System metrics (GPU usage, etc.)
- Run comparison and experiment tracking
- View at: `https://wandb.ai/<username>/yahoo-llm`

### Sequence Length Optimization
Analysis showed:
- Mean: 66 tokens
- 95th percentile: 147 tokens
- 99.44% fit in 256 tokens

Using max_seq_length=256 instead of 512 provides ~40-50% faster training while only truncating extreme outliers.

## Customization

Edit `src/train_question_style_runpod.py` configuration section to adjust:
- `SAMPLE_SIZE`: Number of training examples (max: 660k)
- `BATCH_SIZE`: Batch size per GPU (reduce if OOM errors)
- `NUM_EPOCHS`: Training epochs (3 is typical)
- `LEARNING_RATE`: Learning rate (2e-4 standard for LoRA)
- `WANDB_PROJECT`: Project name or None to disable
- `RANDOM_SEED`: Change for different random initialization

## Hardware Requirements

**Minimum:**
- GPU with 8GB VRAM (QLoRA enables training on RTX 3060 Ti, 3070, etc.)
- 16GB system RAM

**Recommended:**
- GPU with 16-24GB VRAM (RTX 3090/4090, A5000, etc.)
- 32GB system RAM

## RunPod Setup

1. Create a RunPod GPU instance (RTX 3090 or better)
2. Clone this repository
3. Install dependencies: `pip install -r requirements.txt` or use the project.toml
4. Login to wandb (optional): `wandb login YOUR_API_KEY`
5. Run training: `python src/train_question_style_runpod.py`
6. Download model: `runpodctl receive /workspace/yahoo-questions-model`

## Model Inference

Load the trained model with adapters:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
tokenizer = AutoTokenizer.from_pretrained("./yahoo-questions-model")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "./yahoo-questions-model")

# Generate a question
prompt = "<|im_start|>user\nAsk a question<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## License

This project uses the sentence-transformers/yahoo-answers dataset and Qwen2.5 model. Please refer to their respective licenses.