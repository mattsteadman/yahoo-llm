# Running Yahoo Answers LLM Training on RunPod

## Quick Start

### 1. Create a RunPod Instance

1. Go to [RunPod.io](https://www.runpod.io/)
2. Select "GPU Pods"
3. Choose a GPU (recommended: RTX 4090, A100, or similar)
4. Select a PyTorch template (or use the default)
5. Deploy your pod

### 2. Upload Your Code

**Option A: Git Clone (Recommended)**
```bash
cd /workspace
git clone <your-repo-url>
cd yahoo-llm
```

**Option B: Upload Files Manually**
Upload these files to `/workspace/`:
- `train_question_style_runpod.py`
- `runpod_setup.sh` (optional)

### 3. Install Dependencies

```bash
pip install torch transformers datasets trl peft accelerate bitsandbytes sentencepiece protobuf
```

Or use the setup script:
```bash
chmod +x runpod_setup.sh
./runpod_setup.sh
```

### 4. Start Training

```bash
python train_question_style_runpod.py
```

## Configuration

Edit `train_question_style_runpod.py` to adjust:

```python
SAMPLE_SIZE = 100000  # Number of training examples (max: 660k)
BATCH_SIZE = 8        # Increase for larger GPUs (16GB+ VRAM)
NUM_EPOCHS = 3        # Training epochs
```

### GPU Memory Requirements

- **RTX 3090/4090 (24GB)**: `BATCH_SIZE = 8-12`
- **A100 (40GB)**: `BATCH_SIZE = 16-24`
- **A100 (80GB)**: `BATCH_SIZE = 32+`

If you get OOM (Out of Memory) errors, reduce `BATCH_SIZE`.

## Training Time Estimates

With 100k samples:
- **RTX 4090**: ~2-3 hours
- **A100 40GB**: ~1.5-2 hours
- **A100 80GB**: ~1-1.5 hours

## Download Your Model

After training completes, download the model:

**Option 1: RunPod CLI**
```bash
runpodctl receive /workspace/yahoo-questions-model
```

**Option 2: Direct Download**
1. Open the RunPod file browser
2. Navigate to `/workspace/yahoo-questions-model/`
3. Download the files:
   - `adapter_model.safetensors` (LoRA weights, ~70MB)
   - `adapter_config.json`
   - All tokenizer files

## Testing Your Model

Create `test_model.py` on RunPod:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("/workspace/yahoo-questions-model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "/workspace/yahoo-questions-model")

prompt = "<|user|>\nAsk a question\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Run it:
```bash
python test_model.py
```

## Monitoring Training

Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

Check training logs in real-time:
```bash
tail -f /workspace/yahoo-questions-model/runs/*/events.*
```

## Costs

Typical cost with RunPod:
- RTX 4090: ~$0.40-0.60/hour
- A100 40GB: ~$1.00-1.50/hour
- **Total for 100k samples**: $1-3 depending on GPU

## Tips

1. **Start small**: Test with `SAMPLE_SIZE = 1000` first to verify everything works
2. **Save money**: Use Spot instances (cheaper but can be interrupted)
3. **Checkpoints**: Training saves checkpoints every 500 steps - you can resume if interrupted
4. **Full dataset**: For maximum quality, use `SAMPLE_SIZE = None` to train on all 660k examples (~6-8 hours)

## Troubleshooting

**Out of Memory (OOM)**
```python
BATCH_SIZE = 4  # Reduce batch size
GRADIENT_ACCUMULATION_STEPS = 8  # Keep effective batch size
```

**Slow training**
- Increase `BATCH_SIZE` if you have spare GPU memory
- Enable `gradient_checkpointing=False` if you have enough VRAM

**Dataset loading fails**
- Check internet connection
- Try: `export HF_DATASETS_OFFLINE=0`

## After Training

Use the model locally by:
1. Download the adapter files from RunPod
2. Place in `./yahoo-questions-model/`
3. Run `test_question_model.py` on your local machine