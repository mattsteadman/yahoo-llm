# Yahoo Answers Question-Style LLM - RunPod Quick Start

Train an LLM to talk like Yahoo Answers users (question style)!

## Files for RunPod

Upload these to your RunPod instance:
- ✅ `train_question_style_runpod.py` - Main training script
- ✅ `RUNPOD_INSTRUCTIONS.md` - Full instructions
- ⚙️ `runpod_setup.sh` - Optional setup script

## Super Quick Start

1. **Launch RunPod Pod** with GPU (RTX 4090 or A100 recommended)

2. **SSH into your pod** and run:
```bash
cd /workspace
pip install torch transformers datasets trl peft accelerate bitsandbytes sentencepiece protobuf
```

3. **Upload** `train_question_style_runpod.py` to `/workspace/`

4. **Start training**:
```bash
python train_question_style_runpod.py
```

5. **Wait** ~2-3 hours (for 100k samples)

6. **Download** the model from `/workspace/yahoo-questions-model/`

## What It Does

Trains Qwen 2.5-1.5B to generate Yahoo Answers-style questions:
- Casual, rambling style
- Typos and informal language
- Multiple elaborations
- Authentic Yahoo Answers vibe

## Configuration

Edit these in `train_question_style_runpod.py`:

```python
SAMPLE_SIZE = 100000    # How many examples (max: 660k)
BATCH_SIZE = 8          # Adjust for your GPU
NUM_EPOCHS = 3          # Training epochs
```

## Cost Estimate

- RTX 4090: $0.50/hr × 2.5 hrs = **~$1.25**
- A100: $1.25/hr × 2 hrs = **~$2.50**

## Need Help?

See `RUNPOD_INSTRUCTIONS.md` for detailed guide.

---

**Local Testing?** Use `train_question_style.py` (the one without `_runpod`)