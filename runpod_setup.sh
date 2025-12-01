#!/bin/bash
# RunPod setup script for Yahoo Answers LLM training

echo "=========================================="
echo "Yahoo Answers Question-Style LLM Training"
echo "=========================================="

# Update system
apt-get update -qq

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch transformers datasets trl peft accelerate bitsandbytes sentencepiece protobuf

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi

echo ""
echo "✓ Setup complete!"
echo ""
echo "To start training, run:"
echo "  python train_question_style_runpod.py"