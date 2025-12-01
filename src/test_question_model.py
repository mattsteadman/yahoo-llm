#!/usr/bin/env python3
"""Test the Yahoo Answers QUESTION-style model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("./yahoo-questions-model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "./yahoo-questions-model")

print("\n✓ Model loaded!")
print("\nGenerating Yahoo Answers-style questions...\n")

# Generate several questions
for i in range(5):
    prompt = "<|im_start|>user\nAsk a question<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.9,  # Higher temperature for more variety
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the generated question (everything after "Ask a question")
    if "Ask a question" in response:
        question = response.split("Ask a question")[-1].strip()
    else:
        question = response.strip()

    print(f"{i+1}. {question}\n")

print("\n✓ Your model now talks like a Yahoo Answers user!")