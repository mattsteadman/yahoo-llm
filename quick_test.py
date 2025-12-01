#!/usr/bin/env python3
"""Quick test of the fine-tuned model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("./yahoo-answers-model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "./yahoo-answers-model")

question = "How do I make my computer faster?"
prompt = f"""<|user|>
{question}
<|assistant|>
"""

print(f"\nQuestion: {question}\n")
print("Generating answer...")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split("<|assistant|>")[-1].strip()

print(f"Answer: {answer}\n")
print("✓ Model test successful!")
