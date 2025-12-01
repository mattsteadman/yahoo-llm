#!/usr/bin/env python3
"""
Inference script for the Yahoo Answers fine-tuned model.
Ask questions and get answers in Yahoo Answers style!
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path="./yahoo-answers-model", base_model="Qwen/Qwen2.5-1.5B-Instruct"):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading base model: {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA weights from {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()

    return model, tokenizer


def generate_answer(model, tokenizer, question, max_length=512, temperature=0.7):
    """Generate a Yahoo Answers style response to a question."""
    prompt = f"""<|user|>
{question}
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the answer part
    if "<|assistant|>" in response:
        answer = response.split("<|assistant|>")[-1].strip()
    else:
        answer = response

    return answer


def main():
    print("=" * 60)
    print("Yahoo Answers Style LLM - Inference")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model()

    print("\n✓ Model loaded successfully!")
    print("\nAsk questions in Yahoo Answers style (type 'quit' to exit):\n")

    # Example questions
    example_questions = [
        "How do I make my computer faster?",
        "What's the best way to learn programming?",
        "Why is the sky blue?",
        "How can I make money online?",
    ]

    print("Example questions you can try:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        question = input("Question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        print("\nGenerating answer...")
        answer = generate_answer(model, tokenizer, question)
        print(f"\nAnswer: {answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()