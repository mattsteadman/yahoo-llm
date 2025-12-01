#!/usr/bin/env python3
"""
Analyze the actual token lengths of formatted Yahoo Answers questions
to determine optimal max_seq_length for training.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

def format_question_style(example):
    """Same formatting as in training script"""
    title = example['title']
    question_body = example['questions']

    text = f"""<|im_start|>user
Ask a question<|im_end|>
<|im_start|>assistant
{title} {question_body}<|im_end|>"""

    return {"text": text}


def main():
    print("Loading dataset...")
    dataset = load_dataset("sentence-transformers/yahoo-answers", "title-question-pair")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

    # Sample 5000 examples for analysis (enough for good statistics)
    print("\nAnalyzing 5000 examples...")
    sample_size = min(5000, len(dataset['train']))
    sample = dataset['train'].select(range(sample_size))

    lengths = []
    for example in sample:
        formatted = format_question_style(example)
        tokens = tokenizer.encode(formatted['text'])
        lengths.append(len(tokens))

    lengths = np.array(lengths)

    print("\n" + "=" * 60)
    print("TOKEN LENGTH STATISTICS")
    print("=" * 60)
    print(f"Mean:       {lengths.mean():.1f} tokens")
    print(f"Median:     {np.median(lengths):.1f} tokens")
    print(f"Std Dev:    {lengths.std():.1f} tokens")
    print(f"Min:        {lengths.min()} tokens")
    print(f"Max:        {lengths.max()} tokens")
    print()
    print("Percentiles:")
    print(f"  50th (median): {np.percentile(lengths, 50):.0f} tokens")
    print(f"  75th:          {np.percentile(lengths, 75):.0f} tokens")
    print(f"  90th:          {np.percentile(lengths, 90):.0f} tokens")
    print(f"  95th:          {np.percentile(lengths, 95):.0f} tokens")
    print(f"  99th:          {np.percentile(lengths, 99):.0f} tokens")

    print("\n" + "=" * 60)
    print("COVERAGE AT DIFFERENT max_seq_length VALUES")
    print("=" * 60)

    for max_len in [256, 384, 512, 768, 1024]:
        coverage = (lengths <= max_len).sum() / len(lengths) * 100
        truncated = (lengths > max_len).sum()
        avg_tokens_lost = lengths[lengths > max_len].mean() - max_len if truncated > 0 else 0

        print(f"\nmax_seq_length = {max_len}:")
        print(f"  Coverage:      {coverage:.2f}% of examples fit completely")
        print(f"  Truncated:     {truncated} examples ({100-coverage:.2f}%)")
        if truncated > 0:
            print(f"  Avg loss:      {avg_tokens_lost:.1f} tokens per truncated example")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)

    if p95 <= 512:
        print(f"✓ 512 is EXCELLENT - captures 95th percentile ({p95:.0f} tokens)")
    elif p95 <= 768:
        print(f"→ Consider 768 - 512 only captures to {np.percentile(lengths, np.searchsorted(np.percentile(lengths, range(101)), 512)):.0f}th percentile")
        print(f"  95th percentile is at {p95:.0f} tokens")
    else:
        print(f"→ Consider 1024 - 512 truncates significant content")
        print(f"  95th percentile is at {p95:.0f} tokens")

    print(f"\nNote: 99th percentile is at {p99:.0f} tokens")
    print("Very long questions (>99th percentile) are often lower quality anyway.")


if __name__ == "__main__":
    main()