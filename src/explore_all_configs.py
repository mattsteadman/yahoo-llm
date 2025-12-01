#!/usr/bin/env python3
"""Explore all Yahoo Answers dataset configurations"""

from datasets import load_dataset

configs = ['question-answer-pair', 'title-answer-pair', 'title-question-answer-pair', 'title-question-pair']

for config_name in configs:
    print("\n" + "=" * 60)
    print(f"Config: {config_name}")
    print("=" * 60)

    dataset = load_dataset("sentence-transformers/yahoo-answers", config_name)
    print(f"\nColumns: {dataset['train'].column_names}")
    print(f"Size: {len(dataset['train']):,}")

    print("\nSample:")
    sample = dataset['train'][5]
    for key, value in sample.items():
        if isinstance(value, str):
            display_value = value[:200] + "..." if len(value) > 200 else value
        else:
            display_value = value
        print(f"\n  {key}:")
        print(f"    {display_value}")