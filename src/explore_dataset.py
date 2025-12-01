#!/usr/bin/env python3
"""
Explore the Yahoo Answers dataset to understand its structure.
"""

from datasets import load_dataset


def main():
    print("Loading Yahoo Answers dataset...")
    dataset = load_dataset("sentence-transformers/yahoo-answers", "title-question-answer-pair")

    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print(f"\nDataset structure: {dataset}")
    print(f"\nColumn names: {dataset['train'].column_names}")
    print(f"\nTrain size: {len(dataset['train']):,}")

    print("\n" + "=" * 60)
    print("Sample Entries")
    print("=" * 60)

    for i in range(3):
        print(f"\n--- Sample {i + 1} ---")
        sample = dataset['train'][i]
        for key, value in sample.items():
            if isinstance(value, str):
                display_value = value[:300] + "..." if len(value) > 300 else value
            else:
                display_value = value
            print(f"\n{key}:")
            print(f"  {display_value}")
        print()

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    questions = [ex['question'] for ex in dataset['train'].select(range(min(1000, len(dataset['train']))))]
    answers = [ex['answer'] for ex in dataset['train'].select(range(min(1000, len(dataset['train']))))]

    avg_q_len = sum(len(q.split()) for q in questions) / len(questions)
    avg_a_len = sum(len(a.split()) for a in answers) / len(answers)

    print(f"\nAverage question length: {avg_q_len:.1f} words")
    print(f"Average answer length: {avg_a_len:.1f} words")


if __name__ == "__main__":
    main()
