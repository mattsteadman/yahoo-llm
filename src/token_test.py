from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "sentence-transformers/yahoo-answers"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load dataset
dataset = load_dataset(DATASET_NAME, "title-question-pair")

# # Format function (your current broken one, for demonstration)
# def format_question_style(example):
#     text = f"""<|user|>
# Ask a question
# <|assistant|>
# {example['title']} {example['questions']}"""
#     return {"text": text}


def format_question_style(example):
    """
    Format training examples for SFT (Supervised Fine-Tuning).

    FORMATTING STRATEGY:
    We create a simple prompt-completion pair where the model learns to generate
    Yahoo Answers-style questions when prompted with "Ask a question".

    The format uses the ChatML template (Qwen 2.5's native format):
    - <|im_start|>user / <|im_end|>: Marks the user's input/prompt
    - <|im_start|>assistant / <|im_end|>: Marks the assistant's response (what we're training it to generate)

    By combining title + question_body, we teach the model to generate complete
    Yahoo Answers questions with both a title and elaboration in one output.
    """
    title = example["title"]
    question_body = example["questions"]

    # Combine title and body to create the full Yahoo Answers-style question.
    # This teaches the model to generate both parts in sequence, which captures
    # the typical Yahoo Answers pattern of: "Title? Rambling elaboration..."
    # Using ChatML format which is Qwen 2.5's native chat template.
    text = f"""<|im_start|>user
Ask a question<|im_end|>
<|im_start|>assistant
{title} {question_body}<|im_end|>"""

    return {"text": text}


# Create formatted dataset
formatted_dataset = (
    dataset["train"]
    .select(range(100))
    .map(
        format_question_style,
        remove_columns=dataset["train"].column_names,
    )
)

# Inspect
sample = formatted_dataset[0]["text"]
tokens = tokenizer(sample, add_special_tokens=False)

print("=== Raw text ===")
print(sample[:500])

print("\n=== First 30 tokens ===")
for i, tid in enumerate(tokens["input_ids"][:30]):
    decoded = tokenizer.decode([tid])
    print(f"{i:3d}: {tid:6d} -> {repr(decoded)}")
