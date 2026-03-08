from datasets import load_dataset
import re

def add_action_instruction(example):
    # Modify SYSTEM prompt to instruct LLM to output [Action] section.
    prompt = example["prompt"]

    # Insert new instruction inside the system message
    new_instruction = (
        "\nAdditionally, after the [Prediction & Analysis] section, "
        "add a new section titled [Action]:\n"
        "Decide one of: BUY, HOLD, or SELL based on your prediction and reasoning.\n"
        "Format:\n"
        "[Action]: BUY / HOLD / SELL\n"
    )

    # Insert before <</SYS>>
    modified_prompt = prompt.replace("<</SYS>>", new_instruction + "<</SYS>>")
    example["prompt"] = modified_prompt
    return example


def extract_prediction_percent(answer_text):
    # Extracts numeric prediction like 'Up by 3-4%' or 'Down 2%' → return % midpoint.
    text = answer_text.lower()

    # Pattern for ranges (e.g., 3-4%)
    range_pattern = r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%"
    m = re.search(range_pattern, text)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return (low + high) / 2  # midpoint

    # Pattern for single %
    single_pattern = r"(up|down)\s*by?\s*(\d+(?:\.\d+)?)\s*%"
    m = re.search(single_pattern, text)
    if m:
        direction, value = m.group(1), float(m.group(2))
        return value if direction == "up" else -value

    return None  # fallback


def action_from_prediction(pred_percent):
    # Assigns BUY/SELL/HOLD based on predicted % change.
    if pred_percent is None:
        return "HOLD"

    if pred_percent > 1.0:
        return "BUY"
    elif pred_percent < -1.0:
        return "SELL"
    else:
        return "HOLD"


def append_action_to_answer(example):
    # Compute action using real prediction, then append it to answer text.

    answer = example["answer"]
    pred_percent = extract_prediction_percent(answer)
    action = action_from_prediction(pred_percent)

    # Append section (for easy extraction during inference)
    answer_with_action = answer.strip() + f"\n\n[Action]: {action}"

    example["answer"] = answer_with_action
    example["action"] = action
    return example


# Load dataset and apply modifications
def transform_dataset(dataset):
    dataset_modified = dataset.map(add_action_instruction)
    dataset_modified = dataset_modified.map(append_action_to_answer)
    return dataset_modified


def get_dataset(dataset_name: str):
    train_dataset = transform_dataset(load_dataset(dataset_name, split=f"train"))
    test_dataset = transform_dataset(load_dataset(dataset_name, split=f"test"))
    return train_dataset, test_dataset

def tokenize_dataset(tokenizer, example, seq_len):
    prompts = example["prompt"]
    answers = example["answer"] + [tokenizer.eos_token]

    # Tokenize prompt and answer separately
    prompt_tokens = tokenizer(prompts, add_special_tokens=False, padding=False, truncation=True, max_length=seq_len, return_tensors=None)
    answer_tokens = tokenizer(answers, add_special_tokens=False, padding=False, truncation=True, max_length=seq_len, return_tensors=None)

    input_ids = []
    labels = []
    attention_masks = []

    for prompt_ids, answer_ids in zip(prompt_tokens["input_ids"], answer_tokens["input_ids"]):
        # Combine prompt + answer for input
        inputs = prompt_ids + answer_ids

        # Generate Labels (mask prompt with tokenizer.pad_token_id)
        label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + answer_ids.copy()

        # Truncate
        inputs = inputs[:seq_len]
        label_ids = label_ids[:seq_len]

        # Pad
        pad_len = seq_len - len(inputs)
        if pad_len > 0:
            inputs = inputs + [tokenizer.pad_token_id] * pad_len
            label_ids = label_ids + [tokenizer.pad_token_id] * pad_len

        input_ids.append(inputs)
        labels.append(label_ids)
        attention_masks.append([1 if id != tokenizer.pad_token_id else 0 for id in inputs])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_masks
    }


# get_dataset("FinGPT/fingpt-forecaster-dow30-202305-202405")