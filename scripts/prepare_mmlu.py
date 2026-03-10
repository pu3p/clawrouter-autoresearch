"""Prepare MMLU dataset for domain+complexity classifier training."""

import json
import os
from datasets import load_dataset, DatasetDict

RESULTS_DIR = "/workspace/results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "mmlu_processed")


def format_prompt(example):
    """Combine question + choices into a natural prompt."""
    choices = example["choices"]
    letters = ["A", "B", "C", "D"]
    choice_str = "\n".join(
        f"{letters[i]}. {c}" for i, c in enumerate(choices)
    )
    example["text"] = f"{example['question']}\n{choice_str}"
    example["domain"] = example["subject"]
    return example


def main():
    print("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all")

    # Get all 57 subjects
    all_subjects = sorted(
        set(ds["test"]["subject"])
        | set(ds["validation"]["subject"])
        | set(ds["auxiliary_train"]["subject"])
    )
    print(f"Found {len(all_subjects)} subjects")

    # Save domain labels
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "domain_labels.json"), "w") as f:
        json.dump(all_subjects, f, indent=2)
    print(f"Saved {len(all_subjects)} domain labels")

    # Build label maps
    domain2id = {s: i for i, s in enumerate(all_subjects)}

    def process(example):
        example = format_prompt(example)
        example["domain_id"] = domain2id[example["domain"]]
        return example

    # Use auxiliary_train as train, validation as val, test as test
    splits = DatasetDict({
        "train": ds["auxiliary_train"].map(process),
        "validation": ds["validation"].map(process),
        "test": ds["test"].map(process),
    })

    # Keep only needed columns
    keep_cols = ["text", "domain", "domain_id"]
    for split_name in splits:
        drop = [
            c for c in splits[split_name].column_names
            if c not in keep_cols
        ]
        splits[split_name] = splits[split_name].remove_columns(drop)

    for name, split in splits.items():
        print(f"{name}: {len(split)} examples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    splits.save_to_disk(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
