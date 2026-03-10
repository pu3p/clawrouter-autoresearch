"""Prepare MMLU dataset for domain+complexity classifier training.

auxiliary_train has no subject labels, so we split the labeled test+validation
data into train/val/test (80/10/10).
"""

import json
import os
import datasets
from datasets import load_dataset, DatasetDict, concatenate_datasets

RESULTS_DIR = "/workspace/results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "mmlu_processed")


def format_prompt(example):
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

    # Combine all labeled splits (test + validation + dev)
    labeled = concatenate_datasets([
        ds["test"], ds["validation"], ds["dev"]
    ])
    print(f"Total labeled examples: {len(labeled)}")

    # Get 57 real subjects (exclude empty)
    all_subjects = sorted(
        s for s in set(labeled["subject"]) if s
    )
    print(f"Found {len(all_subjects)} subjects")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(
        os.path.join(RESULTS_DIR, "domain_labels.json"), "w"
    ) as f:
        json.dump(all_subjects, f, indent=2)

    domain2id = {s: i for i, s in enumerate(all_subjects)}

    def process(example):
        example = format_prompt(example)
        example["domain_id"] = domain2id[example["domain"]]
        return example

    labeled = labeled.map(process)

    keep_cols = ["text", "domain", "domain_id"]
    drop = [
        c for c in labeled.column_names if c not in keep_cols
    ]
    labeled = labeled.remove_columns(drop)

    # Stratified split: 80/10/10
    from datasets import ClassLabel
    labeled = labeled.cast_column(
        "domain_id",
        ClassLabel(num_classes=len(all_subjects))
    )
    split1 = labeled.train_test_split(
        test_size=0.2, seed=42,
        stratify_by_column="domain_id"
    )
    split2 = split1["test"].train_test_split(
        test_size=0.5, seed=42,
        stratify_by_column="domain_id"
    )
    # Cast back to int
    for k in ["train"]:
        split1[k] = split1[k].cast_column(
            "domain_id", datasets.Value("int64")
        )
    for k in ["train", "test"]:
        split2[k] = split2[k].cast_column(
            "domain_id", datasets.Value("int64")
        )

    splits = DatasetDict({
        "train": split1["train"],
        "validation": split2["train"],
        "test": split2["test"],
    })

    for name, split in splits.items():
        print(f"{name}: {len(split)} examples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    splits.save_to_disk(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
