"""
Data preparation and evaluation harness for ClawRouter autoresearch.
Downloads MMLU dataset, maps subjects to domains and complexity tiers.

Usage:
    python prepare.py              # download and prepare data
    python prepare.py --stats      # show dataset statistics

Data is stored in ~/.cache/clawrouter-autoresearch/.
"""

import os
import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "clawrouter-autoresearch")
DATA_FILE = os.path.join(CACHE_DIR, "mmlu.json")

# MMLU subjects → domain mapping
SUBJECT_TO_DOMAIN = {
    # STEM
    "abstract_algebra": "stem",
    "astronomy": "stem",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_mathematics": "stem",
    "high_school_physics": "stem",
    "high_school_statistics": "stem",
    "machine_learning": "stem",
    # Humanities
    "formal_logic": "humanities",
    "high_school_european_history": "humanities",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_law": "humanities",
    "world_religions": "humanities",
    # Social Sciences
    "econometrics": "social_sciences",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences",
    "human_sexuality": "social_sciences",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    # Other (medical, business, misc)
    "anatomy": "other",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_medicine": "other",
    "global_facts": "other",
    "human_aging": "other",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "nutrition": "other",
    "professional_accounting": "other",
    "professional_medicine": "other",
    "virology": "other",
}

# Complexity tiers based on education level in subject name
# Maps to ClawRouter tiers: SIMPLE, MEDIUM, COMPLEX, REASONING
SUBJECT_TO_TIER = {}
for subj in SUBJECT_TO_DOMAIN:
    if subj.startswith("elementary_") or subj.startswith("high_school_"):
        SUBJECT_TO_TIER[subj] = "SIMPLE" if subj.startswith("elementary_") else "MEDIUM"
    elif subj.startswith("college_"):
        SUBJECT_TO_TIER[subj] = "COMPLEX"
    elif subj.startswith("professional_"):
        SUBJECT_TO_TIER[subj] = "REASONING"
    else:
        # Default: use domain heuristic
        SUBJECT_TO_TIER[subj] = "COMPLEX"

ALL_DOMAINS = sorted(set(SUBJECT_TO_DOMAIN.values()))
ALL_TIERS = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_mmlu():
    """Download MMLU dataset from HuggingFace and save as JSON."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        print(f"Data already exists at {DATA_FILE}")
        return

    print("Downloading MMLU dataset from HuggingFace...")
    from datasets import load_dataset

    records = []
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for split in ["test", "validation"]:
        ds = load_dataset("cais/mmlu", "all", split=split)
        for row in ds:
            subject = row["subject"]
            records.append({
                "question": row["question"],
                "choices": row["choices"],
                "answer": answer_map[row["answer"]],
                "subject": subject,
                "domain": SUBJECT_TO_DOMAIN.get(subject, "other"),
                "tier": SUBJECT_TO_TIER.get(subject, "COMPLEX"),
                "split": split,
            })

    with open(DATA_FILE, "w") as f:
        json.dump(records, f)

    print(f"Saved {len(records)} examples to {DATA_FILE}")


def load_data(split=None):
    """Load MMLU data. Optionally filter by split ('test' or 'validation')."""
    with open(DATA_FILE) as f:
        data = json.load(f)
    if split:
        data = [r for r in data if r["split"] == split]
    return data


def load_val():
    """Load validation split."""
    return load_data("validation")


def load_test():
    """Load test split."""
    return load_data("test")


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(score_fn, data=None):
    """
    Evaluate a scoring function against MMLU validation data.

    score_fn(question, choices) -> dict with keys:
        - "tier": predicted tier (SIMPLE/MEDIUM/COMPLEX/REASONING)
        - "domain": predicted domain (stem/humanities/social_sciences/other)

    Returns dict with:
        - tier_accuracy: fraction of correct tier predictions
        - domain_accuracy: fraction of correct domain predictions
        - combined_score: weighted combination (0.5 * tier + 0.5 * domain)
        - tier_confusion: dict of {true_tier: {predicted_tier: count}}
        - domain_confusion: dict of {true_domain: {predicted_domain: count}}
        - total: number of examples evaluated
    """
    if data is None:
        data = load_val()

    tier_correct = 0
    domain_correct = 0
    tier_confusion = {t: {t2: 0 for t2 in ALL_TIERS} for t in ALL_TIERS}
    domain_confusion = {d: {d2: 0 for d2 in ALL_DOMAINS} for d in ALL_DOMAINS}
    total = len(data)

    for row in data:
        result = score_fn(row["question"], row["choices"])
        pred_tier = result["tier"]
        pred_domain = result["domain"]
        true_tier = row["tier"]
        true_domain = row["domain"]

        if pred_tier == true_tier:
            tier_correct += 1
        if pred_domain == true_domain:
            domain_correct += 1

        if true_tier in tier_confusion and pred_tier in tier_confusion[true_tier]:
            tier_confusion[true_tier][pred_tier] += 1
        if true_domain in domain_confusion and pred_domain in domain_confusion[true_domain]:
            domain_confusion[true_domain][pred_domain] += 1

    tier_acc = tier_correct / total if total else 0
    domain_acc = domain_correct / total if total else 0

    return {
        "tier_accuracy": tier_acc,
        "domain_accuracy": domain_acc,
        "combined_score": 0.5 * tier_acc + 0.5 * domain_acc,
        "tier_confusion": tier_confusion,
        "domain_confusion": domain_confusion,
        "total": total,
    }


def print_results(results):
    """Print evaluation results in the standard output format."""
    print("---")
    print(f"tier_accuracy:    {results['tier_accuracy']:.6f}")
    print(f"domain_accuracy:  {results['domain_accuracy']:.6f}")
    print(f"combined_score:   {results['combined_score']:.6f}")
    print(f"total_examples:   {results['total']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show_stats():
    """Print dataset statistics."""
    data = load_data()
    val = [r for r in data if r["split"] == "validation"]
    test = [r for r in data if r["split"] == "test"]

    print(f"Total examples: {len(data)}")
    print(f"  validation: {len(val)}")
    print(f"  test: {len(test)}")
    print()

    print("Subjects:", len(set(r["subject"] for r in data)))
    print()

    print("Domain distribution (validation):")
    for domain in ALL_DOMAINS:
        count = sum(1 for r in val if r["domain"] == domain)
        print(f"  {domain:20s}: {count}")
    print()

    print("Tier distribution (validation):")
    for tier in ALL_TIERS:
        count = sum(1 for r in val if r["tier"] == tier)
        print(f"  {tier:20s}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MMLU data for ClawRouter autoresearch")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    args = parser.parse_args()

    download_mmlu()
    print()

    if args.stats:
        show_stats()
    else:
        print("Done! Ready to train.")
        print(f"Data at: {DATA_FILE}")
