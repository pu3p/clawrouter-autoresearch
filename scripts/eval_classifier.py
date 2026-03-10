"""Evaluate the trained domain+complexity classifier on MMLU test set."""

import json
import os
import time
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = "/workspace/results"
DATA_DIR = os.path.join(RESULTS_DIR, "mmlu_processed")
CKPT_DIR = os.path.join(RESULTS_DIR, "classifier_checkpoints/best")
MAX_LEN = 256
BATCH_SIZE = 32


class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone_dir, num_domains, num_complexity):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_dir, use_safetensors=True,
            dtype=torch.float32,
        )
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.domain_head = nn.Linear(h, num_domains)
        self.complexity_head = nn.Linear(h, num_complexity)

    def forward(self, input_ids, attention_mask=None):
        out = self.backbone(
            input_ids, attention_mask=attention_mask
        )
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.domain_head(cls), self.complexity_head(cls)


def load_model(device):
    heads = torch.load(
        os.path.join(CKPT_DIR, "heads.pt"),
        map_location=device, weights_only=True,
    )
    model = MultiHeadClassifier(
        CKPT_DIR, heads["num_domains"], heads["num_complexity"]
    ).to(device)
    model.domain_head.load_state_dict(heads["domain_head"])
    model.complexity_head.load_state_dict(
        heads["complexity_head"]
    )
    model.eval()
    return model


def collate_fn(batch, tokenizer):
    texts = [b["text"] for b in batch]
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=MAX_LEN, padding=True
    )
    domain_ids = torch.tensor([b["domain_id"] for b in batch])
    complexity_ids = torch.tensor(
        [b["complexity_id"] for b in batch]
    )
    domains = [b["domain"] for b in batch]
    return enc, domain_ids, complexity_ids, domains


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
    model = load_model(device)
    ds = load_from_disk(DATA_DIR)

    with open(
        os.path.join(RESULTS_DIR, "domain_labels.json")
    ) as f:
        domain_labels = json.load(f)

    test_loader = DataLoader(
        ds["test"], batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=2,
    )

    # Collect predictions
    all_d_preds, all_d_true = [], []
    all_c_preds, all_c_true = [], []
    all_domains = []

    t0 = time.time()
    with torch.no_grad():
        for enc, d_ids, c_ids, domains in test_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            d_logits, c_logits = model(
                enc["input_ids"], enc.get("attention_mask")
            )
            all_d_preds.extend(
                d_logits.argmax(1).cpu().tolist()
            )
            all_d_true.extend(d_ids.tolist())
            all_c_preds.extend(
                c_logits.argmax(1).cpu().tolist()
            )
            all_c_true.extend(c_ids.tolist())
            all_domains.extend(domains)
    inference_time = time.time() - t0

    all_d_preds = np.array(all_d_preds)
    all_d_true = np.array(all_d_true)
    all_c_preds = np.array(all_c_preds)
    all_c_true = np.array(all_c_true)

    # Overall accuracy
    d_acc = (all_d_preds == all_d_true).mean()
    c_acc = (all_c_preds == all_c_true).mean()
    print(f"Domain accuracy: {d_acc:.4f}")
    print(f"Complexity accuracy: {c_acc:.4f}")
    print(
        f"Inference: {inference_time:.1f}s for "
        f"{len(all_d_true)} examples "
        f"({len(all_d_true)/inference_time:.0f} ex/s)"
    )

    # Per-domain accuracy
    per_domain = {}
    for i, name in enumerate(domain_labels):
        mask = all_d_true == i
        if mask.sum() > 0:
            acc = (all_d_preds[mask] == i).mean()
            per_domain[name] = {
                "accuracy": round(float(acc), 4),
                "count": int(mask.sum()),
            }

    # Per-complexity accuracy
    complexity_names = ["simple", "moderate", "complex", "expert"]
    per_complexity = {}
    for i, name in enumerate(complexity_names):
        mask = all_c_true == i
        if mask.sum() > 0:
            acc = (all_c_preds[mask] == i).mean()
            per_complexity[name] = {
                "accuracy": round(float(acc), 4),
                "count": int(mask.sum()),
            }

    # Confusion matrix (domain)
    confusion = np.zeros(
        (len(domain_labels), len(domain_labels)), dtype=int
    )
    for t, p in zip(all_d_true, all_d_preds):
        confusion[t][p] += 1

    # Find most confused pairs
    confused_pairs = []
    for i in range(len(domain_labels)):
        for j in range(len(domain_labels)):
            if i != j and confusion[i][j] > 0:
                confused_pairs.append((
                    domain_labels[i], domain_labels[j],
                    int(confusion[i][j]),
                ))
    confused_pairs.sort(key=lambda x: -x[2])

    # Model size
    n_params = sum(
        p.numel() for p in model.parameters()
    )

    results = {
        "domain_accuracy": round(float(d_acc), 4),
        "complexity_accuracy": round(float(c_acc), 4),
        "per_domain_accuracy": per_domain,
        "per_complexity_accuracy": per_complexity,
        "top_confused_pairs": [
            {"true": t, "pred": p, "count": c}
            for t, p, c in confused_pairs[:20]
        ],
        "confusion_matrix": confusion.tolist(),
        "model_params": n_params,
        "inference_time_s": round(inference_time, 2),
        "examples_per_sec": round(
            len(all_d_true) / inference_time, 1
        ),
        "test_size": len(all_d_true),
    }

    out_path = os.path.join(RESULTS_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    # Print summary
    print(f"\nModel: {n_params:,} params")
    print(f"\nPer-domain accuracy (sorted):")
    sorted_domains = sorted(
        per_domain.items(), key=lambda x: x[1]["accuracy"]
    )
    for name, info in sorted_domains[:10]:
        print(
            f"  {name}: {info['accuracy']:.4f} "
            f"(n={info['count']})"
        )
    print("  ...")
    for name, info in sorted_domains[-5:]:
        print(
            f"  {name}: {info['accuracy']:.4f} "
            f"(n={info['count']})"
        )

    print(f"\nPer-complexity accuracy:")
    for name, info in per_complexity.items():
        print(
            f"  {name}: {info['accuracy']:.4f} "
            f"(n={info['count']})"
        )

    print(f"\nTop confused pairs:")
    for pair in results["top_confused_pairs"][:10]:
        print(
            f"  {pair['true']} -> {pair['pred']}: "
            f"{pair['count']}"
        )


if __name__ == "__main__":
    main()
