"""Evaluate variation B checkpoint on test set."""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = "/workspace/results"
DATA_DIR = os.path.join(RESULTS_DIR, "mmlu_processed")
CKPT_DIR = os.path.join(
    RESULTS_DIR, "classifier_checkpoints/var_b"
)
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
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

    ds = load_from_disk(DATA_DIR)
    with open(
        os.path.join(RESULTS_DIR, "domain_labels.json")
    ) as f:
        domain_labels = json.load(f)

    def collate_fn(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=MAX_LEN, padding=True
        )
        return (
            enc,
            torch.tensor([b["domain_id"] for b in batch]),
            torch.tensor([b["complexity_id"] for b in batch]),
        )

    loader = DataLoader(
        ds["test"], batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    all_d_preds, all_d_true = [], []
    all_c_preds, all_c_true = [], []
    t0 = time.time()
    with torch.no_grad():
        for enc, d_ids, c_ids in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            d_logits, c_logits = model(
                enc["input_ids"], enc.get("attention_mask")
            )
            all_d_preds.extend(d_logits.argmax(1).cpu().tolist())
            all_d_true.extend(d_ids.tolist())
            all_c_preds.extend(c_logits.argmax(1).cpu().tolist())
            all_c_true.extend(c_ids.tolist())
    t1 = time.time()

    d_preds = np.array(all_d_preds)
    d_true = np.array(all_d_true)
    c_preds = np.array(all_c_preds)
    c_true = np.array(all_c_true)

    d_acc = (d_preds == d_true).mean()
    c_acc = (c_preds == c_true).mean()
    print(f"Var B test: domain={d_acc:.4f} complexity={c_acc:.4f}")
    print(f"Inference: {t1-t0:.1f}s ({len(d_true)/(t1-t0):.0f} ex/s)")

    # Per-domain
    per_domain = {}
    for i, name in enumerate(domain_labels):
        mask = d_true == i
        if mask.sum() > 0:
            per_domain[name] = {
                "accuracy": round(float((d_preds[mask] == i).mean()), 4),
                "count": int(mask.sum()),
            }

    # Confusion matrix
    confusion = np.zeros(
        (len(domain_labels), len(domain_labels)), dtype=int
    )
    for t, p in zip(d_true, d_preds):
        confusion[t][p] += 1

    confused_pairs = []
    for i in range(len(domain_labels)):
        for j in range(len(domain_labels)):
            if i != j and confusion[i][j] > 0:
                confused_pairs.append((
                    domain_labels[i], domain_labels[j],
                    int(confusion[i][j]),
                ))
    confused_pairs.sort(key=lambda x: -x[2])

    results = {
        "variant": "B (lr=5e-5, class_weights)",
        "domain_accuracy": round(float(d_acc), 4),
        "complexity_accuracy": round(float(c_acc), 4),
        "per_domain_accuracy": per_domain,
        "top_confused_pairs": [
            {"true": t, "pred": p, "count": c}
            for t, p, c in confused_pairs[:20]
        ],
        "confusion_matrix": confusion.tolist(),
        "inference_time_s": round(t1 - t0, 2),
    }

    out = os.path.join(RESULTS_DIR, "eval_results_var_b.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")

    # Low-accuracy domains
    sorted_d = sorted(
        per_domain.items(), key=lambda x: x[1]["accuracy"]
    )
    print("\nLowest accuracy domains:")
    for name, info in sorted_d[:10]:
        print(f"  {name}: {info['accuracy']:.4f} (n={info['count']})")


if __name__ == "__main__":
    main()
