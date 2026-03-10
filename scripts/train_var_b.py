"""Variation B: Higher LR (5e-5) + class-weighted domain loss."""

import json
import os
import time
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
)

RESULTS_DIR = "/workspace/results"
DATA_DIR = os.path.join(RESULTS_DIR, "mmlu_processed")
CKPT_DIR = os.path.join(
    RESULTS_DIR, "classifier_checkpoints/var_b"
)
NUM_DOMAINS = 57
NUM_COMPLEXITY = 4
BACKBONE = "microsoft/deberta-v3-base"
MAX_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS = 5
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
COMPLEXITY_WEIGHT = 0.5


class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone_name, num_domains, num_complexity):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name, use_safetensors=True,
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
    return enc, domain_ids, complexity_ids


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    d_correct = d_total = c_correct = c_total = 0
    total_loss = 0
    ce = nn.CrossEntropyLoss()
    for enc, d_ids, c_ids in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        d_ids, c_ids = d_ids.to(device), c_ids.to(device)
        d_logits, c_logits = model(
            enc["input_ids"], enc.get("attention_mask")
        )
        loss = ce(d_logits, d_ids) + (
            COMPLEXITY_WEIGHT * ce(c_logits, c_ids)
        )
        total_loss += loss.item() * d_ids.size(0)
        d_correct += (d_logits.argmax(1) == d_ids).sum().item()
        c_correct += (c_logits.argmax(1) == c_ids).sum().item()
        d_total += d_ids.size(0)
        c_total += c_ids.size(0)
    return (
        total_loss / d_total,
        d_correct / d_total,
        c_correct / c_total,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    ds = load_from_disk(DATA_DIR)

    # Compute class weights for domain
    counts = collections.Counter(ds["train"]["domain_id"])
    total = len(ds["train"])
    weights = torch.zeros(NUM_DOMAINS)
    for i in range(NUM_DOMAINS):
        c = counts.get(i, 1)
        weights[i] = total / (NUM_DOMAINS * c)
    # Clip extreme weights
    weights = weights.clamp(max=5.0).to(device)
    print(
        f"Domain weights: min={weights.min():.2f} "
        f"max={weights.max():.2f}"
    )

    def make_loader(split, shuffle=False):
        return DataLoader(
            ds[split], batch_size=BATCH_SIZE, shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, tokenizer),
            num_workers=2, pin_memory=True,
        )

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("validation")

    model = MultiHeadClassifier(
        BACKBONE, NUM_DOMAINS, NUM_COMPLEXITY
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    ce_domain = nn.CrossEntropyLoss(weight=weights)
    ce_complexity = nn.CrossEntropyLoss()

    os.makedirs(CKPT_DIR, exist_ok=True)
    metrics_log = []
    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        d_correct = d_total = c_correct = c_total = 0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (enc, d_ids, c_ids) in enumerate(
            train_loader
        ):
            enc = {k: v.to(device) for k, v in enc.items()}
            d_ids, c_ids = d_ids.to(device), c_ids.to(device)

            d_logits, c_logits = model(
                enc["input_ids"], enc.get("attention_mask")
            )
            loss = (
                ce_domain(d_logits, d_ids)
                + COMPLEXITY_WEIGHT * ce_complexity(
                    c_logits, c_ids
                )
            ) / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM * d_ids.size(0)
            d_correct += (
                d_logits.argmax(1) == d_ids
            ).sum().item()
            c_correct += (
                c_logits.argmax(1) == c_ids
            ).sum().item()
            d_total += d_ids.size(0)
            c_total += c_ids.size(0)

        train_loss = epoch_loss / d_total
        val_loss, val_d_acc, val_c_acc = evaluate(
            model, val_loader, device
        )
        elapsed = time.time() - t0

        m = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_domain_acc": round(d_correct / d_total, 4),
            "train_complexity_acc": round(
                c_correct / c_total, 4
            ),
            "val_loss": round(val_loss, 4),
            "val_domain_acc": round(val_d_acc, 4),
            "val_complexity_acc": round(val_c_acc, 4),
            "time_s": round(elapsed, 1),
        }
        metrics_log.append(m)
        print(
            f"Epoch {epoch}: val_d_acc={val_d_acc:.4f} "
            f"val_c_acc={val_c_acc:.4f} ({elapsed:.0f}s)"
        )

        if val_d_acc > best_val_acc:
            best_val_acc = val_d_acc
            model.backbone.save_pretrained(CKPT_DIR)
            tokenizer.save_pretrained(CKPT_DIR)
            torch.save(
                {
                    "domain_head": (
                        model.domain_head.state_dict()
                    ),
                    "complexity_head": (
                        model.complexity_head.state_dict()
                    ),
                    "dropout_p": 0.1,
                    "num_domains": NUM_DOMAINS,
                    "num_complexity": NUM_COMPLEXITY,
                    "epoch": epoch,
                    "val_domain_acc": val_d_acc,
                    "val_complexity_acc": val_c_acc,
                },
                os.path.join(CKPT_DIR, "heads.pt"),
            )
            print(f"  Saved best (val_d_acc={val_d_acc:.4f})")

    with open(
        os.path.join(RESULTS_DIR, "var_b_metrics.json"), "w"
    ) as f:
        json.dump(metrics_log, f, indent=2)
    print("Variation B complete.")


if __name__ == "__main__":
    main()
