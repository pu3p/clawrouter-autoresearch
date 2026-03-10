"""Label MMLU examples with complexity using NVIDIA's classifier."""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoTokenizer, DebertaV2Model, DebertaV2Config
from safetensors import safe_open
from huggingface_hub import hf_hub_download

RESULTS_DIR = "/workspace/results"
INPUT_DIR = os.path.join(RESULTS_DIR, "mmlu_processed")
MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"
BATCH_SIZE = 64


class Head(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


class ComplexityModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = DebertaV2Config.from_pretrained(
            "microsoft/deberta-v3-base"
        )
        self.backbone = DebertaV2Model(cfg)
        sizes = list(config["target_sizes"].values())
        self.heads = nn.ModuleList(
            [Head(768, s) for s in sizes]
        )
        self.config = config

    def forward(self, input_ids, attention_mask=None,
                token_type_ids=None):
        out = self.backbone(
            input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = out.last_hidden_state[:, 0]
        return [h(cls) for h in self.heads]

    def complexity_score(self, head_outputs):
        wmap = self.config["weights_map"]
        dmap = self.config["divisor_map"]
        # Head indices: 1=creativity, 2=reasoning, 3=context,
        # 4=few_shots, 5=domain_knowledge, 6=no_label, 7=constraint
        names = list(self.config["target_sizes"].keys())
        # Skip task_type (idx 0) and no_label_reason (idx 6)
        weights = {
            "creativity_scope": 0.35,
            "reasoning": 0.25,
            "constraint_ct": 0.15,
            "domain_knowledge": 0.15,
            "contextual_knowledge": 0.05,
            "number_of_few_shots": 0.05,
        }
        total = torch.zeros(
            head_outputs[0].shape[0],
            device=head_outputs[0].device
        )
        for name, w in weights.items():
            idx = names.index(name)
            logits = head_outputs[idx]
            probs = torch.softmax(logits, dim=-1)
            wt = torch.tensor(
                wmap[name], device=logits.device, dtype=torch.float
            )
            div = max(dmap[name], 1)
            total += w * (probs * wt).sum(dim=-1) / div
        return total


def load_model(device):
    cfg_path = hf_hub_download(MODEL_NAME, "config.json")
    with open(cfg_path) as f:
        config = json.load(f)
    model = ComplexityModel(config)
    sf_path = hf_hub_download(MODEL_NAME, "model.safetensors")
    state = {}
    with safe_open(sf_path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.startswith("head_"):
                parts = k.split(".")
                idx = int(parts[0].split("_")[1])
                new_k = f"heads.{idx}." + ".".join(parts[1:])
                state[new_k] = t
            else:
                state[k] = t
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(device)

    ds = load_from_disk(INPUT_DIR)
    all_scores = {}

    for split_name in ["train", "validation", "test"]:
        print(f"Processing {split_name}...")
        texts = ds[split_name]["text"]
        scores = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            ).to(device)
            with torch.no_grad():
                heads = model(**inputs)
                c = model.complexity_score(heads)
            scores.extend(c.cpu().numpy().tolist())
            if (i // BATCH_SIZE) % 100 == 0:
                print(
                    f"  {min(i + BATCH_SIZE, len(texts))}"
                    f"/{len(texts)}"
                )
        all_scores[split_name] = scores

    all_vals = np.concatenate(
        [np.array(v) for v in all_scores.values()]
    )
    print(f"\nStats: min={all_vals.min():.3f} "
          f"max={all_vals.max():.3f} "
          f"mean={all_vals.mean():.3f} std={all_vals.std():.3f}")

    # Quartile-based bins
    t1 = float(np.percentile(all_vals, 25))
    t2 = float(np.percentile(all_vals, 50))
    t3 = float(np.percentile(all_vals, 75))
    print(f"Thresholds: [{t1:.3f}, {t2:.3f}, {t3:.3f}]")

    for split_name in ["train", "validation", "test"]:
        labels = []
        for s in all_scores[split_name]:
            if s < t1:
                labels.append(0)
            elif s < t2:
                labels.append(1)
            elif s < t3:
                labels.append(2)
            else:
                labels.append(3)
        ds[split_name] = ds[split_name].add_column(
            "complexity_id", labels
        )
        ds[split_name] = ds[split_name].add_column(
            "complexity_score", all_scores[split_name]
        )

    ds.save_to_disk(INPUT_DIR)
    print(f"Saved to {INPUT_DIR}")

    meta = {
        "thresholds": [t1, t2, t3],
        "complexity_labels": [
            "simple", "moderate", "complex", "expert"
        ],
        "score_stats": {
            "min": float(all_vals.min()),
            "max": float(all_vals.max()),
            "mean": float(all_vals.mean()),
            "std": float(all_vals.std()),
        },
    }
    with open(
        os.path.join(RESULTS_DIR, "complexity_meta.json"), "w"
    ) as f:
        json.dump(meta, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
