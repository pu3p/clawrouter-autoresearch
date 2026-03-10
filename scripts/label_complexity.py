"""Label MMLU examples with complexity using NVIDIA's classifier.

The NVIDIA model is a custom multi-head DeBERTa model, not loadable via
AutoModel. We build the model class manually and load the safetensors.
"""

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


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


class NVIDIAComplexityModel(nn.Module):
    """Replicate NVIDIA's multi-head complexity classifier."""

    def __init__(self):
        super().__init__()
        cfg = DebertaV2Config.from_pretrained(
            "microsoft/deberta-v3-base"
        )
        self.backbone = DebertaV2Model(cfg)
        # 8 heads matching the safetensors
        head_sizes = [12, 3, 2, 2, 6, 4, 1, 2]
        self.heads = nn.ModuleList([
            ClassificationHead(768, s) for s in head_sizes
        ])
        # Complexity weights from config
        # Order: task_type(12), creativity(3), reasoning(2),
        #   contextual_knowledge(2), few_shots(6),
        #   domain_knowledge(4), no_label(1), constraint(2)
        self.weights_map = {
            1: [2, 1, 0],           # creativity
            2: [0, 1],              # reasoning
            3: [0, 1],              # contextual_knowledge
            4: [0, 1, 2, 3, 4, 5], # few_shots
            5: [3, 1, 2, 0],       # domain_knowledge
            6: [0],                 # no_label
            7: [1, 0],             # constraint
        }
        self.divisors = {
            1: 2, 2: 1, 3: 1, 4: 1, 5: 3, 6: 1, 7: 1
        }
        # Overall complexity = weighted sum
        self.complexity_weights = {
            1: 0.35,  # creativity
            2: 0.25,  # reasoning
            7: 0.15,  # constraint
            5: 0.15,  # domain_knowledge
            3: 0.05,  # contextual_knowledge
            4: 0.05,  # few_shots
        }

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.backbone(
            input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        head_outputs = [h(pooled) for h in self.heads]
        return head_outputs

    def compute_complexity(self, head_outputs):
        """Compute overall complexity score from head outputs."""
        scores = {}
        for idx, w in self.complexity_weights.items():
            logits = head_outputs[idx]
            probs = torch.softmax(logits, dim=-1)
            wmap = torch.tensor(
                self.weights_map[idx], device=logits.device,
                dtype=torch.float
            )
            div = self.divisors[idx]
            score = (probs * wmap).sum(dim=-1) / max(div, 1)
            scores[idx] = score * w
        total = sum(scores.values())
        return total


def load_nvidia_model(device):
    """Load the NVIDIA model from safetensors."""
    model = NVIDIAComplexityModel()
    path = hf_hub_download(MODEL_NAME, "model.safetensors")
    state_dict = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            tensor = f.get_tensor(k)
            # Map head_N.fc.* to heads.N.fc.*
            if k.startswith("head_"):
                parts = k.split(".")
                idx = int(parts[0].split("_")[1])
                new_key = f"heads.{idx}." + ".".join(parts[1:])
                state_dict[new_key] = tensor
            else:
                state_dict[k] = tensor
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading NVIDIA complexity classifier...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_nvidia_model(device)

    ds = load_from_disk(INPUT_DIR)

    all_scores = {}
    for split_name in ["train", "validation", "test"]:
        print(f"\nProcessing {split_name}...")
        texts = ds[split_name]["text"]
        scores = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            ).to(device)
            with torch.no_grad():
                head_outputs = model(**inputs)
                complexity = model.compute_complexity(head_outputs)
            scores.extend(complexity.cpu().numpy().tolist())
            if (i // BATCH_SIZE) % 50 == 0:
                print(f"  {min(i + BATCH_SIZE, len(texts))}"
                      f"/{len(texts)}")
        all_scores[split_name] = scores

    # Analyze distribution
    all_vals = np.concatenate(
        [np.array(v) for v in all_scores.values()]
    )
    print(f"\nComplexity score stats:")
    print(f"  min={all_vals.min():.3f} max={all_vals.max():.3f} "
          f"mean={all_vals.mean():.3f} std={all_vals.std():.3f}")
    for p in [25, 50, 75]:
        print(f"  p{p}={np.percentile(all_vals, p):.3f}")

    # Quartile-based thresholds for balanced bins
    t1 = float(np.percentile(all_vals, 25))
    t2 = float(np.percentile(all_vals, 50))
    t3 = float(np.percentile(all_vals, 75))
    thresholds = [t1, t2, t3]
    print(f"Thresholds: {thresholds}")

    complexity_names = ["simple", "moderate", "complex", "expert"]
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
        "thresholds": thresholds,
        "complexity_labels": complexity_names,
        "score_stats": {
            "min": float(all_vals.min()),
            "max": float(all_vals.max()),
            "mean": float(all_vals.mean()),
            "std": float(all_vals.std()),
        },
    }
    with open(os.path.join(RESULTS_DIR, "complexity_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved complexity metadata")


if __name__ == "__main__":
    main()
