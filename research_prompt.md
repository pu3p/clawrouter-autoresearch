You are an AI research agent.
Your task is to fine-tune a domain + complexity classifier for user prompts, starting from NVIDIA's pre-trained DeBERTa classifiers and evaluating on MMLU.

## Goal
Build a single model that classifies text into:
1. **Domain** — fine-grained topic categories (more specific than NVIDIA's 26 domains)
2. **Complexity** — overall difficulty score

## Background
NVIDIA provides two pre-trained DeBERTa-v3-base models on HuggingFace:
- `nvidia/domain-classifier` — 26 broad domain categories, Apache 2.0
- `nvidia/prompt-task-and-complexity-classifier` — 11 task types + 6 complexity dimensions

MMLU has 57 subjects which serve as fine-grained domain labels — more specific than NVIDIA's 26 broad categories.

## Step 1: Use MMLU's 57 subjects as domain labels
Keep all 57 MMLU subjects as individual domain classes — do NOT group them into broader categories.
The goal is fine-grained domain classification. Save the full list of subjects to /workspace/results/domain_labels.json.

## Step 2: Prepare MMLU data
Write a script at /workspace/simply/scripts/prepare_mmlu.py that:
1. Loads MMLU from HuggingFace: `datasets.load_dataset("cais/mmlu", "all")`
2. Uses the `subject` field directly as the domain label (all 57 subjects).
3. Formats each example as the question text (combine question + choices into a natural prompt).
4. Splits into train/val/test.
5. Saves processed datasets to /workspace/results/mmlu_processed/.

## Step 3: Label complexity with NVIDIA model
Write a script at /workspace/simply/scripts/label_complexity.py that:
1. Loads `nvidia/prompt-task-and-complexity-classifier` from HuggingFace.
2. Runs inference on all MMLU examples to get complexity scores.
3. Discretizes the overall complexity score into 4 bins: simple (<0.2), moderate (0.2-0.4), complex (0.4-0.6), expert (>0.6). Adjust thresholds based on the actual distribution.
4. Saves complexity labels alongside domain labels.

## Step 4: Fine-tune combined model
Write a training script at /workspace/simply/scripts/train_classifier.py that:
1. Loads `nvidia/domain-classifier` as the backbone (DeBERTa-v3-base already fine-tuned on domain classification — better starting point than raw DeBERTa).
2. Replaces the classification head with a multi-head architecture:
   - Head 1: Domain classification (57 classes — all MMLU subjects)
   - Head 2: Complexity classification (4 classes from Step 3)
3. Uses a combined loss: `loss = loss_domain + 0.5 * loss_complexity`
4. Training hyperparameters:
   - Learning rate: 2e-5 with linear warmup (10% of steps)
   - Batch size: 32
   - Epochs: 5
   - Max sequence length: 512
   - Weight decay: 0.01
5. Saves checkpoints to /workspace/results/classifier_checkpoints/.
6. Logs training metrics (loss, accuracy per head) each epoch.

Use HuggingFace Accelerate or plain PyTorch training loop — whichever is simpler.

## Step 5: Evaluate
Write an evaluation script at /workspace/simply/scripts/eval_classifier.py that:
1. Loads the best checkpoint.
2. Runs on the MMLU test split.
3. Computes per-domain accuracy and per-complexity-bin accuracy.
4. Computes a confusion matrix for domains.
5. Compares domain predictions against NVIDIA's original domain-classifier on the same data.
6. Saves results to /workspace/results/eval_results.json.

## Step 6: Iterate
Based on evaluation results:
1. If certain subjects have low accuracy, check class balance and consider oversampling underrepresented subjects.
2. Try at least 2 variations: different learning rates, freezing backbone for first epoch then unfreezing, or adjusting the domain taxonomy.
3. Run each variation and compare.

## Step 7: Report
Write a report to /workspace/results/classifier_report.md with:
- Domain taxonomy used
- Training curves
- Per-domain and per-complexity accuracy tables
- Comparison with NVIDIA's original domain-classifier
- Confusion matrix analysis (which domains get confused)
- Final model size and inference latency
- Recommendations for deployment

## Important notes
- This uses PyTorch/transformers, NOT Simply/JAX.
- All scripts should be runnable standalone: `python scripts/prepare_mmlu.py`, etc.
- Use the GPU — ensure torch.cuda.is_available() and move model/data to GPU.
- Save the final model in HuggingFace format so it can be loaded with from_pretrained().
- Do NOT commit large binary files (.pt, .bin, .safetensors, .arrow) — they are gitignored.
- **CRITICAL: After completing each step**, commit to git:
  ```
  cd ~/sky_workdir && cp -r /workspace/simply/scripts/* scripts/ 2>/dev/null; cp /workspace/results/*.json /workspace/results/*.md results/ 2>/dev/null; git add -A && git commit -m "step N: description" && git push clawrouter mar10-ec2
  ```
- Steps 1-2 are already done. Data is at /workspace/results/mmlu_processed/. Domain labels at /workspace/results/domain_labels.json. Resume from Step 3.
