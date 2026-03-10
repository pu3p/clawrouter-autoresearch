# clawrouter-autoresearch

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), this project applies the same autonomous AI agent experimentation loop to fine-tuning [ClawRouter](https://github.com/pu3p/ClawRouter) using the [MMLU dataset](https://huggingface.co/datasets/cais/mmlu).

An AI agent autonomously modifies the training code, fine-tunes for a fixed time budget, checks if the result improved, keeps or discards, and repeats. You come back to a log of experiments and (hopefully) a better model.

## Setup

- **Hardware:** NVIDIA Jetson board
- **Model:** Qwen3.5:4b
- **Dataset:** MMLU
- **First run:** March 9, 2026

## How it works

Same core loop as autoresearch — the agent only touches `train.py`, trains within a fixed time budget, and uses a single metric to decide whether to keep or revert changes.

- **`prepare.py`** — data prep and runtime utilities (dataloader, evaluation). Not modified by the agent.
- **`train.py`** — model, optimizer, and training loop. **Edited by the agent.**
- **`program.md`** — instructions for the agent. **Edited by the human.**

## Quick start

**Requirements:** NVIDIA Jetson (or compatible NVIDIA GPU), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time)
uv run prepare.py

# 4. Run a single training experiment
uv run train.py
```

## Running the agent

Spin up your preferred AI coding agent in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Acknowledgments

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original autonomous research framework this project is based on.
- [pu3p/ClawRouter](https://github.com/pu3p/ClawRouter) — the proxy being fine-tuned.

## License

MIT
