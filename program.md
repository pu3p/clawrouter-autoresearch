# clawrouter-autoresearch

This is an experiment to have an AI agent autonomously improve [ClawRouter](https://github.com/pu3p/ClawRouter)'s routing logic using the [MMLU dataset](https://huggingface.co/datasets/cais/mmlu).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, MMLU data loading, domain/tier mappings, evaluation harness. Do not modify.
   - `train.py` — the file you modify. Scoring logic, domain keywords, complexity signals, tier thresholds.
4. **Verify data exists**: Check that `~/.cache/clawrouter-autoresearch/mmlu.json` exists. If not, tell the human to run `uv run prepare.py`.
5. **Clone ClawRouter**: Ensure `~/ClawRouter` exists (`git clone https://github.com/pu3p/ClawRouter.git ~/ClawRouter` if needed). Create a matching branch there: `git checkout -b autoresearch/<tag>`.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment evaluates the scoring function against MMLU. You launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: domain keywords, complexity signals, tier thresholds, scoring algorithms, feature extraction, weighting schemes.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation harness, MMLU data loading, and domain/tier ground truth mappings.
- Install new packages or add dependencies.
- Modify the evaluation function. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest combined_score.** This is `0.5 * tier_accuracy + 0.5 * domain_accuracy`. Higher is better.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the scoring as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
tier_accuracy:    0.150000
domain_accuracy:  0.450000
combined_score:   0.300000
total_examples:   1531
```

You can extract the key metric:

```
grep "^combined_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 5 columns:

```
commit	combined_score	tier_acc	domain_acc	status	description
```

1. git commit hash (short, 7 chars)
2. combined_score (e.g. 0.300000)
3. tier_accuracy (e.g. 0.150000)
4. domain_accuracy (e.g. 0.450000)
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

## Syncing to ClawRouter

When an experiment improves combined_score, `train.py` exports a `scoring_config.json`. Copy this to the ClawRouter repo and commit:

```bash
cp scoring_config.json ~/ClawRouter/scoring_config.json
cd ~/ClawRouter
git add scoring_config.json
git commit -m "autoresearch: <description of change>"
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea — modify keywords, thresholds, scoring logic, add new features.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^combined_score:\|^tier_accuracy:\|^domain_accuracy:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (do not commit results.tsv)
8. If combined_score improved (higher), keep the commit and sync scoring_config.json to ClawRouter
9. If combined_score is equal or worse, git reset back to where you started

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try different keyword strategies, different weighting schemes, regex patterns, n-gram features, question structure analysis. The loop runs until the human interrupts you.
