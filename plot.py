"""Generate progress plot from results.tsv"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.tsv", sep="\t")
df = df[df["status"] == "keep"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Combined score
ax1.plot(range(len(df)), df["combined_score"], "o-", linewidth=2, markersize=6)
ax1.set_ylabel("Combined Score", fontsize=12)
ax1.set_title("ClawRouter Autoresearch Progress", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target (0.8)')
ax1.legend()

# Tier and domain accuracy
ax2.plot(range(len(df)), df["tier_acc"], "o-", label="Tier Accuracy", linewidth=2, markersize=6)
ax2.plot(range(len(df)), df["domain_acc"], "s-", label="Domain Accuracy", linewidth=2, markersize=6)
ax2.set_xlabel("Experiment", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print("Saved progress.png")
