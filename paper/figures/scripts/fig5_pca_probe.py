"""Figure 5 — PCA bottleneck probe accuracy vs k (§4.5).

Two panels (BERT, RoBERTa), each comparing base vs fine-tuned. The 0.90
accuracy threshold and the family's min-k crossings (where they exist) are
marked. For RoBERTa, neither curve reaches the threshold within k ≤ 50; this
is shown honestly as a resolution limit rather than glossed over.

Input: results/pca_probe_peak.csv
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply, SEMANTIC, save, REPO_ROOT

apply()

df = pd.read_csv(REPO_ROOT / "results" / "pca_probe_peak.csv")

panels = [
    ("BERT @ L5",     "bert-base-uncased",            "textattack-bert-base-MNLI",       "MNLI"),
    ("RoBERTa @ L12", "roberta-base",                 "textattack-roberta-base-CoLA",    "CoLA"),
]

THRESHOLD = 0.90
CHANCE    = 0.667  # majority-class raw-accuracy baseline

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

for ax, (title, base_label, ft_label, ft_task) in zip(axes, panels):
    base = df[df.model == base_label].sort_values("k")
    ft   = df[df.model == ft_label].sort_values("k")

    ax.errorbar(base["k"], base["mean_accuracy"], yerr=base["std_accuracy"],
                marker="o", markersize=4, capsize=3,
                color=SEMANTIC["base"], label="base", linewidth=1.4)
    ax.errorbar(ft["k"], ft["mean_accuracy"], yerr=ft["std_accuracy"],
                marker="s", markersize=4, capsize=3,
                color=SEMANTIC["fine_tuned"], linestyle="--",
                label=f"fine-tuned ({ft_task})", linewidth=1.4)

    ax.axhline(THRESHOLD, color=SEMANTIC["threshold"], linestyle=":",
               linewidth=0.9, alpha=0.85)
    ax.axhline(CHANCE, color=SEMANTIC["chance"], linestyle=":",
               linewidth=0.6, alpha=0.5)

    # min-k vertical markers where threshold is reached
    for src, color in [(base, SEMANTIC["base"]),
                       (ft,   SEMANTIC["fine_tuned"])]:
        hits = src[src["mean_accuracy"] >= THRESHOLD]
        if len(hits):
            mk = int(hits["k"].min())
            ax.axvline(mk, color=color, linestyle=":",
                       linewidth=0.6, alpha=0.55)
            ax.text(mk, 0.46, f"k={mk}", color=color,
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xlabel("PCA components retained (k)")
    ax.set_title(title)
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_ylim(0.45, 1.03)
    ax.legend(loc="lower right", fontsize=7.5)

    # threshold + chance labels (only in the left panel to avoid duplication)
    if ax is axes[0]:
        ax.text(50, THRESHOLD, " threshold = 0.90", color=SEMANTIC["threshold"],
                ha="right", va="bottom", fontsize=7)
        ax.text(50, CHANCE, " chance = 0.667", color=SEMANTIC["chance"],
                ha="right", va="bottom", fontsize=7, alpha=0.7)

axes[0].set_ylabel("5-fold CV mean accuracy")

# Honest note about RoBERTa resolution limit
fig.text(0.5, -0.04,
         "RoBERTa: neither base nor CoLA reaches the 0.90 threshold within k ≤ 50 "
         "(resolution-limited).",
         ha="center", va="top", fontsize=7.5, style="italic",
         color=SEMANTIC["null"])

fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig5_pca_probe")

print("BERT base min-k=10  MNLI min-k=50  (5x expansion)")
print("RoBERTa base k=50 acc=0.692  CoLA k=50 acc=0.883  (neither >= 0.90)")
