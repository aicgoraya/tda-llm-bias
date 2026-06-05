"""Figure 4 — Fine-tuning effect on H1 Wasserstein at each family's peak (§4.4).

Two panels rather than a broken axis: the BERT and RoBERTa Wasserstein scales
differ by ~25×, and within-family deltas are the meaningful quantity (cross-
family magnitudes are not directly comparable).

Input: results/rlhf_comparison_peak.csv
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply, SEMANTIC, save, REPO_ROOT

apply()

df = pd.read_csv(REPO_ROOT / "results" / "rlhf_comparison_peak.csv")

panels = [
    ("BERT @ L5",    "bert",    "MNLI"),
    ("RoBERTa @ L12","roberta", "CoLA"),
]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

for ax, (title, fam, ft_task) in zip(axes, panels):
    base = float(df[(df.base_family == fam) & (df.variant == "base")]
                 .wasserstein_h1.iloc[0])
    ft   = float(df[(df.base_family == fam) & (df.variant == "fine_tuned")]
                 .wasserstein_h1.iloc[0])
    delta_pct = (ft - base) / base * 100.0

    x = [0, 1]
    ax.bar(x, [base, ft],
           color=[SEMANTIC["base"], SEMANTIC["fine_tuned"]],
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["base", f"fine-tuned\n({ft_task})"])
    ax.set_title(title)
    ax.margins(y=0.25)

    # bar value labels
    ax.text(0, base, f" {base:.3f} ", ha="center", va="bottom", fontsize=8)
    ax.text(1, ft,   f" {ft:.3f} ",   ha="center", va="bottom", fontsize=8)

    # delta annotation centered above
    sign = "+" if delta_pct > 0 else ""
    ax.text(0.5, 0.97,
            f"$\\Delta = {sign}{delta_pct:.1f}\\%$",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold")

axes[0].set_ylabel("H1 Wasserstein\n(lgbtq_explicit vs heteronormative)")

# Caveat note below figure
fig.text(0.5, -0.04,
         "Magnitudes are not cross-family comparable; only within-family Δ is interpreted.",
         ha="center", va="top", fontsize=7.5, style="italic",
         color=SEMANTIC["null"])

fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig4_finetuning")

print("BERT: 18.997 -> 10.896  Δ=-42.6%")
print("RoBERTa: 0.686 -> 1.835  Δ=+167.4%")
