"""Figure 3 — Size-matched baseline H1 Wasserstein distances (§4.3).

Input: results/baseline_sizematched.csv
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply, SEMANTIC, save, REPO_ROOT

apply()

df = pd.read_csv(REPO_ROOT / "results" / "baseline_sizematched.csv")

# fixed display order: nearest neighbour first, then the two controls
order = ["lgbtq_vs_heteronormative",
         "lgbtq_vs_occupation",
         "lgbtq_vs_random_adjectives"]
labels = ["vs.\nheteronormative", "vs.\noccupations", "vs.\nadjectives"]
colors = [SEMANTIC["heteronorm"], SEMANTIC["occupation"], SEMANTIC["adjective"]]

means = [float(df[df.comparison == c].mean_wasserstein_h1.iloc[0]) for c in order]
stds  = [float(df[df.comparison == c].std_wasserstein_h1.iloc[0])  for c in order]

fig, ax = plt.subplots(figsize=(3.5, 2.8))
x = list(range(len(order)))

bars = ax.bar(x, means, yerr=stds, capsize=4,
              color=colors, edgecolor="white", linewidth=0.5)

ymax = max(m + s for m, s in zip(means, stds))
ax.set_ylim(0, ymax + 0.9)

# value labels above each bar
for xi, m, s in zip(x, means, stds):
    text = f"{m:.3f}" + (f"\n± {s:.3f}" if s > 0 else "\n(fixed)")
    ax.text(xi, m + s + 0.05, text, ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("H1 Wasserstein from lgbtq_explicit\n(80 vs. 40, size-matched)")
ax.margins(x=0.05)

# subtle annotation: heteronormative is nearest
ax.annotate("nearest", xy=(0, means[0]), xytext=(0, means[0] + 1.7),
            ha="center", va="bottom", fontsize=7.5, style="italic",
            color=SEMANTIC["heteronorm"],
            arrowprops=dict(arrowstyle="-", color=SEMANTIC["heteronorm"],
                            linewidth=0.6, alpha=0.6))

fig.tight_layout()
save(fig, "fig3_baselines")

print(f"hetero={means[0]:.3f}  occ={means[1]:.3f}±{stds[1]:.3f}  "
      f"adj={means[2]:.3f}±{stds[2]:.3f}")
