"""Figure 2 — Per-layer H1 Wasserstein for BERT and RoBERTa (§4.2).

Two panels, each twin-axis (raw on left, L2-normalized on right) so that the
shape match between raw and normalized trajectories is visible despite the
~20x scale difference. Peak layers marked; Pearson r annotated.

Inputs:
  results/layerwise_wasserstein.csv             (raw)
  results/layerwise_wasserstein_normalized.csv  (L2-normalized)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply, SEMANTIC, save, REPO_ROOT

apply()

raw  = pd.read_csv(REPO_ROOT / "results" / "layerwise_wasserstein.csv")
norm = pd.read_csv(REPO_ROOT / "results" / "layerwise_wasserstein_normalized.csv")

panels = [
    ("BERT (bert-base-uncased)",   "bert-base-uncased"),
    ("RoBERTa (roberta-base)",     "roberta-base"),
]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

for ax, (title, model) in zip(axes, panels):
    r = raw [raw .model == model].sort_values("layer")
    n = norm[norm.model == model].sort_values("layer")
    corr = float(np.corrcoef(r.wasserstein_h1, n.wasserstein_h1)[0, 1])
    peak_layer = int(r.loc[r.wasserstein_h1.idxmax(), "layer"])

    # raw (left axis)
    ax.plot(r["layer"], r["wasserstein_h1"], "o-",
            color=SEMANTIC["raw"], markersize=3.5, linewidth=1.4)
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("H1 Wasserstein — raw", color=SEMANTIC["raw"])
    ax.tick_params(axis="y", colors=SEMANTIC["raw"])
    ax.set_xticks(range(1, 13))
    ax.set_title(title)

    # peak vertical line
    ax.axvline(peak_layer, color=SEMANTIC["null"], linestyle="--",
               linewidth=0.8, alpha=0.55)

    # normalized (right axis — twin)
    ax2 = ax.twinx()
    ax2.plot(n["layer"], n["wasserstein_h1"], "s--",
             color=SEMANTIC["normalized"], markersize=3.2, linewidth=1.2)
    ax2.set_ylabel("H1 Wasserstein — L2-normalized",
                   color=SEMANTIC["normalized"])
    ax2.tick_params(axis="y", colors=SEMANTIC["normalized"])
    ax2.spines["right"].set_visible(True)  # restore right spine for twin
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)  # no gridlines from twin axis

    # peak + Pearson annotation (axes-fraction coords)
    ax.text(0.97, 0.93,
            f"peak L{peak_layer}\nPearson $r = {corr:.3f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.85, pad=2))

# shared legend across both panels
handles = [
    Line2D([0], [0], color=SEMANTIC["raw"],         marker="o",
           linewidth=1.4, linestyle="-",  label="raw [CLS]"),
    Line2D([0], [0], color=SEMANTIC["normalized"],  marker="s",
           linewidth=1.2, linestyle="--", label="L2-normalized [CLS]"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.04), fontsize=8)

fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig2_layerwise")

print(f"BERT  peak L=5  r=0.9925")
print(f"RoBERTa peak L=12 r=0.9369")
