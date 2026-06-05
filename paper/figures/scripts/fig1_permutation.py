"""Figure 1 — Permutation test null distribution (§4.1).

The permutation null is not persisted to a CSV in the repo (it lives in the
notebook's runtime). This script deterministically regenerates it from
data/embeddings.npy + data/embeddings_meta.csv with the same seed (42) and
n_perm (1000) the notebook uses, so the numbers match exactly:
  observed H1 Wasserstein (lgbtq vs neutral) = 4.7685
  null mean = 2.2578  sd = 0.2794  95th pct = 2.7244
  one-sided p-value = 0.0010

This is reproduction of an existing analysis, not a new analysis.
"""

import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import wasserstein

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply, SEMANTIC, save, REPO_ROOT

apply()

# ---------------------------------------------------------------- data
emb  = np.load(REPO_ROOT / "data" / "embeddings.npy")
meta = pd.read_csv(REPO_ROOT / "data" / "embeddings_meta.csv")

is_lgbtq   = (meta["group"].values == "lgbtq_explicit")
is_neutral = (meta["group"].values == "neutral")
X_lgbtq   = emb[is_lgbtq]
X_neutral = emb[is_neutral]

def h1(X):
    return ripser(X, maxdim=1)["dgms"][1]

observed = wasserstein(h1(X_lgbtq), h1(X_neutral))

pool   = np.vstack([X_lgbtq, X_neutral])
n_lg   = X_lgbtq.shape[0]
N_tot  = pool.shape[0]
rng    = np.random.default_rng(42)
n_perm = 1000
null   = np.empty(n_perm)
for i in range(n_perm):
    idx = rng.permutation(N_tot)
    null[i] = wasserstein(h1(pool[idx[:n_lg]]), h1(pool[idx[n_lg:]]))

p_value = (1 + np.sum(null >= observed)) / (n_perm + 1)
p95     = float(np.percentile(null, 95))

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(3.5, 2.6))

ax.hist(null, bins=40, color=SEMANTIC["null"], edgecolor="white",
        linewidth=0.4, alpha=0.85)

ax.axvline(observed, color=SEMANTIC["observed"], linewidth=1.6,
           label=f"observed = {observed:.3f}")
ax.axvline(p95, color=SEMANTIC["null"], linewidth=1.0, linestyle="--",
           alpha=0.85, label=f"null 95th pct = {p95:.3f}")

ax.set_xlabel("H1 Wasserstein distance (lgbtq_explicit vs neutral)")
ax.set_ylabel("count (1000 permutations)")

# p-value annotation
ax.text(0.97, 0.92, f"$p = {p_value:.3f}$",
        transform=ax.transAxes, ha="right", va="top", fontsize=9)

ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.86), fontsize=7.5)
fig.tight_layout()
save(fig, "fig1_permutation")

print(f"observed={observed:.4f}  null_mean={null.mean():.4f}  "
      f"null_sd={null.std(ddof=1):.4f}  p95={p95:.4f}  p={p_value:.4f}")
