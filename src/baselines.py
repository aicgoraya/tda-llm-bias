"""Baseline comparison: is the lgbtq vs heteronormative gap special?

The main analysis shows lgbtq_explicit embeddings are topologically distinct.
But distinct *relative to what*? This module contrasts the lgbtq_explicit group
against three references, all built from the same 20 templates:

  - heteronormative  (the semantically nearest contrast — same identity axis)
  - occupation_terms (a neutral, unrelated semantic category)
  - random_adjectives (generic descriptive variation)

It runs two analyses:

  (A) Full comparison — all 8 control terms (160 sentences each) vs lgbtq.
  (B) Size-matched control — control groups subsampled to 2 terms x 20 templates
      (40 sentences, matching heteronormative), averaged over 20 random 2-term
      subsets, so every comparison is a fair 80-vs-40 contrast.

Analysis (A) is confounded by cloud size (160 vs 40 points inflates Wasserstein);
(B) removes that confound and is the trustworthy verdict.
"""

import warnings
warnings.filterwarnings("ignore")  # ripser shape / persim inf-death notes

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import wasserstein

from embed import get_embeddings
from data_gen import TEMPLATES

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

OCCUPATIONS = ["teacher", "surgeon", "janitor", "engineer",
               "nurse", "lawyer", "plumber", "accountant"]
ADJECTIVES  = ["tall", "quiet", "friendly", "nervous",
               "clever", "tired", "happy", "calm"]

N_SUBSAMPLES = 20
SEED = 42


def h1_diagram(X: np.ndarray):
    """H1 (loop) persistence diagram of a point cloud."""
    return ripser(X, maxdim=1)["dgms"][1]


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def build_with_terms(term_phrases: list[tuple[str, str]]):
    """Fill shared templates with each phrase; return (sentences, term_labels)."""
    sentences, terms = [], []
    for _tid, _domain, template in TEMPLATES:
        for term, phrase in term_phrases:
            s = template.replace("{term}", phrase)
            sentences.append(s[0].upper() + s[1:])
            terms.append(term)
    return sentences, np.array(terms)


def sizematched_distances(dgm_lgbtq, ctrl_emb, ctrl_terms, all_terms, rng):
    """Wasserstein H1 over N_SUBSAMPLES random 2-term (40-sentence) subsets."""
    combos = list(itertools.combinations(all_terms, 2))
    pick = rng.choice(len(combos), size=min(N_SUBSAMPLES, len(combos)), replace=False)
    dists = []
    for i in pick:
        mask = np.isin(ctrl_terms, combos[i])  # 2 terms x 20 templates = 40 rows
        dists.append(wasserstein(dgm_lgbtq, h1_diagram(ctrl_emb[mask])))
    return np.array(dists)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    # existing 200 sentences / embeddings --------------------------------
    emb  = np.load(DATA / "embeddings.npy")
    meta = pd.read_csv(DATA / "embeddings_meta.csv")
    lgbtq  = emb[(meta["group"] == "lgbtq_explicit").values]
    hetero = emb[(meta["group"] == "heteronormative").values]
    dgm_lgbtq = h1_diagram(lgbtq)
    print(f"lgbtq_explicit: {len(lgbtq)} | heteronormative: {len(hetero)}")

    # build + embed both control groups ONCE -----------------------------
    occ_sent, occ_terms = build_with_terms([(w, f"{_article(w)} {w}") for w in OCCUPATIONS])
    adj_sent, adj_terms = build_with_terms([(w, f"{_article(w)} {w} person") for w in ADJECTIVES])
    print(f"Encoding {len(occ_sent)} occupation + {len(adj_sent)} adjective sentences …")
    occ = get_embeddings(occ_sent)
    adj = get_embeddings(adj_sent)

    het_dist = wasserstein(dgm_lgbtq, h1_diagram(hetero))

    # (A) full comparison ------------------------------------------------
    full = {
        "lgbtq_vs_heteronormative":   het_dist,
        "lgbtq_vs_occupation":        wasserstein(dgm_lgbtq, h1_diagram(occ)),
        "lgbtq_vs_random_adjectives": wasserstein(dgm_lgbtq, h1_diagram(adj)),
    }
    full_df = pd.DataFrame({"comparison": list(full), "wasserstein_h1": list(full.values())})
    full_df.to_csv(RESULTS / "baseline_distances.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(["vs\nheteronormative", "vs\noccupations", "vs\nadjectives"],
                  list(full.values()), color=["#d62728", "#1f77b4", "#2ca02c"])
    for b, v in zip(bars, full.values()):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylabel("H1 Wasserstein distance from lgbtq_explicit")
    ax.set_title("Full comparison (unequal cloud sizes)")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_comparison.png", dpi=150)
    plt.close(fig)

    # (B) size-matched control ------------------------------------------
    occ_sm = sizematched_distances(dgm_lgbtq, occ, occ_terms, OCCUPATIONS, rng)
    adj_sm = sizematched_distances(dgm_lgbtq, adj, adj_terms, ADJECTIVES, rng)

    sm_df = pd.DataFrame([
        {"comparison": "lgbtq_vs_heteronormative",   "mean_wasserstein_h1": het_dist,
         "std_wasserstein_h1": 0.0, "n_subsamples": 1},
        {"comparison": "lgbtq_vs_occupation",        "mean_wasserstein_h1": occ_sm.mean(),
         "std_wasserstein_h1": occ_sm.std(ddof=1), "n_subsamples": len(occ_sm)},
        {"comparison": "lgbtq_vs_random_adjectives", "mean_wasserstein_h1": adj_sm.mean(),
         "std_wasserstein_h1": adj_sm.std(ddof=1), "n_subsamples": len(adj_sm)},
    ])
    out_csv = RESULTS / "baseline_sizematched.csv"
    sm_df.to_csv(out_csv, index=False)
    print(f"\nSize-matched results (all comparisons are 80 vs 40 points)")
    print(sm_df.to_string(index=False))
    print(f"Saved → {out_csv}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = ["vs\nheteronormative", "vs\noccupations", "vs\nadjectives"]
    means  = sm_df["mean_wasserstein_h1"].values
    stds   = sm_df["std_wasserstein_h1"].values
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=["#d62728", "#1f77b4", "#2ca02c"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.2f}", ha="center", va="bottom")
    ax.set_ylabel("H1 Wasserstein distance from lgbtq_explicit")
    ax.set_title(f"Size-matched control (40 sentences each, "
                 f"mean ± std over {N_SUBSAMPLES} subsets)")
    ax.margins(y=0.18)
    fig.tight_layout()
    out_png = RESULTS / "baseline_sizematched.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_png}")

    # verdict ------------------------------------------------------------
    ctrl_mean_max = max(occ_sm.mean(), adj_sm.mean())
    print("\n" + "=" * 62)
    print("SIZE-MATCHED VERDICT: lgbtq↔hetero vs lgbtq↔control (all 80 vs 40)")
    print("=" * 62)
    print(f"  heteronormative : {het_dist:.4f}")
    print(f"  occupation      : {occ_sm.mean():.4f} ± {occ_sm.std(ddof=1):.4f}")
    print(f"  adjectives      : {adj_sm.mean():.4f} ± {adj_sm.std(ddof=1):.4f}")
    if het_dist > ctrl_mean_max:
        print(f"\nHYPOTHESIS SUPPORTED: once cloud size is matched, the "
              f"heteronormative\ndistance ({het_dist:.4f}) EXCEEDS both controls "
              f"(max {ctrl_mean_max:.4f}) — an\norientation-specific signal, not a "
              f"sample-size artifact.")
    else:
        print(f"\nHYPOTHESIS STILL NOT SUPPORTED: heteronormative distance "
              f"({het_dist:.4f})\nremains ≤ the size-matched controls "
              f"(max {ctrl_mean_max:.4f}). The lgbtq↔hetero\ngap is not larger "
              f"than generic lexical variation under this metric.")


if __name__ == "__main__":
    main()
