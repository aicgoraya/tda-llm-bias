"""Core TDA analysis: persistent homology and pairwise Wasserstein distances.

Pipeline:
  1. Load embeddings + metadata
  2. Split into per-group embedding arrays
  3. Run Vietoris-Rips (H0 + H1) on each group
  4. Compute pairwise Wasserstein distances for H0 and H1
  5. Persist diagrams and distance matrices; print summary table

CLI:
  python src/tda_pipeline.py             # main pipeline (above)
  python src/tda_pipeline.py --bootstrap # 1000-iteration bootstrap of the
                                         # lgbtq vs heteronormative H1 distance
                                         # (80% subsamples each group)
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tda import compute_persistence, diagram_distance

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

GROUPS = ["lgbtq_explicit", "heteronormative", "neutral", "religious_conservative"]


def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    meta = pd.read_csv(DATA_DIR / "embeddings_meta.csv")
    return embeddings, meta


def split_by_group(
    embeddings: np.ndarray, meta: pd.DataFrame
) -> dict[str, np.ndarray]:
    return {
        group: embeddings[meta["group"].values == group]
        for group in GROUPS
    }


def compute_all_diagrams(
    group_embeddings: dict[str, np.ndarray]
) -> dict[str, list]:
    diagrams: dict[str, list] = {}
    for group, emb in group_embeddings.items():
        print(f"  {group}: {emb.shape[0]} sentences → running ripser …")
        diagrams[group] = compute_persistence(emb, max_dim=1)
    return diagrams


def pairwise_wasserstein(
    diagrams: dict[str, list], dim: int
) -> pd.DataFrame:
    n = len(GROUPS)
    mat = np.zeros((n, n))
    for i, g1 in enumerate(GROUPS):
        for j, g2 in enumerate(GROUPS):
            if i < j:
                d = diagram_distance(diagrams[g1][dim], diagrams[g2][dim])
                mat[i, j] = mat[j, i] = d
    return pd.DataFrame(mat, index=GROUPS, columns=GROUPS)


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    col_w = max(len(c) for c in df.columns) + 2
    row_w = max(len(r) for r in df.index) + 2
    header = " " * row_w + "".join(c.rjust(col_w) for c in df.columns)
    print(header)
    for row in df.index:
        vals = "".join(f"{v:>{col_w}.4f}" for v in df.loc[row])
        print(f"{row:<{row_w}}{vals}")


def bootstrap_stability(
    n_iter: int = 1000,
    frac: float = 0.8,
    seed: int = 42,
) -> None:
    """Resample lgbtq_explicit and heteronormative groups; rebuild the H1
    Wasserstein distance n_iter times to characterize its stability."""
    RESULTS_DIR.mkdir(exist_ok=True)

    embeddings, meta = load_data()
    lgbtq  = embeddings[(meta["group"] == "lgbtq_explicit").values]
    hetero = embeddings[(meta["group"] == "heteronormative").values]
    n_lgbtq_sub  = int(round(frac * lgbtq.shape[0]))
    n_hetero_sub = int(round(frac * hetero.shape[0]))

    # observed (full-sample) distance for reference
    observed = diagram_distance(
        compute_persistence(lgbtq,  max_dim=1)[1],
        compute_persistence(hetero, max_dim=1)[1],
    )

    rng = np.random.default_rng(seed)
    dists = np.empty(n_iter)
    print(f"Bootstrap: {n_iter} iterations | subsample "
          f"lgbtq {lgbtq.shape[0]}→{n_lgbtq_sub}, "
          f"hetero {hetero.shape[0]}→{n_hetero_sub}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ripser shape / persim inf-death notes
        for i in range(n_iter):
            a = lgbtq[rng.choice(lgbtq.shape[0], n_lgbtq_sub, replace=False)]
            b = hetero[rng.choice(hetero.shape[0], n_hetero_sub, replace=False)]
            dists[i] = diagram_distance(
                compute_persistence(a, max_dim=1)[1],
                compute_persistence(b, max_dim=1)[1],
            )

    mean = dists.mean()
    std  = dists.std(ddof=1)
    ci_lo, ci_hi = np.percentile(dists, [2.5, 97.5])

    # per-iteration values for full reproducibility of the histogram
    out_csv = RESULTS_DIR / "bootstrap_stability.csv"
    pd.DataFrame({"iteration": np.arange(n_iter), "wasserstein_h1": dists}) \
        .to_csv(out_csv, index=False)
    print(f"\nSaved per-iteration values → {out_csv}")

    print("\nBootstrap stability — lgbtq_explicit vs heteronormative (H1)")
    print("-" * 60)
    print(f"  observed (full sample)  : {observed:.4f}")
    print(f"  bootstrap mean ± std    : {mean:.4f} ± {std:.4f}")
    print(f"  95% CI (percentile)     : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  n iterations            : {n_iter}")
    print(f"  subsample fraction      : {frac:.0%}")

    # histogram with observed marked
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(dists, bins=40, alpha=0.85, color="steelblue",
            label=f"bootstrap (n={n_iter})")
    ax.axvline(observed, color="crimson", lw=2.5,
               label=f"observed (full) = {observed:.2f}")
    ax.axvline(mean, color="black", ls=":", lw=1.5,
               label=f"bootstrap mean = {mean:.2f}")
    ax.axvspan(ci_lo, ci_hi, alpha=0.12, color="grey",
               label=f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    ax.set_xlabel("H1 Wasserstein distance (lgbtq_explicit vs heteronormative)")
    ax.set_ylabel("count")
    ax.set_title(f"Bootstrap stability — {frac:.0%} subsamples, {n_iter} iterations")
    ax.legend()
    fig.tight_layout()
    out_png = RESULTS_DIR / "bootstrap_distribution.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved histogram → {out_png}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading data …")
    embeddings, meta = load_data()
    print(f"  embeddings shape : {embeddings.shape}")
    print(f"  metadata rows    : {len(meta)}")

    print("\nSplitting by group …")
    group_embeddings = split_by_group(embeddings, meta)
    for g, e in group_embeddings.items():
        print(f"  {g}: {e.shape}")

    print("\nComputing persistence diagrams …")
    diagrams = compute_all_diagrams(group_embeddings)

    pkl_path = RESULTS_DIR / "persistence_diagrams.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(diagrams, f)
    print(f"\nSaved diagrams → {pkl_path}")

    print("\nComputing pairwise Wasserstein distances …")
    dist_h0 = pairwise_wasserstein(diagrams, dim=0)
    dist_h1 = pairwise_wasserstein(diagrams, dim=1)

    # stack into one CSV with a "dim" column
    df_h0 = dist_h0.copy()
    df_h0.insert(0, "dim", 0)
    df_h0.index.name = "group"
    df_h1 = dist_h1.copy()
    df_h1.insert(0, "dim", 1)
    df_h1.index.name = "group"
    combined = pd.concat([df_h0.reset_index(), df_h1.reset_index()])

    csv_path = RESULTS_DIR / "wasserstein_distances.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved distances  → {csv_path}")

    print_table(dist_h0, "Wasserstein distances — H0 (connected components)")
    print_table(dist_h1, "Wasserstein distances — H1 (loops)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run the bootstrap stability analysis instead of "
                             "the main pipeline.")
    parser.add_argument("--n-iter", type=int, default=1000,
                        help="Bootstrap iterations (default: 1000).")
    parser.add_argument("--frac", type=float, default=0.8,
                        help="Per-group subsample fraction (default: 0.8).")
    args = parser.parse_args()
    if args.bootstrap:
        bootstrap_stability(n_iter=args.n_iter, frac=args.frac)
    else:
        main()
