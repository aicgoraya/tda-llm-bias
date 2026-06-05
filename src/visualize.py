"""Visualization of TDA bias-analysis results.

Produces three figures in results/:
  persistence_diagrams.png  2x2 grid of per-group persistence diagrams
  wasserstein_heatmap.png   heatmap of the H1 Wasserstein distance matrix
  umap_embeddings.png       UMAP scatter of all embeddings, colored by group
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import plot_diagrams

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

GROUPS = ["lgbtq_explicit", "heteronormative", "neutral", "religious_conservative"]


def figure_persistence_diagrams(diagrams: dict[str, list]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, group in zip(axes.flat, GROUPS):
        plot_diagrams(diagrams[group], ax=ax)
        ax.set_title(group)
    fig.suptitle("Persistence diagrams by identity group", fontsize=14)
    fig.tight_layout()
    out = RESULTS_DIR / "persistence_diagrams.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_wasserstein_heatmap(dist_df: pd.DataFrame) -> Path:
    h1 = dist_df[dist_df["dim"] == 1].set_index("group").loc[GROUPS, GROUPS]
    mat = h1.values

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="coolwarm")
    fig.colorbar(im, ax=ax, label="Wasserstein distance (H1)")

    ax.set_xticks(range(len(GROUPS)), labels=GROUPS, rotation=45, ha="right")
    ax.set_yticks(range(len(GROUPS)), labels=GROUPS)

    vmid = (mat.max() + mat.min()) / 2
    for i in range(len(GROUPS)):
        for j in range(len(GROUPS)):
            ax.text(
                j, i, f"{mat[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(mat[i, j] - vmid) > (mat.max() - vmid) * 0.5 else "black",
            )

    ax.set_title("Pairwise Wasserstein distances — H1 (loops)")
    fig.tight_layout()
    out = RESULTS_DIR / "wasserstein_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_umap(embeddings: np.ndarray, meta: pd.DataFrame) -> Path:
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 8))
    for group in GROUPS:
        mask = meta["group"].values == group
        ax.scatter(proj[mask, 0], proj[mask, 1], label=group, alpha=0.7, s=40)
    ax.legend(title="group")
    ax.set_title("UMAP projection of sentence embeddings")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    out = RESULTS_DIR / "umap_embeddings.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    with (RESULTS_DIR / "persistence_diagrams.pkl").open("rb") as f:
        diagrams = pickle.load(f)
    dist_df = pd.read_csv(RESULTS_DIR / "wasserstein_distances.csv")
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    meta = pd.read_csv(DATA_DIR / "embeddings_meta.csv")

    for out in (
        figure_persistence_diagrams(diagrams),
        figure_wasserstein_heatmap(dist_df),
        figure_umap(embeddings, meta),
    ):
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
