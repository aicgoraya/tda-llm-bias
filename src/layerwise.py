"""Layer-wise topological bias trajectory through transformer encoders.

Extracts the [CLS] embedding from each of the 12 transformer layers for all
200 stimulus sentences, then tracks the Vietoris-Rips H1 Wasserstein distance
between the lgbtq_explicit and heteronormative groups as a function of depth.
This shows *where* in the network the topological separation emerges.

Models probed:
  - bert-base-uncased
  - roberta-base

Both architectures expose 13 hidden states (index 0 = input embeddings,
1-12 = transformer-layer outputs), so transformer layer i corresponds to
hidden_states[i]. Position 0 is the special start token: [CLS] for BERT,
<s> for RoBERTa — both serve as the sentence-level pooled representation.

Two variants are computed per layer: raw [CLS] vectors and L2-normalized
(unit-sphere) [CLS] vectors. Raw magnitudes partly reflect each layer's
activation scale; the normalized variant removes that scale, so a matching
trajectory *shape* across both confirms each peak is a genuine geometric
feature rather than a norm-growth artifact.

Caveat for cross-family comparison: BERT and RoBERTa have different
tokenizers and were pretrained on different corpora with different
objectives, so absolute Wasserstein magnitudes are not directly comparable
between them. Within each model, the per-layer trajectory shape and its
raw-vs-normalized agreement are the meaningful quantities.
"""

import warnings
warnings.filterwarnings("ignore")  # ripser shape / persim inf-death notes

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from ripser import ripser
from persim import wasserstein

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data"
RESULTS  = ROOT / "results"
MODELS   = {
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base":      "roberta-base",
}
N_LAYERS = 12
BATCH    = 32


def h1_diagram(X: np.ndarray):
    """H1 (loop) persistence diagram of a point cloud."""
    return ripser(X, maxdim=1)["dgms"][1]


def extract_cls_by_layer(model_name: str, sentences: list[str]) -> dict[int, np.ndarray]:
    """Return {layer 1..12 -> (N, hidden_dim) array of [CLS] embeddings}."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    per_layer: dict[int, list[np.ndarray]] = {l: [] for l in range(1, N_LAYERS + 1)}
    with torch.no_grad():
        for start in range(0, len(sentences), BATCH):
            batch = sentences[start:start + BATCH]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            hidden_states = model(**enc).hidden_states  # tuple length 13
            for layer in range(1, N_LAYERS + 1):
                cls = hidden_states[layer][:, 0, :]  # [CLS] (BERT) or <s> (RoBERTa)
                per_layer[layer].append(cls.cpu().numpy())

    return {layer: np.vstack(chunks) for layer, chunks in per_layer.items()}


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Project each row vector onto the unit sphere."""
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def layer_distances(
    cls_by_layer: dict[int, np.ndarray],
    is_a: np.ndarray,
    is_b: np.ndarray,
    normalize: bool,
) -> pd.DataFrame:
    """Per-layer H1 Wasserstein distance between two groups, raw or L2-normalized."""
    rows = []
    for layer in range(1, N_LAYERS + 1):
        emb = cls_by_layer[layer]
        if normalize:
            emb = l2_normalize(emb)
        dist = wasserstein(h1_diagram(emb[is_a]), h1_diagram(emb[is_b]))
        rows.append({"layer": layer, "wasserstein_h1": dist})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    df = pd.read_csv(DATA / "stimuli.csv")
    sentences = df["sentence"].tolist()
    is_lgbtq  = (df["group"] == "lgbtq_explicit").values
    is_hetero = (df["group"] == "heteronormative").values
    print(f"Stimuli: {len(sentences)} | lgbtq {is_lgbtq.sum()} | "
          f"hetero {is_hetero.sum()}\n")

    all_raw:  list[pd.DataFrame] = []
    all_norm: list[pd.DataFrame] = []
    for label, hub_id in MODELS.items():
        print(f"Extracting [CLS] embeddings from {N_LAYERS} layers of "
              f"{hub_id} …")
        cls_by_layer = extract_cls_by_layer(hub_id, sentences)
        raw  = layer_distances(cls_by_layer, is_lgbtq, is_hetero, normalize=False)
        norm = layer_distances(cls_by_layer, is_lgbtq, is_hetero, normalize=True)
        raw.insert(0, "model", label)
        norm.insert(0, "model", label)
        all_raw.append(raw)
        all_norm.append(norm)

        print(f"  layer |     raw  | normalized")
        print(f"  ------+----------+-----------")
        for _, r in raw.iterrows():
            n = norm[norm["layer"] == r["layer"]].iloc[0]
            print(f"   {int(r['layer']):3d}  | {r['wasserstein_h1']:8.4f} | "
                  f"{n['wasserstein_h1']:8.4f}")
        print()

    raw_df  = pd.concat(all_raw,  ignore_index=True)
    norm_df = pd.concat(all_norm, ignore_index=True)

    raw_csv  = RESULTS / "layerwise_wasserstein.csv"
    norm_csv = RESULTS / "layerwise_wasserstein_normalized.csv"
    raw_df.to_csv(raw_csv, index=False)
    norm_df.to_csv(norm_csv, index=False)
    print(f"Saved → {raw_csv}")
    print(f"Saved → {norm_csv}\n")

    # peak summary --------------------------------------------------------
    print("Peak layer per model:")
    print("-" * 80)
    for label in MODELS:
        r = raw_df[raw_df["model"] == label]
        best   = r.loc[r["wasserstein_h1"].idxmax()]
        n      = norm_df[norm_df["model"] == label]
        best_n = n.loc[n["wasserstein_h1"].idxmax()]
        corr   = np.corrcoef(r["wasserstein_h1"], n["wasserstein_h1"])[0, 1]
        print(f"  {label:<22s}  raw peak @ layer {int(best['layer'])} "
              f"({best['wasserstein_h1']:.4f})  |  norm peak @ layer "
              f"{int(best_n['layer'])} ({best_n['wasserstein_h1']:.4f})  |  "
              f"Pearson r = {corr:.3f}")

    # plot 1: side-by-side raw trajectories -------------------------------
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 5), sharex=True)
    if len(MODELS) == 1:
        axes = [axes]
    for ax, label in zip(axes, MODELS):
        r = raw_df[raw_df["model"] == label]
        best = r.loc[r["wasserstein_h1"].idxmax()]
        ax.plot(r["layer"], r["wasserstein_h1"], "o-", color="#d62728", lw=2)
        ax.axvline(best["layer"], color="grey", ls="--", alpha=0.7,
                   label=f"max @ layer {int(best['layer'])}")
        ax.set_xticks(range(1, N_LAYERS + 1))
        ax.set_xlabel("transformer layer")
        ax.set_title(label)
        ax.legend()
    axes[0].set_ylabel(
        "H1 Wasserstein distance\n(lgbtq_explicit vs heteronormative)")
    fig.suptitle("Layer-wise topological separation — BERT vs RoBERTa")
    fig.tight_layout()
    out_traj = RESULTS / "layerwise_trajectory.png"
    fig.savefig(out_traj, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {out_traj}")

    # plot 2: side-by-side raw vs normalized per model --------------------
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 5))
    if len(MODELS) == 1:
        axes = [axes]
    for ax1, label in zip(axes, MODELS):
        ax2 = ax1.twinx()
        r = raw_df[raw_df["model"] == label]
        n = norm_df[norm_df["model"] == label]
        corr = np.corrcoef(r["wasserstein_h1"], n["wasserstein_h1"])[0, 1]
        l1, = ax1.plot(r["layer"], r["wasserstein_h1"], "o-",
                       color="#d62728", lw=2, label="raw [CLS]")
        l2, = ax2.plot(n["layer"], n["wasserstein_h1"], "s--",
                       color="#1f77b4", lw=2, label="L2-normalized [CLS]")
        ax1.set_xticks(range(1, N_LAYERS + 1))
        ax1.set_xlabel("transformer layer")
        ax1.set_ylabel("H1 Wasserstein — raw", color="#d62728")
        ax2.set_ylabel("H1 Wasserstein — L2-normalized", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#d62728")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.set_title(f"{label}\nPearson r = {corr:.3f}")
        ax1.legend(handles=[l1, l2], loc="upper right", fontsize=9)
    fig.suptitle("Raw vs L2-normalized [CLS] trajectory — BERT vs RoBERTa")
    fig.tight_layout()
    out_cmp = RESULTS / "layerwise_comparison.png"
    fig.savefig(out_cmp, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_cmp}")


if __name__ == "__main__":
    main()
