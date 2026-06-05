"""Tokenization control: is the LGBTQ+ topological signal a subword-length artifact?

A natural confound for the main finding is that identity terms might fragment
into different numbers of subword tokens (e.g. a rare term splitting into many
WordPieces), which alone could perturb a sentence embedding. This module:

  1. tokenizes every identity_term with the model's own tokenizer,
  2. tags each sentence with its term's subword token count,
  3. recomputes the lgbtq_explicit vs heteronormative H1 Wasserstein distance
     *within* each token-count stratum (length-matched comparison),
  4. reports whether the signal survives length matching.

token_count depends only on identity_term, so the Wasserstein step maps counts
onto embeddings_meta.csv (row-aligned with embeddings.npy) — no fragile
positional join with stimuli.csv is required.
"""

import warnings
warnings.filterwarnings("ignore")  # ripser shape / persim inf-death notes

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from ripser import ripser
from persim import wasserstein

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data"
RESULTS  = ROOT / "results"
MODEL    = "sentence-transformers/all-MiniLM-L6-v2"

GROUP_COLORS = {
    "lgbtq_explicit":         "#d62728",
    "heteronormative":        "#1f77b4",
    "neutral":                "#7f7f7f",
    "religious_conservative": "#2ca02c",
}


def h1_diagram(X: np.ndarray):
    """H1 (loop) persistence diagram of a point cloud."""
    return ripser(X, maxdim=1)["dgms"][1]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    # 1. load stimuli ----------------------------------------------------
    df = pd.read_csv(DATA / "stimuli.csv")

    # 2. subword token count per identity_term ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    unique_terms = sorted(df["identity_term"].unique())
    term_token_count = {t: len(tokenizer.tokenize(t)) for t in unique_terms}

    # 3. attach token_count column ---------------------------------------
    df["token_count"] = df["identity_term"].map(term_token_count)

    # 4. per-term table --------------------------------------------------
    term_tbl = (
        df[["identity_term", "group", "token_count"]]
        .drop_duplicates()
        .sort_values(["group", "token_count", "identity_term"])
        .reset_index(drop=True)
    )
    print("Subword token count per identity term")
    print("-" * 55)
    print(term_tbl.to_string(index=False))

    # 5. token-count-matched Wasserstein H1 (lgbtq vs heteronormative) ---
    emb  = np.load(DATA / "embeddings.npy")
    meta = pd.read_csv(DATA / "embeddings_meta.csv")
    meta["token_count"] = meta["identity_term"].map(term_token_count)

    is_lgbtq  = (meta["group"] == "lgbtq_explicit").values
    is_hetero = (meta["group"] == "heteronormative").values

    # overall (unmatched) reference distance
    overall = wasserstein(h1_diagram(emb[is_lgbtq]), h1_diagram(emb[is_hetero]))

    rows = []
    all_counts = sorted(set(meta.loc[is_lgbtq | is_hetero, "token_count"]))
    for tc in all_counts:
        a_mask = is_lgbtq  & (meta["token_count"].values == tc)
        b_mask = is_hetero & (meta["token_count"].values == tc)
        na, nb = int(a_mask.sum()), int(b_mask.sum())
        if na > 0 and nb > 0:
            dist = wasserstein(h1_diagram(emb[a_mask]), h1_diagram(emb[b_mask]))
        else:
            dist = np.nan  # cannot match this stratum
        rows.append({
            "token_count":       tc,
            "n_lgbtq":           na,
            "n_heteronormative": nb,
            "lgbtq_terms":       ",".join(sorted(meta.loc[a_mask, "identity_term"].unique())),
            "hetero_terms":      ",".join(sorted(meta.loc[b_mask, "identity_term"].unique())),
            "wasserstein_h1":    dist,
        })
    control = pd.DataFrame(rows)

    # 6. save ------------------------------------------------------------
    out_csv = RESULTS / "token_control.csv"
    control.to_csv(out_csv, index=False)
    print(f"\nMatched-stratum results → {out_csv}")
    print(control.to_string(index=False))

    # 7. bar chart of token counts by term -------------------------------
    plot_df = term_tbl.sort_values(["group", "identity_term"]).reset_index(drop=True)
    colors = [GROUP_COLORS[g] for g in plot_df["group"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(plot_df["identity_term"], plot_df["token_count"], color=colors)
    ax.set_xlabel("identity term")
    ax.set_ylabel("subword token count")
    ax.set_title("Subword token count per identity term (all-MiniLM-L6-v2 tokenizer)")
    ax.set_yticks(range(0, plot_df["token_count"].max() + 1))
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    ax.legend(handles, GROUP_COLORS.keys(), title="group")
    fig.tight_layout()
    out_png = RESULTS / "token_distribution.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nToken-count bar chart → {out_png}")

    # 8. verdict ---------------------------------------------------------
    matched = control.dropna(subset=["wasserstein_h1"])
    print("\n" + "=" * 60)
    print("DOES THE WASSERSTEIN SIGNAL PERSIST AFTER TOKEN MATCHING?")
    print("=" * 60)
    print(f"Unmatched lgbtq vs hetero H1 distance : {overall:.4f}")

    lgbtq_counts  = set(meta.loc[is_lgbtq,  "token_count"])
    hetero_counts = set(meta.loc[is_hetero, "token_count"])
    if lgbtq_counts == hetero_counts and len(lgbtq_counts) == 1:
        only = next(iter(lgbtq_counts))
        print(f"\nAll lgbtq_explicit and heteronormative terms tokenize to "
              f"exactly {only} subword token(s).")
        print("The two groups are therefore *already* perfectly matched on "
              "token length, so the\nmatched comparison is identical to the "
              "unmatched one — the topological difference\nCANNOT be a "
              "subword-fragmentation artifact.")
        print(f"\nVERDICT: signal PERSISTS (matched distance "
              f"{matched['wasserstein_h1'].iloc[0]:.4f} == unmatched {overall:.4f}).")
    else:
        mean_matched = matched["wasserstein_h1"].mean()
        print(f"Mean within-token-count matched distance : {mean_matched:.4f}")
        persists = mean_matched >= 0.5 * overall and matched["wasserstein_h1"].min() > 0
        verdict = "PERSISTS" if persists else "ATTENUATED / possibly confounded"
        print(f"\nVERDICT: signal {verdict} "
              f"(mean matched {mean_matched:.4f} vs unmatched {overall:.4f}).")


if __name__ == "__main__":
    main()
