"""Does downstream fine-tuning preserve or reduce the orientation cluster?

The layer-wise probe (src/layerwise.py) localized the lgbtq_explicit ↔
heteronormative H1 Wasserstein peak to layer 5 of bert-base-uncased and to
layer 12 of roberta-base. This module asks whether fine-tuned descendants of
base encoders still carry that mid-network orientation structure, or whether
the fine-tuning objective dampens / preserves / amplifies it.

Two families are probed:
  - BERT:    bert-base-uncased   vs textattack/bert-base-uncased-MNLI
  - RoBERTa: roberta-base        vs textattack/roberta-base-CoLA

CLI:
  python src/rlhf_compare.py
      Probe both families at the fixed LAYER = 5 (BERT empirical peak).
      Writes results/rlhf_comparison.csv and .png.

  python src/rlhf_compare.py --per-family-peak
      For each family, look up the base model's peak layer from
      results/layerwise_wasserstein.csv and probe both base + fine-tuned at
      that family's own peak. Writes results/rlhf_comparison_peak.csv and
      .png. This is the within-family-fair comparison: each architecture is
      evaluated where its orientation cluster is most concentrated.

Caveat 1 — fine-tuning ≠ RLHF. textattack/* models are supervised fine-tunes
on a downstream task (MNLI / CoLA), used here as tractable proxies for
"the same base after a downstream training pass." Real preference-optimized
models (DPO/RLHF) remain future work.

Caveat 2 — BERT-MNLI and RoBERTa-CoLA were fine-tuned on *different* tasks
(3-way entailment, sentence-pair vs. binary acceptability, single-sentence),
so within-family base-vs-fine-tuned deltas are the clean comparison.
Cross-family Wasserstein magnitudes confound architecture + tokenizer +
fine-tuning task and should NOT be compared directly.
"""

import argparse
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

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

LAYER   = 5    # default fixed layer (BERT empirical peak)
BATCH   = 32

# MODELS kept as the BERT pair for backward compatibility with
# src/linear_probe.py, which imports MODELS and probes those 2 models. To
# extend the linear probe to RoBERTa, edit linear_probe.py to import
# ALL_MODELS instead.
MODELS = {
    "bert-base-uncased":              "bert-base-uncased",
    "textattack-bert-base-MNLI":      "textattack/bert-base-uncased-MNLI",
}
ROBERTA_PAIR = {
    "roberta-base":                   "roberta-base",
    "textattack-roberta-base-CoLA":   "textattack/roberta-base-CoLA",
}
ALL_MODELS = {**MODELS, **ROBERTA_PAIR}

# label -> (base_family, variant)
MODEL_META = {
    "bert-base-uncased":              ("bert",    "base"),
    "textattack-bert-base-MNLI":      ("bert",    "fine_tuned"),
    "roberta-base":                   ("roberta", "base"),
    "textattack-roberta-base-CoLA":   ("roberta", "fine_tuned"),
}

# family -> base model label (used to look up that family's peak layer)
FAMILY_BASE_LABEL = {"bert": "bert-base-uncased", "roberta": "roberta-base"}


def h1_diagram(X: np.ndarray):
    """H1 (loop) persistence diagram of a point cloud."""
    return ripser(X, maxdim=1)["dgms"][1]


def layer_cls(model_name: str, sentences: list[str], layer: int) -> np.ndarray:
    """Return (N, hidden_dim) [CLS] embeddings at the given transformer layer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(sentences), BATCH):
            batch = sentences[start:start + BATCH]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            hs = model(**enc).hidden_states  # tuple len 13 (0=input emb, 1-12=layers)
            chunks.append(hs[layer][:, 0, :].cpu().numpy())
    return np.vstack(chunks)


def family_peak_layers() -> dict[str, int]:
    """Read each base model's raw peak layer from results/layerwise_wasserstein.csv."""
    csv = RESULTS / "layerwise_wasserstein.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — run `python src/layerwise.py` first to "
            f"compute per-model peak layers.")
    df = pd.read_csv(csv)
    peaks: dict[str, int] = {}
    for fam, base_label in FAMILY_BASE_LABEL.items():
        sub = df[df["model"] == base_label]
        if sub.empty:
            raise ValueError(
                f"No rows for base model {base_label!r} in {csv}; can't "
                f"determine {fam} peak layer.")
        peak_row = sub.loc[sub["wasserstein_h1"].idxmax()]
        peaks[fam] = int(peak_row["layer"])
    return peaks


def run(per_family_peak: bool) -> None:
    RESULTS.mkdir(exist_ok=True)

    if per_family_peak:
        family_layer = family_peak_layers()
        suffix = "_peak"
        mode_label = "per-family peak layer"
        print(f"Mode: per-family peak layer (from layerwise CSV)")
        print(f"  bert    → layer {family_layer['bert']}")
        print(f"  roberta → layer {family_layer['roberta']}\n")
    else:
        family_layer = {"bert": LAYER, "roberta": LAYER}
        suffix = ""
        mode_label = f"fixed layer {LAYER}"
        print(f"Mode: fixed layer {LAYER} for all models\n")

    df = pd.read_csv(DATA / "stimuli.csv")
    sentences = df["sentence"].tolist()
    is_lgbtq  = (df["group"] == "lgbtq_explicit").values
    is_hetero = (df["group"] == "heteronormative").values
    print(f"Loaded {len(sentences)} sentences | "
          f"lgbtq {is_lgbtq.sum()} | hetero {is_hetero.sum()}")
    print(f"Probing {len(ALL_MODELS)} models …\n")

    rows = []
    for label, hub_id in ALL_MODELS.items():
        fam, variant = MODEL_META[label]
        layer = family_layer[fam]
        print(f"  • {label}  [{fam}/{variant}]  @ layer {layer}")
        cls = layer_cls(hub_id, sentences, layer)
        dist = wasserstein(h1_diagram(cls[is_lgbtq]),
                           h1_diagram(cls[is_hetero]))
        rows.append({
            "model":          label,
            "hub_id":         hub_id,
            "base_family":    fam,
            "variant":        variant,
            "layer":          layer,
            "wasserstein_h1": dist,
        })
        print(f"    → H1 Wasserstein (lgbtq vs hetero): {dist:.4f}\n")

    res = pd.DataFrame(rows)
    out_csv = RESULTS / f"rlhf_comparison{suffix}.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    # side-by-side table -------------------------------------------------
    print(f"\nH1 Wasserstein (lgbtq_explicit vs heteronormative) — {mode_label}")
    print("-" * 78)
    for _, r in res.iterrows():
        print(f"  {r['base_family']:<8s} {r['variant']:<11s} "
              f"L{int(r['layer']):<3d} {r['model']:<32s} "
              f"{r['wasserstein_h1']:.4f}")

    # bar plot: 2 families × 2 conditions --------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    families = ["bert", "roberta"]
    x = np.arange(len(families))
    width = 0.35
    base_vals = [res[(res.base_family == f) & (res.variant == "base")]
                 .wasserstein_h1.iloc[0] for f in families]
    ft_vals   = [res[(res.base_family == f) & (res.variant == "fine_tuned")]
                 .wasserstein_h1.iloc[0] for f in families]
    fam_layers = [family_layer[f] for f in families]
    bars_b = ax.bar(x - width / 2, base_vals, width,
                    label="base", color="#d62728")
    bars_f = ax.bar(x + width / 2, ft_vals,   width,
                    label="fine-tuned (supervised downstream)",
                    color="#1f77b4")
    for bars, vals in [(bars_b, base_vals), (bars_f, ft_vals)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f.upper()} (L{l})" for f, l in zip(families, fam_layers)])
    ax.set_ylabel("H1 Wasserstein distance "
                  "(lgbtq_explicit vs heteronormative)")
    ax.set_title(f"Orientation cluster — base vs fine-tuned ({mode_label})")
    ax.legend()
    ax.margins(y=0.20)
    fig.tight_layout()
    out_png = RESULTS / f"rlhf_comparison{suffix}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_png}")

    # per-family verdict -------------------------------------------------
    print("\nPer-family interpretation")
    print("-" * 78)
    for fam_label_h, fam in [("BERT", "bert"), ("RoBERTa", "roberta")]:
        base_row = res[(res.base_family == fam) & (res.variant == "base")].iloc[0]
        ft_row   = res[(res.base_family == fam) & (res.variant == "fine_tuned")].iloc[0]
        delta_abs = ft_row.wasserstein_h1 - base_row.wasserstein_h1
        delta_pct = delta_abs / base_row.wasserstein_h1 * 100
        print(f"\n  {fam_label_h}  (layer {family_layer[fam]}):")
        print(f"    base       : {base_row.wasserstein_h1:.4f}")
        print(f"    fine-tuned : {ft_row.wasserstein_h1:.4f}")
        print(f"    Δ          : {delta_abs:+.4f}  ({delta_pct:+.1f}%)")
        if abs(delta_pct) < 10:
            print(f"    → PRESERVED (within ±10%)")
        elif delta_pct < 0:
            print(f"    → REDUCED by {-delta_pct:.1f}%")
        else:
            print(f"    → AMPLIFIED by {delta_pct:.1f}%")

    print("\nCaveats")
    print("-" * 78)
    print("• Cross-family magnitudes are NOT directly comparable (different")
    print("  tokenizers / pretraining / fine-tuning task: MNLI vs CoLA).")
    print("• Absolute H1 magnitudes are N-sensitive (bootstrap CV 5.7%); each")
    print("  comparison above uses the same N (80 vs 40) so within-family")
    print("  deltas are meaningful.")
    if not per_family_peak:
        print("• Probe is at layer 5 (BERT empirical peak). For the fair")
        print("  within-family-at-peak comparison run with --per-family-peak.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--per-family-peak", action="store_true",
        help="Probe each model at its base family's peak layer (read from "
             "results/layerwise_wasserstein.csv) instead of the fixed "
             f"layer {LAYER}. Writes rlhf_comparison_peak.csv/.png.",
    )
    args = parser.parse_args()
    run(per_family_peak=args.per_family_peak)


if __name__ == "__main__":
    main()
