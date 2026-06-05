"""Linear-probe follow-up to rlhf_compare.py.

For the BERT family, MNLI fine-tuning shrinks the layer-5 lgbtq↔heteronormative
H1 Wasserstein distance by ~43%. For the RoBERTa family, CoLA fine-tuning
*amplifies* it by ~167% at its own peak layer (L12). Wasserstein alone can't
say whether either change reflects:

  (1) genuine info loss / gain — orientation information is actually weaker
      (BERT) or stronger (RoBERTa) in the fine-tuned representation, or
  (2) feature reorganization — orientation info is preserved but the cloud's
      geometric layout changed (axes rotated, signal dispersed or compressed),
      or
  (3) catastrophic forgetting / representational specialization — generic
      structure shifts as a side effect of the downstream objective.

A linear probe distinguishes these: if a logistic regression on the [CLS]
vector can still classify lgbtq_explicit vs heteronormative with the same
cross-validated accuracy after fine-tuning, the information is linearly
recoverable and the Wasserstein change reflects *geometric reorganization*,
not loss/gain. A large accuracy drop points to attenuation; a large gain
points to enhanced linear separability.

CLI:
  python src/linear_probe.py
      Probe the BERT pair at fixed LAYER=5. Writes results/linear_probe.csv,
      results/pca_probe.csv, results/pca_probe.png.

  python src/linear_probe.py --per-family-peak
      Probe all 4 models (BERT pair + RoBERTa pair) each at its base family's
      peak layer (BERT L5, RoBERTa L12; read from layerwise_wasserstein.csv).
      Writes results/linear_probe_peak.csv, results/pca_probe_peak.csv,
      results/pca_probe_peak.png.

Setup: 80 lgbtq + 40 heteronormative = 120 samples in 768-dim space. Class
imbalance handled with stratified 5-fold CV; both raw and balanced accuracy
are reported alongside the majority-class chance baseline (66.7%).
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from rlhf_compare import (layer_cls, LAYER, MODELS, ALL_MODELS, MODEL_META,
                          family_peak_layers)

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"
SEED    = 42
N_FOLDS = 5
PCA_KS  = [2, 5, 10, 20, 50]
ACC_THRESHOLD = 0.90

# styles for the 4-curve peak-mode plot
PALETTE   = {"bert": "#d62728", "roberta": "#1f77b4"}
LINESTYLE = {"base": "-",       "fine_tuned": "--"}
MARKER    = {"base": "o",       "fine_tuned": "s"}


def probe(X: np.ndarray, y: np.ndarray) -> dict:
    """Stratified 5-fold logistic regression on (X, y); return CV summary."""
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                             random_state=SEED)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = cross_validate(clf, X, y, cv=cv,
                         scoring=("accuracy", "balanced_accuracy"),
                         return_train_score=False)
    return {
        "mean_accuracy":          out["test_accuracy"].mean(),
        "std_accuracy":           out["test_accuracy"].std(ddof=1),
        "mean_balanced_accuracy": out["test_balanced_accuracy"].mean(),
        "std_balanced_accuracy":  out["test_balanced_accuracy"].std(ddof=1),
        "fold_accuracies":        out["test_accuracy"].tolist(),
    }


def pca_probe(X: np.ndarray, y: np.ndarray, ks: list[int]) -> pd.DataFrame:
    """Logistic regression on the top-k principal components for each k.

    PCA is fit inside each CV fold (via Pipeline) so no test variance leaks
    into the training projection.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for k in ks:
        pipe = Pipeline([
            ("pca", PCA(n_components=k, random_state=SEED)),
            ("lr",  LogisticRegression(max_iter=2000, C=1.0,
                                       solver="lbfgs", random_state=SEED)),
        ])
        out = cross_validate(pipe, X, y, cv=cv,
                             scoring=("accuracy", "balanced_accuracy"))
        rows.append({
            "k":                      k,
            "mean_accuracy":          out["test_accuracy"].mean(),
            "std_accuracy":           out["test_accuracy"].std(ddof=1),
            "mean_balanced_accuracy": out["test_balanced_accuracy"].mean(),
            "std_balanced_accuracy":  out["test_balanced_accuracy"].std(ddof=1),
        })
    return pd.DataFrame(rows)


def min_k_for_threshold(df: pd.DataFrame, threshold: float) -> int | None:
    """Smallest k where mean_accuracy ≥ threshold, or None if never reached."""
    hits = df[df["mean_accuracy"] >= threshold]["k"].tolist()
    return int(min(hits)) if hits else None


def run(per_family_peak: bool) -> None:
    RESULTS.mkdir(exist_ok=True)

    if per_family_peak:
        models_to_probe = ALL_MODELS
        peaks = family_peak_layers()
        layer_for = lambda label: peaks[MODEL_META[label][0]]
        suffix = "_peak"
        mode_label = "per-family peak layer"
        print(f"Mode: per-family peak layer "
              f"(bert→L{peaks['bert']}, roberta→L{peaks['roberta']})\n")
    else:
        models_to_probe = MODELS
        layer_for = lambda label: LAYER
        suffix = ""
        mode_label = f"fixed layer {LAYER}"
        print(f"Mode: fixed layer {LAYER}\n")

    df = pd.read_csv(DATA / "stimuli.csv")
    sentences = df["sentence"].tolist()
    mask = (df["group"] == "lgbtq_explicit") | (df["group"] == "heteronormative")
    y = (df.loc[mask, "group"] == "lgbtq_explicit").astype(int).values
    chance = max(y.mean(), 1 - y.mean())
    print(f"Samples: {mask.sum()} (lgbtq {y.sum()} / hetero {(1 - y).sum()})  "
          f"| chance (majority) = {chance:.3f}\n")

    # ---------------- full-rank probe -----------------------------------
    rows = []
    X_by_model: dict[str, np.ndarray] = {}
    for label, hub_id in models_to_probe.items():
        fam, variant = MODEL_META[label]
        layer = layer_for(label)
        print(f"  • {label}  [{fam}/{variant}]  @ layer {layer}")
        cls = layer_cls(hub_id, sentences, layer)
        X = cls[mask.values]
        X_by_model[label] = X
        r = probe(X, y)
        print(f"    accuracy          : {r['mean_accuracy']:.3f} ± "
              f"{r['std_accuracy']:.3f}  folds={[f'{a:.2f}' for a in r['fold_accuracies']]}")
        print(f"    balanced accuracy : {r['mean_balanced_accuracy']:.3f} ± "
              f"{r['std_balanced_accuracy']:.3f}\n")
        rows.append({
            "model":                  label,
            "hub_id":                 hub_id,
            "base_family":            fam,
            "variant":                variant,
            "layer":                  layer,
            "n_samples":              int(mask.sum()),
            "n_features":             int(X.shape[1]),
            "n_folds":                N_FOLDS,
            "mean_accuracy":          r["mean_accuracy"],
            "std_accuracy":           r["std_accuracy"],
            "mean_balanced_accuracy": r["mean_balanced_accuracy"],
            "std_balanced_accuracy":  r["std_balanced_accuracy"],
            "chance_majority":        chance,
        })

    res = pd.DataFrame(rows)
    out_csv = RESULTS / f"linear_probe{suffix}.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    # ---------------- full-rank verdict (per family) --------------------
    print(f"\nFull-rank linear probe — {mode_label}")
    print("-" * 72)
    for _, r in res.iterrows():
        print(f"  {r['base_family']:<8s} {r['variant']:<11s} "
              f"L{int(r['layer']):<3d} {r['model']:<32s} "
              f"acc {r['mean_accuracy']:.3f} | bal-acc "
              f"{r['mean_balanced_accuracy']:.3f}")
    print(f"  chance (majority): {chance:.3f}")

    families_present = sorted(res["base_family"].unique())
    print("\nInterpretation (per family)")
    print("-" * 72)
    for fam in families_present:
        fam_label_h = {"bert": "BERT", "roberta": "RoBERTa"}[fam]
        fam_rows = res[res["base_family"] == fam]
        base_row = fam_rows[fam_rows["variant"] == "base"].iloc[0]
        ft_row   = fam_rows[fam_rows["variant"] == "fine_tuned"].iloc[0]
        delta_bal = ft_row["mean_balanced_accuracy"] - base_row["mean_balanced_accuracy"]
        layer = int(base_row["layer"])
        big_drop = delta_bal < -0.10
        big_gain = delta_bal > 0.10
        both_high = (base_row["mean_balanced_accuracy"] > 0.80 and
                     ft_row["mean_balanced_accuracy"]   > 0.80)
        print(f"\n  {fam_label_h}  (layer {layer}):")
        print(f"    base  bal-acc : {base_row['mean_balanced_accuracy']:.3f}")
        print(f"    FT    bal-acc : {ft_row['mean_balanced_accuracy']:.3f}")
        print(f"    Δ             : {delta_bal*100:+.1f} pp")
        if big_drop:
            print(f"    → GENUINE ATTENUATION (info loss) — FT lost "
                  f"{-delta_bal*100:.1f} pp of recoverability.")
        elif big_gain:
            print(f"    → INFO GAIN — FT *added* {delta_bal*100:.1f} pp of "
                  f"recoverability; not just geometric stretch.")
        elif both_high:
            print(f"    → FEATURE REORGANIZATION — both retain high "
                  f"recoverability; the Wasserstein\n      change reflects "
                  f"geometric layout, not information content.")
        else:
            print(f"    → MIXED / inconclusive: balanced accuracy moved by "
                  f"{delta_bal*100:+.1f} pp.")

    # ---------------- PCA bottleneck ------------------------------------
    print("\n" + "=" * 72)
    print(f"PCA bottleneck probe — accuracy vs top-k principal components")
    print("=" * 72)

    pca_rows = []
    pca_curves: dict[str, pd.DataFrame] = {}
    for label, X in X_by_model.items():
        layer = layer_for(label)
        print(f"\n  • {label}  @ L{layer}  (k ∈ {PCA_KS})")
        curve = pca_probe(X, y, PCA_KS)
        pca_curves[label] = curve
        evr = PCA(n_components=max(PCA_KS), random_state=SEED).fit(X).explained_variance_ratio_
        cum = np.cumsum(evr)
        for _, row in curve.iterrows():
            k = int(row["k"])
            print(f"    k={k:3d}  acc {row['mean_accuracy']:.3f} ± "
                  f"{row['std_accuracy']:.3f}  |  bal-acc "
                  f"{row['mean_balanced_accuracy']:.3f}  "
                  f"|  PCA cum-var {cum[k - 1]:.3f}")
            fam, variant = MODEL_META[label]
            pca_rows.append({
                "model":                  label,
                "base_family":            fam,
                "variant":                variant,
                "layer":                  layer,
                "k":                      k,
                "mean_accuracy":          row["mean_accuracy"],
                "std_accuracy":           row["std_accuracy"],
                "mean_balanced_accuracy": row["mean_balanced_accuracy"],
                "std_balanced_accuracy":  row["std_balanced_accuracy"],
                "cumulative_variance":    float(cum[k - 1]),
            })

    pca_df = pd.DataFrame(pca_rows)
    pca_csv = RESULTS / f"pca_probe{suffix}.csv"
    pca_df.to_csv(pca_csv, index=False)
    print(f"\nSaved → {pca_csv}")

    min_ks = {label: min_k_for_threshold(c, ACC_THRESHOLD)
              for label, c in pca_curves.items()}
    print(f"\nMinimum k with mean accuracy ≥ {ACC_THRESHOLD:.2f}:")
    for label, k in min_ks.items():
        print(f"  {label:<32s}  k = "
              f"{k if k is not None else f'> {max(PCA_KS)}'}")

    # ---------------- plot ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for label, curve in pca_curves.items():
        fam, variant = MODEL_META[label]
        ax.errorbar(
            curve["k"], curve["mean_accuracy"],
            yerr=curve["std_accuracy"],
            marker=MARKER[variant], linestyle=LINESTYLE[variant],
            lw=2, capsize=4, color=PALETTE[fam],
            label=f"{label} (L{layer_for(label)})",
        )
    ax.axhline(ACC_THRESHOLD, color="grey", ls="--", alpha=0.5,
               label=f"threshold = {ACC_THRESHOLD}")
    ax.axhline(chance, color="black", ls=":", alpha=0.5,
               label=f"chance (majority) = {chance:.3f}")
    ax.set_xlabel("number of PCA components (k)")
    ax.set_ylabel("5-fold CV accuracy  (lgbtq vs heteronormative)")
    ax.set_title(f"PCA bottleneck probe — {mode_label}")
    ax.set_xticks(PCA_KS)
    ax.set_ylim(min(0.45, chance - 0.05), 1.02)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_png = RESULTS / f"pca_probe{suffix}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_png}")

    # ---------------- PCA verdict (per family) --------------------------
    print("\nInterpretation (PCA bottleneck, per family)")
    print("-" * 72)
    for fam in families_present:
        fam_label_h = {"bert": "BERT", "roberta": "RoBERTa"}[fam]
        base_label = next(l for l, m in MODEL_META.items()
                          if m[0] == fam and m[1] == "base"
                          and l in pca_curves)
        ft_label = next(l for l, m in MODEL_META.items()
                        if m[0] == fam and m[1] == "fine_tuned"
                        and l in pca_curves)
        k_base = min_ks[base_label]
        k_ft   = min_ks[ft_label]
        print(f"\n  {fam_label_h}:")
        print(f"    base       min-k : {k_base if k_base is not None else f'> {max(PCA_KS)}'}")
        print(f"    fine-tuned min-k : {k_ft   if k_ft   is not None else f'> {max(PCA_KS)}'}")
        if k_base is None or k_ft is None:
            print(f"    → at least one model didn't reach {ACC_THRESHOLD:.2f} "
                  f"within k ≤ {max(PCA_KS)}.")
        elif k_ft < k_base:
            print(f"    → FT needs FEWER components ({k_ft} < {k_base}) — "
                  f"orientation signal\n      CONCENTRATED into a "
                  f"lower-dimensional subspace by fine-tuning.")
        elif k_ft > k_base:
            print(f"    → FT needs MORE components ({k_ft} > {k_base}) — "
                  f"orientation signal\n      DIFFUSED across more directions "
                  f"by fine-tuning.")
        else:
            print(f"    → Equal k ({k_base}). Dimensionality unchanged at "
                  f"this resolution.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--per-family-peak", action="store_true",
        help="Probe all 4 models at their base family's peak layer (read "
             "from results/layerwise_wasserstein.csv). Writes "
             "linear_probe_peak.csv, pca_probe_peak.csv, pca_probe_peak.png.",
    )
    args = parser.parse_args()
    run(per_family_peak=args.per_family_peak)


if __name__ == "__main__":
    main()
