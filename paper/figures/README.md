# paper/figures

Publication figures for `paper/paper.md`. Each figure is produced by one
self-contained script in `scripts/`, reading data from `results/*.csv` at the
repo root and writing both `.pdf` (vector, for LaTeX) and `.png` (raster, for
preview) at 300 dpi.

## Regenerate all figures

From the repo root, with the project venv active:

```bash
source .venv/bin/activate          # or: .venv/bin/python -m ...
for f in paper/figures/scripts/fig*.py; do
    python "$f"
done
```

Outputs land in `paper/figures/fig{1..5}_*.{pdf,png}`. Scripts are
deterministic; rerunning overwrites in place with identical bytes (modulo
PDF/PNG metadata timestamps).

## Figure → source mapping

| Figure | Source                                                           | Script                                |
|--------|------------------------------------------------------------------|---------------------------------------|
| 1      | `data/embeddings.npy` + `data/embeddings_meta.csv` (see note ↓)  | `scripts/fig1_permutation.py`         |
| 2      | `results/layerwise_wasserstein{,_normalized}.csv`                | `scripts/fig2_layerwise.py`           |
| 3      | `results/baseline_sizematched.csv`                               | `scripts/fig3_baselines.py`           |
| 4      | `results/rlhf_comparison_peak.csv`                               | `scripts/fig4_finetuning.py`          |
| 5      | `results/pca_probe_peak.csv`                                     | `scripts/fig5_pca_probe.py`           |

Shared style (palette, rcParams, save helper) lives in `scripts/_style.py`.

## Note on Fig 1

The permutation null distribution is not persisted to disk anywhere in the
repo — it lives only in `notebooks/analysis.ipynb` §6 at runtime. The
script regenerates it deterministically from the embeddings using the same
seed (`np.random.default_rng(42)`) and `n_perm = 1000` as the notebook, so
the values reproduce the notebook's output exactly (observed = 4.7685,
null mean = 2.2578, p = 0.001). Figures 2–5 read existing CSVs only.

## Dependencies

All required packages (`matplotlib`, `pandas`, `numpy`, `ripser`, `persim`)
are already pinned in the repo's top-level `requirements.txt`; no
figure-specific extras are needed.
