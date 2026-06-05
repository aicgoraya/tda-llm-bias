# Downstream Fine-Tuning Reorganizes Identity-Coded Geometry in Transformer Embeddings: A Cross-Architecture Topological Probe

## Key Findings

- **LGBTQ+ embeddings are topologically distinct.** The `lgbtq_explicit` group's
  Wasserstein H1 distance from the identity-neutral baseline is significantly
  greater than chance (**p = 0.001**, permutation test, n = 1000). Bootstrap
  stability analysis (n = 1000, 80% subsample) confirms low resampling variance
  (CV = 5.7%); absolute H1 Wasserstein magnitudes are N-sensitive and should be
  interpreted relative to same-N comparators.
- **Heteronormative and neutral language is treated as the default.** These two
  groups show the smallest mutual topological separation, consistent with
  heteronormative phrasing being represented like unmarked, neutral phrasing.
- **The identity-coded representational signal is subtle.** It surfaces
  *topologically* (in the H1 loop structure of the embedding cloud) rather
  than as gross cluster separation — UMAP organizes the space by sentence
  template, not by identity group, so the effect is invisible to naive
  visual inspection or pairwise cosine similarity.
- **Topological separation is not attributable to subword tokenization
  artifacts** — all primary identity terms tokenize to exactly 1 subword
  (matched), ruling out the most common confound in embedding geometry studies.
- **Orientation is a coherent latent axis, not an isolated outlier.** Against
  size-matched controls, the `lgbtq_explicit` ↔ `heteronormative` distance (3.53)
  is *not* anomalously large — it is the **smallest** of the contrasts, sitting
  ~2.4 std below the random-adjective baseline (4.46 ± 0.38) and below the
  occupation baseline (4.72 ± 0.41). The model encodes sexual orientation as a
  single coherent dimension on which LGBTQ+ and heteronormative terms are nearest
  semantic neighbors. The layer-wise probe shows this orientation cluster
  crystallizes at **different depths across architectures**: layer 5 in
  `bert-base-uncased` (Pearson r = 0.993 raw vs L2-normalized) and layer 12 in
  `roberta-base` (r = 0.937).
- **Cross-architecture contrast: fine-tuning compresses the orientation
  cluster in BERT and amplifies it in RoBERTa, with a quantifiable
  dimensionality cost.** Supervised downstream fine-tuning moves the
  lgbtq ↔ heteronormative H1 Wasserstein distance in *opposite* directions
  at each family's own peak layer — **BERT + MNLI** compresses it by 43%
  (18.99 → 10.90), **RoBERTa + CoLA** amplifies it by 167%
  (0.69 → 1.83). A PCA bottleneck quantifies the BERT direction: base BERT
  supports a 0.90-accuracy probe on the top **k = 10** principal components
  (70% of variance), while MNLI-tuned BERT requires **k = 50** components
  (99% of variance) to clear the same threshold — a **~5× expansion** of
  the effective dimensionality of the orientation axis. Linear-probe accuracy
  on the full [CLS] vector clarifies what each direction means
  representationally (BERT compression is geometric reorganization with
  information preserved; RoBERTa amplification reflects new information
  gained — see Mechanism section). Any "fine-tuning debiases" claim must be
  scoped to a specific (architecture, task) pair.

## Mechanism: Architecture- and Task-Dependent Effects of Fine-Tuning

**Headline.** Supervised downstream fine-tuning does not affect identity-coded
representational geometry in a uniform way. Probing two encoder families with
their respective downstream proxies, we observe *opposite-direction* effects at
each family's own peak layer: fine-tuning **disperses** the orientation cluster
in BERT under MNLI (Wasserstein down, information preserved), but **introduces
linearly recoverable orientation information** to it in RoBERTa under CoLA
(Wasserstein up, probe accuracy up). Any general claim about "downstream
fine-tuning debiases" needs to be scoped to a specific (architecture, task) pair.

### Common foundation — TDA detects an orientation cluster in both base models

Persistent homology on `all-MiniLM-L6-v2` shows the LGBTQ+ embedding cloud is
topologically distinct from the neutral baseline (H1 Wasserstein p = 0.001,
n = 1000 permutations). Layer-wise probing of two encoder families
(`src/layerwise.py`) shows the orientation cluster crystallizes at *different
depths*: layer 5 of `bert-base-uncased` (raw H1 = **18.99**, Pearson r = 0.993
raw vs L2-normalized) and layer 12 of `roberta-base` (raw H1 = **0.69**,
r = 0.937). Both peaks are real geometric features, not activation-scale
artifacts; their different depths reflect different pretraining recipes.

### Case A — BERT + MNLI: dispersion without erasure

Loading the same base architecture after MNLI fine-tuning
(`textattack/bert-base-uncased-MNLI`), the layer-5 H1 Wasserstein distance
drops by **43%** to 10.90. At first glance this looks like attenuation. But a
5-fold CV logistic regression (`src/linear_probe.py`) decodes lgbtq vs
heteronormative from the layer-5 [CLS] vector at **100% balanced accuracy in
both base and fine-tuned models** — the orientation information is fully
preserved. The Wasserstein change must therefore reflect *geometric
reorganization*. A PCA bottleneck quantifies it: base BERT reaches 93%
accuracy at **k = 10** components (70% of variance), while MNLI-tuned BERT
needs **k = 50** components (99% of variance) to clear the same threshold —
the orientation signal got pushed off the top variance directions (repurposed
for the NLI task) and dispersed into the smaller-eigenvalue tail. *Same
information, geometrically rearranged across ~5× more dimensions.*

### Case B — RoBERTa + CoLA: information gain at the peak layer

The same pipeline run on RoBERTa tells the opposite story. At RoBERTa's own
peak layer (L12), the H1 Wasserstein distance *grows* by **+167%** under CoLA
fine-tuning (0.69 → 1.83). The linear probe (`src/linear_probe.py
--per-family-peak`) shows this is not stretched-cloud-same-information — it
is a real **information gain**. Base RoBERTa's layer-12 [CLS] vector is
essentially at chance for orientation classification (balanced accuracy
**0.537** vs. majority baseline 0.500); the CoLA-tuned model reaches balanced
accuracy **0.825** — a **+28.7 pp** gain. CoLA fine-tuning *added* linearly
recoverable orientation information to a layer that previously carried almost
none, even though the CoLA objective is grammatical acceptability with no
identity-related supervision.

### AI safety implication, revised

The combined picture is more concerning than either case in isolation:

- **Downstream fine-tuning is not a reliable debiasing operation.** It can
  preserve identity-coded structure while making it harder to detect with
  surface tests (BERT+MNLI), or it can *introduce* identity-coded structure
  that wasn't strongly present in the base model (RoBERTa+CoLA). Both
  outcomes look bad for surface bias audits that rely on top-K outputs or
  pairwise similarity.
- **Identity-coded geometry can emerge as a side effect of unrelated
  supervised objectives.** CoLA has no identity-related labels, yet
  CoLA-tuned RoBERTa encodes orientation at L12 with 82.5% linear
  recoverability where the base model encodes it at chance. A surface bias
  audit of base RoBERTa would correctly report "no orientation signal here"
  and would be wrong about the fine-tuned descendant.
- **The dispersion mechanism is real where it applies, but it is
  architecture- and task-dependent.** Any safety claim that quantifies
  representational bias should report the (architecture, task) pair and not
  generalize across families. We demonstrate differential geometric encoding
  of identity-coded language across model families — we make no direct claim
  that this constitutes harmful bias without behavioral validation.

## Project Goal

This project investigates whether large language model (LLM) embedding spaces encode
differential geometric structure for LGBTQ+-coded vs. heteronormative or identity-neutral
language, using tools from topological data analysis (TDA).

Rather than relying on cosine similarity or classification-based probes, we use **persistent
homology** to characterize the geometric and topological structure of embedding clouds —
comparing how otherwise-identical sentences cluster and connect when only the identity term
varies (e.g., "a gay person" vs. "a person"). Identity-coded representational separability
manifests as asymmetric topology: different Betti numbers, persistence diagrams, or
Wasserstein distances between groups.

## Research Questions

1. Do LGBTQ+-coded sentences occupy a topologically distinct region of embedding space
   compared to heteronormative or identity-neutral sentences?
2. Is the persistent homology of embeddings invariant to surface-level phrasing, or do
   lexical identity terms create measurable topological signatures?
3. Which transformer layers encode the most topologically discriminative representations?

## Approach

1. **Stimulus construction** — paired sentence sets differing only in identity terms
2. **Embedding extraction** — via `sentence-transformers` (and layer-wise via HuggingFace)
3. **Dimensionality reduction** — UMAP for visualization
4. **Persistent homology** — Vietoris-Rips filtration via `ripser` / `gudhi`
5. **Comparison** — Wasserstein / bottleneck distance between persistence diagrams (`persim`)
6. **Statistical testing** — permutation tests on topological summaries

## Methods

Each group of sentence embeddings is treated as a point cloud in 384-dimensional
space. We build a **Vietoris–Rips filtration**: imagine growing a ball of radius
ε around every point and connecting points whose balls overlap. As ε increases
from 0 to ∞, connected components (H0) merge and one-dimensional loops (H1) are
born and later fill in. Recording the ε at which each topological feature appears
and disappears yields a **persistence diagram** — a coordinate-free, multi-scale
fingerprint of the cloud's shape. To compare two groups we compute the
**Wasserstein distance** between their persistence diagrams, the optimal-transport
cost of matching one diagram's birth–death points to the other's; a large distance
means the two clouds are shaped differently. This is why TDA detects group-level
geometric structure that cosine similarity misses: cosine is a *pairwise, local*
measure that compares two vectors at a time and is blind to the global arrangement
of a set of points. The identity-coded representational separability here is not a
uniform directional shift (which cosine could catch) but a change in how a group's
embeddings are *connected and clustered* across scales — exactly the structure
persistent homology is built to quantify. We assess
significance with a permutation test that shuffles group labels (n = 1000) to build
a null distribution of Wasserstein distances.

## Project Structure

```
tda-llm-bias/
├── data/               # Raw stimuli, embeddings (numpy arrays), metadata CSVs
├── notebooks/          # Exploratory analysis and visualization
├── src/                # Reusable modules (embedding, TDA pipeline, stats)
├── results/            # Persistence diagrams, plots, summary tables
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Results

Run the four pipeline scripts from the project root, in order. Each consumes the
previous step's output:

```bash
# 1. Generate the stimulus dataset → data/stimuli.csv
.venv/bin/python src/data_gen.py

# 2. Encode sentences with all-MiniLM-L6-v2 → data/embeddings.npy + metadata
.venv/bin/python src/embeddings.py

# 3. Persistent homology + pairwise Wasserstein distances
#    → results/persistence_diagrams.pkl, results/wasserstein_distances.csv
.venv/bin/python src/tda_pipeline.py

# 4. Render figures → results/{persistence_diagrams,wasserstein_heatmap,umap_embeddings}.png
.venv/bin/python src/visualize.py
```

To reproduce the full write-up (including the permutation test), execute the
notebook:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb \
  --output analysis.ipynb --ExecutePreprocessor.timeout=600
```

## Dependencies

- `sentence-transformers` — embedding extraction
- `ripser` — fast Vietoris-Rips persistent homology
- `gudhi` — full TDA toolkit (cubical complexes, alpha complexes, mappers)
- `persim` — persistence diagram distances (Wasserstein, bottleneck)
- `umap-learn` — dimensionality reduction for visualization
- `matplotlib` / `pandas` — plotting and data wrangling
- `jupyter` — running the analysis notebook

> **Note:** `giotto-tda` requires CMake and has no Python 3.14 wheels yet.
> `gudhi` + `ripser` + `persim` cover equivalent functionality.

## References

- Caliskan et al. (2017) — Semantics derived automatically from language corpora contain human-like biases
- Garg et al. (2018) — Word embeddings quantify 100 years of gender and ethnic stereotypes
- Edelsbrunner & Harer (2010) — *Computational Topology: An Introduction*
- Carlsson (2009) — Topology and data, *Bulletin of the AMS*
