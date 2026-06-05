"""Shared paper-figure style and helpers.

Provides:
  - apply()          : install paper-quality matplotlib rcParams
  - COLORS, SEMANTIC : Wong (2011) colorblind-safe palette + semantic aliases
  - save(fig, stem)  : write both stem.pdf (vector) and stem.png (300 dpi)
  - REPO_ROOT        : Path to repository root (for reading data/ and results/)

Import from sibling figN_*.py scripts via `from _style import ...`.
"""

from pathlib import Path
import matplotlib as mpl

# ---------------------------------------------------------------- paths
# This file lives at: <repo>/paper/figures/scripts/_style.py
PAPER_FIGURES_DIR = Path(__file__).parent.parent          # <repo>/paper/figures
REPO_ROOT         = PAPER_FIGURES_DIR.parent.parent       # <repo>

# ---------------------------------------------------------------- palette
# Wong (2011) — Nature Methods 8(6):441 — eight hues distinguishable to all
# common colour-vision deficiencies.
COLORS = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "vermilion": "#D55E00",
    "green":     "#009E73",
    "skyblue":   "#56B4E9",
    "yellow":    "#F0E442",
    "purple":    "#CC79A7",
    "grey":      "#777777",
    "black":     "#000000",
}

# Semantic aliases for reuse across figures.
SEMANTIC = {
    "observed":     COLORS["vermilion"],
    "null":         COLORS["grey"],
    "raw":          COLORS["blue"],
    "normalized":   COLORS["orange"],
    "base":         COLORS["blue"],
    "fine_tuned":   COLORS["vermilion"],
    "threshold":    COLORS["grey"],
    "chance":       COLORS["black"],
    "heteronorm":   COLORS["blue"],
    "occupation":   COLORS["green"],
    "adjective":    COLORS["purple"],
}


# ---------------------------------------------------------------- rcParams
def apply() -> None:
    """Install paper-quality matplotlib defaults."""
    mpl.rcParams.update({
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linewidth":     0.5,
        "legend.frameon":     False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        # editable text in PDF/PS exports (avoids type-3 outlines)
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "lines.linewidth":    1.5,
        "axes.linewidth":     0.6,
    })


# ---------------------------------------------------------------- save
def save(fig, stem: str) -> None:
    """Save fig as <PAPER_FIGURES_DIR>/<stem>.{pdf,png}."""
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
