"""Visualization helpers."""

import numpy as np
import matplotlib.pyplot as plt
from persim import plot_diagrams


def plot_persistence(diagrams, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots()
    plot_diagrams(diagrams, ax=ax)
    ax.set_title(title)
    return fig


def plot_umap(embeddings: np.ndarray, labels: list[str], title: str = "") -> plt.Figure:
    import umap
    reducer = umap.UMAP(random_state=42)
    proj = reducer.fit_transform(embeddings)
    unique = sorted(set(labels))
    fig, ax = plt.subplots()
    for lbl in unique:
        mask = np.array(labels) == lbl
        ax.scatter(proj[mask, 0], proj[mask, 1], label=lbl, alpha=0.7)
    ax.legend()
    ax.set_title(title)
    return fig
