"""TDA pipeline: persistent homology and diagram comparison."""

import numpy as np
from ripser import ripser
from persim import wasserstein, bottleneck


def compute_persistence(embeddings: np.ndarray, max_dim: int = 1) -> dict:
    """Run Vietoris-Rips filtration and return persistence diagrams."""
    result = ripser(embeddings, maxdim=max_dim)
    return result["dgms"]  # list of (birth, death) arrays per dimension


def diagram_distance(dgm_a, dgm_b, metric: str = "wasserstein") -> float:
    fn = wasserstein if metric == "wasserstein" else bottleneck
    return fn(dgm_a, dgm_b)
