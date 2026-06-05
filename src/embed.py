"""Embedding extraction utilities."""

from sentence_transformers import SentenceTransformer
import numpy as np


def get_embeddings(sentences: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(sentences, normalize_embeddings=True)
