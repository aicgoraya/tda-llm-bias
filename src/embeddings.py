"""Encode stimuli.csv with sentence-transformers and persist the results.

Outputs:
  data/embeddings.npy          float32 array, shape (N, D)
  data/embeddings_meta.csv     sentence / group / template_id / identity_term
"""

from pathlib import Path

import numpy as np
import pandas as pd

from embed import get_embeddings

DATA_DIR = Path(__file__).parent.parent / "data"
STIMULI_PATH  = DATA_DIR / "stimuli.csv"
EMB_PATH      = DATA_DIR / "embeddings.npy"
META_PATH     = DATA_DIR / "embeddings_meta.csv"

META_COLS = ["sentence", "group", "template_id", "identity_term"]


def main() -> None:
    df = pd.read_csv(STIMULI_PATH)

    print(f"Encoding {len(df)} sentences with all-MiniLM-L6-v2 …")
    embeddings = get_embeddings(df["sentence"].tolist())

    np.save(EMB_PATH, embeddings.astype(np.float32))
    df[META_COLS].to_csv(META_PATH, index=False)

    print(f"embeddings  → {EMB_PATH}  shape={embeddings.shape}")
    print(f"metadata    → {META_PATH}")


if __name__ == "__main__":
    main()
