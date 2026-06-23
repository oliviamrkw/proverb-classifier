"""
01_build_split_and_embeddings.py
--------------------------------
Builds the train/val/test split on the labeled MAPS rows and the shared LaBSE
embeddings. Run ONCE (re-run whenever master.csv changes).

ALL artifacts are written INTO the embedding_classifier/ folder (next to this
script), regardless of where you run from, so every script agrees on locations:
    embedding_classifier/train_indices.npy
    embedding_classifier/val_indices.npy
    embedding_classifier/test_indices.npy
    embedding_classifier/maps_embeddings.npy

Run:  python embedding_classifier/01_build_split_and_embeddings.py
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))      # embedding_classifier/
ROOT = os.path.dirname(HERE)                           # project root
MASTER = os.path.join(ROOT, "data", "processed", "master.csv")
EMB   = os.path.join(HERE, "maps_embeddings.npy")
TRAIN = os.path.join(HERE, "train_indices.npy")
VAL   = os.path.join(HERE, "val_indices.npy")
TEST  = os.path.join(HERE, "test_indices.npy")


def load_maps(path=MASTER):
    """The ONE definition of the labeled MAPS frame. Every script imports/copies
    this so embeddings and indices line up positionally. Filters to rows with a
    usable 0/1 label so an empty label can never crash the split."""
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    m = d[d.source_dataset == "MAPS"].copy().reset_index(drop=True)
    m["label"] = pd.to_numeric(m["fig_or_literal"], errors="coerce")
    n_before = len(m)
    m = m[m["label"].notna()].copy().reset_index(drop=True)
    if len(m) != n_before:
        print(f"NOTE: dropped {n_before - len(m)} MAPS rows with no/invalid label.")
    m["label"] = m["label"].astype(int)
    return m


def main():
    maps = load_maps()
    maps["strat_key"] = maps["language"] + "_" + maps["label"].astype(str)
    print(f"labeled MAPS rows: {len(maps)}")
    print(maps["strat_key"].value_counts(), "\n")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trainval_local, test_local = next(sss.split(maps, maps["strat_key"]))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tv = maps.iloc[trainval_local]
    train_in_tv, val_in_tv = next(sss2.split(tv, tv["strat_key"]))

    train_idx = trainval_local[train_in_tv]
    val_idx = trainval_local[val_in_tv]
    test_idx = test_local
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    np.save(TRAIN, train_idx); np.save(VAL, val_idx); np.save(TEST, test_idx)
    print(f"saved indices -> {HERE}\n")

    model = SentenceTransformer("sentence-transformers/LaBSE")
    emb = model.encode(maps["proverb_native"].tolist(), show_progress_bar=True)
    np.save(EMB, emb)
    print(f"embeddings: {emb.shape}  ->  {EMB}")


if __name__ == "__main__":
    main()