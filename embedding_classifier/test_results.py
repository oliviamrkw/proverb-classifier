import numpy as np, pandas as pd
d = pd.read_csv("data/processed/master.csv", encoding="utf-8-sig", dtype=str).fillna("")
m = d[d.source_dataset=="MAPS"].reset_index(drop=True)
m = m[pd.to_numeric(m["fig_or_literal"],errors="coerce").notna()].reset_index(drop=True)
key = m["proverb_native"].str.strip().str.lower()
tr = list(set(np.load("embedding_classifier/train_indices.npy")) | set(np.load("embedding_classifier/val_indices.npy")))
te = list(np.load("embedding_classifier/test_indices.npy"))
print("shared proverbs train/val ∩ test:", len(set(key[tr]) & set(key[te])))   # want 0