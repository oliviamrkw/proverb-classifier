"""
02_umap_diagnostic.py
=====================
DIAGNOSTIC ONLY -- not a classifier, never touches the locked test set.

THREE MODES (run from project root):
  python embedding_classifier/02_umap_diagnostic.py            # all langs, 2 panels
  python embedding_classifier/02_umap_diagnostic.py zh         # one language
  python embedding_classifier/02_umap_diagnostic.py --facet    # per-language grid

All artifacts (.npy in, .png out) live in the embedding_classifier/ folder.
Requires: pip install umap-learn matplotlib scikit-learn
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MASTER = os.path.join(ROOT, "data", "processed", "master.csv")
EMB   = os.path.join(HERE, "maps_embeddings.npy")
TRAIN = os.path.join(HERE, "train_indices.npy")
VAL   = os.path.join(HERE, "val_indices.npy")
MIN_ROWS = 20


def load_maps(path=MASTER):
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    m = d[d.source_dataset == "MAPS"].copy().reset_index(drop=True)
    m["label"] = pd.to_numeric(m["fig_or_literal"], errors="coerce")
    m = m[m["label"].notna()].copy().reset_index(drop=True)
    m["label"] = m["label"].astype(int)
    return m


def load_trainval():
    maps = load_maps()
    emb = np.load(EMB)
    assert emb.shape[0] == len(maps), (
        f"embedding rows ({emb.shape[0]}) != labeled MAPS rows ({len(maps)}). "
        "Re-run 01 after rebuilding the data so they align.")
    idx = np.concatenate([np.load(TRAIN), np.load(VAL)])
    return emb[idx], maps["label"].values[idx], maps["language"].values[idx]


def umap_2d(X, seed=SEED):
    nn = min(15, max(2, len(X) - 1))
    return umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=seed,
                     metric="cosine").fit_transform(X)


def silhouette(X, y):
    from sklearn.metrics import silhouette_score
    if len(set(y)) < 2 or len(y) < 3:
        return None
    try:
        return float(silhouette_score(X, y, metric="cosine"))
    except Exception:
        return None


def scatter_fig(ax, X2, y, title):
    for lab, name, color in [(0, "literal", "#2166ac"), (1, "figurative", "#b2182b")]:
        m = y == lab
        ax.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.5, c=color,
                   label=f"{name} (n={int(m.sum())})")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def mode_all():
    X, labels, langs = load_trainval()
    print(f"plotting {len(X)} rows | figurative={int((labels==1).sum())} "
          f"literal={int((labels==0).sum())}")
    X2 = umap_2d(X)
    s = silhouette(X, labels)
    if s is not None:
        print(f"label silhouette (ALL langs mixed): {s:.3f} "
              "-- inflated by the language confound")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    scatter_fig(ax1, X2, labels, "colored by fig_or_literal")
    for lang in sorted(set(langs)):
        m = langs == lang
        ax2.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.5, label=f"{lang} (n={int(m.sum())})")
    ax2.set_title("colored by language")
    ax2.legend(loc="best", fontsize=8); ax2.set_xticks([]); ax2.set_yticks([])
    fig.suptitle("UMAP of MAPS proverbs in LaBSE space (train+val)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "umap_fig_literal.png")
    fig.savefig(out, dpi=150); print("saved", out)


def mode_one(lang):
    X, labels, langs = load_trainval()
    m = langs == lang
    if m.sum() == 0:
        print(f"language '{lang}' not found. available: {sorted(set(langs))}")
        return
    Xl, yl = X[m], labels[m]
    s = silhouette(Xl, yl)
    print(f"{lang}: n={int(m.sum())} | figurative={int((yl==1).sum())} "
          f"literal={int((yl==0).sum())} | silhouette="
          f"{'n/a' if s is None else f'{s:.3f}'}")
    fig, ax = plt.subplots(figsize=(7, 6))
    title = f"{lang} only — colored by fig_or_literal"
    if s is not None:
        title += f"  (silhouette={s:.2f})"
    scatter_fig(ax, umap_2d(Xl), yl, title)
    fig.tight_layout()
    out = os.path.join(HERE, f"umap_{lang}.png")
    fig.savefig(out, dpi=150); print("saved", out)


def mode_facet():
    X, labels, langs = load_trainval()
    use = [l for l in sorted(set(langs)) if (langs == l).sum() >= MIN_ROWS]
    skipped = [l for l in sorted(set(langs)) if l not in use]
    if skipped:
        print(f"skipped (under {MIN_ROWS} rows): {skipped}")
    if not use:
        print("no language has enough rows."); return
    cols = min(3, len(use)); rows = (len(use) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, lang in enumerate(use):
        m = langs == lang
        Xl, yl = X[m], labels[m]
        s = silhouette(Xl, yl)
        ax = axes[k // cols][k % cols]; ax.axis("on")
        title = f"{lang}  (n={int(m.sum())}"
        title += f", sil={s:.2f})" if s is not None else ")"
        scatter_fig(ax, umap_2d(Xl), yl, title)
        print(f"  {lang}: n={int(m.sum())} silhouette="
              f"{'n/a' if s is None else f'{s:.3f}'}")
    fig.suptitle("Per-language UMAP, colored by fig_or_literal "
                 "(separation here = real figurativity signal)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "umap_by_language.png")
    fig.savefig(out, dpi=150); print("saved", out)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        mode_all()
    elif arg == "--facet":
        mode_facet()
    else:
        mode_one(arg)


if __name__ == "__main__":
    main()