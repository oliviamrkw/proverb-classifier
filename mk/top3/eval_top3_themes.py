"""
mk/top3/eval_top3_themes.py
==============================================================================
MEASURE how accurate the top-3 THEME guesses are, the honest (leak-free) way.

Plain description:
  For each proverb, kNN finds its nearest reference proverbs and votes on the
  theme. Instead of keeping only the single winner, we keep the top-3 themes by
  vote. We then ask: is the TRUE theme among the top-1 / top-2 / top-3? That is
  exactly what your mk_theme1 / mk_theme2 / mk_theme3 columns will contain, so
  this tells you how often those columns will be right.

  Evaluated with the same grouped cross-validation as 02 (group by kuusi_id so a
  type's translated variants never split train/test; stratify by theme). Every
  example is tested once.

This only MEASURES accuracy. The companion script label_master_top3.py is what
actually writes the labels onto your 21k dataset.

INPUT   (from 01, one level up)  mk/npy/ref_embeddings.npy , mk/ref_metadata.csv
OUTPUT  mk/top3/results/eval_top3_predictions.csv , eval_top3_summary.csv
RUN     python mk/top3/eval_top3_themes.py
REQUIRES pip install scikit-learn pandas numpy
==============================================================================
"""

import os                                              # paths
import numpy as np                                     # math
import pandas as pd                                    # tables
from collections import defaultdict                    # vote tallies

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))      # .../mk/top3
MK   = os.path.dirname(HERE)                            # .../mk
EMB  = os.path.join(MK, "npy", "ref_embeddings.npy")   # embeddings from 01
META = os.path.join(MK, "ref_metadata.csv")            # labels from 01
RESULTS_DIR = os.path.join(HERE, "results")            # outputs

K_NEIGHBORS = 15                                       # neighbours that vote
K_FOLDS     = 5                                        # cross-validation folds


def l2_normalize(x):                                   # unit-length rows -> dot = cosine
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return x / n


def load_index():
    """Load 01's index; return example points (tested) and ALL points' theme labels
    (examples + definitions both carry a theme and serve as reference voters)."""
    emb = l2_normalize(np.load(EMB).astype("float32"))            # normalized vectors
    meta = pd.read_csv(META, encoding="utf-8-sig", dtype=str).fillna("")
    assert len(meta) == emb.shape[0], "metadata/embeddings mismatch (re-run 01)"
    is_ex = (meta["source_type"] == "example").values            # example rows
    ex = dict(vec=emb[is_ex],                                     # example embeddings
              theme=meta["theme"].values[is_ex],                 # gold theme
              kid=meta["kuusi_id"].values[is_ex],                # group key
              text=meta["text"].values[is_ex])                   # proverb text
    ref_vec = emb                                                 # every point can be a voter
    ref_theme = meta["theme"].values                             # its theme label
    ref_is_ex = is_ex                                            # flag: example or definition
    return ex, ref_vec, ref_theme, ref_is_ex


def top_themes(qvec, cand_vec, cand_theme, k=K_NEIGHBORS, n=3):
    """Return the top-n themes (by similarity-weighted vote) plus their scores."""
    sims = cand_vec @ qvec                                        # similarity to every voter
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]                       # k nearest voters
    tally = defaultdict(float)                                    # theme -> summed similarity
    for idx in top:
        tally[cand_theme[idx]] += max(0.0, sims[idx])            # add weighted vote
    ranked = sorted(tally, key=tally.get, reverse=True)          # strongest theme first
    total = sum(tally.values()) or 1.0                           # for turning votes into shares
    themes = ranked[:n]                                          # top-n themes
    scores = [round(tally[t] / total, 3) for t in themes]        # their confidence shares
    return themes, scores


def run_cv():
    ex, ref_vec, ref_theme, ref_is_ex = load_index()
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splits = list(StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
                      .split(ex["vec"], ex["theme"], groups=ex["kid"]))
        name = "StratifiedGroupKFold(group=kuusi_id, stratify=theme)"
    except Exception:
        from sklearn.model_selection import GroupKFold
        splits = list(GroupKFold(n_splits=K_FOLDS).split(ex["vec"], ex["theme"], groups=ex["kid"]))
        name = "GroupKFold(group=kuusi_id)"
    print(f"CV: {name}, folds={K_FOLDS}, k-NN={K_NEIGHBORS}\n")

    # map each example to its row in the global reference arrays, so we can exclude
    # the test fold's examples (and their kuusi_id group) from the voter pool.
    ex_rows = np.where(ref_is_ex)[0]                              # global indices of examples
    rows, fold_scores = [], []
    for f, (tr, te) in enumerate(splits):
        assert not (set(ex["kid"][tr]) & set(ex["kid"][te])), "LEAK"   # no type spans the split
        # voter pool = all reference points EXCEPT the held-out examples this fold
        keep = np.ones(len(ref_vec), bool)                       # start with everything
        keep[ex_rows[te]] = False                                # drop this fold's test examples
        cand_vec, cand_theme = ref_vec[keep], ref_theme[keep]    # the voters

        t1 = t2 = t3 = 0                                          # top-1/2/3 hit counters
        for i in te:                                             # each held-out proverb
            themes, scores = top_themes(ex["vec"][i], cand_vec, cand_theme, n=3)
            gold = ex["theme"][i]
            t1 += gold in themes[:1]; t2 += gold in themes[:2]; t3 += gold in themes[:3]
            rows.append(dict(fold=f, text=ex["text"][i], true_theme=gold,
                             theme1=themes[0] if len(themes)>0 else "",
                             theme2=themes[1] if len(themes)>1 else "",
                             theme3=themes[2] if len(themes)>2 else "",
                             score1=scores[0] if len(scores)>0 else 0,
                             in_top1=int(gold in themes[:1]),
                             in_top3=int(gold in themes[:3])))
        nteb = len(te)
        fold_scores.append((t1/nteb, t2/nteb, t3/nteb))
        print(f"  fold {f}: top1={t1/nteb:.3f}  top2={t2/nteb:.3f}  top3={t3/nteb:.3f}")
    return pd.DataFrame(rows), np.array(fold_scores)


def main():
    pred_df, fold_scores = run_cv()
    mean, std = fold_scores.mean(axis=0), fold_scores.std(axis=0)
    print("\n================ TOP-N THEME ACCURACY ================")
    for label, m, s in zip(["top-1", "top-2", "top-3"], mean, std):
        print(f"  {label}: {m:.3f} +/- {s:.3f}")
    print("  (single-guess majority baseline = 0.173)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_df.to_csv(os.path.join(RESULTS_DIR, "eval_top3_predictions.csv"),
                   index=False, encoding="utf-8-sig")
    pd.DataFrame(dict(metric=["top1", "top2", "top3"],
                      mean_acc=np.round(mean, 4), std_acc=np.round(std, 4))
                 ).to_csv(os.path.join(RESULTS_DIR, "eval_top3_summary.csv"), index=False)
    print(f"\nsaved results to {RESULTS_DIR}")
    print("If top-3 looks good, run label_master_top3.py to write the columns onto your 21k.")


if __name__ == "__main__":
    main()