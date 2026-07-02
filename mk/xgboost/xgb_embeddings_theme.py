"""
xgb_embeddings_theme.py
=======================

EXPERIMENT A of the classifier comparison: XGBoost on e5 embeddings ONLY,
classifying Kuusi example proverbs into the 13 theme categories
(A, B, C, D, E, F, G, H, J, K, L, M, T).

WHAT THIS SCRIPT DOES
---------------------
1. Loads the e5 embeddings + parallel metadata that
   01_build_reference_index.py already produced.
2. Keeps ONLY real example proverbs (drops definition rows).
3. Runs XGBoost with leak-free grouped cross-validation (examples from the
   same Kuusi type never span train and test folds).
4. Reports macro-F1, accuracy, a per-class table, and a confusion matrix.

WHAT THIS SCRIPT DOES *NOT* DO
------------------------------
- It does NOT build embeddings. It reuses ref_embeddings.npy. If that file
  does not exist, run 01_build_reference_index.py --encoder e5 first.
- It uses embeddings as the ONLY features. This is the deliberate "Experiment
  A" baseline. Tree models usually do POORLY on raw embeddings because they
  split one dimension at a time and no single embedding dimension is
  meaningful. A low number here is expected and is itself the finding:
  it motivates Experiment B (engineered features).

INPUT (all produced by 01_build_reference_index.py --encoder e5):
  mk/npy/ref_embeddings.npy   float array, one row per reference item
  mk/ref_metadata.csv         columns: ref_idx, source_type, text,
                              theme, main_class, subgroup_code, kuusi_id
                              (row i here == row i of the .npy)

OUTPUT:
  mk/xgboost/results/xgb_emb_theme_perclass.csv    per-theme precision/recall/F1
  mk/xgboost/results/xgb_emb_theme_confusion.csv   13x13 confusion matrix
  mk/xgboost/results/xgb_emb_theme_summary.txt     headline numbers
  (also printed to the console)

RUN:
  python mk/xgboost/xgb_embeddings_theme.py
  # optional flags:
  #   --folds 5           number of CV folds (default 5)
  #   --estimators 400    number of boosting rounds (default 400)
  #   --max-depth 6       tree depth (default 6)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PART A -- file locations
# =============================================================================
# This file lives in mk/xgboost/. The embeddings + metadata live one level up
# in mk/. We resolve paths relative to THIS file so the script works no matter
# what directory you launch it from.
HERE      = os.path.dirname(os.path.abspath(__file__))   # .../mk/xgboost
MK_DIR    = os.path.dirname(HERE)                         # .../mk
NPY_DIR   = os.path.join(MK_DIR, "npy")
EMB_FILE  = os.path.join(NPY_DIR, "ref_embeddings.npy")
META_FILE = os.path.join(MK_DIR, "ref_metadata.csv")
ENC_FILE  = os.path.join(NPY_DIR, "encoder.txt")

RESULTS_DIR = os.path.join(HERE, "results")

# The 13 Kuusi theme codes, in a fixed order so the confusion matrix rows and
# columns always line up the same way.
THEMES = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "T"]


# =============================================================================
# PART B -- load and sanity-check the input
# =============================================================================
def load_data():
    """Load embeddings + metadata, keep only example proverbs, align them.

    Returns
    -------
    X      : np.ndarray  (n_examples, embedding_dim)   the features
    y      : np.ndarray  (n_examples,)                 integer theme labels 0..12
    groups : np.ndarray  (n_examples,)                 kuusi_id per row (for CV)
    meta   : pd.DataFrame                               the kept metadata rows
    """
    # --- check the files exist BEFORE doing anything else -------------------
    if not os.path.exists(EMB_FILE):
        sys.exit(
            f"ERROR: embeddings not found at {EMB_FILE}\n"
            f"Run this first:  python mk/01_build_reference_index.py --encoder e5"
        )
    if not os.path.exists(META_FILE):
        sys.exit(f"ERROR: metadata not found at {META_FILE}")

    # --- load ---------------------------------------------------------------
    emb = np.load(EMB_FILE)                       # shape (N_all, dim)
    meta = pd.read_csv(META_FILE, encoding="utf-8-sig")

    # Report which encoder built these vectors, so you never accidentally mix
    # a LaBSE index with an "e5" experiment.
    if os.path.exists(ENC_FILE):
        with open(ENC_FILE, encoding="utf-8") as fh:
            enc = fh.read().strip()
        print(f"[info] embeddings were built with encoder: {enc}")

    # --- the row-count contract --------------------------------------------
    # Row i of the metadata MUST correspond to row i of the embedding array.
    # If these two lengths disagree, the index is corrupt -- stop now rather
    # than train on mislabeled data.
    if len(emb) != len(meta):
        sys.exit(
            f"ERROR: {len(emb)} embedding rows but {len(meta)} metadata rows. "
            f"The index is out of sync -- rebuild it."
        )

    # --- keep only real example proverbs ------------------------------------
    # The index also contains definition rows (source_type starts with 'def_').
    # Those are typology definitions, not proverbs, so we exclude them from
    # both training and evaluation.
    is_example = ~meta["source_type"].astype(str).str.startswith("def_")
    meta = meta[is_example].reset_index(drop=True)
    emb = emb[is_example.to_numpy()]
    print(f"[info] kept {len(meta)} example proverbs "
          f"(dropped {int((~is_example).sum())} definition rows)")

    # --- build the label vector y ------------------------------------------
    # Map each theme letter to an integer 0..12. XGBoost needs integer classes.
    theme_to_int = {t: i for i, t in enumerate(THEMES)}

    # Any row whose theme is not one of the 13 known codes is dropped and
    # reported -- we never silently guess a label.
    known = meta["theme"].isin(THEMES)
    if not known.all():
        bad = meta.loc[~known, "theme"].unique().tolist()
        print(f"[warn] dropping {int((~known).sum())} rows with unknown theme "
              f"codes: {bad}")
        meta = meta[known].reset_index(drop=True)
        emb = emb[known.to_numpy()]

    y = meta["theme"].map(theme_to_int).to_numpy()

    # --- groups for leak-free CV -------------------------------------------
    # Examples that came from the SAME Kuusi type are near-duplicates. If one
    # lands in train and another in test, the model "cheats". Grouping by
    # kuusi_id keeps a whole type entirely inside one fold.
    groups = meta["kuusi_id"].astype(str).to_numpy()

    X = emb.astype(np.float32)
    return X, y, groups, meta


# =============================================================================
# PART C -- the cross-validated training loop
# =============================================================================
def run_cv(X, y, groups, n_folds, n_estimators, max_depth):
    """Train + evaluate XGBoost with StratifiedGroupKFold.

    We collect the model's prediction for every proverb (each proverb is
    predicted exactly once, by the fold in which it was in the test set).
    Those out-of-fold predictions are what we score -- an honest estimate of
    how the model does on proverbs it never saw during training.

    Returns
    -------
    y_true : np.ndarray  the true labels, in prediction order
    y_pred : np.ndarray  the model's predicted labels, same order
    """
    # Imports are inside the function so the file at least loads and shows help
    # even on a machine that hasn't installed xgboost yet.
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    n_classes = len(THEMES)

    # StratifiedGroupKFold gives us BOTH properties we need:
    #   - stratified: each fold keeps roughly the same theme proportions
    #   - grouped:    a kuusi_id never appears in two folds
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # We accumulate out-of-fold predictions here. Start as -1 so we can assert
    # at the end that every row got filled in exactly once.
    oof_pred = np.full(len(y), -1, dtype=int)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y, groups), start=1):
        # Class imbalance handling: themes are very uneven (some have many
        # proverbs, some few). We compute a per-sample weight = inverse of that
        # class's frequency in the TRAINING fold, so rare themes are not ignored.
        counts = np.bincount(y[tr_idx], minlength=n_classes)
        # avoid divide-by-zero for a class absent from this training fold
        inv_freq = np.where(counts > 0, len(tr_idx) / (n_classes * counts), 0.0)
        sample_weight = inv_freq[y[tr_idx]]

        clf = XGBClassifier(
            objective="multi:softprob",  # multiclass, returns per-class probs
            num_class=n_classes,
            n_estimators=n_estimators,    # number of boosting rounds (trees)
            max_depth=max_depth,          # depth of each tree
            learning_rate=0.1,            # step size per round
            subsample=0.8,                # row sampling -> less overfitting
            colsample_bytree=0.8,         # column sampling -> less overfitting
            reg_lambda=1.0,               # L2 regularization (your "lambda")
            tree_method="hist",           # fast histogram algorithm
            eval_metric="mlogloss",
            n_jobs=-1,                    # use all CPU cores
            random_state=42,
        )

        clf.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weight)
        oof_pred[te_idx] = clf.predict(X[te_idx])

        # Per-fold progress so a long run shows signs of life.
        from sklearn.metrics import f1_score
        fold_f1 = f1_score(y[te_idx], oof_pred[te_idx], average="macro")
        print(f"[fold {fold}/{n_folds}] "
              f"train={len(tr_idx)}  test={len(te_idx)}  "
              f"macro-F1={fold_f1:.3f}")

    # Every proverb should have been predicted exactly once.
    assert (oof_pred >= 0).all(), "some rows never got an out-of-fold prediction"
    return y, oof_pred


# =============================================================================
# PART D -- score and save
# =============================================================================
def report(y_true, y_pred, out_dir):
    """Compute headline metrics + per-class table + confusion matrix; save all."""
    from sklearn.metrics import (accuracy_score, f1_score,
                                  classification_report, confusion_matrix)

    os.makedirs(out_dir, exist_ok=True)

    # --- headline numbers ---------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")   # unweighted mean F1
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    summary = (
        f"XGBoost on e5 embeddings only -- 13-way theme classification\n"
        f"examples scored : {len(y_true)}\n"
        f"accuracy        : {acc:.4f}\n"
        f"macro-F1        : {macro_f1:.4f}\n"
        f"weighted-F1     : {weighted_f1:.4f}\n"
    )
    print("\n" + summary)

    # --- per-class table ----------------------------------------------------
    # precision/recall/F1 for each of the 13 themes -> a DataFrame we can save.
    rep = classification_report(
        y_true, y_pred,
        labels=list(range(len(THEMES))),
        target_names=THEMES,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(rep).transpose()
    per_class_path = os.path.join(out_dir, "xgb_emb_theme_perclass.csv")
    per_class.to_csv(per_class_path, encoding="utf-8-sig")

    # --- confusion matrix ---------------------------------------------------
    # Row = TRUE theme, Column = PREDICTED theme. The diagonal is correct
    # predictions; big off-diagonal cells show which themes get confused.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(THEMES))))
    cm_df = pd.DataFrame(cm, index=THEMES, columns=THEMES)
    cm_df.index.name = "true\\pred"
    cm_path = os.path.join(out_dir, "xgb_emb_theme_confusion.csv")
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    # --- save the summary ---------------------------------------------------
    with open(os.path.join(out_dir, "xgb_emb_theme_summary.txt"),
              "w", encoding="utf-8") as fh:
        fh.write(summary)

    print(f"[saved] {per_class_path}")
    print(f"[saved] {cm_path}")
    print(f"[saved] {os.path.join(out_dir, 'xgb_emb_theme_summary.txt')}")

    # Show the confusion matrix inline too -- it's small enough to read.
    print("\nConfusion matrix (row = true theme, column = predicted):")
    print(cm_df.to_string())


# =============================================================================
# PART E -- entry point
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="XGBoost on e5 embeddings only, 13-way theme classification.")
    ap.add_argument("--folds", type=int, default=5,
                    help="number of cross-validation folds (default 5)")
    ap.add_argument("--estimators", type=int, default=400,
                    help="number of boosting rounds / trees (default 400)")
    ap.add_argument("--max-depth", type=int, default=6,
                    help="max depth of each tree (default 6)")
    args = ap.parse_args()

    X, y, groups, meta = load_data()

    # A quick look at how many proverbs each theme has. Very uneven counts are
    # exactly why we use class weighting and macro-F1 rather than accuracy.
    counts = pd.Series(y).value_counts().sort_index()
    print("\n[info] examples per theme:")
    for i, t in enumerate(THEMES):
        print(f"   {t}: {int(counts.get(i, 0))}")

    y_true, y_pred = run_cv(
        X, y, groups,
        n_folds=args.folds,
        n_estimators=args.estimators,
        max_depth=args.max_depth,
    )
    report(y_true, y_pred, RESULTS_DIR)


if __name__ == "__main__":
    main()