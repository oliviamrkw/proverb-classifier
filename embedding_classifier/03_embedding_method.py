"""
03_method_a.py  --  METHOD A: LaBSE embeddings + logistic regression
====================================================================
Trains on the train split, tunes the regularization strength C on the val
split, then evaluates ONCE on the locked test split. Reports the headline
macro-F1, per-class metrics, per-language macro-F1, and the dummy baseline.

Reads the shared artifacts produced by 01 (all inside embedding_classifier/):
    maps_embeddings.npy, train_indices.npy, val_indices.npy, test_indices.npy
Writes:
    embedding_classifier/method_a_results.csv      (per-language summary)

Run:  python embedding_classifier/03_method_a.py
Requires: pip install scikit-learn
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MASTER = os.path.join(ROOT, "data", "processed", "master.csv")
EMB   = os.path.join(HERE, "maps_embeddings.npy")
TRAIN = os.path.join(HERE, "train_indices.npy")
VAL   = os.path.join(HERE, "val_indices.npy")
TEST  = os.path.join(HERE, "test_indices.npy")

# LaBSE is a cosine-geometry encoder, so L2-normalizing the vectors before a
# linear model is the geometrically correct default and usually helps a little.
# Flip to False to reproduce the doc's raw-embedding setup.
L2_NORMALIZE = True
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def load_maps(path=MASTER):
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    m = d[d.source_dataset == "MAPS"].copy().reset_index(drop=True)
    m["label"] = pd.to_numeric(m["fig_or_literal"], errors="coerce")
    m = m[m["label"].notna()].copy().reset_index(drop=True)
    m["label"] = m["label"].astype(int)
    return m


def main():
    maps = load_maps()
    emb = np.load(EMB)
    assert emb.shape[0] == len(maps), (
        f"embedding rows ({emb.shape[0]}) != labeled MAPS rows ({len(maps)}). "
        "Re-run 01 so they align.")
    if L2_NORMALIZE:
        emb = normalize(emb)

    tr, va, te = np.load(TRAIN), np.load(VAL), np.load(TEST)
    y = maps["label"].values
    langs = maps["language"].values
    Xtr, ytr = emb[tr], y[tr]
    Xva, yva = emb[va], y[va]
    Xte, yte = emb[te], y[te]
    print(f"train={len(tr)} val={len(va)} test={len(te)} | "
          f"L2_NORMALIZE={L2_NORMALIZE}")

    # --- tune C on the validation split (headline metric = macro-F1) ---
    print("\ntuning C on val:")
    best_C, best_f = None, -1.0
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                                 random_state=SEED)
        clf.fit(Xtr, ytr)
        f = f1_score(yva, clf.predict(Xva), average="macro", zero_division=0)
        print(f"  C={C:<7} val macro-F1={f:.3f}")
        if f > best_f:
            best_C, best_f = C, f
    print(f"chosen C={best_C} (val macro-F1={best_f:.3f})")

    # --- final model: refit on train+val with chosen C, evaluate ONCE on test ---
    Xtv = np.vstack([Xtr, Xva]); ytv = np.concatenate([ytr, yva])
    clf = LogisticRegression(C=best_C, max_iter=1000, class_weight="balanced",
                             random_state=SEED).fit(Xtv, ytv)
    pred = clf.predict(Xte)

    # TWO baselines on purpose:
    #  - most_frequent: the ACCURACY floor (always predicts the majority class).
    #  - stratified:    the MACRO-F1 floor (random guessing at the class priors).
    # A class_weight="balanced" model beats most_frequent on macro-F1 for free by
    # hedging across classes, so the honest test is beating the STRATIFIED floor.
    mf = DummyClassifier(strategy="most_frequent").fit(Xtv, ytv)
    st = DummyClassifier(strategy="stratified", random_state=SEED).fit(Xtv, ytv)
    mf_pred, st_pred = mf.predict(Xte), st.predict(Xte)

    macro = f1_score(yte, pred, average="macro", zero_division=0)
    mf_macro = f1_score(yte, mf_pred, average="macro", zero_division=0)
    st_macro = f1_score(yte, st_pred, average="macro", zero_division=0)

    print("\n================ TEST RESULTS (locked) ================")
    print(f"Dummy most-frequent   macro-F1={mf_macro:.3f}  acc={accuracy_score(yte, mf_pred):.3f}  (accuracy floor)")
    print(f"Dummy stratified      macro-F1={st_macro:.3f}  acc={accuracy_score(yte, st_pred):.3f}  (macro-F1 floor)")
    print(f"Method A LaBSE+LogReg macro-F1={macro:.3f}  acc={accuracy_score(yte, pred):.3f}")
    lift = macro - st_macro                       # lift over the HONEST floor
    acc_ok = accuracy_score(yte, pred) >= accuracy_score(yte, mf_pred)
    if lift > 0.03 and acc_ok:
        verdict = "real signal (beats stratified floor AND holds accuracy)"
    elif lift > 0.03:
        verdict = ("beats stratified floor on macro-F1 but accuracy is BELOW the "
                   "majority dummy -- treat as weak/ambiguous signal")
    else:
        verdict = "ties the stratified floor -- no usable figurativity signal"
    print(f"  -> {verdict}\n     macro-F1 lift over stratified floor = {lift:+.3f}")
    dpred = st_pred   # per-language comparison uses the honest floor

    print("\nclassification report (Method A):")
    print(classification_report(yte, pred, target_names=["literal", "figurative"],
                                zero_division=0))

    print("per-language macro-F1 (test, vs stratified floor):")
    rows = []
    for lang in sorted(set(langs[te])):
        m = langs[te] == lang
        mf2 = f1_score(yte[m], pred[m], average="macro", zero_division=0)
        df_ = f1_score(yte[m], dpred[m], average="macro", zero_division=0)
        n = int(m.sum()); fig = int((yte[m] == 1).sum())
        print(f"  {lang}: n={n:<4} figurative={fig:<4} methodA={mf2:.3f}  strat-floor={df_:.3f}")
        rows.append(dict(language=lang, n=n, figurative=fig,
                         methodA_macroF1=round(mf2, 3), strat_floor_macroF1=round(df_, 3)))

    pred_df = maps.iloc[te].copy()
    pred_df["pred"] = pred
    pred_df["method"] = "MethodA_LaBSE_LogReg"
    pcols = ["id", "language", "proverb_native", "proverb_en", "label", "pred", "method"]
    pred_path = os.path.join(HERE, "method_a_predictions.csv")
    pred_df[[c for c in pcols if c in pred_df.columns]].to_csv(
        pred_path, index=False, encoding="utf-8-sig")
    print(f"saved {pred_path}")

    out = pd.DataFrame(rows)
    summ = os.path.join(HERE, "method_a_results.csv")
    out.to_csv(summ, index=False)
    print(f"saved {summ}")


if __name__ == "__main__":
    main()