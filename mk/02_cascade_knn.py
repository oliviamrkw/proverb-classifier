"""
mk/02_cascade_knn.py
==============================================================================
STEP 2 OF THE MATTI KUUSI PIPELINE  --  cascade kNN classifier + honest CV.

WHAT THIS SCRIPT DOES
---------------------
It answers one question for your supervisor: "how accurately can we assign a
proverb to its Kuusi theme / main_class / subgroup?" -- measured the rigorous way.

For every reference EXAMPLE proverb it predicts a code by k-Nearest-Neighbours in
LaBSE space, using a 3-level CASCADE:
    1. predict the THEME   (A..T, 13 options)        -- vote over the whole index
    2. predict the MAIN    (A1.., restricted to the predicted theme)
    3. predict the SUBGROUP (A1a.., restricted to the predicted main)
Narrowing at each step means the final vote is among ~5-6 sibling subgroups, not
all 316 at once. It also honours your priority order: theme is decided first and
is never corrupted by a fine-grained mistake further down.

HOW IT IS EVALUATED (this is the part your supervisor flagged)
--------------------------------------------------------------
A single 80/20 split is noisy and, worse, it would LEAK here. The Kuusi "variants"
of one type are often translations of the SAME proverb (e.g. "Let things take their
course" / "Man muss den Dingen ihren Lauf lassen"). If one translation sits in train
and the other in test, the model trivially matches a sentence to its own translation
and accuracy looks far better than it really is.

So we use leave-TYPES-out cross-validation:
  * Folds are built with StratifiedGroupKFold: grouped by kuusi_id (so ALL variants
    of a type stay together in one fold -> no translation leak), and stratified by
    theme (so every fold sees every theme in proportion).
  * Every example is tested exactly once, across K folds -> we "test on more data",
    and report mean +/- standard deviation, which is the publishable, stable number.

Definitions (from 01) are ALWAYS in the reference set, never tested, never grouped.
They act as a backstop -- especially for tiny subgroups whose only example got held
out -- and you can switch them off with USE_DEFINITIONS to measure their lift.

IMPORTANT HONESTY CAVEAT (state this in the paper)
--------------------------------------------------
This CV measures "can we re-find a held-out Kuusi example's code from OTHER types'
examples + definitions". That is a PROXY for labelling your 21k. It is still
somewhat optimistic, because curated Kuusi examples are cleaner and more typical
than arbitrary real-world proverbs. The human-validated sample (later step) is what
turns this proxy into a defensible real-world number. Do not publish the CV figure
as if it were the 21k accuracy.

INPUTS  (from 01)
    mk/npy/ref_embeddings.npy
    mk/ref_metadata.csv
OUTPUTS
    mk/results/cv_predictions.csv   (every example, its true vs predicted codes)
    mk/results/cv_summary.csv       (per-level mean/std accuracy)

RUN       python mk/02_cascade_knn.py
REQUIRES  pip install scikit-learn pandas numpy
==============================================================================
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

SEED = 42

# -----------------------------------------------------------------------------
# PATHS -- anchored to this script's folder (mk/), same convention as 01.
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
EMB  = os.path.join(HERE, "npy", "ref_embeddings.npy")
META = os.path.join(HERE, "ref_metadata.csv")
RESULTS_DIR = os.path.join(HERE, "results")

# -----------------------------------------------------------------------------
# KNOBS you can tune (and SHOULD experiment with for the write-up).
# -----------------------------------------------------------------------------
K_NEIGHBORS    = 15      # how many nearest references vote
WEIGHTED       = True    # weight each vote by cosine similarity (closer = stronger)
USE_DEFINITIONS = True   # include definition anchors in the reference set?
K_FOLDS        = 5       # cross-validation folds (5 or 10 are both standard)


# =============================================================================
# LOAD + PREP THE INDEX
# =============================================================================
def l2_normalize(x):
    """Scale each row to unit length so a dot product == cosine similarity.
    LaBSE is a cosine-geometry encoder, so this is the geometrically correct step
    (same choice as Method A in the figurative/literal task)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def load_index():
    """Load embeddings + metadata from 01, normalize, and split into the EXAMPLE
    points (the things we test) and the DEFINITION points (fixed anchors)."""
    emb = l2_normalize(np.load(EMB).astype("float32"))
    meta = pd.read_csv(META, encoding="utf-8-sig", dtype=str).fillna("")
    assert len(meta) == emb.shape[0], "metadata/embedding length mismatch (re-run 01)"

    is_ex = (meta["source_type"] == "example").values
    is_def = ~is_ex

    ex = dict(
        vec=emb[is_ex],
        theme=meta["theme"].values[is_ex],
        main=meta["main_class"].values[is_ex],
        sub=meta["subgroup_code"].values[is_ex],
        kid=meta["kuusi_id"].values[is_ex],      # group key for leak-free folds
        text=meta["text"].values[is_ex],
        ref_idx=meta["ref_idx"].values[is_ex],
    )
    de = dict(
        vec=emb[is_def],
        stype=meta["source_type"].values[is_def],   # def_theme / def_main / def_subgroup
        theme=meta["theme"].values[is_def],
        main=meta["main_class"].values[is_def],
        sub=meta["subgroup_code"].values[is_def],
    )
    print(f"loaded {len(ex['vec'])} example points, {len(de['vec'])} definition points")
    return ex, de


# =============================================================================
# THE kNN VOTE -- the single primitive used at every cascade level
# =============================================================================
def knn_vote(qvec, cand_vec, cand_lab, k=K_NEIGHBORS, weighted=WEIGHTED):
    """Find the k references closest to qvec, then vote on their labels.

    cand_vec : (n, d) candidate vectors (already L2-normalized)
    cand_lab : (n,)   the code each candidate carries at THIS level
    returns  : (winning_label, confidence) where confidence = winner's share of
               the total vote weight (1.0 = unanimous, ~0 = a coin toss).
    If there are no candidates we return ("", 0.0) so the caller can mark it
    'unassigned' instead of inventing a label.
    """
    n = len(cand_lab)
    if n == 0:
        return "", 0.0
    sims = cand_vec @ qvec                       # cosine similarity to every candidate
    k = min(k, n)
    # argpartition grabs the top-k by similarity without fully sorting (fast).
    top = np.argpartition(-sims, k - 1)[:k]
    top_sims = sims[top]
    top_labs = cand_lab[top]
    # Vote weights: similarity-weighted (clip negatives to 0) or plain 1-per-vote.
    if weighted:
        w = np.clip(top_sims, 0.0, None)
        if w.sum() == 0:                         # all neighbours pointed away -> fall back
            w = np.ones_like(top_sims)
    else:
        w = np.ones_like(top_sims)
    tally = defaultdict(float)
    for lab, weight in zip(top_labs, w):
        tally[lab] += weight
    winner = max(tally, key=tally.get)
    confidence = tally[winner] / sum(tally.values())
    return winner, float(confidence)


# =============================================================================
# PER-FOLD CANDIDATE SETS -- precomputed once per fold for speed + clarity
# =============================================================================
def build_candidates(ex, de, train_mask):
    """Assemble the reference pools the cascade will draw on, for ONE fold.

    train_mask selects which EXAMPLE points are 'training' this fold (the test
    fold's examples are excluded). Definition points are always included (if
    USE_DEFINITIONS). We pre-slice by theme and by main so each query just does a
    dictionary lookup instead of re-filtering the whole index every time.
    """
    exv, exth, exmn, exsg = ex["vec"], ex["theme"], ex["main"], ex["sub"]
    tr = train_mask

    # ---- definition slices (constant across folds) ----
    if USE_DEFINITIONS:
        d_has_theme = np.ones(len(de["vec"]), bool)                 # all defs carry a theme
        d_has_main  = np.isin(de["stype"], ["def_main", "def_subgroup"])
        d_is_sub    = de["stype"] == "def_subgroup"
    else:
        d_has_theme = d_has_main = d_is_sub = np.zeros(len(de["vec"]), bool)

    # ---- LEVEL 1 (theme): all training examples + all definitions ----
    theme_vec = np.vstack([exv[tr], de["vec"][d_has_theme]]) if d_has_theme.any() else exv[tr]
    theme_lab = np.concatenate([exth[tr], de["theme"][d_has_theme]])

    # ---- LEVEL 2 (main), pre-sliced per theme T ----
    by_theme = {}
    themes = np.unique(np.concatenate([exth, de["theme"]]))
    for T in themes:
        ex_m = tr & (exth == T)
        dv = de["vec"][d_has_main & (de["theme"] == T)]
        dl = de["main"][d_has_main & (de["theme"] == T)]
        v = np.vstack([exv[ex_m], dv]) if len(dv) else exv[ex_m]
        l = np.concatenate([exmn[ex_m], dl])
        by_theme[T] = (v, l)

    # ---- LEVEL 3 (subgroup), pre-sliced per main M ----
    by_main = {}
    mains = np.unique(np.concatenate([exmn, de["main"][d_has_main]]))
    for M in mains:
        ex_m = tr & (exmn == M)
        dv = de["vec"][d_is_sub & (de["main"] == M)]
        dl = de["sub"][d_is_sub & (de["main"] == M)]
        v = np.vstack([exv[ex_m], dv]) if len(dv) else exv[ex_m]
        l = np.concatenate([exsg[ex_m], dl])
        by_main[M] = (v, l)

    # ---- FLAT subgroup baseline (diagnostic): one direct vote over all subgroups ----
    flat_vec = np.vstack([exv[tr], de["vec"][d_is_sub]]) if d_is_sub.any() else exv[tr]
    flat_lab = np.concatenate([exsg[tr], de["sub"][d_is_sub]])

    return theme_vec, theme_lab, by_theme, by_main, flat_vec, flat_lab


# =============================================================================
# THE CASCADE for one query
# =============================================================================
def cascade_predict(qvec, theme_vec, theme_lab, by_theme, by_main):
    """Run theme -> main -> subgroup, each restricted by the previous decision."""
    t_pred, t_conf = knn_vote(qvec, theme_vec, theme_lab)
    mv, ml = by_theme.get(t_pred, (np.empty((0, qvec.shape[0])), np.array([])))
    m_pred, m_conf = knn_vote(qvec, mv, ml)
    sv, sl = by_main.get(m_pred, (np.empty((0, qvec.shape[0])), np.array([])))
    s_pred, s_conf = knn_vote(qvec, sv, sl)
    return (t_pred, m_pred, s_pred), (t_conf, m_conf, s_conf)


# =============================================================================
# CROSS-VALIDATION DRIVER
# =============================================================================
def run_cv(ex, de):
    # StratifiedGroupKFold = stratify by theme, group by kuusi_id (no translation leak).
    # Fall back to GroupKFold if the installed sklearn is too old to have it.
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        splits = splitter.split(ex["vec"], ex["theme"], groups=ex["kid"])
        split_name = "StratifiedGroupKFold (group=kuusi_id, stratify=theme)"
    except Exception:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=K_FOLDS)
        splits = splitter.split(ex["vec"], ex["theme"], groups=ex["kid"])
        split_name = "GroupKFold (group=kuusi_id)"
    print(f"CV: {split_name}, K={K_FOLDS}, k-neighbours={K_NEIGHBORS}, "
          f"weighted={WEIGHTED}, definitions={USE_DEFINITIONS}\n")

    n = len(ex["vec"])
    rows = []
    fold_scores = []   # (theme_acc, main_acc, sub_acc, flat_acc) per fold

    for fold, (tr_idx, te_idx) in enumerate(splits):
        train_mask = np.zeros(n, bool); train_mask[tr_idx] = True

        # SAFETY: prove no type leaks across the split (no shared kuusi_id).
        assert not (set(ex["kid"][tr_idx]) & set(ex["kid"][te_idx])), \
            "LEAK: a kuusi_id appears in both train and test"

        theme_vec, theme_lab, by_theme, by_main, flat_vec, flat_lab = \
            build_candidates(ex, de, train_mask)

        tc = mc = sc = fc = 0
        for i in te_idx:
            q = ex["vec"][i]
            (tp, mp, sp), (tcf, mcf, scf) = cascade_predict(
                q, theme_vec, theme_lab, by_theme, by_main)
            fp, _ = knn_vote(q, flat_vec, flat_lab)   # flat baseline (no cascade)

            t_ok = tp == ex["theme"][i]
            m_ok = mp == ex["main"][i]
            s_ok = sp == ex["sub"][i]
            f_ok = fp == ex["sub"][i]
            tc += t_ok; mc += m_ok; sc += s_ok; fc += f_ok

            rows.append(dict(
                fold=fold, ref_idx=ex["ref_idx"][i], kuusi_id=ex["kid"][i],
                text=ex["text"][i],
                true_theme=ex["theme"][i], true_main=ex["main"][i], true_sub=ex["sub"][i],
                pred_theme=tp, pred_main=mp, pred_sub=sp,
                theme_conf=round(tcf, 3), main_conf=round(mcf, 3), sub_conf=round(scf, 3),
                theme_ok=int(t_ok), main_ok=int(m_ok), sub_ok=int(s_ok),
                flat_sub=fp, flat_ok=int(f_ok),
            ))

        nt = len(te_idx)
        fold_scores.append((tc/nt, mc/nt, sc/nt, fc/nt))
        print(f"  fold {fold}: n={nt:<5} theme={tc/nt:.3f}  main={mc/nt:.3f}  "
              f"subgroup={sc/nt:.3f}  (flat subgroup={fc/nt:.3f})")

    return pd.DataFrame(rows), np.array(fold_scores)


# =============================================================================
# REPORTING
# =============================================================================
def report(pred_df, fold_scores):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mean = fold_scores.mean(axis=0); std = fold_scores.std(axis=0)
    labels = ["theme (A)", "main (A1)", "subgroup (A1a)", "flat subgroup (no cascade)"]

    print("\n================ CROSS-VALIDATED ACCURACY ================")
    print("(cascade accuracy compounds: each level needs the one above it right)\n")
    for lab, m, s in zip(labels, mean, std):
        print(f"  {lab:<28} {m:.3f}  +/- {s:.3f}")

    # Conditional diagnostics: WHERE does the cascade lose proverbs?
    t_ok = pred_df["theme_ok"] == 1
    m_ok = pred_df["main_ok"] == 1
    main_given_theme = pred_df.loc[t_ok, "main_ok"].mean()
    sub_given_main   = pred_df.loc[m_ok, "sub_ok"].mean()
    print("\nconditional (diagnostic):")
    print(f"  main correct | theme correct   : {main_given_theme:.3f}")
    print(f"  subgroup correct | main correct: {sub_given_main:.3f}")

    # Per-theme subgroup accuracy: which themes are intrinsically hard?
    print("\nper-theme subgroup accuracy (cascade):")
    pt = pred_df.groupby("true_theme")["sub_ok"].agg(["mean", "size"])
    for th, r in pt.iterrows():
        print(f"  {th}: {r['mean']:.3f}  (n={int(r['size'])})")

    # Save artefacts.
    pred_path = os.path.join(RESULTS_DIR, "cv_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summ = pd.DataFrame(dict(
        level=["theme", "main", "subgroup", "flat_subgroup"],
        mean_acc=np.round(mean, 4), std_acc=np.round(std, 4)))
    summ_path = os.path.join(RESULTS_DIR, "cv_summary.csv")
    summ.to_csv(summ_path, index=False)
    print(f"\nsaved {pred_path}")
    print(f"saved {summ_path}")
    print("\nnext: mk/03_llm_reranker.py (LLM picks among the top-5 subgroup candidates)")


def main():
    ex, de = load_index()
    pred_df, fold_scores = run_cv(ex, de)
    report(pred_df, fold_scores)


if __name__ == "__main__":
    main()