"""
mk/fine_tune/definition_anchored.py
==============================================================================
DEFINITION-ANCHORED CLASSIFICATION

THE IDEA
-------------------------
Instead of matching a proverb to other example proverbs (what 02 did), we match
it to the TYPOLOGY DEFINITIONS themselves. We embed every category definition
("A1a = water and fire as natural elements") and then, for each proverb, ask:
which definition's embedding is nearest? That definition's code is the prediction.

The definition is a clean, single "prototype" for its category, so this can help
where a category has almost no example proverbs to match against. It also needs
NO training and NO cross-validation: the proverbs are never used as references,
only the fixed definitions are, so there is no leakage to guard against.

We test three ways of using the definitions:
  A) FLAT THEME      : nearest of the 13 theme definitions -> theme.
  B) FLAT SUBGROUP   : nearest of the 325 subgroup definitions -> subgroup,
                       then read theme/main off that code (A1a -> A1 -> A).
  C) CASCADE         : nearest theme def -> nearest main def *within that theme*
                       -> nearest subgroup def *within that main*.

It reuses the embeddings 01 already produced (examples AND definitions are both
in there), so you do NOT need to re-embed anything.

INPUT   (produced by 01)
    mk/npy/ref_embeddings.npy
    mk/ref_metadata.csv
OUTPUT
    mk/results/defanchor_predictions.csv
    mk/results/defanchor_summary.csv

RUN       python mk/fine_tune/definition_anchored.py
REQUIRES  pip install numpy pandas
==============================================================================
"""

import os                                              # paths
import numpy as np                                     # vectors / math
import pandas as pd                                    # tables

# This script lives in mk/fine_tune/, but the shared pipeline files live in mk/.
HERE = os.path.dirname(os.path.abspath(__file__))      # .../mk/fine_tune
MK   = os.path.dirname(HERE)                           # .../mk
EMB  = os.path.join(MK, "npy", "ref_embeddings.npy")  # embeddings from 01
META = os.path.join(MK, "ref_metadata.csv")           # labels from 01
RESULTS_DIR = os.path.join(MK, "results")             # where we write our outputs

# The pure-kNN (02) numbers, hard-coded only so the printout can show them side by
# side for comparison. These are YOUR measured results, not assumptions.
KNN_REFERENCE = {"theme": 0.332, "main": 0.202, "subgroup": 0.083}


def l2_normalize(x):                                   # set each row to length 1...
    n = np.linalg.norm(x, axis=1, keepdims=True)       # ...so dot product == cosine similarity
    n[n == 0] = 1.0                                     # guard against zero-length rows
    return x / n


def load_index():
    """Load 01's index and split it into example points (to be classified) and the
    three groups of definition points (the prototypes we classify against)."""
    emb = l2_normalize(np.load(EMB).astype("float32")) # embeddings, normalized
    meta = pd.read_csv(META, encoding="utf-8-sig", dtype=str).fillna("")  # labels
    assert len(meta) == emb.shape[0], "metadata/embeddings mismatch (re-run 01)"

    st = meta["source_type"].values                    # source_type per row
    is_ex = st == "example"                             # the proverbs we want to label

    ex = dict(                                          # EXAMPLE proverbs (the test items)
        vec=emb[is_ex],                                 #   their vectors
        theme=meta["theme"].values[is_ex],             #   gold theme
        main=meta["main_class"].values[is_ex],         #   gold main
        sub=meta["subgroup_code"].values[is_ex],       #   gold subgroup
        text=meta["text"].values[is_ex],               #   the proverb text
    )
    # Definition prototypes, one group per level. Each is (vectors, codes).
    def grab(stype, code_col):                         # helper: pull a definition group
        m = st == stype                                #   rows of this definition type
        return emb[m], meta[code_col].values[m]        #   their vectors + their codes
    theme_defs = grab("def_theme", "theme")            # 13 theme prototypes
    main_defs  = grab("def_main", "main_class")        # 52 main prototypes
    sub_defs   = grab("def_subgroup", "subgroup_code") # 325 subgroup prototypes

    print(f"examples={len(ex['vec'])} | theme_defs={len(theme_defs[1])} "
          f"main_defs={len(main_defs[1])} sub_defs={len(sub_defs[1])}")
    return ex, theme_defs, main_defs, sub_defs


def nearest(ex_vec, def_vec, def_codes):
    """For every proverb, return the code of the single nearest definition.
    sims is (n_proverbs x n_defs); argmax along axis=1 picks the closest def."""
    sims = ex_vec @ def_vec.T                           # cosine similarity to every definition
    best = sims.argmax(axis=1)                          # index of the nearest definition per proverb
    return def_codes[best]                              # map those indices to their codes


def cascade(ex_vec, theme_defs, main_defs, sub_defs):
    """Definition cascade: pick theme, then the best main INSIDE that theme, then
    the best subgroup INSIDE that main. We mask out definitions that don't belong
    to the already-chosen parent by setting their similarity to -infinity."""
    tdv, tdc = theme_defs                               # theme def vectors + codes
    mdv, mdc = main_defs                                # main def vectors + codes
    sdv, sdc = sub_defs                                 # subgroup def vectors + codes

    # parent code of each main/sub definition, used for masking
    main_theme = np.array([c[0] for c in mdc])          # "A1" -> "A"
    sub_main   = np.array([c[:2] for c in sdc])         # "A1a" -> "A1"

    pred_t = nearest(ex_vec, tdv, tdc)                  # step 1: nearest theme definition

    sims_m = ex_vec @ mdv.T                             # similarity to every main definition
    for i in range(len(ex_vec)):                        # for each proverb...
        sims_m[i, main_theme != pred_t[i]] = -np.inf    #   forbid mains outside its predicted theme
    pred_m = mdc[sims_m.argmax(axis=1)]                 # step 2: best main within the theme

    sims_s = ex_vec @ sdv.T                             # similarity to every subgroup definition
    for i in range(len(ex_vec)):                        # for each proverb...
        sims_s[i, sub_main != pred_m[i]] = -np.inf      #   forbid subgroups outside its predicted main
    pred_s = sdc[sims_s.argmax(axis=1)]                 # step 3: best subgroup within the main
    return pred_t, pred_m, pred_s


def main():
    ex, theme_defs, main_defs, sub_defs = load_index()  # load everything from 01
    gold_t, gold_m, gold_s = ex["theme"], ex["main"], ex["sub"]  # gold labels

    # --- METHOD A: flat theme (nearest theme definition) ---
    A_theme = nearest(ex["vec"], *theme_defs)           # predicted theme for each proverb

    # --- METHOD B: flat subgroup (nearest subgroup def), derive theme/main from it ---
    B_sub = nearest(ex["vec"], *sub_defs)               # predicted subgroup
    B_theme = np.array([c[0] for c in B_sub])           # theme = first char
    B_main  = np.array([c[:2] for c in B_sub])          # main  = first two chars

    # --- METHOD C: definition cascade (theme -> main -> subgroup) ---
    C_theme, C_main, C_sub = cascade(ex["vec"], theme_defs, main_defs, sub_defs)

    # --- accuracies ---
    def acc(pred, gold):                                # fraction correct
        return float(np.mean(pred == gold))

    results = {
        "A_flat_theme":      {"theme": acc(A_theme, gold_t)},
        "B_flat_subgroup":   {"theme": acc(B_theme, gold_t),
                              "main":  acc(B_main, gold_m),
                              "subgroup": acc(B_sub, gold_s)},
        "C_cascade":         {"theme": acc(C_theme, gold_t),
                              "main":  acc(C_main, gold_m),
                              "subgroup": acc(C_sub, gold_s)},
    }

    print("\n================ DEFINITION-ANCHORED RESULTS ================")
    print(f"(compare to example-kNN from 02:  theme={KNN_REFERENCE['theme']:.3f}  "
          f"main={KNN_REFERENCE['main']:.3f}  subgroup={KNN_REFERENCE['subgroup']:.3f})\n")
    for method, scores in results.items():
        line = "  ".join(f"{lvl}={scores[lvl]:.3f}" for lvl in ["theme", "main", "subgroup"]
                         if lvl in scores)
        print(f"  {method:18} {line}")

    # --- save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_df = pd.DataFrame(dict(
        text=ex["text"], true_theme=gold_t, true_main=gold_m, true_sub=gold_s,
        A_theme=A_theme,
        B_theme=B_theme, B_main=B_main, B_sub=B_sub,
        C_theme=C_theme, C_main=C_main, C_sub=C_sub,
    ))
    pred_path = os.path.join(RESULTS_DIR, "defanchor_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    rows = [{"method": m, "level": lvl, "acc": round(v, 4)}
            for m, sc in results.items() for lvl, v in sc.items()]
    summ_path = os.path.join(RESULTS_DIR, "defanchor_summary.csv")
    pd.DataFrame(rows).to_csv(summ_path, index=False)
    print(f"\nsaved {pred_path}")
    print(f"saved {summ_path}")
    print("\nIf the best of these beats kNN's theme 0.332, definitions help and we "
          "build on it. If not, fine-tuning is the next script in this folder.")


if __name__ == "__main__":
    main()