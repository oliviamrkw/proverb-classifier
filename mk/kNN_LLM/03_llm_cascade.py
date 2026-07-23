"""
mk/03_llm_cascade.py
==============================================================================
kNN SHORTLIST  ->  LLM PICK.   (This is the method we actually agreed on.)

Plain description of what this script does, step by step:
  1. kNN finds the 15 nearest reference proverbs to the one we're classifying.
  2. We collect the distinct Kuusi codes those 15 neighbours carry -> a shortlist.
  3. The LLM is shown ONLY that shortlist (each code + its definition) and picks one.
  4. We do this three times, narrowing each time: theme -> main -> subgroup.

Why this should work where the earlier tries didn't:
  - Pure kNN (02) got ~0.33 at theme because it blindly takes the single nearest
    neighbour. But the recall table showed the RIGHT theme is inside the top-15
    neighbours ~80-90% of the time. So the answer is usually in the shortlist.
  - Pure LLM got ~0.10 because it had to choose from all 13/52/316 categories cold.
    Here the LLM only chooses among the few codes kNN already retrieved, with their
    definitions in front of it -- a much easier job.
  - So: kNN supplies RECALL (gets the right code into the shortlist), the LLM
    supplies PRECISION (picks it out). Neither does the whole job alone.

HONEST LIMITS (read before trusting any number this prints):
  - I (the author of this file) cannot run LaBSE or an LLM in my environment, so
    this code has only been checked for logic with a --mock run that makes RANDOM
    picks. The real accuracy numbers can ONLY come from YOU running it on your
    machine with Ollama. A --mock number is meaningless; ignore it.
  - This needs 01 to have been run first (it reads 01's saved embeddings).
  - subgroup accuracy is capped by retrieval recall (~0.36 at top-20), so do not
    expect subgroup to be high no matter how good the LLM picker is.

INPUTS  (produced by 01)
    mk/npy/ref_embeddings.npy      embeddings for every reference point
    mk/ref_metadata.csv            the label for every reference point
    mk/definitions.csv             the text definition for every code
OUTPUTS
    mk/results/rerank_predictions.csv   one row per evaluated proverb
    mk/results/rerank_summary.csv       theme/main/subgroup accuracy

RUN (real, needs Ollama running + a pulled model):
    python mk/03_llm_cascade.py --model qwen2.5:7b --sample 200 --workers 4
RUN (logic test only, no server, RANDOM picks -- number is meaningless):
    python mk/03_llm_cascade.py --mock --sample 50
REQUIRES  pip install scikit-learn pandas numpy requests
==============================================================================
"""

import os                                              # file paths
import re                                              # salvage a code from messy LLM text
import json                                            # parse the LLM's JSON answer
import random                                          # reproducible mock picks
import argparse                                        # command-line flags
import numpy as np                                     # vectors / math
import pandas as pd                                    # tables / CSV
from collections import defaultdict                    # group codes -> weights
from concurrent.futures import ThreadPoolExecutor, as_completed  # run queries in parallel

SEED = 42                                              # fixed seed = reproducible runs
HERE = os.path.dirname(os.path.abspath(__file__))      # .../mk/kNN_LLM
MK = os.path.dirname(HERE)                             # .../mk
EMB  = os.path.join(MK, "npy", "ref_embeddings.npy")  # 01's embeddings
META = os.path.join(MK, "ref_metadata.csv")           # 01's labels (same order as EMB)
DEFS_CSV = os.path.join(MK, "definitions.csv")        # code -> definition text
RESULTS_DIR = os.path.join(MK, "results")             # where we write outputs
OLLAMA_URL = "http://localhost:11434/api/generate"     # local Ollama endpoint

K_NEIGHBORS    = 15     # how many nearest neighbours form the shortlist
MAX_CANDIDATES = 10     # never show the LLM more than this many distinct codes
K_FOLDS        = 5      # cross-validation folds (for leak-free evaluation)


# =============================================================================
# LOAD 01's INDEX  (embeddings + labels), and split it into examples vs defs
# =============================================================================
def l2_normalize(x):                                   # make every row length 1...
    n = np.linalg.norm(x, axis=1, keepdims=True)       # ...so a dot product == cosine similarity
    n[n == 0] = 1.0                                     # avoid divide-by-zero on any all-zero row
    return x / n                                        # return the normalized vectors


def load_index():                                      # read the reference index from disk
    emb = l2_normalize(np.load(EMB).astype("float32")) # load + normalize the embeddings
    meta = pd.read_csv(META, encoding="utf-8-sig", dtype=str).fillna("")  # load the labels
    assert len(meta) == emb.shape[0], "metadata/embedding length mismatch (re-run 01)"  # they must align

    is_ex = (meta["source_type"] == "example").values  # rows that are real example proverbs
    is_def = ~is_ex                                     # rows that are definition anchors

    ex = dict(                                          # the EXAMPLE points (these get tested)
        vec=emb[is_ex],                                 #   their embeddings
        theme=meta["theme"].values[is_ex],             #   gold theme  (A)
        main=meta["main_class"].values[is_ex],         #   gold main   (A1)
        sub=meta["subgroup_code"].values[is_ex],       #   gold subgroup (A1a)
        kid=meta["kuusi_id"].values[is_ex],            #   type id (used to prevent leakage)
        text=meta["text"].values[is_ex],               #   the proverb string (shown to the LLM)
    )
    de = dict(                                          # the DEFINITION points (always reference-only)
        vec=emb[is_def],                               #   their embeddings
        stype=meta["source_type"].values[is_def],      #   def_theme / def_main / def_subgroup
        theme=meta["theme"].values[is_def],            #   theme this definition belongs to
        main=meta["main_class"].values[is_def],        #   main this definition belongs to
        sub=meta["subgroup_code"].values[is_def],      #   subgroup this definition belongs to
    )
    print(f"loaded {len(ex['vec'])} example points, {len(de['vec'])} definition points")
    return ex, de                                       # hand both back to the caller


# =============================================================================
# LOAD THE DEFINITIONS the LLM reads when choosing among shortlist codes
# =============================================================================
def load_typology(defs_csv=DEFS_CSV):                  # build code -> human-readable description
    d = pd.read_csv(defs_csv, encoding="utf-8-sig", dtype=str).fillna("")  # read definitions.csv
    theme_desc, main_desc, sub_desc = {}, {}, {}       # one dict per level
    for _, r in d.iterrows():                          # walk every definition row
        if r["level"] == "theme":   theme_desc[r["code"]] = r["text"]   # "A"  -> description
        elif r["level"] == "main":  main_desc[r["code"]]  = r["text"]   # "A1" -> description
        elif r["level"] == "subgroup": sub_desc[r["code"]] = r["text"]  # "A1a"-> description
    return theme_desc, main_desc, sub_desc             # return the three lookup tables


# =============================================================================
# kNN SHORTLIST  -- the candidate codes from the 15 nearest neighbours
# =============================================================================
def knn_shortlist(qvec, cand_vec, cand_codes, k=K_NEIGHBORS, max_cand=MAX_CANDIDATES):
    """Return the distinct codes among the k nearest candidates, best first.

    qvec       : the query proverb's embedding
    cand_vec   : (n,d) embeddings of all reference points allowed at this level
    cand_codes : (n,)  the code each reference point carries at this level
    """
    n = len(cand_codes)                                # how many candidates exist
    if n == 0:                                         # nothing to retrieve from...
        return []                                      # ...return an empty shortlist
    sims = cand_vec @ qvec                             # cosine similarity to every candidate
    k = min(k, n)                                      # can't take more neighbours than exist
    top = np.argpartition(-sims, k - 1)[:k]            # indices of the k most similar (unsorted)
    weight = defaultdict(float)                        # code -> summed similarity (its "vote")
    for idx in top:                                    # for each of the k neighbours...
        weight[cand_codes[idx]] += max(0.0, sims[idx]) #   add its similarity to that code's vote
    ranked = sorted(weight, key=weight.get, reverse=True)  # codes ordered by strongest vote first
    return ranked[:max_cand]                           # cap the shortlist length and return it


# =============================================================================
# LLM PICK  -- choose ONE code out of the shortlist (with definitions shown)
# =============================================================================
PROMPT = (                                             # frozen prompt template
    "You are classifying a proverb into the Matti Kuusi typology.\n"
    "Pick the ONE category whose description best matches the proverb's GENERAL "
    "POINT or underlying lesson (not its surface images).\n\n"
    "Proverb:\n{proverb}\n\n"
    "Candidate categories:\n{options}\n\n"
    "Answer with ONLY JSON: {{\"code\": \"<one code from the list above>\"}}"
)


def llm_pick(proverb, options, model, mock=False, rng=None):
    """options is a list of (code, description). Return the chosen code, or "".

    If the shortlist has only one code we skip the LLM entirely and take it.
    In --mock mode we return a RANDOM code from the shortlist (logic test only).
    """
    codes = [c for c, _ in options]                    # the allowed codes
    if len(codes) == 0:                                # empty shortlist...
        return ""                                      # ...nothing to pick
    if len(codes) == 1:                                # only one option...
        return codes[0]                                # ...no need to ask the LLM
    if mock:                                           # logic-test mode...
        return rng.choice(codes)                       # ...pick at random (meaningless number)

    import requests                                    # lazy import so --mock needs no requests
    opts_text = "\n".join(f"  {c} = {d}" for c, d in options)  # render the shortlist for the prompt
    prompt = PROMPT.format(proverb=proverb, options=opts_text) # fill the template
    try:                                               # the model call can fail; guard it
        resp = requests.post(OLLAMA_URL, json={        # POST to local Ollama
            "model": model,                            #   which model
            "prompt": prompt,                          #   the prompt
            "stream": False,                           #   want the whole answer at once
            "format": "json",                          #   force valid JSON output
            "options": {"temperature": 0.0, "num_predict": 32},  # deterministic, short answer
            "keep_alive": "30m",                       #   keep model warm between calls
        }, timeout=120)                                #   give up after 120s
        raw = resp.json().get("response", "")          # the model's text response
    except Exception as e:                             # network/server error...
        print("  ! ollama error:", e)                  #   log it
        return ""                                      #   treat as "no pick"

    allowed = set(codes)                               # valid answers
    code = ""                                          # default: nothing picked
    try:                                               # try clean JSON first
        code = json.loads(raw).get("code", "").strip() #   read the "code" field
    except Exception:                                  # JSON was malformed...
        pass                                           #   fall through to salvage
    if code not in allowed:                            # if that wasn't a valid code...
        m = re.search(r"[A-Z]\d[a-z]?", raw)           #   look for a code-shaped substring
        code = m.group(0) if (m and m.group(0) in allowed) else ""  # accept only if valid
    return code if code in allowed else ""             # return a valid code or ""


# =============================================================================
# BUILD the per-fold retrieval pools  (which reference points each level may use)
# =============================================================================
def build_pools(ex, de, train_mask):
    """For ONE cross-validation fold, assemble the reference points the cascade
    retrieves from. train_mask selects the EXAMPLE points that are 'training' this
    fold (the held-out fold's examples are excluded). Definitions are always in."""
    exv, exth, exmn, exsg = ex["vec"], ex["theme"], ex["main"], ex["sub"]  # shorthands
    tr = train_mask                                    # boolean mask of training examples

    d_theme = np.ones(len(de["vec"]), bool)            # every definition carries a theme
    d_main  = np.isin(de["stype"], ["def_main", "def_subgroup"])  # these carry a main
    d_sub   = de["stype"] == "def_subgroup"            # only these carry a subgroup

    # LEVEL 1 pool (theme): all training examples + all definitions, labelled by theme
    theme_vec = np.vstack([exv[tr], de["vec"][d_theme]])           # stack their vectors
    theme_lab = np.concatenate([exth[tr], de["theme"][d_theme]])   # stack their theme codes

    # LEVEL 2 pools (main), one per theme T: training examples in T + defs in T with a main
    by_theme = {}                                      # theme -> (vectors, main-codes)
    for T in np.unique(np.concatenate([exth, de["theme"]])):       # every theme that exists
        m = tr & (exth == T)                           # training examples in theme T
        dv = de["vec"][d_main & (de["theme"] == T)]    # definition vectors in theme T (with main)
        dl = de["main"][d_main & (de["theme"] == T)]   # their main codes
        by_theme[T] = (np.vstack([exv[m], dv]) if len(dv) else exv[m],  # vectors
                       np.concatenate([exmn[m], dl]))                    # main codes

    # LEVEL 3 pools (subgroup), one per main M: training examples in M + subgroup defs in M
    by_main = {}                                       # main -> (vectors, subgroup-codes)
    for M in np.unique(np.concatenate([exmn, de["main"][d_main]])):     # every main that exists
        m = tr & (exmn == M)                           # training examples in main M
        dv = de["vec"][d_sub & (de["main"] == M)]      # subgroup-definition vectors in main M
        dl = de["sub"][d_sub & (de["main"] == M)]      # their subgroup codes
        by_main[M] = (np.vstack([exv[m], dv]) if len(dv) else exv[m],   # vectors
                      np.concatenate([exsg[m], dl]))                     # subgroup codes

    return theme_vec, theme_lab, by_theme, by_main     # hand the pools back


# =============================================================================
# THE CASCADE for one proverb:  shortlist + LLM pick, three times
# =============================================================================
def classify(qvec, text, pools, typ, model, mock, rng):
    theme_vec, theme_lab, by_theme, by_main = pools     # unpack the fold's pools
    theme_desc, main_desc, sub_desc = typ               # unpack the definition lookups

    # ---- THEME ----
    cand = knn_shortlist(qvec, theme_vec, theme_lab)    # 15-NN -> distinct theme candidates
    t_in = None                                         # (filled in by caller's gold check)
    t = llm_pick(text, [(c, theme_desc.get(c, c)) for c in cand], model, mock, rng)  # LLM picks theme
    if not t:                                           # if no valid pick...
        return ("", "", ""), (cand, [], [])             #   stop; return what we had

    # ---- MAIN (restricted to the chosen theme) ----
    mv, ml = by_theme.get(t, (np.empty((0, qvec.shape[0])), np.array([])))  # that theme's pool
    cand_m = knn_shortlist(qvec, mv, ml)                # 15-NN within the theme -> main candidates
    m = llm_pick(text, [(c, main_desc.get(c, c)) for c in cand_m], model, mock, rng)  # LLM picks main
    if not m:                                           # if no valid pick...
        return (t, "", ""), (cand, cand_m, [])          #   stop after theme

    # ---- SUBGROUP (restricted to the chosen main) ----
    sv, sl = by_main.get(m, (np.empty((0, qvec.shape[0])), np.array([])))  # that main's pool
    cand_s = knn_shortlist(qvec, sv, sl)                # 15-NN within the main -> subgroup candidates
    s = llm_pick(text, [(c, sub_desc.get(c, c)) for c in cand_s], model, mock, rng)   # LLM picks subgroup
    return (t, m, s), (cand, cand_m, cand_s)            # return predictions + the shortlists


# =============================================================================
# CHOOSE which examples to evaluate (optionally a theme-stratified subsample)
# =============================================================================
def pick_eval_indices(ex, sample, seed=SEED):
    n = len(ex["vec"])                                  # total example points
    idx = np.arange(n)                                  # all their indices
    if sample <= 0 or sample >= n:                      # 0 (or >=n) means evaluate everything
        return idx                                      # return all indices
    rng = np.random.default_rng(seed)                   # seeded RNG for reproducibility
    chosen = []                                         # collected indices
    frac = sample / n                                   # fraction to keep
    for T in np.unique(ex["theme"]):                    # spread the sample across themes
        pool = idx[ex["theme"] == T]                    # indices in this theme
        k = max(1, round(len(pool) * frac))             # how many to take from it
        chosen.append(rng.choice(pool, size=min(k, len(pool)), replace=False))  # take them
    return np.concatenate(chosen)                       # the stratified subset of indices


# =============================================================================
# CROSS-VALIDATION DRIVER  (leak-free folds, like 02)
# =============================================================================
def run_cv(ex, de, typ, model, sample, workers, mock):
    # Build folds grouped by kuusi_id (so a type's translated variants never split
    # across train/test) and stratified by theme. Falls back if sklearn is old.
    try:
        from sklearn.model_selection import StratifiedGroupKFold          # preferred splitter
        splits = list(StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
                      .split(ex["vec"], ex["theme"], groups=ex["kid"]))
        split_name = "StratifiedGroupKFold(group=kuusi_id, stratify=theme)"
    except Exception:
        from sklearn.model_selection import GroupKFold                     # fallback splitter
        splits = list(GroupKFold(n_splits=K_FOLDS).split(ex["vec"], ex["theme"], groups=ex["kid"]))
        split_name = "GroupKFold(group=kuusi_id)"

    n = len(ex["vec"])                                  # number of example points
    fold_of = np.empty(n, int)                          # which fold each example is a TEST item in
    for f, (_, te) in enumerate(splits):                # for each fold...
        fold_of[te] = f                                 #   tag its test indices with the fold id

    eval_idx = pick_eval_indices(ex, sample)            # which examples we'll actually score
    eval_set = set(eval_idx.tolist())                   # fast membership test
    print(f"CV: {split_name}, K={K_FOLDS} | evaluating {len(eval_idx)} of {n} examples | "
          f"k-NN={K_NEIGHBORS} | model={model} | mock={mock} | workers={workers}\n")

    rows = []                                           # collected per-proverb results
    for f, (tr, te) in enumerate(splits):               # process one fold at a time
        train_mask = np.zeros(n, bool); train_mask[tr] = True          # mark this fold's training examples
        assert not (set(ex["kid"][tr]) & set(ex["kid"][te])), "LEAK"   # prove no type spans train+test
        pools = build_pools(ex, de, train_mask)         # build this fold's retrieval pools once
        todo = [i for i in te if i in eval_set]         # the test items in this fold we will score
        if not todo:                                    # nothing to do this fold...
            continue                                    #   skip it

        def task(i):                                    # classify ONE proverb (run in a thread)
            rng = random.Random(SEED + int(i))          # per-item RNG (keeps --mock reproducible)
            (t, m, s), (ct, cm, cs) = classify(         # run the shortlist+pick cascade
                ex["vec"][i], ex["text"][i], pools, typ, model, mock, rng)
            return dict(                                # one output record
                fold=f, text=ex["text"][i],            #   bookkeeping
                true_theme=ex["theme"][i], true_main=ex["main"][i], true_sub=ex["sub"][i],  # gold
                pred_theme=t, pred_main=m, pred_sub=s,  #   predictions
                theme_ok=int(t == ex["theme"][i]),      #   was theme right?
                main_ok=int(m == ex["main"][i]),        #   was main right?
                sub_ok=int(s == ex["sub"][i]),          #   was subgroup right?
                theme_in_short=int(ex["theme"][i] in ct),   # was the gold theme even retrieved?
                main_in_short=int(ex["main"][i] in cm),     # was the gold main retrieved?
                sub_in_short=int(ex["sub"][i] in cs),       # was the gold subgroup retrieved?
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:   # run this fold's items in parallel
            for rec in pool.map(task, todo):            # collect each finished record
                rows.append(rec)                        #   store it
        print(f"  fold {f}: scored {len(todo)} proverbs")  # progress line

    return pd.DataFrame(rows)                            # all results as a table


# =============================================================================
# REPORT  -- print accuracy and save the two output files
# =============================================================================
def report(df):
    os.makedirs(RESULTS_DIR, exist_ok=True)             # ensure results/ exists
    theme = df["theme_ok"].mean()                       # theme accuracy
    main  = df["main_ok"].mean()                        # main accuracy
    sub   = df["sub_ok"].mean()                         # subgroup accuracy

    print("\n================ kNN->LLM RERANK ACCURACY ================")
    print(f"  theme (A)        {theme:.3f}")            # headline theme number
    print(f"  main  (A1)       {main:.3f}")            # headline main number
    print(f"  subgroup (A1a)   {sub:.3f}")            # headline subgroup number

    # Diagnostic: how often was the gold answer in the shortlist at all (kNN recall),
    # and -- when it was -- how often did the LLM then pick it (LLM precision)?
    print("\nretrieval vs picker (diagnostic):")
    for lvl, ok, ins in [("theme", "theme_ok", "theme_in_short"),
                         ("main",  "main_ok",  "main_in_short"),
                         ("sub",   "sub_ok",   "sub_in_short")]:
        recall = df[ins].mean()                         # gold present in shortlist
        sub_df = df[df[ins] == 1]                        # rows where it WAS present
        precision = sub_df[ok].mean() if len(sub_df) else float("nan")  # LLM picked it
        print(f"  {lvl:6} shortlist-recall={recall:.3f}   LLM-picks-it={precision:.3f}")

    pred_path = os.path.join(RESULTS_DIR, "rerank_predictions.csv")     # per-proverb file
    df.to_csv(pred_path, index=False, encoding="utf-8-sig")             # save it
    summ = pd.DataFrame(dict(level=["theme", "main", "subgroup"],       # summary file
                             acc=[round(theme, 4), round(main, 4), round(sub, 4)]))
    summ_path = os.path.join(RESULTS_DIR, "rerank_summary.csv")         # summary path
    summ.to_csv(summ_path, index=False)                                 # save it
    print(f"\nsaved {pred_path}")                       # tell the user where files went
    print(f"saved {summ_path}")
    print("\nThe 'shortlist-recall' is kNN's ceiling; 'LLM-picks-it' is how much of "
          "that ceiling the LLM captures. Compare theme here against pure-kNN 0.33.")


def main():
    ap = argparse.ArgumentParser(description="kNN shortlist -> LLM pick classifier.")  # CLI
    ap.add_argument("--model", default="qwen2.5:7b", help="Ollama model tag")          # which model
    ap.add_argument("--sample", type=int, default=200, help="proverbs to score (0=all)")  # eval size
    ap.add_argument("--workers", type=int, default=4, help="parallel requests")        # concurrency
    ap.add_argument("--mock", action="store_true", help="random picks; no server (logic test)")  # mock
    args = ap.parse_args()                              # parse the flags

    ex, de = load_index()                               # load 01's embeddings + labels
    typ = load_typology()                               # load the definition text
    df = run_cv(ex, de, typ, args.model, args.sample, args.workers, args.mock)  # run evaluation
    report(df)                                          # print + save results


if __name__ == "__main__":                              # only run main() when executed directly
    main()
