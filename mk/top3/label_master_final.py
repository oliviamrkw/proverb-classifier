"""
mk/top3/label_master_final.py
==============================================================================
THE FINAL LABELER. Adds mk_theme1, mk_theme2, mk_theme3, mk_confidence to your
master CSV by FUSING two independent methods (e5 kNN + mBERT) and reading their
agreement as a confidence signal.

THE PIPELINE (what happens for every proverb in the master)
  Method 1 - e5 kNN:
     embed the proverb with multilingual-e5, find its nearest Kuusi reference
     proverbs, and turn the neighbours' themes into a 13-way vote distribution.
  Method 2 - mBERT:
     a bert-base-multilingual-cased model (trained here on all Kuusi examples,
     class-weighted so it can't collapse) outputs its own 13-way probability.
  FUSE:
     average the two distributions. The 3 highest themes become
     mk_theme1 / mk_theme2 / mk_theme3 (highest-confidence first).
  CONFIDENCE (mk_confidence):
     HIGH   - both methods' #1 theme is the SAME   (~0.51 theme1 / ~0.70 top-3)
     MEDIUM - mBERT's #1 is inside the fused top-3  (~0.29 theme1 / ~0.59 top-3)
     LOW    - the methods disagree                  (~0.28 theme1 / ~0.55 top-3)
     (tier accuracies measured on held-out Kuusi CV data.)
  Overall: top-3 ~0.61, theme1 ~0.35. So mk_theme1/2/3 are CANDIDATE themes, not
  a single answer, and mk_confidence tells a user how much to trust them.

WHICH TEXT IS CLASSIFIED
  By default the English column (proverb_en) when it's non-empty, else the native
  text -- the Kuusi reference is ~99% English so English matches more reliably.
  Override with --text-col / --no-english-fallback.

COLUMN PLACEMENT
  The four columns are inserted right AFTER the existing 'mk_theme' column (kept,
  not overwritten). Every row in the master gets values -> consistent schema.

ORDER OF OPERATIONS
  python mk/01_build_reference_index.py --encoder e5        # builds the index
  python mk/top3/label_master_final.py --master data/processed/master.csv

INPUT   mk/npy/ref_embeddings.npy , mk/ref_metadata.csv , mk/npy/encoder.txt  (from 01)
        mk/kuusi_proverb_types_clean.csv  (mBERT training data)
        --master  your master CSV
OUTPUT  mk/top3/results/master_labeled.csv
HARDWARE  GPU recommended (mBERT training). CPU works but is slow.
INSTALL  pip install sentence-transformers torch transformers scikit-learn pandas numpy
==============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
# torch/transformers/sentence_transformers imported inside the functions that use
# them, so the data + fusion + confidence logic can be inspected without them.

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
MK   = os.path.dirname(HERE)
EMB  = os.path.join(MK, "npy", "ref_embeddings.npy")
META = os.path.join(MK, "ref_metadata.csv")
ENC_FILE = os.path.join(MK, "npy", "encoder.txt")
KUUSI_CSV = os.path.join(MK, "kuusi_proverb_types_clean.csv")
RESULTS_DIR = os.path.join(HERE, "results")

K_NEIGHBORS = 15
EXAMPLE_COLS = ["proverb_variant_1", "proverb_variant_2", "proverb_variant_3",
                "proverb_variant_4", "proverb_variant_extra", "proverb_primary_en"]


def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return x / n


# -----------------------------------------------------------------------------
# pick the text to classify: English when present (best match to the reference),
# else the native proverb.
# -----------------------------------------------------------------------------
def pick_texts(master, text_col, en_fallback):
    if text_col:                                        # explicit column requested
        return master[text_col].astype(str).values
    if en_fallback and "proverb_en" in master and "proverb_native" in master:
        en = master["proverb_en"].astype(str)
        nat = master["proverb_native"].astype(str)
        return np.where(en.str.strip() != "", en, nat).astype(str)   # English, else native
    return master["proverb_native"].astype(str).values


# -----------------------------------------------------------------------------
# METHOD 1: e5 kNN -> a 13-way theme distribution for each master proverb
# -----------------------------------------------------------------------------
def e5_distributions(texts, themes_order):
    from sentence_transformers import SentenceTransformer
    if os.path.exists(ENC_FILE):
        key, hf_id, prefix = open(ENC_FILE, encoding="utf-8").read().strip().split("\t")
    else:
        hf_id, prefix = "intfloat/multilingual-e5-large", "query: "
        print("WARNING: encoder.txt missing; assuming e5.")
    print(f"e5 encoder: {hf_id} prefix={prefix!r}")

    ref = l2(np.load(EMB).astype("float32"))
    meta = pd.read_csv(META, encoding="utf-8-sig", dtype=str).fillna("")
    is_ex = (meta["source_type"] == "example").values
    ref_vec, ref_theme = ref[is_ex], meta["theme"].values[is_ex]      # example voters

    model = SentenceTransformer(hf_id)
    q = l2(model.encode([prefix + t for t in texts], batch_size=64,
                        show_progress_bar=True, convert_to_numpy=True).astype("float32"))

    ti = {t: i for i, t in enumerate(themes_order)}
    dist = np.zeros((len(texts), len(themes_order)), dtype="float32")
    for r in range(len(texts)):
        sims = ref_vec @ q[r]
        top = np.argpartition(-sims, K_NEIGHBORS - 1)[:K_NEIGHBORS]
        for idx in top:
            dist[r, ti[ref_theme[idx]]] += max(0.0, sims[idx])       # weighted vote
    dist /= dist.sum(1, keepdims=True).clip(min=1e-9)                # normalize to a distribution
    return dist


# -----------------------------------------------------------------------------
# METHOD 2: mBERT trained on all Kuusi examples -> a 13-way probability per master row
# -----------------------------------------------------------------------------
def mbert_distributions(texts, themes_order, args):
    import torch
    import torch.nn.functional as F
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n*** mBERT DEVICE = {device} ***" +
          ("" if device == "cuda" else "  <-- CPU is slow; install CUDA torch."))

    # training data: explode the Kuusi CSV into (text, theme)
    kdf = pd.read_csv(KUUSI_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    tr_text, tr_theme = [], []
    for _, r in kdf.iterrows():
        seen = set()
        for c in EXAMPLE_COLS:
            t = r[c].strip()
            if t and t not in seen:
                seen.add(t); tr_text.append(t); tr_theme.append(r["theme"])
    ti = {t: i for i, t in enumerate(themes_order)}
    y = np.array([ti[t] for t in tr_theme])
    n_labels = len(themes_order)

    counts = np.bincount(y, minlength=n_labels).astype("float32")    # class weights
    w = torch.tensor(counts.sum() / (n_labels * np.maximum(counts, 1)),
                     dtype=torch.float32, device=device)

    tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=n_labels).to(device)

    enc = tok(tr_text, truncation=True, padding=True, max_length=args.max_len, return_tensors="pt")
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"],
                                      torch.tensor(y)), batch_size=args.batch, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total), total)
    use_amp = device == "cuda"; scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for ep in range(args.epochs):
        for ids, mask, yb in loader:
            ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = F.cross_entropy(logits, yb, weight=w)         # class-weighted
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update(); sched.step()
        print(f"  mBERT epoch {ep+1}/{args.epochs} done")

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            chunk = list(texts[i:i+args.batch])
            enc = tok(chunk, truncation=True, padding=True, max_length=args.max_len,
                      return_tensors="pt").to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**enc).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


# -----------------------------------------------------------------------------
# FUSE + CONFIDENCE
# -----------------------------------------------------------------------------
def fuse_and_label(e5_dist, mb_dist, themes_order):
    fused = (e5_dist + mb_dist) / 2.0                   # average the two opinions
    order = np.argsort(-fused, axis=1)                  # rank themes per row
    t1 = np.array([themes_order[order[i, 0]] for i in range(len(fused))])
    t2 = np.array([themes_order[order[i, 1]] for i in range(len(fused))])
    t3 = np.array([themes_order[order[i, 2]] for i in range(len(fused))])
    e5_top = np.array([themes_order[e5_dist[i].argmax()] for i in range(len(fused))])
    mb_top = np.array([themes_order[mb_dist[i].argmax()] for i in range(len(fused))])

    conf = []
    for i in range(len(fused)):
        if e5_top[i] == mb_top[i]:                       # both methods' #1 agree
            conf.append("HIGH")
        elif mb_top[i] in (t1[i], t2[i], t3[i]):         # mBERT's #1 is within the top-3
            conf.append("MEDIUM")
        else:
            conf.append("LOW")
    return t1, t2, t3, np.array(conf)


def insert_after(df, anchor, newcols):
    """Insert new columns right after `anchor` (if present), else append at the end."""
    cols = list(df.columns)
    for c in newcols:                                   # avoid duplicates on re-run
        if c in cols: cols.remove(c)
    if anchor in cols:
        at = cols.index(anchor) + 1
        cols = cols[:at] + newcols + cols[at:]
    else:
        cols = cols + newcols
    return df[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--text-col", default="", help="force a specific text column")
    ap.add_argument("--no-english-fallback", action="store_true",
                    help="classify proverb_native instead of preferring English")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=48)
    args = ap.parse_args()

    master = pd.read_csv(args.master, encoding="utf-8-sig", dtype=str).fillna("")
    themes_order = sorted(pd.read_csv(META, encoding="utf-8-sig", dtype=str)
                          .fillna("")["theme"].unique())              # canonical 13-theme order
    print(f"master rows: {len(master)} | themes: {themes_order}")

    texts = pick_texts(master, args.text_col or None, not args.no_english_fallback)
    e5_dist = e5_distributions(texts, themes_order)                   # method 1
    mb_dist = mbert_distributions(texts, themes_order, args)          # method 2
    t1, t2, t3, conf = fuse_and_label(e5_dist, mb_dist, themes_order) # fuse + confidence

    master["mk_theme1"], master["mk_theme2"], master["mk_theme3"] = t1, t2, t3
    master["mk_confidence"] = conf
    master = insert_after(master, "mk_theme",
                          ["mk_theme1", "mk_theme2", "mk_theme3", "mk_confidence"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "master_labeled.csv")
    master.to_csv(out, index=False, encoding="utf-8-sig")
    dist = pd.Series(conf).value_counts(normalize=True).round(3).to_dict()
    print(f"\nsaved {out}\nconfidence distribution: {dist}")
    print("HIGH -> trust mk_theme1 | MEDIUM -> use the top-3 set | LOW -> uncertain")


if __name__ == "__main__":
    main()