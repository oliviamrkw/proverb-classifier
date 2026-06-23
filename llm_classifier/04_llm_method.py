"""
04_method_c.py  --  METHOD C: LLM classification via Ollama
===========================================================
Sends the SAME locked test rows Method A was scored on to a local LLM and asks
for a figurative/literal label. Same test set => fair head-to-head with A.
 
PROMPT (MAPS-aligned definition, frozen):
  Matches the MAPS annotation definition -- a proverb is FIGURATIVE when its
  intended meaning differs from the literal words, LITERAL when they match.
 
CONDITIONS:
  zero-shot  (default, --shots 0): definition only, no labeled examples.
  few-shot   (--shots K): prepend K balanced labeled examples drawn from the
             TRAIN split only (never test). K is rounded to an even number.
 
VARIANTS (does English touch the proverb?):
  native_only   native proverb only  (primary; matches Method A)
  with_english  adds the English gloss  (ablation)
 
Reproducibility: temperature=0, prompt frozen, model version recorded, parse
failures logged + excluded from metrics and reported as a rate, examples seeded.
 
Examples:
  python llm_classifier/04_method_c.py --model llama3.1 --limit 5            # pilot
  python llm_classifier/04_method_c.py --model llama3.1 --limit 0            # full zero-shot
  python llm_classifier/04_method_c.py --model llama3.1 --shots 4 --limit 0  # few-shot
  python llm_classifier/04_method_c.py --model aya-expanse:8b --limit 0
  python llm_classifier/04_method_c.py --mock --limit 6                      # plumbing, no LLM
 
Requires: pip install requests scikit-learn   (and Ollama running)
"""
 
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
 
HERE = os.path.dirname(os.path.abspath(__file__))          # llm_classifier/
ROOT = os.path.dirname(HERE)
MASTER = os.path.join(ROOT, "data", "processed", "master.csv")
SPLIT_DIR = os.path.join(ROOT, "embedding_classifier")     # shared split
TEST = os.path.join(SPLIT_DIR, "test_indices.npy")
TRAIN = os.path.join(SPLIT_DIR, "train_indices.npy")
OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42

DEF_BLOCK = (
    "You are classifying proverbs as FIGURATIVE or LITERAL.\n\n"
    "FIGURATIVE: the interpreted meaning of the proverb is different from the "
    "expressed literal meaning -- the surface describes one thing but the "
    "message is about another.\n"
    "LITERAL: the proverb's intended meaning matches what the words say "
    "-- a direct statement or piece of advice.\n\n"
)
OUTPUT_INSTR = ('Respond with ONLY a JSON object: {"label": 0} for literal or '
                '{"label": 1} for figurative. No explanation.')
 
 
def query_block(variant, row):
    if variant == "with_english":
        return (f"Proverb (original language): {row['proverb_native']}\n"
                f"English translation: {row['proverb_en']}\n\n")
    return f"Proverb: {row['proverb_native']}\n\n"
 
 
def load_maps(path=MASTER):
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    m = d[d.source_dataset == "MAPS"].copy().reset_index(drop=True)
    m["label"] = pd.to_numeric(m["fig_or_literal"], errors="coerce")
    m = m[m["label"].notna()].copy().reset_index(drop=True)
    m["label"] = m["label"].astype(int)
    return m
 
 
def few_shot_block(maps, variant, shots):
    """Balanced labeled examples from the TRAIN split only (never test)."""
    if shots <= 0:
        return ""
    per = max(1, shots // 2)
    tr = maps.iloc[np.load(TRAIN)]
    rng = np.random.default_rng(SEED)
    picks = []
    for lab in (1, 0):
        pool = tr[tr["label"] == lab]
        take = min(per, len(pool))
        picks += list(pool.sample(n=take, random_state=SEED).index)
    rng.shuffle(picks)
    out = ["Examples:"]
    for idx in picks:
        r = maps.loc[idx]
        out.append(query_block(variant, r).rstrip())
        out.append(json.dumps({"label": int(r["label"])}))
        out.append("")
    out.append("Now classify this proverb:\n")
    return "\n".join(out)
 
 
def build_prompt(variant, row, shots_block):
    return DEF_BLOCK + shots_block + query_block(variant, row) + OUTPUT_INSTR
 
 
def call_ollama(model, prompt, mock=False, mock_i=0):
    if mock:
        if mock_i % 4 == 3:
            return "Sure! I'd say this proverb is figurative."     # parse failure
        return '{"label": %d}' % (mock_i % 2)
    import requests
    try:
        r = requests.post(OLLAMA_URL, timeout=120, json={
            "model": model, "prompt": prompt, "stream": False,
            "format": "json",                 # constrain output to valid JSON
            "keep_alive": "30m",              # don't unload the model between calls
            "options": {"temperature": 0,     # deterministic
                        "num_predict": 16}})  # cap output; label needs ~5 tokens
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return ""                              # network/timeout -> counts as a miss
 
 
def parse_label(text):
    if not text:
        return None
    for c in [text.strip()] + re.findall(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and "label" in obj:
                v = int(obj["label"])
                if v in (0, 1):
                    return v
        except Exception:
            pass
    m = re.search(r"label\D{0,5}([01])", text, re.I)
    return int(m.group(1)) if m else None
 
 
def summarize(test):
    """Compact metrics on parsed rows: macro-F1, acc, per-class recall, per-lang."""
    from sklearn.metrics import f1_score, recall_score, accuracy_score
    scored = test[test["pred"].notna()].copy()
    if len(scored) < 2:
        print("(too few parsed rows for metrics)")
        return
    y = scored["label"].astype(int).values
    p = scored["pred"].astype(int).values
    print(f"\nmacro-F1={f1_score(y,p,average='macro',zero_division=0):.3f}  "
          f"acc={accuracy_score(y,p):.3f}  "
          f"recall[literal]={recall_score(y,p,pos_label=0,zero_division=0):.3f}  "
          f"recall[figurative]={recall_score(y,p,pos_label=1,zero_division=0):.3f}")
    print("per-language macro-F1:")
    for lang in sorted(scored["language"].unique()):
        m = scored["language"] == lang
        print(f"  {lang}: n={int(m.sum()):<4} "
              f"macroF1={f1_score(y[m.values],p[m.values],average='macro',zero_division=0):.3f}")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--variant", default="native_only",
                    choices=["native_only", "with_english"])
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples (0=zero-shot)")
    ap.add_argument("--limit", type=int, default=5, help="rows to run; 0 = full test set")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests to ONE model (needs GPU headroom + "
                         "OLLAMA_NUM_PARALLEL set on the server). Start at 4.")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
 
    maps = load_maps()
    test = maps.iloc[np.load(TEST)].reset_index(drop=True)
    if args.limit and args.limit > 0:
        test = test.iloc[:args.limit].copy()
 
    shots_block = few_shot_block(maps, args.variant, args.shots)
    cond = f"{args.shots}-shot" if args.shots > 0 else "zero-shot"
    tag = f"{args.model.replace(':','_')}_{args.variant}_{cond}"
    print(f"model={args.model} variant={args.variant} {cond} rows={len(test)} "
          f"workers={args.workers} mock={args.mock}\n")
 
    rows = list(test.iterrows())
 
    def work(item):
        i, row = item
        raw = call_ollama(args.model, build_prompt(args.variant, row, shots_block),
                          mock=args.mock, mock_i=i)
        return i, parse_label(raw)
 
    preds = [None] * len(rows)
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, lab in ex.map(work, rows):
                preds[i] = lab
                done += 1
                if done % 25 == 0 or done == len(rows):
                    print(f"  {done}/{len(rows)} done")
    else:
        for item in rows:
            i, lab = work(item)
            preds[i] = lab
            row = item[1]
            gold = int(row["label"])
            ok = "" if lab is None else ("Y" if lab == gold else "n")
            nat = (row["proverb_native"][:40] + "…") if len(row["proverb_native"]) > 40 else row["proverb_native"]
            print(f"[{i:>3}] {row['language']} gold={gold} pred={str(lab):>4} {ok}  | {nat}")
 
    fails = sum(1 for p in preds if p is None)
    test = test.copy()
    test["pred"] = preds
    test["method"] = tag
    print(f"\nrows={len(test)}  parse/miss failures={fails} ({fails/len(test)*100:.0f}%)")
    summarize(test)
 
    out = os.path.join(HERE, f"method_c_{tag}.csv")
    cols = ["id", "language", "proverb_native", "proverb_en", "label", "pred", "method"]
    test[[c for c in cols if c in test.columns]].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nsaved {out}")
    if args.limit and args.limit > 0:
        print("LIMITED pilot run. Use --limit 0 for the full locked test set.")
 
 
if __name__ == "__main__":
    main()