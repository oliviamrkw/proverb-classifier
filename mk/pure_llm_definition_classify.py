"""
mk/pure_llm_definition_classify.py
==============================================================================
PURE LLM, DEFINITION-BASED CLASSIFICATION  (no retrieval, no embeddings)

THIS IS NOT `03_llm_cascade.py` AND NOT `fine_tune/definition_anchored.py`.
Read this if you're not sure which file you want:

  - definition_anchored.py : NO LLM AT ALL. Embeds the definitions, embeds the
                             proverb, picks the definition with the closest
                             cosine similarity. Pure math, zero reasoning.
  - 03_llm_cascade.py      : USES KNN. Finds the 15 nearest EXAMPLE proverbs
                             first, builds a shortlist of the codes they carry,
                             and only THEN asks the LLM to pick among that
                             shortlist. The LLM never sees the full definition
                             list, and retrieval does most of the work.
  - THIS FILE              : No embeddings, no shortlist, no nearest-neighbor
                             anything. The LLM is handed the actual human-
                             written definitions (all of them, or all of them
                             at one level) and reasons about which one the
                             proverb fits, cold. This is the "Pure LLM: 0.095"
                             result mentioned in the project notes — this is
                             the file that produces that number (or a real,
                             reproducible version of it, since that number
                             apparently was never saved as a clean script).

WHY 316 SUBGROUP DEFINITIONS IS A PROBLEM, AND THE TWO MODES THIS SCRIPT OFFERS
--------------------------------------------------------------------------
Handing an LLM all 316 subgroup definitions in one prompt is a lot of text,
and is exactly the "choose from all 13/52/316 categories cold" setup that
scored ~0.10 on theme in earlier testing. So this script supports two modes:

  --mode cascade (DEFAULT, recommended):
      Ask the LLM to freely pick among all 13 THEME definitions.
      Then, ask it to freely pick among only THAT THEME's ~4 MAIN definitions.
      Then, ask it to freely pick among only THAT MAIN's ~6 SUBGROUP definitions.
      IMPORTANT: this narrowing comes from the typology's own nested structure
      (a subgroup only ever belongs to one main, which belongs to one theme) —
      it is NOT retrieval, NOT embeddings, NOT a kNN shortlist. The LLM is
      still choosing freely and only sees definitions, never other proverbs.

  --mode flat-subgroup:
      Hands the LLM ALL ~316 subgroup definitions in a single prompt, cold.
      This is the closest reproduction of the original "0.095" setup. Expect
      this to do WORSE than cascade and to cost far more tokens — it exists
      so you can measure the ceiling of "definitions alone, no narrowing" and
      compare it honestly against the cascade version.

WHY NO CROSS-VALIDATION IS NEEDED (same reasoning as definition_anchored.py)
--------------------------------------------------------------------------
This method never uses other example proverbs as references — only the fixed,
human-written definitions. There is no training and nothing to leak, so every
example proverb can be evaluated directly. No CV, no train/test split.

HONEST LIMITS (read before trusting any number this prints)
--------------------------------------------------------------------------
- I cannot run Ollama in my environment, so this has only been checked for
  LOGIC using --mock (random valid-code picks). The real accuracy can ONLY
  come from running this on your machine with Ollama up and a model pulled.
  A --mock number is meaningless; ignore it.
- Needs ONLY mk/definitions.csv and mk/ref_metadata.csv. It does NOT need
  mk/npy/ref_embeddings.npy — this method never touches embeddings, which is
  the whole point of it being a different method from 02/03/definition_anchored.
- The LLM can hallucinate a code that doesn't exist in the definitions list
  (e.g. invent "A9" when only A1-A4 were shown). Those are counted as
  "invalid", not silently matched to anything — check the invalid rate before
  trusting the accuracy number, since a high invalid rate means the LLM isn't
  reliably following the closed-list instruction, not that it's "guessing wrong".

INPUT
    mk/definitions.csv     level,code,text   (theme / main / subgroup rows)
    mk/ref_metadata.csv    source_type,theme,main_class,subgroup_code,kuusi_id,text
OUTPUT
    mk/results/pure_llm_predictions.csv
    mk/results/pure_llm_summary.csv

RUN (real, needs Ollama running + a pulled model):
    python mk/pure_llm_definition_classify.py --model qwen2.5:7b --sample 200 --mode cascade
RUN (logic test only, no server, RANDOM valid picks -- number is meaningless):
    python mk/pure_llm_definition_classify.py --mock --sample 20
REQUIRES  pip install pandas numpy requests
==============================================================================
"""

import os                                              # file paths
import re                                              # salvage a code from messy LLM text
import random                                          # reproducible mock picks + sampling
import argparse                                        # command-line flags
import numpy as np                                     # only used for sampling, not embeddings
import pandas as pd                                    # tables / CSV
import requests                                        # talk to local Ollama server

SEED = 42                                              # fixed seed = reproducible runs
HERE = os.path.dirname(os.path.abspath(__file__))      # .../mk  (put this file directly in mk/)
MK = HERE
META = os.path.join(MK, "ref_metadata.csv")           # example proverbs + gold labels
DEFS_CSV = os.path.join(MK, "definitions.csv")        # code -> definition text, all levels
RESULTS_DIR = os.path.join(MK, "results")             # where we write outputs
OLLAMA_URL = "http://localhost:11434/api/generate"     # local Ollama endpoint
TEMPERATURE = 0                                        # deterministic, matches rest of project


# =============================================================================
# LOAD DATA — only definitions + example proverbs. NO embeddings needed here.
# =============================================================================
def load_examples():
    meta = pd.read_csv(META, encoding="utf-8-sig", dtype=str).fillna("")
    is_ex = (meta["source_type"] == "example").values
    ex = dict(
        theme=meta["theme"].values[is_ex],
        main=meta["main_class"].values[is_ex],
        sub=meta["subgroup_code"].values[is_ex],
        text=meta["text"].values[is_ex],               # CONFIRM this is proverb_en in your file
    )
    print(f"loaded {len(ex['text'])} example proverbs (no embeddings loaded — not needed)")
    return ex


def load_definitions():
    """Returns dict: level -> list of (code, text), e.g. defs['theme'] = [("A","..."), ...]"""
    d = pd.read_csv(DEFS_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    out = {"theme": [], "main": [], "subgroup": []}
    for _, r in d.iterrows():
        lvl = r["level"]
        if lvl in out:
            out[lvl].append((r["code"], r["text"]))
    for lvl in out:
        print(f"  {lvl}: {len(out[lvl])} definitions loaded")
    return out


# =============================================================================
# PROMPT CONSTRUCTION — the LLM sees ONLY the candidate definitions + the proverb
# =============================================================================
def build_prompt(proverb, candidates, level_name):
    """candidates: list of (code, definition_text) the LLM must choose ONE from."""
    listing = "\n".join(f"  {code}: {text}" for code, text in candidates)
    valid_codes = ", ".join(code for code, _ in candidates)
    return f"""You are classifying a proverb into the Matti Kuusi proverb typology.

Below are the possible {level_name} categories and their definitions:
{listing}

Proverb: "{proverb}"

Which {level_name} category does this proverb best fit? Choose exactly ONE code
from this list: {valid_codes}

Respond in EXACTLY this format and nothing else:
CODE: <the code you chose>"""


CODE_RE = re.compile(r"code\s*:\s*([A-Za-z0-9]+)", re.IGNORECASE)


def call_llm(prompt, model, mock, rng, valid_codes):
    """Returns a code string, or '' if the LLM's answer couldn't be parsed/validated."""
    if mock:
        # random VALID pick, for logic-testing only — never a real signal
        return rng.choice(valid_codes) if rng.random() > 0.1 else ""  # occasionally simulate a miss

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": TEMPERATURE},
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            print(f"  ollama error: {data['error']}")
            return ""
        text = data.get("response", "")
    except Exception as e:
        print(f"  request failed: {e}")
        return ""

    m = CODE_RE.search(text)
    if not m:
        return ""
    code = m.group(1)
    return code if code in valid_codes else ""   # reject hallucinated codes explicitly


# =============================================================================
# CLASSIFY ONE PROVERB — cascade mode or flat-subgroup mode
# =============================================================================
def classify_cascade(proverb, defs, model, mock, rng):
    theme_cands = defs["theme"]
    pred_theme = call_llm(build_prompt(proverb, theme_cands, "theme"),
                          model, mock, rng, [c for c, _ in theme_cands])
    if not pred_theme:
        return "", "", ""

    main_cands = [(c, t) for c, t in defs["main"] if c.startswith(pred_theme)]
    if not main_cands:
        return pred_theme, "", ""
    pred_main = call_llm(build_prompt(proverb, main_cands, "main"),
                         model, mock, rng, [c for c, _ in main_cands])
    if not pred_main:
        return pred_theme, "", ""

    sub_cands = [(c, t) for c, t in defs["subgroup"] if c.startswith(pred_main)]
    if not sub_cands:
        return pred_theme, pred_main, ""
    pred_sub = call_llm(build_prompt(proverb, sub_cands, "subgroup"),
                        model, mock, rng, [c for c, _ in sub_cands])
    return pred_theme, pred_main, pred_sub


def classify_flat_subgroup(proverb, defs, model, mock, rng):
    """Cold, all-316-at-once. Theme/main are DERIVED from the chosen subgroup code
    (A1a -> main A1 -> theme A), not predicted separately."""
    sub_cands = defs["subgroup"]
    pred_sub = call_llm(build_prompt(proverb, sub_cands, "subgroup (all categories, no narrowing)"),
                        model, mock, rng, [c for c, _ in sub_cands])
    if not pred_sub:
        return "", "", ""
    pred_main = pred_sub[:2]
    pred_theme = pred_sub[:1]
    return pred_theme, pred_main, pred_sub


# =============================================================================
# MAIN EVAL LOOP
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--sample", type=int, default=200, help="proverbs to score (0 = all)")
    ap.add_argument("--mode", choices=["cascade", "flat-subgroup"], default="cascade")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    ex = load_examples()
    defs = load_definitions()

    n = len(ex["text"])
    idx = list(range(n))
    if args.sample and args.sample < n:
        random.Random(SEED).shuffle(idx)
        idx = idx[:args.sample]

    print(f"\nEvaluating {len(idx)} of {n} proverbs | mode={args.mode} | "
          f"model={args.model} | mock={args.mock}\n")

    rows = []
    for i in idx:
        rng = random.Random(SEED + i)   # per-item RNG, keeps --mock reproducible
        proverb = ex["text"][i]
        if args.mode == "cascade":
            pt, pm, ps = classify_cascade(proverb, defs, args.model, args.mock, rng)
        else:
            pt, pm, ps = classify_flat_subgroup(proverb, defs, args.model, args.mock, rng)

        rows.append(dict(
            text=proverb,
            true_theme=ex["theme"][i], true_main=ex["main"][i], true_sub=ex["sub"][i],
            pred_theme=pt, pred_main=pm, pred_sub=ps,
            theme_ok=int(pt == ex["theme"][i]),
            main_ok=int(pm == ex["main"][i]),
            sub_ok=int(ps == ex["sub"][i]),
            theme_invalid=int(pt == ""), main_invalid=int(pm == ""), sub_invalid=int(ps == ""),
        ))
        if (len(rows)) % 25 == 0:
            print(f"  ...{len(rows)}/{len(idx)}")

    df = pd.DataFrame(rows)
    report(df, args.mode)


def report(df, mode):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n================ PURE LLM (DEFINITIONS ONLY) ACCURACY ================")
    for lvl in ["theme", "main", "sub"]:
        acc = df[f"{lvl}_ok"].mean()
        invalid = df[f"{lvl}_invalid"].mean()
        label = "subgroup" if lvl == "sub" else lvl
        print(f"  {label:10} acc={acc:.3f}   invalid/unparsed={invalid:.3f}")
    print("\nCheck invalid rate before trusting accuracy: a high invalid rate means the "
          "model isn't following the closed-list format, not that it's genuinely guessing wrong.")

    pred_path = os.path.join(RESULTS_DIR, "pure_llm_predictions.csv")
    df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summ = pd.DataFrame([
        {"level": "theme", "mode": mode, "acc": round(df["theme_ok"].mean(), 4),
         "invalid": round(df["theme_invalid"].mean(), 4)},
        {"level": "main", "mode": mode, "acc": round(df["main_ok"].mean(), 4),
         "invalid": round(df["main_invalid"].mean(), 4)},
        {"level": "subgroup", "mode": mode, "acc": round(df["sub_ok"].mean(), 4),
         "invalid": round(df["sub_invalid"].mean(), 4)},
    ])
    summ_path = os.path.join(RESULTS_DIR, "pure_llm_summary.csv")
    summ.to_csv(summ_path, index=False)
    print(f"\nsaved {pred_path}")
    print(f"saved {summ_path}")
    print("\nCompare this theme number against: kNN 0.332 (02), definition-anchored "
          "cascade (definition_anchored.py), and kNN-shortlist+LLM-pick (03).")


if __name__ == "__main__":
    main()