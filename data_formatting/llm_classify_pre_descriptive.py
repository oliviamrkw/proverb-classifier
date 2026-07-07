"""
llm_classify_pre_descriptive.py
===============================

STEP 2 of the prescriptive-vs-descriptive experiment.

WHAT THIS SCRIPT DOES
---------------------
Reads the 50 proverbs from random_proverbs_50.csv and asks one or more LLMs
to classify each one as "prescriptive" or "descriptive". Writes every model's
guess to pre_descriptive_llm_guess.csv. If you have already filled in the
'prescriptive_or_descriptive' column, it also prints each model's accuracy against your labels.

THE PROMPT
----------
System role: "You are a multilingual computational paremiology scholar."
Task: classify the proverb as prescriptive or descriptive, ONE word, no
explanation. (Prescriptive = tells you what to do / gives advice or a command;
descriptive = states how the world is.)

MODELS
------
Uses your existing local Ollama setup (same endpoint as 03_llm_cascade.py).
List the models you want with --models, e.g.:
    --models qwen2.5:7b llama3.1:8b aya:8b
Each model becomes its own column in the output so you can compare them.

OFFLINE TESTING
---------------
--mock returns a fake but deterministic label per proverb WITHOUT calling any
model, so you can verify the plumbing (reading, prompting, writing, scoring)
before spending GPU time. The mock labels are NOT real predictions.

INPUT:
  data_formatting/random_proverbs_50.csv
    needs at least: row_id, proverb_en   (prescriptive_or_descriptive optional, used for scoring)

OUTPUT:
  data_formatting/pre_descriptive_llm_guess.csv
    columns: row_id, proverb_en, prescriptive_or_descriptive(if present),
             <model>_guess  (one column per model),
             and if prescriptive_or_descriptive present: <model>_correct (1/0)

RUN:
  # offline plumbing check first:
  python data_formatting/llm_classify_pre_descriptive.py --mock

  # real run (needs `ollama serve` running and models pulled):
  python data_formatting/llm_classify_pre_descriptive.py --models qwen2.5:7b llama3.1:8b
"""

import argparse
import os
import re
import sys

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))          # .../data_formatting
IN_FILE = os.path.join(HERE, "random_proverbs_50.csv")
OUT_FILE = os.path.join(HERE, "pre_descriptive_llm_guess.csv")

OLLAMA_URL = "http://localhost:11434/api/generate"         # same as 03_llm_cascade.py

# The two allowed answers. Kept in one place so scoring + salvage stay consistent.
LABELS = ("prescriptive", "descriptive")

# The instruction sent to the model. We keep it tight: role, task, output rule.
SYSTEM_PROMPT = (
    "You are a multilingual computational paremiology scholar. "
    "Classify each phrase as either 'prescriptive' (it gives advice, a rule, "
    "or a command about what one should do) or 'descriptive' (it states how the "
    "world or people are, without prescribing action). "
    "Answer with exactly one word: prescriptive or descriptive. No explanation."
)

# The label vocabulary this whole script speaks. If you ever hand-label with
# 0/1 instead of words, this map converts them so the human column, the model
# guesses, and the scoring all line up. 0 = prescriptive, 1 = descriptive.
NUM_TO_WORD = {"0": "prescriptive", "1": "descriptive"}


def build_prompt(proverb):
    """Assemble the full text sent to the model for one proverb."""
    # Ollama's /api/generate takes a single prompt string, so we prepend the
    # system instruction rather than using a separate system field.
    return f"{SYSTEM_PROMPT}\n\nProverb: {proverb}\nAnswer:"


def salvage_label(raw):
    """Turn a model's raw text into 'prescriptive' / 'descriptive' / '' (unknown).

    Models sometimes add stray words despite instructions, so we search the
    text for either label rather than requiring an exact match. We also accept
    a bare 0/1 in case a model answers numerically (0=prescriptive,
    1=descriptive), so a numeric reply is never silently dropped.
    """
    low = raw.strip().lower()
    # direct word hits first
    for lab in LABELS:
        if lab in low:
            return lab
    # common shorthand the model might emit
    if re.search(r"\bpre\b", low):
        return "prescriptive"
    if re.search(r"\bdesc\b", low):
        return "descriptive"
    # numeric fallback: a lone 0 or 1 anywhere with no digit neighbours
    m = re.search(r"(?<!\d)([01])(?!\d)", low)
    if m:
        return NUM_TO_WORD[m.group(1)]
    return ""                                    # unknown -> flagged, never guessed


def ask_model(proverb, model):
    """Send one proverb to one Ollama model, return a cleaned label.

    Returns '' on any network/parse error (logged), so one bad call never
    crashes the whole run.
    """
    import requests                              # lazy import: --mock needs no requests
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": build_prompt(proverb),
            "stream": False,
            "options": {"temperature": 0.0},     # deterministic: same input -> same label
            "keep_alive": "30m",                 # keep the model warm between calls
        }, timeout=120)
        raw = resp.json().get("response", "")
    except Exception as e:
        print(f"  ! ollama error ({model}): {e}")
        return ""
    return salvage_label(raw)


def mock_label(proverb):
    """Deterministic fake label for offline plumbing tests.

    Rule: if the proverb contains an obvious command-ish cue, call it
    prescriptive, else descriptive. This is a STUB, not a real prediction --
    it only exists so you can test the file flow without a GPU.
    """
    cues = ("don't", "do not", "never", "always", "must", "should",
            "let ", "keep ", "make ", "take ")
    low = " " + proverb.lower() + " "
    return "prescriptive" if any(c in low for c in cues) else "descriptive"


def main():
    ap = argparse.ArgumentParser(
        description="Classify 50 proverbs as prescriptive/descriptive with LLM(s).")
    ap.add_argument("--models", nargs="+", default=["qwen2.5:7b"],
                    help="one or more Ollama model tags (default qwen2.5:7b)")
    ap.add_argument("--mock", action="store_true",
                    help="use offline stub labels instead of calling any model")
    args = ap.parse_args()

    # --- load the 50 proverbs ----------------------------------------------
    if not os.path.exists(IN_FILE):
        sys.exit(f"ERROR: {IN_FILE} not found.\n"
                 f"Run select_random_proverbs.py first.")
    df = pd.read_csv(IN_FILE, encoding="utf-8-sig", dtype=str).fillna("")
    if "proverb_en" not in df.columns:
        sys.exit("ERROR: input is missing the 'proverb_en' column.")
    print(f"[info] loaded {len(df)} proverbs from {os.path.basename(IN_FILE)}")

    # Is there a human gold column to score against?
    has_human = "prescriptive_or_descriptive" in df.columns and (df["prescriptive_or_descriptive"].str.len() > 0).any()
    if has_human:
        n_labeled = int((df["prescriptive_or_descriptive"].str.len() > 0).sum())
        print(f"[info] found {n_labeled} human labels -> will report accuracy")
    else:
        print("[info] no human labels yet -> will only write guesses "
              "(fill 'prescriptive_or_descriptive' to get accuracy)")

    # --- start building the output frame ------------------------------------
    out = pd.DataFrame({"row_id": df["row_id"], "proverb_en": df["proverb_en"]})
    if "prescriptive_or_descriptive" in df.columns:
        # Normalize the human column to the word vocabulary. This accepts BOTH
        # hand-labeling styles: words ('prescriptive'/'descriptive') OR numbers
        # (0/1). Numbers are mapped via NUM_TO_WORD so they compare correctly
        # against the models' word answers. Without this, human '1' would never
        # equal model 'descriptive' and every score would be 0.
        human = df["prescriptive_or_descriptive"].str.strip().str.lower()
        human = human.replace(NUM_TO_WORD)        # '0'->prescriptive, '1'->descriptive
        out["prescriptive_or_descriptive"] = human

    models = ["MOCK"] if args.mock else args.models

    # We collect correctness columns separately and append them AFTER all the
    # guess columns, so the final CSV reads: human label -> every model's guess
    # -> every model's correct flag (instead of interleaving guess/correct).
    correctness = {}          # model -> list of "1"/"0"/"" per row
    accuracies = {}           # model -> float accuracy (for the closing summary)

    # --- classify with each model ------------------------------------------
    for model in models:
        col = f"{model}_guess"
        print(f"\n[run] model = {model}")
        guesses = []
        for i, text in enumerate(df["proverb_en"], start=1):
            label = mock_label(text) if args.mock else ask_model(text, model)
            guesses.append(label)
            # light progress dot every 10 proverbs
            if i % 10 == 0:
                print(f"   {i}/{len(df)} done")
        out[col] = guesses

        # how many did the model refuse / return unparseable?
        n_unknown = sum(g == "" for g in guesses)
        if n_unknown:
            print(f"   [warn] {n_unknown} proverbs got no usable label from {model}")

        # --- score against human labels, if available ----------------------
        if has_human:
            gold = out["prescriptive_or_descriptive"]
            pred = pd.Series(guesses, index=out.index)
            # only score rows that have BOTH a human label and a model label
            scorable = (gold.str.len() > 0) & (pred.str.len() > 0)
            correct = (gold[scorable] == pred[scorable])
            # store as "1"/"0" strings: the output frame is string-typed
            # (dtype=str on load), so assigning ints would raise. CSV looks
            # identical either way. Held aside now, appended after all guesses.
            flags = pd.Series([""] * len(out), index=out.index)
            flags.loc[scorable] = correct.astype(int).astype(str)
            correctness[model] = flags
            acc = correct.mean() if scorable.any() else float("nan")
            accuracies[model] = acc
            print(f"   accuracy vs human ({int(scorable.sum())} scorable rows): "
                  f"{acc:.3f}")

    # --- append correctness columns AFTER all guess columns -----------------
    # This produces the requested order: ... human, <all guesses>, <all correct>.
    for model in models:
        if model in correctness:
            out[f"{model}_correct"] = correctness[model]

    # --- save ---------------------------------------------------------------
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {OUT_FILE}")

    # closing accuracy summary, sorted best-first, so you don't have to scroll
    if accuracies:
        print("\n=== accuracy summary (vs human labels) ===")
        for model, acc in sorted(accuracies.items(),
                                 key=lambda kv: (kv[1] != kv[1], -kv[1])):
            shown = "nan" if acc != acc else f"{acc:.3f}"   # acc!=acc catches NaN
            print(f"   {model:<16} {shown}")
    if not has_human:
        print("TIP: fill the 'prescriptive_or_descriptive' column in random_proverbs_50.csv, "
              "re-run, and you'll get per-model accuracy.")


if __name__ == "__main__":
    main()