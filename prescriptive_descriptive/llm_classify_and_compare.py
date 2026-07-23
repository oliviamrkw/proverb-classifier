#!/usr/bin/env python3
"""
llm_classify_and_compare.py
---------------------------
1. Reads the verified human gold file (human_labelled_verified_proverbs_100.csv).
2. Asks several FREE local LLMs (via Ollama) to classify EACH proverb on TWO
   dimensions:
     (a) prescriptive_or_descriptive
     (b) has_implicit_prescription
3. Writes llm_classified_proverbs_100.csv (per-model guesses + ensemble majority
   for both dimensions).
4. Compares LLM guesses to the human labels (human = ground truth) and prints
   accuracy for both dimensions, per model + ensemble.

FREE / OFFLINE: everything runs locally through Ollama. No API keys, no cost.

-------------------------------------------------------------------------------
!!! CONFIRM THIS BEFORE TRUSTING THE ACCURACY NUMBERS !!!

  The code mappings below are UNCONFIRMED — inferred from row 1 of your CSV
  ("Lightly come, lightly gone." -> prescriptive_or_descriptive=1,
  has_implicit_prescription=0). That row is descriptive with no hidden advice,
  so the best guess is:
        prescriptive_or_descriptive:  1 = descriptive , 0 = prescriptive
        has_implicit_prescription:    1 = yes (has one), 0 = no
  If either is backwards, flip the two numbers in that CODE_TO_WORD dict below
  — nothing else changes. Tell (sanity check): a mapping that's flipped shows
  up as accuracy that's the WRONG way round (e.g. 0.2 instead of 0.8).
-------------------------------------------------------------------------------

MODEL CHOICES (why these):
  - qwen2.5:7b   -> your strongest LLM on the earlier fig/lit task; good default.
  - llama3.1:8b  -> strong instruction-following, different family = useful vote.
  - gemma2:9b    -> good reasoner; NOTE ~5.4GB in q4, tight on a 6GB GPU. If it
                    won't load, swap for a lighter one (see LIGHT_ALTERNATIVES).
  Three models with different training = a meaningful ensemble, and gives you a
  per-model comparison table for the paper.

  Pull them first:
     ollama pull qwen2.5:7b
     ollama pull llama3.1:8b
     ollama pull gemma2:9b

Usage:
     python llm_classify_and_compare.py
     python llm_classify_and_compare.py --mock      # no Ollama; tests plumbing
     python llm_classify_and_compare.py --models qwen2.5:7b mistral:7b
"""

import argparse
import csv
import re
import sys
from collections import Counter

# ---- CONFIG ------------------------------------------------------------------
MODELS = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]
LIGHT_ALTERNATIVES = ["gemma2:2b", "mistral:7b", "phi3.5", "qwen2.5:3b"]  # if 6GB is tight

# CONFIRM THESE (see banner above). code -> word.
FORM_CODE_TO_WORD = {"0": "prescriptive", "1": "descriptive"}
IMPLICIT_CODE_TO_WORD = {"0": "no", "1": "yes"}

FORM_WORD_TO_CODE = {v: k for k, v in FORM_CODE_TO_WORD.items()}
IMPLICIT_WORD_TO_CODE = {v: k for k, v in IMPLICIT_CODE_TO_WORD.items()}

FORM_COL = "prescriptive_or_descriptive"
IMPLICIT_COL = "has_implicit_prescription"
GOLD_FILE = "human_labelled_verified_proverbs_100.csv"
PRED_FILE = "llm_classified_proverbs_100.csv"
METRICS_FILE = "llm_vs_human_metrics.txt"

TEMPERATURE = 0  # deterministic, matches the rest of the project

# ---- PROMPT -------------------------------------------------------------------
# Two questions in one call (cheaper + lets the model reason about form before
# implicit-content, which mirrors how a human annotator would naturally think).
# Strict output format at the end makes parsing robust regardless of how much
# the model reasons beforehand.
PROMPT = """You are annotating a proverb for a linguistics dataset. Answer two \
questions about it, using these exact definitions:

1. FORM — is the proverb prescriptive or descriptive?
   - "prescriptive": it gives advice, a rule, or a command about what one \
SHOULD do. It tells the reader/listener how to act.
   - "descriptive": it states how the world or people ARE or BEHAVE, as an \
observation, without prescribing any action.
   Judge only the literal surface phrasing — is it grammatically framed as an \
instruction/recommendation, or as a statement of fact/observation?

2. IMPLICIT PRESCRIPTION — does the proverb carry an implicit prescription?
   - This means: even though the proverb is not explicitly phrased as advice \
(i.e. even if you classified it as "descriptive" above), there is an \
UNDERLYING piece of advice or a lesson about what one should do that a \
listener would reasonably infer from it.
   - A proverb classified as "prescriptive" in Q1 is already explicit, so it \
normally does NOT also carry a separate hidden prescription — answer "no" for \
those unless there's a genuinely distinct hidden lesson beyond the stated one.
   - A "descriptive" proverb may or may not carry an implicit prescription: \
some are pure observations with no actionable lesson (answer "no"), others \
clearly imply "therefore, you should/shouldn't ..." even though that's never \
said (answer "yes").

Worked examples:
  Proverb: "Look before you leap."
  FORM: prescriptive        (explicit command)
  IMPLICIT: no              (already explicit, nothing further hidden)

  Proverb: "The early bird catches the worm."
  FORM: descriptive         (states an observed outcome, no command)
  IMPLICIT: yes             (clearly implies: therefore, act/arrive early)

  Proverb: "Lightly come, lightly gone."
  FORM: descriptive         (states a general pattern about easy gains)
  IMPLICIT: no              (a comment on how things tend to go, not a \
recommendation to act any particular way)

  Proverb: "Out of the abundance of the heart the mouth speaketh."
  FORM: descriptive         (states how speech reflects inner feeling)
  IMPLICIT: no              (pure observation about human nature, no \
actionable "you should" follows from it)

Now classify this proverb:
Proverb: "{proverb}"

Respond in EXACTLY this two-line format and nothing else:
FORM: <prescriptive or descriptive>
IMPLICIT: <yes or no>"""

FORM_RE = re.compile(r"form\s*:\s*(prescriptive|descriptive)", re.IGNORECASE)
IMPLICIT_RE = re.compile(r"implicit\s*:\s*(yes|no)", re.IGNORECASE)


# ---- LLM call (isolated so it can be mocked / swapped) ------------------------
def classify_one(model, proverb, mock=False):
    """Return dict: {'form': 'prescriptive'|'descriptive'|'unknown',
                     'implicit': 'yes'|'no'|'unknown'}"""
    if mock:
        # deterministic pseudo-answers for pipeline testing only
        s = sum(map(ord, proverb))
        return {
            "form": "prescriptive" if s % 2 == 0 else "descriptive",
            "implicit": "yes" if s % 3 == 0 else "no",
        }

    import ollama  # imported lazily so --mock works with no ollama installed
    for attempt in range(2):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": PROMPT.format(proverb=proverb)}],
                options={"temperature": TEMPERATURE},
            )
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "no such model" in msg or "pull" in msg:
                sys.exit(f"ERROR: model '{model}' not available. Run:  ollama pull {model}")
            if attempt == 0:
                continue
            sys.exit(f"ERROR calling Ollama for '{model}': {e}\n"
                     f"Is Ollama running? (start it, or run `ollama serve`)")

        text = resp.get("message", {}).get("content") or ""
        fm = FORM_RE.search(text)
        im = IMPLICIT_RE.search(text)
        form = fm.group(1).lower() if fm else None
        implicit = im.group(1).lower() if im else None

        if form and implicit:
            return {"form": form, "implicit": implicit}
        # partial/garbled response -> retry once, then give up on the missing bit(s)
        if attempt == 1:
            return {"form": form or "unknown", "implicit": implicit or "unknown"}
    return {"form": "unknown", "implicit": "unknown"}


# ---- IO ----------------------------------------------------------------------
def read_gold(path):
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        sys.exit(f"ERROR: {path} not found. Run majority_vote.py first.")
    keep = []
    for r in rows:
        keep.append({
            "row_id": r.get("row_id", ""),
            "id": r.get("id", ""),
            "language": r.get("language", ""),
            "proverb_en": (r.get("proverb_en") or "").strip(),
            "human_form_code": (r.get(FORM_COL) or "").strip(),        # '' if tie/blank
            "human_implicit_code": (r.get(IMPLICIT_COL) or "").strip(),  # '' if tie/blank
        })
    return keep


def ensemble_vote(values, valid_values):
    """Majority across model answers; ignores 'unknown'. Tie/none -> 'unknown'."""
    valid = [v for v in values if v in valid_values]
    if not valid:
        return "unknown"
    c = Counter(valid)
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "unknown"  # tie
    return top[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=GOLD_FILE)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--mock", action="store_true",
                    help="Skip Ollama; use fake answers to test the pipeline.")
    ap.add_argument("--limit", type=int, default=0, help="Only first N proverbs (debug).")
    args = ap.parse_args()

    rows = read_gold(args.gold)
    if args.limit:
        rows = rows[:args.limit]

    print(f"Classifying {len(rows)} proverbs (FORM + IMPLICIT) with: "
          f"{', '.join(args.models)}" + ("   [MOCK MODE]" if args.mock else ""))

    # --- run each model over all proverbs -------------------------------------
    for m in args.models:
        print(f"  -> {m} ...", flush=True)
        for r in rows:
            result = classify_one(m, r["proverb_en"], mock=args.mock)
            r[f"form_pred__{m}"] = result["form"]
            r[f"form_code__{m}"] = FORM_WORD_TO_CODE.get(result["form"], "")
            r[f"implicit_pred__{m}"] = result["implicit"]
            r[f"implicit_code__{m}"] = IMPLICIT_WORD_TO_CODE.get(result["implicit"], "")

    # --- ensemble -------------------------------------------------------------
    for r in rows:
        form_words = [r[f"form_pred__{m}"] for m in args.models]
        implicit_words = [r[f"implicit_pred__{m}"] for m in args.models]

        form_ens = ensemble_vote(form_words, {"prescriptive", "descriptive"})
        implicit_ens = ensemble_vote(implicit_words, {"yes", "no"})

        r["form_pred__ensemble"] = form_ens
        r["form_code__ensemble"] = FORM_WORD_TO_CODE.get(form_ens, "")
        r["implicit_pred__ensemble"] = implicit_ens
        r["implicit_code__ensemble"] = IMPLICIT_WORD_TO_CODE.get(implicit_ens, "")

    # --- correctness flags (for spot-checking in the CSV) ----------------------
    for r in rows:
        form_gradable = r["human_form_code"] in ("0", "1")
        r["form_ensemble_correct"] = (
            "" if not form_gradable or r["form_code__ensemble"] == "" else
            ("1" if r["form_code__ensemble"] == r["human_form_code"] else "0")
        )
        imp_gradable = r["human_implicit_code"] in ("0", "1")
        r["implicit_ensemble_correct"] = (
            "" if not imp_gradable or r["implicit_code__ensemble"] == "" else
            ("1" if r["implicit_code__ensemble"] == r["human_implicit_code"] else "0")
        )

    # --- write predictions csv ------------------------------------------------
    fields = ["row_id", "id", "language", "proverb_en",
              "human_form_code", "human_implicit_code"]
    for m in args.models:
        fields += [f"form_pred__{m}", f"form_code__{m}",
                   f"implicit_pred__{m}", f"implicit_code__{m}"]
    fields += ["form_pred__ensemble", "form_code__ensemble", "form_ensemble_correct",
               "implicit_pred__ensemble", "implicit_code__ensemble", "implicit_ensemble_correct"]

    with open(PRED_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # --- compare to human gold (both dimensions) --------------------------------
    def score_dimension(rows, human_key, code_col_fn):
        gradable = [r for r in rows if r[human_key] in ("0", "1")]
        out = {}
        for m in args.models + ["ensemble"]:
            code_col = code_col_fn(m)
            correct = unknown = 0
            for r in gradable:
                code = r[code_col]
                if code == "":
                    unknown += 1
                    continue
                if code == r[human_key]:
                    correct += 1
            answered = len(gradable) - unknown
            acc = correct / answered if answered else 0.0
            out[m] = (correct, acc, unknown, len(gradable))
        return out

    form_scores = score_dimension(rows, "human_form_code", lambda m: f"form_code__{m}")
    implicit_scores = score_dimension(rows, "human_implicit_code", lambda m: f"implicit_code__{m}")

    def render_table(title, scores, code_to_word):
        lines = [title, f"(code mapping used: {code_to_word} — confirm this is correct)", ""]
        header = f"{'model':<16} {'correct':>8} {'acc':>7} {'unknown':>8} {'gradable':>9}"
        lines.append(header)
        lines.append("-" * len(header))
        for m in args.models:
            c, a, u, g = scores[m]
            lines.append(f"{m:<16} {c:>8} {a:>7.3f} {u:>8} {g:>9}")
        c, a, u, g = scores["ensemble"]
        lines.append(f"{'ENSEMBLE':<16} {c:>8} {a:>7.3f} {u:>8} {g:>9}")
        return lines

    lines = []
    lines.append("LLM vs HUMAN (human = ground truth)")
    lines.append(f"Proverbs total: {len(rows)}")
    lines.append("")
    lines += render_table("== PRESCRIPTIVE vs DESCRIPTIVE ==", form_scores, FORM_CODE_TO_WORD)
    lines.append("")
    lines += render_table("== HAS IMPLICIT PRESCRIPTION ==", implicit_scores, IMPLICIT_CODE_TO_WORD)
    lines.append("")
    lines.append("Note: acc is over human-gradable proverbs the model actually answered "
                 "(unknowns excluded from the denominator; 'gradable' shows the base).")

    report = "\n".join(lines)
    print("\n" + report)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\nWrote {PRED_FILE} and {METRICS_FILE}")


if __name__ == "__main__":
    main()