#!/usr/bin/env python3
"""
feature_engineering/add_imperative_flag.py
------------------------------------------
Adds a `has_imperative` column (1/0) to the master proverb dataset, flagging
whether the proverb is phrased in the imperative mood ("Look before you leap",
"Don't count your chickens", "Waste not, want not").

METHOD (and why): imperative mood is a GRAMMATICAL property, so a parser detects
it far more reliably than a hand-written verb list and far more cheaply/
deterministically than an LLM. Determinism matters because this column feeds
XGBoost — the same proverb must always get the same flag.

Two engines:
  --engine spacy  (DEFAULT, recommended): spaCy morphology (Mood=Imp) + a parse
                  rule (subject-less base-form verb at clause start) + a small
                  negation/let prefix rule. Run on proverb_en (~99% English, so
                  no multilingual model needed).
  --engine rules  (fallback / offline smoke test): pure-Python regex + a short
                  imperative-starter list. NO dependencies, deterministic, but
                  APPROXIMATE — this is the "hardcoding" approach, kept only so
                  you can test the plumbing or run without spaCy. Not for the
                  final feature.

HONEST LIMITATION (state up front): short, context-free proverbs are the hard
case for any parser — a sentence-initial verb can be mistagged ("Time flies").
Expect ~80-90% precision, not perfect. Fine for a FEATURE (it just needs to
carry signal); don't treat it as ground truth. Sanity-check later against your
human prescriptive/descriptive labels — imperatives correlate strongly with
"prescriptive".

Setup for the real run (on your machine):
    pip install spacy pandas
    python -m spacy download en_core_web_sm        # or en_core_web_trf for best accuracy

Usage:
    # offline smoke test (no spaCy model needed):
    python feature_engineering/add_imperative_flag.py --master master.csv --engine rules

    # real run:
    python feature_engineering/add_imperative_flag.py --master master.csv --engine spacy
"""

import argparse
import re
import sys

import pandas as pd

# ---- CONFIG ------------------------------------------------------------------
TEXT_COL = "proverb_en"
NEW_COL = "has_imperative"
DEFAULT_MODEL = "en_core_web_sm"

# Clause-initial negation / hortative cues that reliably signal imperative mood.
# Applied at the start of the whole phrase AND right after , ; : (clause breaks).
# Used by BOTH engines (it's the one pattern spaCy is weakest on).
PREFIX_CUES = [
    r"don'?t\b", r"do not\b", r"never\b", r"let\b", r"let'?s\b",
]
_PREFIX_RE = re.compile(
    r"(?:^|[,;:.]\s*)\s*(?:" + "|".join(PREFIX_CUES) + r")",
    re.IGNORECASE,
)

# APPROXIMATE imperative-starter verbs, ONLY for the rules fallback engine.
# (Deliberately short; a real list is impossible — almost any verb can start an
# imperative. This is why spaCy is the recommended engine.)
_RULES_STARTERS = {
    "look", "make", "keep", "waste", "want", "strike", "cut", "count", "cast",
    "spare", "seize", "know", "mind", "measure", "hope", "live", "give", "take",
    "put", "catch", "save", "beware", "trust", "speak", "act", "think", "leave",
    "hold", "walk", "run", "fear", "love", "hate", "judge", "praise", "waste",
}
_FIRST_WORD_RE = re.compile(r"[a-zA-Z']+")


def prefix_flag(text):
    return bool(_PREFIX_RE.search(text or ""))


# ---- Engine: rules-only (offline, approximate) -------------------------------
def rules_flag(text):
    t = (text or "").strip()
    if not t:
        return 0
    if prefix_flag(t):
        return 1
    # "X not, Y not" imperative pattern (e.g. "Waste not, want not")
    if re.search(r"\bnot\b.*,.*\bnot\b", t, re.IGNORECASE):
        return 1
    m = _FIRST_WORD_RE.search(t)
    if m and m.group(0).lower() in _RULES_STARTERS:
        return 1
    return 0


# ---- Engine: spaCy (recommended, real) ---------------------------------------
def build_spacy(model_name):
    try:
        import spacy
    except ImportError:
        sys.exit("ERROR: pip install spacy   (or use --engine rules for an offline pass)")
    try:
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
    except OSError:
        sys.exit(
            f"ERROR: spaCy model '{model_name}' not installed.\n"
            f"  Run:  python -m spacy download {model_name}\n"
            f"  (or use --engine rules for a quick offline pass)"
        )
    return nlp


def spacy_flag(doc):
    """True if any evidence of imperative mood in the parsed doc."""
    SUBJECT_DEPS = {"nsubj", "nsubjpass", "expl", "csubj", "csubjpass"}
    for sent in doc.sents:
        # skip questions — not imperatives
        if sent.text.strip().endswith("?"):
            continue

        # 1) morphology: the model tagged an imperative verb
        for tok in sent:
            if "Imp" in tok.morph.get("Mood"):
                return True

        # 2) parse rule: clause root is a base-form verb with NO subject
        #    ("Look before you leap": root=Look, VB, no nsubj)
        root = sent.root
        if root.pos_ == "VERB" and root.tag_ == "VB":
            has_subject = any(c.dep_ in SUBJECT_DEPS for c in root.children)
            if not has_subject:
                return True

        # 3) also check the first verb of the sentence (handles some parses
        #    where root isn't the leading verb)
        for tok in sent:
            if tok.is_alpha:
                if tok.pos_ == "VERB" and tok.tag_ == "VB" and \
                        not any(c.dep_ in SUBJECT_DEPS for c in tok.children):
                    return True
                break  # only inspect the first real word

    return False


def run_spacy(texts, model_name):
    nlp = build_spacy(model_name)
    out = []
    # nlp.pipe is the fast batched path for large data
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        flag = 1 if (spacy_flag(doc) or prefix_flag(doc.text)) else 0
        out.append(flag)
        if (i + 1) % 5000 == 0:
            print(f"  ...{i + 1} rows", flush=True)
    return out


# ---- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True, help="master.csv to add the column to.")
    ap.add_argument("--out", default=None, help="Output path (default: overwrite in place with _imperative suffix).")
    ap.add_argument("--engine", choices=["spacy", "rules"], default="spacy")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    df = pd.read_csv(args.master, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if TEXT_COL not in df.columns:
        sys.exit(f"ERROR: master has no '{TEXT_COL}' column.")
    texts = df[TEXT_COL].fillna("").str.strip().tolist()
    print(f"Loaded {len(texts)} rows. Engine: {args.engine}"
          + (f" ({args.model})" if args.engine == "spacy" else " [approximate fallback]"))

    if args.engine == "spacy":
        flags = run_spacy(texts, args.model)
    else:
        flags = [rules_flag(t) for t in texts]

    df[NEW_COL] = flags   # appended; existing columns untouched (no blast radius)

    out = args.out or args.master.replace(".csv", "_imperative.csv")
    if out == args.master:
        out = args.master.rsplit(".", 1)[0] + "_imperative.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    n1 = sum(flags)
    print(f"\nFlagged {n1}/{len(flags)} ({n1/len(flags)*100:.1f}%) as imperative.")
    print(f"Wrote {out}  (new column: {NEW_COL})")
    if args.engine == "rules":
        print("NOTE: rules engine is approximate — rerun with --engine spacy for the real feature.")


if __name__ == "__main__":
    main()