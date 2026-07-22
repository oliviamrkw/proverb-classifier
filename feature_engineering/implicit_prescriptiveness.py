"""
feature_engineering/implicit_prescriptiveness.py
==============================================================================
IMPLICIT PRESCRIPTIVENESS — feature heuristic + hand-check sample generator

TARGET (per our working definition, implicit and explicit merged):
    implicit_prescriptive = 1 when the proverb names a situation /
    diagnoses a condition and directs a person toward a wise response —
    whether stated outright or underlying.
    implicit_prescriptive = 0 when it explains how the world works
    without steering the reader toward any response.

DESIGN — 7 compact feature columns (regex/lexicon, fully offline, fast on 10k):
    f_imperative    command surface form ("Make good flour...", "Don't...")
                    [reuses the tested rules engine: P=1.00 R=0.88 on 14 gold]
    f_prohibition   negative command ("never", "don't", "let not")
    f_conditional   if/when/he-who structures (86% precision in canon test)
    f_causal        X causes/brings/draws/leads-to Y
    f_valuation     better-than / best / as-good-as / is-worth / is-a-precious
    f_agency        a person-controllable action is depicted (2nd person or
                    agentive "he who VERBs" or advice-verb presence)
    f_valence       pos/neg/neut of the OUTCOME clause via compact lexicons
                    (crude by design — a transformer would be better but
                    isn't needed for a first hand-check round)

SCORING (transparent, ordered by measured precision from prior validation):
    label=1 if imperative or prohibition          (command IS direction)
    label=1 if valuation                          (ranking IS direction)
    label=1 if (conditional or causal) and agency and valence != neut
                                                  (our consequential chain)
    label=1 if agency and valence != neut         (weaker consequential)
    else label=0
    confidence: HIGH if >=2 independent triggers or a command trigger;
                MED if exactly one non-command trigger; LOW for label=0 rows
                that still had any single feature fire (nearly missed).

OUTPUTS
    <master>_imppresc.csv   all rows + 7 feature cols + label + trigger + conf
    handcheck_sample_50.csv  stratified 50 rows for human verification, with
                             an empty `hand_label` column to fill in

HONEST LIMITS (read before trusting):
    - English-only by construction (run on master_en.csv).
    - f_valence is lexicon-based: it will miss subtle/ironic valence. Expect
      most errors to trace to it; that's the column to upgrade first
      (transformer sentiment) if hand-check accuracy lands short.
    - This is a HEURISTIC BASELINE to be hand-checked and then, with your
      verified labels, superseded by a trained classifier. It is not the
      final method; it is how we cheaply get gold labels + an interpretable
      baseline number.

USAGE
    python feature_engineering/implicit_prescriptiveness.py \
        --master master_en.csv --sample 50
==============================================================================
"""
import argparse
import random
import re
import sys

import pandas as pd

SEED = 42
TEXT_COL = "proverb_en"

# ---- feature regexes ---------------------------------------------------------
# imperative starters (from the tested add_imperative_flag rules engine)
_IMP_START = re.compile(
    r"(?i)^(?:make|keep|look|take|give|put|let|strike|cut|count|cast|spare|seize"
    r"|know|mind|measure|hope|live|save|beware|trust|speak|act|think|leave|hold"
    r"|walk|run|judge|praise|help|prepare|choose|learn|honou?r|respect|obey"
    r"|forgive|remember|ask|answer|catch|marry|waste|want|do|be|have|use|try"
    r"|call|eat|drink|buy|sell|pay|work|strike)\b")
_PROHIB = re.compile(r"(?i)(?:^|[,;:]\s*)(?:never|don'?t|do not|let not|not to be)\b")
_NOT_PATTERN = re.compile(r"(?i)\b\w+ not, \w+ not\b")   # "Waste not, want not"
_COND = re.compile(r"(?i)(?:^(?:if|when|where)\b|\b(?:he|she|one|they|who(?:ever)?) who\b|^who(?:ever)? \w+)")
_CAUSAL = re.compile(r"(?i)\b(?:causes?|brings?|breeds?|draws?|makes?|leads? to"
                     r"|begets?|creates?|produces?|results? in|ends? in)\b")
_VALUATION = re.compile(r"(?i)(?:\bbetter\b.*\bthan\b|\bworse\b.*\bthan\b|\bbest\b"
                        r"|\bas good as\b|\bis worth\b|\bworth more\b|\benough\b.*\bthan\b"
                        r"|\bis (?:a |an )?(?:rich|precious|valuable|great|treasure)\b)")
# agency: 2nd person address, or agentive he-who, or a controllable-action verb
_AGENCY = re.compile(r"(?i)(?:\byou(?:r)?\b|\bthyself?\b|\bthy\b|\bthee\b"
                     r"|\b(?:he|she|one|they) who\b|\bwhoever\b"
                     r"|\b(?:marry|marries|lie[sd]?|speak|work|give|take|help|save"
                     r"|spend|borrow|lend|steal|plant|sow|reap|build|learn|teach"
                     r"|wait|hurry|rush|persist|practice)\b)")

# outcome valence lexicons (compact; crude by design — see header)
_POS = re.compile(r"(?i)\b(?:good|well|reward|success|succeed|gain|profit|wealth"
                  r"|rich(?:es)?|wise|wisdom|blessing|blessed|honou?r|fortune|thrive"
                  r"|prosper|safe|safety|peace|friend|win|victor|heal|health|strong"
                  r"|sweet|golden|catch(?:es)? the)\b")
_NEG = re.compile(r"(?i)\b(?:repent|regret|shame|ruin|harm|hurt|loss|lose|poor"
                  r"|poverty|fool(?:ish)?|danger|fall|fail|starve|hunger|weep|cry"
                  r"|sorrow|grief|pain|punish|beaten|eaten|devour|trap|fleas|sad"
                  r"|bitter|die[sd]?|death|sting|burn|drown|thorn|enemy|spoil)\b")


def valence(text):
    """Valence of the OUTCOME: prefer the clause after a conditional split,
    else the whole text. Returns pos/neg/neut; ties -> neut (safety valve)."""
    parts = re.split(r",|;| then ", text, maxsplit=1)
    target = parts[1] if len(parts) > 1 and len(parts[1].split()) >= 2 else text
    p = len(_POS.findall(target)); n = len(_NEG.findall(target))
    if p > n:
        return "pos"
    if n > p:
        return "neg"
    return "neut"


def featurize(text):
    t = (text or "").strip()
    t = t.replace("\u2019", "'").replace("\u2018", "'")   # curly -> straight apostrophes
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    f = {
        "f_imperative": int(bool(_IMP_START.search(t)) and not t.endswith("?")),
        "f_prohibition": int(bool(_PROHIB.search(t)) or bool(_NOT_PATTERN.search(t))),
        "f_conditional": int(bool(_COND.search(t))),
        "f_causal": int(bool(_CAUSAL.search(t))),
        "f_valuation": int(bool(_VALUATION.search(t))),
        "f_agency": int(bool(_AGENCY.search(t))),
        "f_valence": valence(t),
    }
    return f


def score(f):
    """Returns (label, trigger, confidence) per the ordered rules in the header."""
    triggers = []
    if f["f_imperative"] or f["f_prohibition"]:
        triggers.append("command")
    if f["f_valuation"]:
        triggers.append("valuation")
    if f["f_conditional"] and f["f_agency"]:
        triggers.append("consequential")            # cond+agency: 86% measured precision, no valence gate
    elif f["f_causal"] and f["f_agency"] and f["f_valence"] != "neut":
        triggers.append("causal+valence")           # causal alone: 40% -> keep valence gate
    elif f["f_agency"] and f["f_valence"] != "neut":
        triggers.append("agency+valence")

    if triggers:
        conf = "HIGH" if ("command" in triggers or len(triggers) >= 2) else "MED"
        return 1, "+".join(triggers), conf
    # label 0 — LOW conf if anything fired at all (near miss), else HIGH-0
    any_fire = any(f[k] for k in ("f_imperative", "f_prohibition", "f_conditional",
                                   "f_causal", "f_valuation", "f_agency"))
    return 0, "", ("LOW" if any_fire else "HIGH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True, help="master_en.csv (English rows only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", type=int, default=50,
                    help="rows in the hand-check sample file (0 to skip)")
    args = ap.parse_args()

    df = pd.read_csv(args.master, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if TEXT_COL not in df.columns:
        sys.exit(f"ERROR: no '{TEXT_COL}' column in {args.master}")
    print(f"Loaded {len(df)} rows")

    feats, labels, trigs, confs = [], [], [], []
    for t in df[TEXT_COL].fillna(""):
        f = featurize(t)
        lab, trig, conf = score(f)
        feats.append(f); labels.append(lab); trigs.append(trig); confs.append(conf)

    fdf = pd.DataFrame(feats)
    for c in fdf.columns:
        df[c] = fdf[c].values
    df["implicit_prescriptive"] = labels
    df["ip_trigger"] = trigs
    df["ip_confidence"] = confs

    n1 = sum(labels)
    print(f"\nimplicit_prescriptive: {n1}/{len(df)} flagged 1 ({n1/len(df)*100:.1f}%)")
    print("trigger breakdown:", pd.Series([t for t in trigs if t]).value_counts().to_dict())
    print("confidence:", pd.Series(confs).value_counts().to_dict())

    out = args.out or args.master.rsplit(".", 1)[0] + "_imppresc.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {out}")

    # ---- stratified hand-check sample ---------------------------------------
    if args.sample:
        rng = random.Random(SEED)
        # stratify: half predicted-1 (spread across triggers), half predicted-0
        # (mix of HIGH-0 clear negatives and LOW-0 near-misses — the near-misses
        # are where the heuristic most likely erred, so oversample them)
        idx1 = [i for i, l in enumerate(labels) if l == 1]
        idx0_low = [i for i, (l, c) in enumerate(zip(labels, confs)) if l == 0 and c == "LOW"]
        idx0_high = [i for i, (l, c) in enumerate(zip(labels, confs)) if l == 0 and c == "HIGH"]
        rng.shuffle(idx1); rng.shuffle(idx0_low); rng.shuffle(idx0_high)
        k = args.sample
        take = idx1[:k // 2] + idx0_low[:k // 4] + idx0_high[:k - k // 2 - k // 4]
        take = take[:k]
        rng.shuffle(take)

        cols = ["id", "language", TEXT_COL, "implicit_prescriptive",
                "ip_trigger", "ip_confidence"]
        sample = df.iloc[take][cols].copy()
        sample["hand_label"] = ""          # you fill this in: 1 or 0
        sample["notes"] = ""               # optional: why, for disagreements
        sample.to_csv("handcheck_sample_%d.csv" % k, index=False, encoding="utf-8-sig")
        print(f"Wrote handcheck_sample_{k}.csv "
              f"({len(idx1[:k//2])} predicted-1, {len(idx0_low[:k//4])} near-miss-0, "
              f"rest clear-0). Fill `hand_label`, then we measure accuracy.")


if __name__ == "__main__":
    main()
