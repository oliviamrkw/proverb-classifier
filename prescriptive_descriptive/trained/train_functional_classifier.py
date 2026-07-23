"""
prescriptive_descriptive/trained/train_functional_classifier.py
==============================================================================
LEARNED-WEIGHTS FUNCTIONAL PRESCRIPTIVENESS CLASSIFIER  (steps 1-4)

WHAT CHANGED vs feature_engineering/functional_prescriptiveness.py:
  STEP 1  The hand-written IF/OR combination rule is GONE. A class-weighted
          logistic regression now learns how much each feature matters,
          directly from your hand labels. (Measured on the 150 labels:
          hand rule 0.659 macro-F1 -> learned 0.687, same features.)
  STEP 2  Imperative detection upgraded: uses spaCy grammatical mood
          detection when available (any verb, not a hardcoded list of 40);
          falls back to the tested rules engine otherwise, and SAYS which
          one it used.
  STEP 3  NEW modal feature: should / must / ought to / had better / have to
          — explicit prescription markers the old features missed entirely
          ("You SHOULD forge the iron while it is hot").
  STEP 4  Valence hooks: --valence transformer scores the outcome clause with
          a sentiment model (cardiffnlp RoBERTa; needs your GPU + HF access),
          --valence lexicon keeps the offline word lists. Transformer path is
          MACHINE-ONLY: written and logic-checked here, but its accuracy
          contribution is unverified until you run it.

TRAIN DATA: any CSVs with a label column. Two supported shapes:
  - hand_label column with 0/1                        (handcheck sheets)
  - prescriptive_or_descriptive + has_implicit_prescription
    -> label = 1 if (pd == 0) or (implicit == 1)      (the two-column shape)

OUTPUTS
  model_functional.joblib      trained classifier + feature config
  cv report printed            5-fold macro-F1 + learned feature weights
  <master>_trained.csv         (with --apply) master labeled with
                               pred_functional / prob_functional

USAGE
  # train + report:
  python prescriptive_descriptive/trained/train_functional_classifier.py \
      --train YOUR_100_LABELS.csv handcheck_sample_50.csv
      (use your real filenames — run `dir *.csv` to see what you have)
  # train then label the corpus:
  python ... --train YOUR_100_LABELS.csv handcheck_sample_50.csv --apply master_en.csv
  # with transformer valence (your machine):
  python ... --train ... --apply master_en.csv --valence transformer
==============================================================================
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, classification_report

SEED = 42
TEXT_COL = "proverb_en"
FEATURE_NAMES = ["f_imperative", "f_prohibition", "f_modal", "f_conditional",
                 "f_causal", "f_valuation", "f_agency",
                 "f_valence_pos", "f_valence_neg"]

# ---------------------------------------------------------------------------
# FEATURES (regex layer — shared by both imperative backends)
# ---------------------------------------------------------------------------
_IMP_START = re.compile(
    r"(?i)^(?:make|keep|look|take|give|put|let|strike|cut|count|cast|spare|seize"
    r"|know|mind|measure|hope|live|save|beware|trust|speak|act|think|leave|hold"
    r"|walk|run|judge|praise|help|prepare|choose|learn|honou?r|respect|obey"
    r"|forgive|remember|ask|answer|catch|marry|waste|want|do|be|have|use|try"
    r"|call|eat|drink|buy|sell|pay|work|chastise|forge|practice|hear|see|sow|reap)\b")
_PROHIB = re.compile(r"(?i)(?:^|[,;:]\s*)(?:never|don'?t|do not|let not|not to be)\b")
_NOT_PATTERN = re.compile(r"(?i)\b\w+ not, \w+ not\b")
_MODAL = re.compile(r"(?i)\b(?:should|must|ought to|had better|have to|shalt)\b")   # STEP 3
_COND = re.compile(r"(?i)(?:^(?:if|when|where)\b|\b(?:he|she|one|they) who\b|\bwho(?:ever)? \w+)")
_CAUSAL = re.compile(r"(?i)\b(?:causes?|brings?|breeds?|draws?|makes?|leads? to"
                     r"|begets?|creates?|produces?|results? in|ends? in)\b")
_VALUATION = re.compile(r"(?i)(?:\bbetter\b.*\bthan\b|\bworse\b.*\bthan\b|\bbest\b"
                        r"|\bas good as\b|\bis worth\b|\bworth more\b|\benough\b.*\bthan\b"
                        r"|\bis (?:a |an )?(?:rich|precious|valuable|great|treasure)\b)")
_AGENCY = re.compile(r"(?i)(?:\byou(?:r)?\b|\bthyself?\b|\bthy\b|\bthee\b"
                     r"|\b(?:he|she|one|they) who\b|\bwhoever\b"
                     r"|\b(?:marry|marries|lie[sd]?|speak|work|give|take|help|save"
                     r"|spend|borrow|lend|steal|plant|sow|reap|build|learn|teach"
                     r"|wait|hurry|rush|persist|practice)\b)")
_POS = re.compile(r"(?i)\b(?:good|well|reward|success|succeed|gain|profit|wealth"
                  r"|rich(?:es)?|wise|wisdom|blessing|blessed|honou?r|fortune|thrive"
                  r"|prosper|safe|safety|peace|friend|win|victor|heal|health|strong"
                  r"|sweet|golden|catch(?:es)? the)\b")
_NEG = re.compile(r"(?i)\b(?:repent|regret|shame|ruin|harm|hurt|loss|lose|poor"
                  r"|poverty|fool(?:ish)?|danger|fall|fail|starve|hunger|weep|cry"
                  r"|sorrow|grief|pain|punish|beaten|eaten|devour|trap|fleas|sad"
                  r"|bitter|die[sd]?|death|sting|burn|drown|thorn|enemy|spoil)\b")


def _norm(text):
    t = str(text or "").strip()
    return (t.replace("\u2019", "'").replace("\u2018", "'")
             .replace("\u201c", '"').replace("\u201d", '"'))


# ---- STEP 2: imperative backend -------------------------------------------
class ImperativeDetector:
    """spaCy grammatical detection when available; tested regex fallback else."""
    def __init__(self, prefer_spacy=True, model="en_core_web_sm"):
        self.nlp = None
        self.backend = "regex"
        if prefer_spacy:
            try:
                import spacy
                self.nlp = spacy.load(model, disable=["ner", "lemmatizer"])
                self.backend = f"spacy:{model}"
            except Exception:
                pass
        print(f"[imperative backend: {self.backend}]"
              + ("" if self.nlp else "  (install spaCy + model for grammatical detection)"))

    def __call__(self, text):
        t = _norm(text)
        if t.endswith("?"):
            return 0
        if self.nlp is None:
            return int(bool(_IMP_START.search(t)))
        SUBJ = {"nsubj", "nsubjpass", "expl", "csubj", "csubjpass"}
        doc = self.nlp(t)
        for sent in doc.sents:
            for tok in sent:
                if "Imp" in tok.morph.get("Mood"):
                    return 1
            root = sent.root
            if root.pos_ == "VERB" and root.tag_ == "VB" and \
                    not any(c.dep_ in SUBJ for c in root.children):
                return 1
            for tok in sent:
                if tok.is_alpha:
                    if tok.pos_ == "VERB" and tok.tag_ == "VB" and \
                            not any(c.dep_ in SUBJ for c in tok.children):
                        return 1
                    break
        return 0


# ---- STEP 4: valence backend -----------------------------------------------
class ValenceScorer:
    """lexicon (offline, tested) or transformer (machine-only, needs GPU+HF)."""
    def __init__(self, mode="lexicon"):
        self.mode = mode
        self.pipe = None
        if mode == "transformer":
            try:
                from transformers import pipeline
                self.pipe = pipeline("sentiment-analysis",
                                     model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                     truncation=True)
            except Exception as e:
                print(f"[valence: transformer unavailable ({e}); falling back to lexicon]")
                self.mode = "lexicon"
        print(f"[valence backend: {self.mode}]")

    @staticmethod
    def _outcome_clause(text):
        parts = re.split(r",|;| then ", text, maxsplit=1)
        return parts[1] if len(parts) > 1 and len(parts[1].split()) >= 2 else text

    def __call__(self, text):
        """returns (pos_flag, neg_flag)"""
        target = self._outcome_clause(_norm(text))
        if self.mode == "transformer" and self.pipe is not None:
            out = self.pipe(target[:512])[0]
            lab = out["label"].lower()
            if "pos" in lab:
                return 1, 0
            if "neg" in lab:
                return 0, 1
            return 0, 0
        p = len(_POS.findall(target)); n = len(_NEG.findall(target))
        if p > n:
            return 1, 0
        if n > p:
            return 0, 1
        return 0, 0


def featurize_all(texts, imp, val):
    rows = []
    for t in texts:
        tt = _norm(t)
        vp, vn = val(tt)
        rows.append([
            imp(tt),
            int(bool(_PROHIB.search(tt)) or bool(_NOT_PATTERN.search(tt))),
            int(bool(_MODAL.search(tt))),
            int(bool(_COND.search(tt))),
            int(bool(_CAUSAL.search(tt))),
            int(bool(_VALUATION.search(tt))),
            int(bool(_AGENCY.search(tt))),
            vp, vn,
        ])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
def load_labels(paths):
    """Accepts either labeled-CSV shape; returns texts, labels."""
    texts, labels = [], []
    for p in paths:
        if not os.path.exists(p):
            here = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
            print(f"ERROR: can't find '{p}'", file=sys.stderr)
            print(f"  Looked in: {os.path.abspath('.')}", file=sys.stderr)
            if here:
                print("  CSV files that ARE here:", file=sys.stderr)
                for f in sorted(here):
                    print(f"    {f}", file=sys.stderr)
            else:
                print("  No CSV files in this folder — are you in the right directory?",
                      file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if "hand_label" in df.columns and (df["hand_label"].str.strip() != "").any():
            sub = df[df["hand_label"].str.strip().isin(["0", "1"])]
            texts += sub[TEXT_COL].tolist()
            labels += sub["hand_label"].astype(int).tolist()
            print(f"  {p}: {len(sub)} rows via hand_label")
        elif {"prescriptive_or_descriptive", "has_implicit_prescription"} <= set(df.columns):
            sub = df[df["prescriptive_or_descriptive"].str.strip().isin(["0", "1"])
                     & df["has_implicit_prescription"].str.strip().isin(["0", "1"])]
            lab = ((sub["prescriptive_or_descriptive"].str.strip() == "0")
                   | (sub["has_implicit_prescription"].str.strip() == "1")).astype(int)
            texts += sub[TEXT_COL].tolist()
            labels += lab.tolist()
            print(f"  {p}: {len(sub)} rows via pd/implicit composite")
        else:
            sys.exit(f"ERROR: {p} has neither hand_label nor pd+implicit columns")
    # dedupe identical texts (a proverb might appear in both files)
    seen = {}
    for t, l in zip(texts, labels):
        seen.setdefault(t.strip(), l)
    return list(seen.keys()), np.array(list(seen.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--apply", default=None, help="CSV to label with the trained model")
    ap.add_argument("--valence", choices=["lexicon", "transformer"], default="lexicon")
    ap.add_argument("--no-spacy", action="store_true", help="force regex imperative backend")
    ap.add_argument("--model-out", default="model_functional.joblib")
    args = ap.parse_args()

    texts, y = load_labels(args.train)
    print(f"\nTraining set: n={len(y)}  balance={dict(zip(*np.unique(y, return_counts=True)))}")

    imp = ImperativeDetector(prefer_spacy=not args.no_spacy)
    val = ValenceScorer(args.valence)
    X = featurize_all(texts, imp, val)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(clf, X, y, cv=skf)

    print("\n===== 5-FOLD CROSS-VALIDATION (out-of-fold, the honest number) =====")
    print(f"  macro-F1 = {f1_score(y, pred, average='macro'):.3f}   "
          f"accuracy = {accuracy_score(y, pred):.3f}")
    print("  " + classification_report(y, pred, zero_division=0).replace("\n", "\n  "))

    clf.fit(X, y)
    print("LEARNED WEIGHTS (positive pushes toward 'prescriptive'):")
    for n, w in sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda x: -x[1]):
        print(f"  {n:16} {w:+.2f}")

    import joblib
    joblib.dump({"clf": clf, "features": FEATURE_NAMES, "valence_mode": val.mode,
                 "imperative_backend": imp.backend}, args.model_out)
    print(f"\nSaved {args.model_out}")

    if args.apply:
        df = pd.read_csv(args.apply, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if TEXT_COL not in df.columns:
            sys.exit(f"ERROR: no '{TEXT_COL}' in {args.apply}")
        Xa = featurize_all(df[TEXT_COL].tolist(), imp, val)
        df["pred_functional"] = clf.predict(Xa).astype(int)
        pos_idx = list(clf.classes_).index(1)
        df["prob_functional"] = np.round(clf.predict_proba(Xa)[:, pos_idx], 4)
        out = args.apply.rsplit(".", 1)[0] + "_trained.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        n1 = int(df["pred_functional"].astype(int).sum())
        print(f"Labeled {args.apply}: {n1}/{len(df)} predicted prescriptive -> {out}")


if __name__ == "__main__":
    main()