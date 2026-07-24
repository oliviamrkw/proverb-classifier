"""
prescriptive_descriptive/trained/train_ip_classifier.py
==============================================================================
LEARNED-WEIGHTS IMPLICIT PRESCRIPTION (IP) CLASSIFIER

Predicts whether a proverb carries an implicit prescription — advice or a
steer toward a wise response, whether stated outright or only implied.

WHAT THIS DOES
  STEP 1  No hand-written IF/OR rule. A class-weighted logistic regression
          learns how much each feature matters from your hand labels.
  STEP 2  Imperative detection can use spaCy grammatical mood (any verb) or
          the regex word-list fallback.
  STEP 3  Modal feature: should / must / ought to / had better / have to.
  STEP 4  Valence can use a transformer sentiment model or offline word lists.

>>> ABLATION MODE (--ablate) <<<
  Runs ALL FOUR backend combinations on the same labels and prints one
  comparison table, so you can see which upgrade helped and which hurt.
  This is the right way to test them: changing two things at once tells you
  nothing about which one caused the change.

      combo                          what it isolates
      regex     + lexicon            the baseline (no upgrades)
      spacy     + lexicon            spaCy's effect ALONE
      regex     + transformer        the transformer's effect ALONE
      spacy     + transformer        both together

  IMPORTANT: with ~150 labels the measurement wobble is roughly +/- 7 points.
  Treat differences smaller than that as "can't tell", not as real gains.

TRAIN DATA: CSVs with either
  - a hand_label column of 0/1, or
  - prescriptive_or_descriptive + has_implicit_prescription
    -> IP label = 1 if (prescriptive_or_descriptive == 0) or (implicit == 1)

OUTPUTS
  model_implicit_prescription.joblib    trained model + which backends it used
  <input>_ip.csv                        (with --apply) adds pred_ip / prob_ip

USAGE
  # 1. TEST THE BACKENDS SEPARATELY (do this first):
  python prescriptive_descriptive/trained/train_ip_classifier.py \
      --train YOUR_100_LABELS.csv handcheck_sample_50.csv --ablate

  # 2. Then train with whichever combo won, e.g. spaCy on, lexicon valence:
  python ... --train YOUR_100_LABELS.csv handcheck_sample_50.csv \
      --valence lexicon --apply master_en.csv

  (run `dir *.csv` to see your real filenames)
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
# FEATURE PATTERNS
# ---------------------------------------------------------------------------
_IMP_START = re.compile(
    r"(?i)^(?:make|keep|look|take|give|put|let|strike|cut|count|cast|spare|seize"
    r"|know|mind|measure|hope|live|save|beware|trust|speak|act|think|leave|hold"
    r"|walk|run|judge|praise|help|prepare|choose|learn|honou?r|respect|obey"
    r"|forgive|remember|ask|answer|catch|marry|waste|want|do|be|have|use|try"
    r"|call|eat|drink|buy|sell|pay|work|chastise|forge|practice|hear|see|sow|reap)\b")
_PROHIB = re.compile(r"(?i)(?:^|[,;:]\s*)(?:never|don'?t|do not|let not|not to be)\b")
_NOT_PATTERN = re.compile(r"(?i)\b\w+ not, \w+ not\b")
_MODAL = re.compile(r"(?i)\b(?:should|must|ought to|had better|have to|shalt)\b")
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


# ---- imperative backends ---------------------------------------------------
class ImperativeDetector:
    def __init__(self, use_spacy=True, model="en_core_web_sm", quiet=False):
        self.nlp = None
        self.backend = "regex"
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(model, disable=["ner", "lemmatizer"])
                self.backend = "spacy"
            except Exception:
                if not quiet:
                    print("  [spaCy unavailable -> regex fallback; "
                          "pip install spacy && python -m spacy download en_core_web_sm]")
        if not quiet:
            print(f"[imperative backend: {self.backend}]")

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


# ---- valence backends ------------------------------------------------------
class ValenceScorer:
    def __init__(self, mode="lexicon", quiet=False):
        self.mode = mode
        self.pipe = None
        if mode == "transformer":
            try:
                from transformers import pipeline
                self.pipe = pipeline("sentiment-analysis",
                                     model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                     truncation=True)
            except Exception as e:
                if not quiet:
                    print(f"  [transformer unavailable ({type(e).__name__}) -> lexicon fallback]")
                self.mode = "lexicon"
        if not quiet:
            print(f"[valence backend: {self.mode}]")

    @staticmethod
    def _outcome_clause(text):
        parts = re.split(r",|;| then ", text, maxsplit=1)
        return parts[1] if len(parts) > 1 and len(parts[1].split()) >= 2 else text

    def __call__(self, text):
        target = self._outcome_clause(_norm(text))
        if self.mode == "transformer" and self.pipe is not None:
            lab = self.pipe(target[:512])[0]["label"].lower()
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


def cv_score(X, y):
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    pred = cross_val_predict(clf, X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return f1_score(y, pred, average="macro"), accuracy_score(y, pred), pred


# ---------------------------------------------------------------------------
def load_labels(paths):
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
                print("  No CSV files here — wrong directory?", file=sys.stderr)
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
            print(f"  {p}: {len(sub)} rows via prescriptive/implicit columns")
        else:
            sys.exit(f"ERROR: {p} has no usable label column")
    seen = {}
    for t, l in zip(texts, labels):
        seen.setdefault(t.strip(), l)
    return list(seen.keys()), np.array(list(seen.values()))


def run_ablation(texts, y):
    """Test each backend upgrade SEPARATELY. This is the point of the script."""
    print("\nBuilding backends (transformer load may take a minute the first time)...")
    imp_regex = ImperativeDetector(use_spacy=False, quiet=True)
    imp_spacy = ImperativeDetector(use_spacy=True, quiet=True)
    val_lex = ValenceScorer("lexicon", quiet=True)
    val_trans = ValenceScorer("transformer", quiet=True)

    spacy_ok = imp_spacy.backend == "spacy"
    trans_ok = val_trans.mode == "transformer"
    print(f"  spaCy available: {spacy_ok}   transformer available: {trans_ok}")
    if not spacy_ok or not trans_ok:
        print("  NOTE: an unavailable backend silently falls back, so its rows below")
        print("        will duplicate the baseline. Install it to get a real comparison.")

    combos = [
        ("regex   + lexicon    ", imp_regex, val_lex, "baseline, no upgrades"),
        ("spacy   + lexicon    ", imp_spacy, val_lex, "spaCy effect ALONE"),
        ("regex   + transformer", imp_regex, val_trans, "transformer effect ALONE"),
        ("spacy   + transformer", imp_spacy, val_trans, "both together"),
    ]

    print("\n" + "=" * 78)
    print("ABLATION — each upgrade tested separately (5-fold CV, n=%d)" % len(y))
    print("=" * 78)
    hdr = f"{'combo':<24} {'macro-F1':>9} {'accuracy':>9}   what it isolates"
    print(hdr); print("-" * len(hdr))

    results = []
    base = None
    for name, imp, val, note in combos:
        X = featurize_all(texts, imp, val)
        f1, acc, _ = cv_score(X, y)
        if base is None:
            base = f1
        delta = f1 - base
        dstr = "  (baseline)" if abs(delta) < 1e-9 else f"  ({delta:+.3f})"
        print(f"{name:<24} {f1:>9.3f} {acc:>9.3f}   {note}{dstr}")
        results.append((name.strip(), f1, acc, delta))

    print("\nHOW TO READ THIS:")
    print("  With ~150 labels the wobble is about +/- 7 points (0.07).")
    print("  A change smaller than that is NOT measurable — do not treat it as real.")
    best = max(results, key=lambda r: r[1])
    print(f"\n  Best combo here: {best[0]}  (macro-F1 {best[1]:.3f})")
    if abs(best[3]) < 0.07:
        print("  ...but it is within the noise band of the baseline, so the honest")
        print("     conclusion is 'no measurable difference between these options'.")
    pd.DataFrame(results, columns=["combo", "macro_f1", "accuracy", "delta_vs_baseline"]) \
        .to_csv("ip_ablation_results.csv", index=False)
    print("\nWrote ip_ablation_results.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--apply", default=None)
    ap.add_argument("--ablate", action="store_true",
                    help="test each backend upgrade separately and stop")
    ap.add_argument("--valence", choices=["lexicon", "transformer"], default="lexicon")
    ap.add_argument("--no-spacy", action="store_true")
    ap.add_argument("--model-out", default="model_implicit_prescription.joblib")
    args = ap.parse_args()

    texts, y = load_labels(args.train)
    print(f"\nTraining set: n={len(y)}  balance={dict(zip(*np.unique(y, return_counts=True)))}")

    if args.ablate:
        run_ablation(texts, y)
        return

    imp = ImperativeDetector(use_spacy=not args.no_spacy)
    val = ValenceScorer(args.valence)
    X = featurize_all(texts, imp, val)

    f1, acc, pred = cv_score(X, y)
    print("\n===== 5-FOLD CROSS-VALIDATION (out-of-fold) =====")
    print(f"  macro-F1 = {f1:.3f}   accuracy = {acc:.3f}")
    print("  " + classification_report(y, pred, zero_division=0).replace("\n", "\n  "))

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    clf.fit(X, y)
    print("LEARNED WEIGHTS (positive pushes toward 'has implicit prescription'):")
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
        df["pred_ip"] = clf.predict(Xa).astype(int)
        df["prob_ip"] = np.round(clf.predict_proba(Xa)[:, list(clf.classes_).index(1)], 4)
        out = args.apply.rsplit(".", 1)[0] + "_ip.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Labeled {args.apply}: {int(df['pred_ip'].sum())}/{len(df)} "
              f"predicted to have an implicit prescription -> {out}")


if __name__ == "__main__":
    main()