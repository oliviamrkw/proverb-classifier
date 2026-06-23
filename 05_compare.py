"""
05_compare.py  --  aggregate Method A + all Method C runs into ONE table
========================================================================
Reads every per-row prediction CSV produced by 03 and 04, scores each on the
locked test set with identical metrics, and writes a single comparison table
for the poster. Recompute anytime without re-running any model.

Reads:
  embedding_classifier/method_a_predictions.csv     (Method A)
  llm_classifier/method_c_*.csv                      (each Method C run)
Only FULL-test-set runs are included; smaller files (pilots) are skipped.

Writes:
  comparison_table.csv   (project root)  + a printed table

Metrics per method: parse-failure rate, macro-F1 (headline), accuracy,
literal/figurative recall, and per-language macro-F1. Two dummy baselines
(most-frequent = accuracy floor, stratified = macro-F1 floor) are added as rows.

Run:  python 05_compare.py
Requires: pip install scikit-learn
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.dummy import DummyClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, "embedding_classifier")) else os.path.dirname(HERE)
EMB_DIR = os.path.join(ROOT, "embedding_classifier")
LLM_DIR = os.path.join(ROOT, "llm_classifier")
TEST = os.path.join(EMB_DIR, "test_indices.npy")
SEED = 42


def metrics_row(name, df, n_expected):
    """df has columns label, pred (pred may be empty = parse failure), language."""
    n_total = len(df)
    parsed = df[df["pred"].astype(str).str.strip().ne("") & df["pred"].notna()].copy()
    if len(parsed) == 0:
        return None
    y = parsed["label"].astype(int).values
    p = parsed["pred"].astype(float).astype(int).values
    row = {
        "method": name,
        "n": n_total,
        "parse_fail_%": round((n_total - len(parsed)) / n_total * 100, 1),
        "macroF1": round(f1_score(y, p, average="macro", zero_division=0), 3),
        "acc": round(accuracy_score(y, p), 3),
        "rec_literal": round(recall_score(y, p, pos_label=0, zero_division=0), 3),
        "rec_figur": round(recall_score(y, p, pos_label=1, zero_division=0), 3),
    }
    for lang in sorted(parsed["language"].unique()):
        m = (parsed["language"] == lang).values
        row[f"F1_{lang}"] = round(f1_score(y[m], p[m], average="macro", zero_division=0), 3)
    return row


def main():
    test_idx = np.load(TEST)
    n_expected = len(test_idx)
    print(f"locked test size: {n_expected}\n")

    files = []
    pa = os.path.join(EMB_DIR, "method_a_predictions.csv")
    if os.path.exists(pa):
        files.append(pa)
    files += sorted(glob.glob(os.path.join(LLM_DIR, "method_c_*.csv")))

    rows = []
    gold_ref = None
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
        if len(df) != n_expected:
            print(f"skip (not full test set, {len(df)} rows): {os.path.basename(f)}")
            continue
        df["label"] = df["label"].astype(int)
        name = df["method"].iloc[0] if "method" in df.columns and df["method"].iloc[0] else os.path.basename(f)
        r = metrics_row(name, df, n_expected)
        if r:
            rows.append(r)
        if gold_ref is None:
            gold_ref = df[["language", "label"]].copy()

    if gold_ref is None:
        print("No full-test prediction files found. Run 03 and 04 (--limit 0) first.")
        return

    # baselines on the full test gold
    y = gold_ref["label"].values
    for strat, label in [("most_frequent", "DUMMY most-frequent"),
                         ("stratified", "DUMMY stratified")]:
        d = DummyClassifier(strategy=strat, random_state=SEED).fit(
            np.zeros((len(y), 1)), y)
        p = d.predict(np.zeros((len(y), 1)))
        b = {"method": label, "n": len(y), "parse_fail_%": 0.0,
             "macroF1": round(f1_score(y, p, average="macro", zero_division=0), 3),
             "acc": round(accuracy_score(y, p), 3),
             "rec_literal": round(recall_score(y, p, pos_label=0, zero_division=0), 3),
             "rec_figur": round(recall_score(y, p, pos_label=1, zero_division=0), 3)}
        for lang in sorted(gold_ref["language"].unique()):
            m = (gold_ref["language"] == lang).values
            b[f"F1_{lang}"] = round(f1_score(y[m], p[m], average="macro", zero_division=0), 3)
        rows.append(b)

    table = pd.DataFrame(rows).sort_values("macroF1", ascending=False)
    out = os.path.join(ROOT, "comparison_table.csv")
    table.to_csv(out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(table.to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()