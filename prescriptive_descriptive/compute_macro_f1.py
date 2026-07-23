#!/usr/bin/env python3
"""
compute_macro_f1.py
--------------------
Adds macro-F1 (and per-class precision/recall) on top of the accuracy report,
using the PREDICTIONS ALREADY SAVED in llm_classified_proverbs_100.csv.
Does NOT call Ollama / does NOT rerun any model — just re-scores what's there.

WHY THIS MATTERS: on an imbalanced task (e.g. 83 descriptive / 17 prescriptive),
raw accuracy can look good just from guessing the majority class every time.
Macro-F1 weights both classes equally, so it exposes whether a model is
actually distinguishing the two classes or just riding the base rate. This is
the same metric your figurative/literal sub-project used (0.826 macro-F1),
so this is the right point of comparison for these two tasks too.

It auto-detects every model column in the file (form_code__X, implicit_code__X)
so it works whether you ran 1 model or 5, and includes 'ensemble' automatically.

Usage:
    python compute_macro_f1.py
    python compute_macro_f1.py --pred llm_classified_proverbs_100.csv
"""

import argparse
import csv
import re
import sys
from collections import defaultdict

PRED_FILE_DEFAULT = "llm_classified_proverbs_100.csv"


def f1_binary_macro(y_true, y_pred, labels=("0", "1")):
    """
    Macro-F1 for binary 0/1 labels, no sklearn dependency required.
    Returns (macro_f1, per_class_dict) where per_class_dict[label] = (P, R, F1, support).
    """
    per_class = {}
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        support = sum(1 for t in y_true if t == lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[lbl] = (prec, rec, f1, support)
    macro_f1 = sum(v[2] for v in per_class.values()) / len(labels)
    return macro_f1, per_class


def discover_models(fieldnames, prefix):
    """Find every X in '<prefix>__X' columns, preserving first-seen order."""
    pat = re.compile(rf"^{re.escape(prefix)}__(.+)$")
    seen = []
    for fn in fieldnames:
        m = pat.match(fn)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def score_task(rows, human_key, code_prefix, models):
    """Returns {model: (macro_f1, per_class_dict, n_used, n_dropped_unknown)}"""
    out = {}
    for m in models:
        code_col = f"{code_prefix}__{m}"
        y_true, y_pred = [], []
        dropped = 0
        for r in rows:
            ht = r.get(human_key, "").strip()
            pc = r.get(code_col, "").strip()
            if ht not in ("0", "1"):
                continue  # not human-gradable (blank/tie) -> skip, same as accuracy report
            if pc not in ("0", "1"):
                dropped += 1  # model answered 'unknown' -> excluded, same convention as before
                continue
            y_true.append(ht)
            y_pred.append(pc)
        if not y_true:
            out[m] = (0.0, {}, 0, dropped)
            continue
        macro_f1, per_class = f1_binary_macro(y_true, y_pred)
        out[m] = (macro_f1, per_class, len(y_true), dropped)
    return out


def render(title, code_to_word, scores, models):
    lines = [title, f"(labels: 0='{code_to_word['0']}'  1='{code_to_word['1']}')", ""]
    header = f"{'model':<16} {'macro-F1':>9} {'n_used':>7} {'unknown':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for m in models + ["ensemble"]:
        if m not in scores:
            continue
        f1, per_class, n_used, dropped = scores[m]
        lines.append(f"{m:<16} {f1:>9.3f} {n_used:>7} {dropped:>8}")
    lines.append("")
    # per-class breakdown for the best model only, to keep this readable
    if scores:
        best_model = max(scores, key=lambda k: scores[k][0])
        f1, per_class, n_used, _ = scores[best_model]
        if per_class:
            lines.append(f"Per-class breakdown for best model ({best_model}):")
            for lbl in ("0", "1"):
                if lbl in per_class:
                    p, r, f, sup = per_class[lbl]
                    lines.append(f"  {code_to_word[lbl]:<14} precision={p:.3f}  recall={r:.3f}  "
                                 f"F1={f:.3f}  support={sup}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=PRED_FILE_DEFAULT)
    ap.add_argument("--form-labels", nargs=2, default=["prescriptive", "descriptive"],
                    metavar=("LABEL_FOR_0", "LABEL_FOR_1"),
                    help="What 0/1 mean for prescriptive_or_descriptive (confirm against your mapping).")
    ap.add_argument("--implicit-labels", nargs=2, default=["no", "yes"],
                    metavar=("LABEL_FOR_0", "LABEL_FOR_1"),
                    help="What 0/1 mean for has_implicit_prescription.")
    args = ap.parse_args()

    try:
        with open(args.pred, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        sys.exit(f"ERROR: {args.pred} not found. Run llm_classify_and_compare.py first "
                 f"(no need to rerun it now if it already exists).")

    if not rows:
        sys.exit(f"ERROR: {args.pred} is empty.")

    form_models = discover_models(rows[0].keys(), "form_code")
    implicit_models = discover_models(rows[0].keys(), "implicit_code")
    # 'ensemble' is discovered too, but we want it printed last -> keep separate
    form_models = [m for m in form_models if m != "ensemble"]
    implicit_models = [m for m in implicit_models if m != "ensemble"]

    print(f"Loaded {len(rows)} rows from {args.pred}")
    print(f"Models found: {form_models}  (+ ensemble)\n")

    form_scores = score_task(rows, "human_form_code", "form_code", form_models)
    implicit_scores = score_task(rows, "human_implicit_code", "implicit_code", implicit_models)

    form_code_map = {"0": args.form_labels[0], "1": args.form_labels[1]}
    implicit_code_map = {"0": args.implicit_labels[0], "1": args.implicit_labels[1]}

    lines = []
    lines += render("== PRESCRIPTIVE vs DESCRIPTIVE (macro-F1) ==", form_code_map,
                     form_scores, form_models)
    lines.append("")
    lines += render("== HAS IMPLICIT PRESCRIPTION (macro-F1) ==", implicit_code_map,
                     implicit_scores, implicit_models)
    lines.append("")
    lines.append("Reminder: macro-F1 weights both classes equally, unlike accuracy — "
                 "a model that just guesses the majority class every time scores near 0 "
                 "on the minority class's F1, dragging macro-F1 down even if accuracy looked high.")

    report = "\n".join(lines)
    print(report)

    with open("llm_macro_f1_metrics.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote llm_macro_f1_metrics.txt")


if __name__ == "__main__":
    main()