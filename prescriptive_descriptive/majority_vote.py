#!/usr/bin/env python3
"""
majority_vote.py
----------------
Merge N hand-labeled proverb CSVs (default: 3 reviewers) into ONE verified
gold file by majority vote, for BOTH label columns:
  - prescriptive_or_descriptive
  - has_implicit_prescription

Output: human_labelled_verified_proverbs_100.csv

STEP 1 — DIAGNOSTIC (always runs first, before any merging):
  Computes pairwise raw agreement % between every pair of reviewers, for both
  label columns. This is what catches a reviewer using a REVERSED coding
  convention (e.g. 0=descriptive instead of 1=descriptive) before it silently
  corrupts your gold file. A reversed rater shows up as agreement far BELOW
  chance (well under ~35-40%) rather than normal annotator disagreement
  (which is usually 65-90%). The script prints a clear WARNING and does NOT
  proceed to merge if it detects this, unless you've applied --flip to fix it.

STEP 2 — FLIP (optional, once you've confirmed a reversal):
  --flip file.csv:column   inverts 0<->1 in that column of that file, in
  memory, before joining. Reusable for any future reviewer, not just one case.

STEP 3 — MERGE:
  Same majority-vote logic as before: joins on proverb `id`, requires all
  files to cover the same ids, never silently breaks a tie (flags it for you
  instead), keeps a full audit trail of raw votes + agreement per row.

Usage:
  # first pass: just diagnose, don't merge yet
  python majority_vote.py --inputs kristin.csv ishan.csv daphne.csv --diagnose-only

  # after confirming ishan's prescriptive_or_descriptive is reversed:
  python majority_vote.py --inputs kristin.csv ishan.csv daphne.csv \\
      --flip ishan.csv:prescriptive_or_descriptive
"""

import argparse
import csv
import sys
from collections import Counter, OrderedDict
from itertools import combinations

# ---- Column names in the source CSVs -----------------------------------------
META_COLS = ["row_id", "id", "language", "proverb_native", "proverb_en"]
PD_COL = "prescriptive_or_descriptive"
IMP_COL = "has_implicit_prescription"
JOIN_KEY = "id"  # stable proverb id used to align reviewers

OUTPUT = "human_labelled_verified_proverbs_100.csv"

# below this pairwise agreement %, warn that the pair might be using
# opposite coding conventions rather than genuinely disagreeing
REVERSAL_WARNING_THRESHOLD = 0.40


def read_labeled(path):
    """Read one reviewer CSV -> OrderedDict keyed by proverb id."""
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        sys.exit(f"ERROR: file not found: {path}")

    if not rows:
        sys.exit(f"ERROR: {path} is empty.")
    missing = [c for c in (META_COLS + [PD_COL, IMP_COL]) if c not in rows[0]]
    if missing:
        sys.exit(f"ERROR: {path} is missing columns: {missing}")

    d = OrderedDict()
    for r in rows:
        rid = (r.get(JOIN_KEY) or "").strip()
        if not rid:
            continue
        d[rid] = r
    return d


def apply_flips(reviewers, paths, flips):
    """flips: list of 'file.csv:column' strings. Inverts 0<->1 in place."""
    FLIP_MAP = {"0": "1", "1": "0"}
    for spec in flips:
        if ":" not in spec:
            sys.exit(f"ERROR: --flip expects file.csv:column, got '{spec}'")
        fname, col = spec.rsplit(":", 1)
        if col not in (PD_COL, IMP_COL):
            sys.exit(f"ERROR: --flip column must be '{PD_COL}' or '{IMP_COL}', got '{col}'")
        try:
            idx = [i for i, p in enumerate(paths) if p == fname or p.endswith("/" + fname)][0]
        except IndexError:
            sys.exit(f"ERROR: --flip target '{fname}' not found among --inputs {paths}")
        n_flipped = 0
        for row in reviewers[idx].values():
            v = row.get(col, "").strip()
            if v in FLIP_MAP:
                row[col] = FLIP_MAP[v]
                n_flipped += 1
        print(f"Flipped {n_flipped} values in {fname}:{col}")


def pairwise_agreement(reviewers, paths, col):
    """Returns list of (nameA, nameB, agree_pct, n_compared) for every reviewer pair."""
    out = []
    for (ia, ra), (ib, rb) in combinations(enumerate(reviewers), 2):
        common = set(ra) & set(rb)
        match = total = 0
        for rid in common:
            va = ra[rid].get(col, "").strip()
            vb = rb[rid].get(col, "").strip()
            if va in ("0", "1") and vb in ("0", "1"):
                total += 1
                if va == vb:
                    match += 1
        pct = (match / total) if total else None
        out.append((ia, ib, pct, total))
    return out


def run_diagnostic(reviewers, paths):
    print("=" * 70)
    print("STEP 1: PAIRWISE AGREEMENT DIAGNOSTIC (run before any merging)")
    print("=" * 70)
    any_warning = False
    for col, label in [(PD_COL, "prescriptive_or_descriptive"),
                        (IMP_COL, "has_implicit_prescription")]:
        print(f"\n[{label}]")
        for ia, ib, pct, n in pairwise_agreement(reviewers, paths, col):
            name_a, name_b = paths[ia], paths[ib]
            if pct is None:
                print(f"  {name_a} vs {name_b}: no comparable rows")
                continue
            flag = ""
            if pct < REVERSAL_WARNING_THRESHOLD:
                flag = "  <-- WARNING: looks REVERSED (opposite coding), not normal disagreement"
                any_warning = True
            print(f"  {name_a} vs {name_b}: {pct*100:5.1f}% agreement (n={n}){flag}")
    print()
    if any_warning:
        print("ACTION NEEDED: one or more pairs above look reversed, not just in disagreement.")
        print("Confirm with the reviewer, then re-run with e.g.:")
        print("  --flip reviewerfile.csv:prescriptive_or_descriptive")
    else:
        print("No reversal pattern detected. Agreement levels look like normal disagreement.")
    return any_warning


def majority(votes):
    """
    votes: list of raw cell strings (may include '' for blanks).
    returns (winner_value, agreement_flag, raw_votes_joined)
    """
    clean = [v.strip() for v in votes if v is not None and v.strip() != ""]
    raw = ";".join(clean) if clean else ""
    if not clean:
        return "", "insufficient (0 votes)", raw

    counts = Counter(clean)
    ranked = counts.most_common()
    best_val, best_n = ranked[0]
    tied = [v for v, n in ranked if n == best_n]

    if len(tied) > 1:
        return "", f"TIE {dict(counts)} — needs adjudication", raw

    n_valid = len(clean)
    if n_valid == 1:
        agree = "single vote only"
    elif best_n == n_valid:
        agree = "unanimous"
    else:
        agree = "majority"
    return best_val, agree, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--inputs", nargs="+",
        default=["olivia_label_100_proverbs.csv", "daphne_label_100_proverbs.csv", "ishan_label_100_proverbs.csv"],
        help="The hand-labeled reviewer CSVs (2 or more).",
    )
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--flip", nargs="*", default=[],
                    help="file.csv:column pairs to invert 0<->1 before merging.")
    ap.add_argument("--diagnose-only", action="store_true",
                    help="Run the agreement diagnostic and stop (no merge, no output file).")
    ap.add_argument("--force", action="store_true",
                    help="Merge even if the diagnostic finds an unresolved reversal warning.")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        sys.exit("ERROR: give at least 2 reviewer files to --inputs.")

    reviewers = [read_labeled(p) for p in args.inputs]

    # --- Input-chain check: all files must cover the same proverbs -------------
    id_sets = [set(r.keys()) for r in reviewers]
    base = id_sets[0]
    for path, s in zip(args.inputs[1:], id_sets[1:]):
        if s != base:
            only_base = sorted(base - s)[:10]
            only_other = sorted(s - base)[:10]
            print("ERROR: reviewer files do not cover the same proverb ids.",
                  file=sys.stderr)
            print(f"  In {args.inputs[0]} but not {path}: {only_base}", file=sys.stderr)
            print(f"  In {path} but not {args.inputs[0]}: {only_other}", file=sys.stderr)
            sys.exit(1)

    # --- Diagnostic runs BEFORE any flip, so you see the raw problem -----------
    had_warning = run_diagnostic(reviewers, args.inputs)

    if args.flip:
        print()
        apply_flips(reviewers, args.inputs, args.flip)
        print("\nRe-running diagnostic after flip:")
        had_warning = run_diagnostic(reviewers, args.inputs)

    if args.diagnose_only:
        return

    if had_warning and not args.force:
        print("\nSTOPPING before merge: unresolved reversal warning above.")
        print("Fix with --flip, or re-run with --force if you're sure this is genuine disagreement.")
        sys.exit(1)

    # --- Merge -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: MERGING (majority vote)")
    print("=" * 70)

    ordered_ids = list(reviewers[0].keys())
    out_fields = (META_COLS
                  + [PD_COL, "pd_votes", "pd_agreement"]
                  + [IMP_COL, "imp_votes", "imp_agreement"])

    stats = {"pd": Counter(), "imp": Counter(), "text_mismatch": 0}
    out_rows = []

    for rid in ordered_ids:
        base_row = reviewers[0][rid]

        texts = {(rev[rid].get("proverb_en") or "").strip() for rev in reviewers}
        if len(texts) > 1:
            stats["text_mismatch"] += 1

        pd_votes = [rev[rid].get(PD_COL, "") for rev in reviewers]
        imp_votes = [rev[rid].get(IMP_COL, "") for rev in reviewers]

        pd_val, pd_agree, pd_raw = majority(pd_votes)
        imp_val, imp_agree, imp_raw = majority(imp_votes)

        stats["pd"][pd_agree.split(" ")[0]] += 1
        stats["imp"][imp_agree.split(" ")[0]] += 1

        row = {c: base_row.get(c, "") for c in META_COLS}
        row.update({
            PD_COL: pd_val, "pd_votes": pd_raw, "pd_agreement": pd_agree,
            IMP_COL: imp_val, "imp_votes": imp_raw, "imp_agreement": imp_agree,
        })
        out_rows.append(row)

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    pd_ties = sum(1 for r in out_rows if r["pd_agreement"].startswith("TIE"))
    imp_ties = sum(1 for r in out_rows if r["imp_agreement"].startswith("TIE"))
    print(f"\nWrote {args.output}  ({n} proverbs)")
    print(f"  prescriptive/descriptive: {dict(stats['pd'])}  | ties needing you: {pd_ties}")
    print(f"  implicit prescription:    {dict(stats['imp'])} | ties needing you: {imp_ties}")
    if stats["text_mismatch"]:
        print(f"  WARNING: {stats['text_mismatch']} proverbs had different proverb_en "
              f"text across reviewers (same id). Check those.")
    if pd_ties or imp_ties:
        print("  -> Open the file, filter *_agreement for 'TIE', break those by hand.")


if __name__ == "__main__":
    main()