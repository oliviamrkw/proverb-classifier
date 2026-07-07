#!/usr/bin/env python3
"""
majority_vote.py
----------------
Merge N hand-labeled proverb CSVs (default: 3 reviewers) into ONE verified
gold file by majority vote, for BOTH label columns:
  - prescriptive_or_descriptive
  - has_implicit_prescription

Output: human_labelled_verified_proverbs_100.csv

Design notes (why it's built this way):
- Joins reviewers on the stable proverb `id` (e.g. gutenberg_06651), NOT row order.
- Verifies all files contain the SAME set of ids before doing anything
  (input-chain check). Exits loudly on mismatch instead of silently merging.
- Ties are NEVER silently broken. With 3 valid votes you always get a 2-1 or
  3-0 winner; but if a reviewer left a cell blank you can get a 1-1 tie on the
  remaining two. Those are flagged "TIE — needs adjudication" with the cell left
  blank, so you decide.
- Keeps a full audit trail (the raw votes + an agreement flag per field).

Usage (Windows / anywhere):
  python majority_vote.py --inputs kristin.csv reviewerB.csv reviewerC.csv
  # or just drop the 3 files in this folder, rename to match the defaults below,
  # and run:  python majority_vote.py
"""

import argparse
import csv
import sys
from collections import Counter, OrderedDict

# ---- Column names in the source CSVs -----------------------------------------
META_COLS = ["row_id", "id", "language", "proverb_native", "proverb_en"]
PD_COL = "prescriptive_or_descriptive"
IMP_COL = "has_implicit_prescription"
JOIN_KEY = "id"  # stable proverb id used to align reviewers

OUTPUT = "human_labelled_verified_proverbs_100.csv"


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
        default=["reviewer1.csv", "reviewer2.csv", "reviewer3.csv"],
        help="The hand-labeled reviewer CSVs (2 or more).",
    )
    ap.add_argument("--output", default=OUTPUT)
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

    # Row order follows the first file.
    ordered_ids = list(reviewers[0].keys())

    out_fields = (META_COLS
                  + [PD_COL, "pd_votes", "pd_agreement"]
                  + [IMP_COL, "imp_votes", "imp_agreement"])

    stats = {"pd": Counter(), "imp": Counter(), "text_mismatch": 0}
    out_rows = []

    for rid in ordered_ids:
        base_row = reviewers[0][rid]

        # integrity: warn if the proverb text differs across reviewers for same id
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

    # --- Summary (bottom-line-first) ------------------------------------------
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