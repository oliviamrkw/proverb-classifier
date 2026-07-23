"""
wisdom_extractor/apply_to_master.py
==============================================================================
APPLY THE (EXTENDED) WISDOM-EXTRACTOR CANONICALIZATION TO THE MASTER DATASET

Reads master.csv, canonicalizes every proverb_en, and writes a new master CSV
with 7 added columns (existing columns untouched):

    canonical_claim         the normalized proposition ("Better X than Y.")
    canon_rule_family       comparison|valuation|prohibition|conditional|
                            causal|imperative|none
    canon_rule_id           which specific rule fired (we_* = Wisdom Extractor
                            verbatim, ext_* = our extension) — audit trail
    canon_is_command        1 if imperative/prohibition template matched
    canon_has_valuation     1 if comparison/valuation template matched
    canon_has_cause_result  1 if conditional/causal template matched
    canon_implicit_hint     1 if any of the above (structural prescription
                            signal; 0 = NO TEMPLATE MATCHED, not "no
                            prescription" — see canonical_to_heuristic.py)

USAGE
    python wisdom_extractor/apply_to_master.py --master data/processed/master.csv
    python wisdom_extractor/apply_to_master.py --master master.csv --out custom_name.csv
    # validation mode: also compare canon_implicit_hint against a hand-labeled
    # column if present (e.g. on the 100-proverb gold file):
    python wisdom_extractor/apply_to_master.py --master olivia_label_100_proverbs_RELABELED_v2.csv --validate

ATTRIBUTION: canonicalization engine + we_* rules from Belciug & Pelican,
ConsILR-2025, github.com/ovladon/wisdom-extractor. Cite in the manuscript.
==============================================================================
"""
import os
import sys
import argparse
import pandas as pd

# allow running from repo root OR from inside wisdom_extractor/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonicalizer import canonicalize_with_family          # noqa: E402
from wisdom_extractor.canonical_to_heuristic import family_to_columns        # noqa: E402

TEXT_COL = "proverb_en"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", default=None,
                    help="default: <master>_canonicalized.csv next to the input")
    ap.add_argument("--validate", action="store_true",
                    help="if a has_implicit_prescription column exists, report "
                         "how the structural hint compares against it")
    args = ap.parse_args()

    df = pd.read_csv(args.master, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if TEXT_COL not in df.columns:
        sys.exit(f"ERROR: no '{TEXT_COL}' column in {args.master}")

    print(f"Loaded {len(df)} rows from {args.master}")

    # ---- canonicalize every proverb ----------------------------------------
    canon, fams, rids = [], [], []
    hint_cols = {"canon_is_command": [], "canon_has_valuation": [],
                 "canon_has_cause_result": [], "canon_implicit_hint": []}
    for text in df[TEXT_COL].fillna(""):
        c, fam, rid = canonicalize_with_family(text)
        canon.append(c); fams.append(fam); rids.append(rid)
        for k, v in family_to_columns(fam).items():
            hint_cols[k].append(v)

    df["canonical_claim"] = canon
    df["canon_rule_family"] = fams
    df["canon_rule_id"] = rids
    for k, v in hint_cols.items():
        df[k] = v

    # ---- coverage report (the honest number) --------------------------------
    n = len(df)
    matched = sum(1 for f in fams if f != "none")
    print(f"\nCOVERAGE: {matched}/{n} ({matched/n*100:.1f}%) matched a structural rule; "
          f"{n-matched} passed through unchanged (family='none').")
    fam_counts = pd.Series(fams).value_counts()
    for fam, cnt in fam_counts.items():
        print(f"  {fam:12} {cnt}")
    we_hits = sum(1 for r in rids if r.startswith("we_"))
    ext_hits = sum(1 for r in rids if r.startswith("ext_"))
    print(f"  (rule provenance: {we_hits} from Wisdom Extractor originals, "
          f"{ext_hits} from our extensions)")

    # ---- optional validation vs hand labels ---------------------------------
    if args.validate and "has_implicit_prescription" in df.columns:
        gold = df["has_implicit_prescription"].astype(str).str.strip()
        mask = gold.isin(["0", "1"])
        g = gold[mask].astype(int)
        h = pd.Series(hint_cols["canon_implicit_hint"])[mask.values].astype(int)
        tp = int(((h == 1) & (g == 1)).sum())
        fp = int(((h == 1) & (g == 0)).sum())
        fn = int(((h == 0) & (g == 1)).sum())
        tn = int(((h == 0) & (g == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"\nVALIDATION vs hand-labeled has_implicit_prescription (n={mask.sum()}):")
        print(f"  hint=1 & gold=1: {tp}   hint=1 & gold=0: {fp}")
        print(f"  hint=0 & gold=1: {fn}   hint=0 & gold=0: {tn}")
        print(f"  PRECISION of hint=1: {prec:.2f}   RECALL: {rec:.2f}")
        print("  (Low recall is EXPECTED — hint=0 means 'no template matched', "
              "not 'no prescription'. Judge this column by its precision.)")
    elif args.validate:
        print("\n--validate requested but no has_implicit_prescription column found; skipped.")

    # ---- write --------------------------------------------------------------
    out = args.out or args.master.rsplit(".", 1)[0] + "_canonicalized.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nWrote {out}  ({len(df)} rows, 7 new columns; originals untouched)")


if __name__ == "__main__":
    main()
