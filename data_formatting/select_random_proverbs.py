"""
select_random_proverbs.py
=========================

STEP 1 of the prescriptive-vs-descriptive experiment.

WHAT THIS SCRIPT DOES
---------------------
Picks 50 random proverbs from the master CSV and writes them to
random_proverbs_50.csv with an EMPTY 'human_label' column that YOU fill in
by hand (with "prescriptive" or "descriptive"). That hand-labeled file is
your gold standard -- the ground truth the LLM guesses get scored against in
step 2.

WHY A SEPARATE HUMAN FILE
-------------------------
Keeping your labels and the machine's labels in different files means the LLM
never sees your answer, and the comparison in step 2 stays honest.

TEXT SELECTION
--------------
Only proverbs with English translations (proverb_en) are selected. Proverbs
without English translations are skipped entirely. The English text shown to
a human / LLM is saved in a 'text_used' column so there is no ambiguity about
what was labeled.

INPUT:
  data/processed/master.csv   (columns include: id, language,
                               proverb_native, proverb_en, ...)

OUTPUT:
  data_formatting/random_proverbs_50.csv
    columns: row_id, id, language, proverb_native, proverb_en,
             text_used, human_label(empty)

RUN:
  python data_formatting/select_random_proverbs.py
  # optional flags:
  #   --master path/to/master.csv   (default data/processed/master.csv)
  #   --n 50                          how many to sample (default 50)
  #   --seed 42                       reproducible sample (default 42)
"""

import argparse
import os
import sys

import pandas as pd


# This file lives in data_formatting/. The default master path is relative to
# the project root, which is the parent of this folder.
HERE = os.path.dirname(os.path.abspath(__file__))          # .../data_formatting
ROOT = os.path.dirname(HERE)                               # project root
DEFAULT_MASTER = os.path.join(ROOT, "data", "processed", "master.csv")
OUT_FILE = os.path.join(HERE, "random_proverbs_50.csv")


# def pick_text(row):
#     """Return the English text if present, else the native text.

#     Mirrors the pipeline's convention so the human labels the SAME string the
#     LLM will later see.
#     """
#     en = str(row.get("proverb_en", "") or "").strip()
#     if en and en.lower() != "nan":
#         return en
#     return str(row.get("proverb_native", "") or "").strip()



def load_english_pool(master_path):
    """Load master.csv and return only rows that have a usable English text.

    Shared by both the initial sampler and add_proverbs so the "must have an
    English translation" rule is defined in exactly one place.
    """
    if not os.path.exists(master_path):
        sys.exit(f"ERROR: master not found at {master_path}\n"
                 f"Pass the correct path with --master.")
    master = pd.read_csv(master_path, encoding="utf-8-sig", dtype=str).fillna("")
    pool = master[master["proverb_en"].str.strip() != ""].copy()
    pool = pool[pool["proverb_en"].str.strip().str.lower() != "nan"]
    return pool


def add_proverbs(n_add, existing_file=OUT_FILE, master_path=DEFAULT_MASTER,
                 out_file=None, seed=42):
    """Add N NEW English-translation proverbs to an existing sample.

    WHAT IT DOES
    ------------
    - Reads the existing sample file. Every row already there -- INCLUDING any
      human labels you have entered -- is carried over completely unchanged.
    - Draws n_add new proverbs from master, considering ONLY rows that (a) have
      an English translation and (b) are not already in the existing sample
      (deduped by id, with text as a backup key). So no duplicates.
    - Writes the combined result (old + new) to a NEW file. The original file
      is never modified, so it stays as a safe backup.

    The new rows continue the row_id numbering (51, 52, ...) and get an EMPTY
    label column for you to fill in -- your existing labels are untouched.

    Parameters
    ----------
    n_add        : how many new proverbs to add
    existing_file: the sample to extend (default random_proverbs_50.csv)
    master_path  : master.csv to draw from
    out_file     : where to write the combined file (default: auto-named
                   random_proverbs_<total>.csv next to the existing file)
    seed         : random seed for a reproducible draw of the new rows
    """
    # --- read the existing sample: it is the source of truth for both the ---
    # --- rows to preserve AND the set of proverbs already used --------------
    if not os.path.exists(existing_file):
        sys.exit(f"ERROR: {existing_file} not found.\n"
                 f"Create the initial sample first (run without --add).")
    existing = pd.read_csv(existing_file, encoding="utf-8-sig", dtype=str).fillna("")
    print(f"[info] existing sample: {len(existing)} rows (kept unchanged)")

    # --- candidate pool: English rows NOT already in the existing sample ----
    pool = load_english_pool(master_path)
    used_ids = set(existing["id"]) if "id" in existing.columns else set()
    used_text = set(existing["text_used"]) if "text_used" in existing.columns else set()
    # Exclude by id first; also exclude by English text as a backup so blank
    # ids can never sneak a duplicate through.
    mask = ~pool["id"].isin(used_ids) & ~pool["proverb_en"].isin(used_text)
    candidates = pool[mask]
    print(f"[info] {len(candidates)} unused English proverbs available to draw from")

    if len(candidates) < n_add:
        sys.exit(f"ERROR: only {len(candidates)} unused English proverbs "
                 f"available, cannot add {n_add}. Use a smaller --add value.")

    # --- draw the new rows (reproducible via seed) --------------------------
    new = candidates.sample(n=n_add, random_state=seed).reset_index(drop=True)

    # --- build new rows in the SAME schema as the existing file -------------
    start_id = len(existing) + 1                 # continue the row_id counter
    new_rows = pd.DataFrame({
        "row_id": range(start_id, start_id + len(new)),
        "id": new["id"],
        "language": new["language"],
        "proverb_native": new["proverb_native"],
        "text_used": new["proverb_en"],          # English translation (guaranteed present)
        "prescriptive_or_descriptive": "",        # empty -> you fill these in
    })

    # Defensive: if the existing file has any extra/different columns, align to
    # it so concat lines up cleanly (missing new columns become blank).
    for col in existing.columns:
        if col not in new_rows.columns:
            new_rows[col] = ""
    new_rows = new_rows[existing.columns]

    combined = pd.concat([existing, new_rows], ignore_index=True)

    # --- write to a NEW file, leaving the original untouched ----------------
    if out_file is None:
        out_file = os.path.join(HERE, f"random_proverbs_{len(combined)}.csv")
    combined.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[saved] {out_file}")
    print(f"        {len(existing)} original + {len(new)} new = {len(combined)} total")
    print(f"[info] original file left unchanged: {existing_file}")
    print("\nNEXT: open the new file and fill 'prescriptive_or_descriptive' for "
          "the new rows (the originals keep whatever you already labeled).")
    return out_file


def main():
    ap = argparse.ArgumentParser(
        description="Sample N random proverbs from master.csv for hand-labeling.")
    ap.add_argument("--master", default=DEFAULT_MASTER,
                    help="path to master.csv (default data/processed/master.csv)")
    ap.add_argument("--n", type=int, default=50,
                    help="how many proverbs to sample (default 50)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for a reproducible sample (default 42)")
    ap.add_argument("--add", type=int, default=None,
                    help="ADD this many NEW proverbs to an existing sample "
                         "instead of creating a fresh one. Existing rows and "
                         "labels are preserved; result goes to a new file.")
    ap.add_argument("--existing", default=OUT_FILE,
                    help="sample file to extend when using --add "
                         "(default random_proverbs_50.csv)")
    ap.add_argument("--out", default=None,
                    help="output path for the combined file when using --add "
                         "(default auto-named random_proverbs_<total>.csv)")
    args = ap.parse_args()

    # --- branch: extend an existing sample instead of creating a new one ----
    if args.add is not None:
        add_proverbs(
            n_add=args.add,
            existing_file=args.existing,
            master_path=args.master,
            out_file=args.out,
            seed=args.seed,
        )
        return

    # --- verify the input exists before doing anything ----------------------
    if not os.path.exists(args.master):
        sys.exit(f"ERROR: master not found at {args.master}\n"
                 f"Pass the correct path with --master.")

    # dtype=str + fillna keeps everything as clean strings (no NaN surprises).
    master = pd.read_csv(args.master, encoding="utf-8-sig", dtype=str).fillna("")
    print(f"[info] loaded master with {len(master)} rows")

    # --- get rows with English translations --------------------------------
    # Filter for proverbs that have English translations
    master_with_en = master[master["proverb_en"].str.strip() != ""].copy()
    master_with_en = master_with_en[master_with_en["proverb_en"].str.lower() != "nan"]
    
    if len(master_with_en) < args.n:
        sys.exit(f"ERROR: master has only {len(master_with_en)} rows with English "
                 f"translations, cannot sample {args.n}.")

    # --- the random sample --------------------------------------------------
    # Shuffle with seed for reproducibility, then take the first N
    sample = master_with_en.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    # --- build the output frame --------------------------------------------
    # Only keep columns we know exist; add any missing ones as blank so the
    # output schema is always the same.
    for col in ["id", "language", "proverb_native", "proverb_en"]:
        if col not in sample.columns:
            sample[col] = ""

    out = pd.DataFrame({
        "row_id": range(1, len(sample) + 1),       # 1..50, a stable local key
        "id": sample["id"],
        "language": sample["language"],
        "proverb_native": sample["proverb_native"],
        "text_used": sample["proverb_en"],         # English translation (always present)
        "prescriptive_or_descriptive": "",                         # <-- YOU fill this in by hand
    })

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[saved] {OUT_FILE}  ({len(out)} proverbs)")
    print("\nNEXT: open that file and fill the 'human_label' column with "
          "'prescriptive' or 'descriptive' for each row, then run "
          "llm_classify_pre_descriptive.py.")


if __name__ == "__main__":
    main()