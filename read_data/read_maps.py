"""
read_maps.py  --  MAPS BUILDER (all languages) + figurativity labels
====================================================================
REPLACES the old patcher. The previous version only *filled labels* on rows
that already existed in maps.csv (zh/bn/id). It could not ADD languages.

This version BUILDS maps.csv from the raw MAPS folders directly, one language
at a time, so every language MAPS ships (bn, de, en, id, ru, zh, ...) is
included with native text + English + the figurative/literal label.

Per language:
  - proverb_native + fig_or_literal  <- test_proverbs.xlsx  (same row = safe)
        is_figurative is already 0/1 and matches the schema (1=fig, 0=literal).
  - proverb_en  <- human_translation_2en.xlsx  (preferred),
                   else machine_translation_2en.xlsx  (fallback),
                   and for 'en' the native text IS English.
        Some languages only ship machine translations -- the builder reports
        which source it used per language so you can see it.

SAFETY: native + label always come from test_proverbs (never mispaired).
English is paired by row position and sanity-checked against the shared
answer_key column; if a language's English file is a different length or out of
order, English is left blank / flagged rather than silently mismatched. The
label -- the thing ML depends on -- is never at risk.

VERIFIED here: zh builds 334 rows (143 fig / 191 literal, human English), and
the machine-only fallback path works. de/en/ru/bn/id depend on your local raw
folders -- run it and read the per-language report.

Run:  python read_maps.py
Or for one language in a notebook:  build_language("zh")
"""

import os
import pandas as pd
from proverb_schema import clean, save

# --- paths / constants -----------------------------------------------------
RAW_DIR = "data/raw/MAPS"               # holds per-language folders: zh/ bn/ de/ en/ id/ ru/
OUTPUT  = "data/processed/maps.csv"     # built fresh from raw each run
URL = "https://github.com/UKPLab/maps"
LICENSE = "gated-research-only-do-not-redistribute"

LABEL_COL  = "is_figurative"            # 0/1 -> fig_or_literal directly (1=fig, 0=literal)
NATIVE_COL = "proverb"                  # test_proverbs: native | *_2en: English
TEST_FILE  = "test_proverbs.xlsx"       # carries native + label together
EN_FILES   = [("human_translation_2en.xlsx", "human"),
              ("machine_translation_2en.xlsx", "machine")]   # order = preference

# Resource level per your project taxonomy. NOTE: the OLD maps.csv tagged zh as
# "high", which contradicted your doc (zh/bn/id are low-resource). Set per doc;
# edit this dict if you disagree.
RESOURCE_LEVEL = {"zh": "low", "bn": "low", "id": "low",
                  "de": "high", "en": "high", "ru": "high"}

SCHEMA = ["id", "language", "resource_level", "proverb_native", "proverb_en",
          "fig_or_literal", "mk_theme", "proverb_type",
          "prescriptive_or_descriptive", "structural_pattern",
          "crosslingual_type_id", "source_dataset", "source_url", "license",
          "human_machine_labelled"]


def _read_english(lang_dir):
    """Return (english_dataframe_or_None, source_tag). Prefers human over machine."""
    for fname, tag in EN_FILES:
        p = os.path.join(lang_dir, fname)
        if os.path.exists(p):
            return pd.read_excel(p, dtype=str).fillna(""), tag
    return None, "none"


def build_language(lang, raw_dir=RAW_DIR):
    """Build ALL MAPS rows for ONE language from its raw files.
    Returns a 15-column DataFrame, or None if the folder/test file is missing.
    Use this directly for single-language work, e.g. build_language("zh")."""
    lang_dir = os.path.join(raw_dir, lang)
    test_path = os.path.join(lang_dir, TEST_FILE)
    if not os.path.exists(test_path):
        print(f"  [SKIP] {lang}: no {TEST_FILE} in {lang_dir}")
        return None

    t = pd.read_excel(test_path, dtype=str).fillna("")
    if NATIVE_COL not in t.columns or LABEL_COL not in t.columns:
        print(f"  [SKIP] {lang}: {TEST_FILE} missing '{NATIVE_COL}'/'{LABEL_COL}'")
        return None

    native = t[NATIVE_COL].map(clean)
    label = t[LABEL_COL].str.strip()

    bad = sorted(set(label) - {"0", "1"})
    if bad:
        print(f"  [WARN] {lang}: non-0/1 label values present: {bad}")

    # --- English column ---
    if lang == "en":
        prov_en, en_src = native, "native(en)"
    else:
        e, en_src = _read_english(lang_dir)
        if e is None:
            prov_en = pd.Series([""] * len(t))
        elif len(e) != len(t):
            print(f"  [WARN] {lang}: English rows ({len(e)}) != native ({len(t)}); "
                  f"English left blank")
            prov_en, en_src = pd.Series([""] * len(t)), en_src + "(len-mismatch)"
        else:
            if ("answer_key" in e.columns and "answer_key" in t.columns
                    and not (e["answer_key"].values == t["answer_key"].values).all()):
                print(f"  [WARN] {lang}: answer_key order differs between test and "
                      f"{en_src} file -- English may be mispaired")
            prov_en = e[NATIVE_COL].map(clean)

    # --- assemble schema frame ---
    n = len(t)
    out = pd.DataFrame({c: [""] * n for c in SCHEMA})
    out["language"] = lang
    out["resource_level"] = RESOURCE_LEVEL.get(lang, "")
    out["proverb_native"] = native.values
    out["proverb_en"] = prov_en.values
    out["fig_or_literal"] = label.values
    out["source_dataset"] = "MAPS"
    out["source_url"] = URL
    out["license"] = LICENSE
    out["human_machine_labelled"] = "human"          # MAPS labels are human gold
    out["id"] = [f"{lang}_maps_{i + 1:05d}" for i in range(n)]

    figc, litc = int((label == "1").sum()), int((label == "0").sum())
    print(f"  {lang}: {n} rows | figurative={figc} literal={litc} | English={en_src}")
    return out[SCHEMA]


def main(raw_dir=RAW_DIR, output_path=OUTPUT):
    # auto-discover any language folder that has a test_proverbs.xlsx
    langs = sorted(d for d in os.listdir(raw_dir)
                   if os.path.isdir(os.path.join(raw_dir, d))
                   and os.path.exists(os.path.join(raw_dir, d, TEST_FILE)))
    print(f"languages found: {langs}")

    frames = [f for f in (build_language(l, raw_dir) for l in langs) if f is not None]
    if not frames:
        print("No languages built. Check RAW_DIR and the folder layout.")
        return

    allrows = pd.concat(frames, ignore_index=True)
    blanks = (allrows["fig_or_literal"] == "").sum()
    en_blank = (allrows["proverb_en"] == "").sum()
    print(f"\nTOTAL: {len(allrows)} rows / {len(frames)} languages "
          f"| label blanks: {blanks} | English blanks: {en_blank}")
    save(allrows, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()