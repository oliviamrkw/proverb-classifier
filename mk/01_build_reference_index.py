"""
mk/01_build_reference_index.py
==============================================================================
STEP 1 OF THE MATTI KUUSI CLASSIFICATION PIPELINE  --  build the reference index.

WHAT THIS SCRIPT IS FOR
-----------------------
We want to label each proverb in your big 21k dataset with a Kuusi code at three
levels: theme (e.g. "A"), main_class ("A1"), subgroup ("A1a"). We have NO labels
on the 21k. What we DO have is the Kuusi typology itself, which already contains:
    (1) example proverbs, each one tagged with its theme/main/subgroup, and
    (2) a one-line definition for every code (parsed from the website).
Those two things together ARE our labeled reference data. This script turns them
into a single searchable "reference index": a big array of embedding vectors plus
a parallel table that says, for each vector, which Kuusi code it belongs to.

Later scripts (02 = cascade kNN, 03 = LLM re-ranker) read this index and, for any
new proverb, find the nearest reference vectors and read off their codes. So this
script does NOT classify anything yet -- it only builds the lookup table.

This is the Kuusi-pipeline equivalent of 01_build_split_and_embeddings.py from the
figurative/literal task, EXCEPT:
  - there is no train/val/test split here (that happens in 02 via cross-validation),
  - we embed TWO kinds of text: real example proverbs AND abstract definitions.

INPUTS  (both expected to sit next to this script, inside the mk/ folder)
    mk/kuusi_proverb_types_clean.csv   <- the Kuusi types + example proverbs
    mk/definitions.csv                 <- code -> definition (level,code,text)

OUTPUTS (written into mk/npy/ and mk/, next to this script)
    mk/npy/ref_embeddings.npy          <- float array, one row per reference item
    mk/ref_metadata.csv                <- one row per reference item, SAME order:
                                          ref_idx, source_type, text,
                                          theme, main_class, subgroup_code, kuusi_id

RUN
    python mk/01_build_reference_index.py
REQUIRES
    pip install sentence-transformers pandas numpy
==============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# PATHS. We anchor everything to THIS script's folder (HERE = mk/) so the script
# behaves identically no matter what directory you run it from. ROOT is the
# project root (one level up), kept for consistency with your other scripts.
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))     # .../mk
ROOT = os.path.dirname(HERE)                           # project root

KUUSI_CSV = os.path.join(HERE, "kuusi_proverb_types_clean.csv")  # examples + codes
DEFS_CSV  = os.path.join(HERE, "definitions.csv")                # code -> definition
NPY_DIR   = os.path.join(HERE, "npy")                            # output folder
EMB_OUT   = os.path.join(NPY_DIR, "ref_embeddings.npy")          # vectors
META_OUT  = os.path.join(HERE, "ref_metadata.csv")              # parallel labels

# -----------------------------------------------------------------------------
# ENCODER REGISTRY. Pick one with --encoder (default labse). LaBSE was built for
# translation matching and underperformed on this typology, so e5 / bge-m3 are
# here to test whether a stronger *semantic* encoder lifts accuracy.
#   "prefix" is text prepended before encoding. multilingual-e5 REQUIRES a
#   "query:" / "passage:" prefix or it silently degrades; for a symmetric
#   classification/similarity task we use "query: " on everything. bge-m3 and
#   LaBSE need no prefix.
# IMPORTANT: whatever you choose here, the script that embeds your 21k proverbs
# later MUST use the same encoder + prefix, or the two live in different spaces.
# We record the choice in npy/encoder.txt so downstream steps can check.
# -----------------------------------------------------------------------------
ENCODERS = {
    "labse": ("sentence-transformers/LaBSE",            ""),
    "e5":    ("intfloat/multilingual-e5-large",         "query: "),
    "bge":   ("BAAI/bge-m3",                             ""),
}
ENC_FILE = os.path.join(NPY_DIR, "encoder.txt")

# Which CSV columns hold example proverbs. The variant_* columns are the SAME
# proverb split out per language (English / German / French / ...), so embedding
# each one separately gives us multilingual anchors for the same Kuusi type.
# proverb_primary_en is the English representative. proverb_original is the raw
# concatenated "A / B / C" string -- we only fall back to it if a row somehow has
# no split variants, so we don't embed a multi-language blob as one vector.
EXAMPLE_COLS = [
    "proverb_variant_1", "proverb_variant_2", "proverb_variant_3",
    "proverb_variant_4", "proverb_variant_extra", "proverb_primary_en",
]


# =============================================================================
# PART A -- turn the Kuusi example proverbs into labeled reference points
# =============================================================================
def load_examples(path=KUUSI_CSV):
    """Read the Kuusi CSV and EXPLODE it into one row per example proverb.

    The CSV has one row per Kuusi *type* (e.g. A1a-13), and each type carries up
    to ~6 example proverbs spread across the variant columns. We want one
    reference POINT per individual proverb, each carrying that type's three codes,
    so that later a nearest-neighbour hit on any single proverb tells us the code.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    rows = []
    for _, r in df.iterrows():
        # Collect this type's example proverbs from the variant columns, keeping
        # only non-empty ones and removing duplicates WITHIN the row (the English
        # variant is often identical to proverb_primary_en, so we'd double-count).
        seen, examples = set(), []
        for col in EXAMPLE_COLS:
            txt = r[col].strip()
            if txt and txt not in seen:
                seen.add(txt)
                examples.append(txt)
        # Safety net: if a row had no split variants at all, fall back to the raw
        # "original" field, splitting on " / " so we still get separate proverbs.
        if not examples and r["proverb_original"].strip():
            for piece in r["proverb_original"].split("/"):
                piece = piece.strip()
                if piece and piece not in seen:
                    seen.add(piece)
                    examples.append(piece)
        # Emit one reference point per example proverb, tagged with all 3 codes.
        for txt in examples:
            rows.append(dict(
                source_type="example",
                text=txt,
                theme=r["theme"],
                main_class=r["main_class"],
                subgroup_code=r["subgroup_code"],
                kuusi_id=r["kuusi_id"],
            ))
    out = pd.DataFrame(rows)
    print(f"examples: {len(out)} reference points "
          f"from {len(df)} Kuusi types "
          f"({out['subgroup_code'].nunique()} subgroups)")
    return out


# =============================================================================
# PART B -- turn the code definitions into labeled reference points (anchors)
# =============================================================================
def load_definitions(path=DEFS_CSV):
    """Read definitions.csv and make one reference point per definition.

    These are ABSTRACT anchors. A proverb whose wording matches no specific
    example may still land near the *definition* of its category (e.g. an abstract
    saying about unchanging character landing near "X's basic nature will be
    unchanged"). We tag each definition with whatever codes it implies:
        theme def     -> theme only          (source_type = def_theme)
        main def      -> theme + main_class   (source_type = def_main)
        subgroup def  -> theme + main + sub   (source_type = def_subgroup)
    Deriving the parent codes from the child code is trivial: for "A1a",
    theme = "A", main_class = "A1", subgroup_code = "A1a".
    """
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    rows = []
    for _, r in d.iterrows():
        level, code, text = r["level"], r["code"], r["text"].strip()
        if not text:
            continue
        if level == "theme":          # code like "A"
            theme, main, sub = code, "", ""
            stype = "def_theme"
        elif level == "main":         # code like "A1"
            theme, main, sub = code[0], code, ""
            stype = "def_main"
        elif level == "subgroup":     # code like "A1a"
            theme, main, sub = code[0], code[:2], code
            stype = "def_subgroup"
        else:
            continue                  # unknown level -> skip
        rows.append(dict(
            source_type=stype,
            text=text,
            theme=theme,
            main_class=main,
            subgroup_code=sub,
            kuusi_id="",              # definitions have no specific type id
        ))
    out = pd.DataFrame(rows)
    print(f"definitions: {len(out)} reference points "
          f"({(out.source_type=='def_subgroup').sum()} subgroup, "
          f"{(out.source_type=='def_main').sum()} main, "
          f"{(out.source_type=='def_theme').sum()} theme)")
    return out


# =============================================================================
# PART C -- assemble the full reference frame (examples + definitions)
# =============================================================================
def build_reference_frame(kuusi_csv=KUUSI_CSV, defs_csv=DEFS_CSV):
    """Stack examples on top of definitions into ONE table, in a fixed order.

    The row order here is the contract for the whole pipeline: row i of
    ref_metadata.csv corresponds to row i of ref_embeddings.npy. We add an
    explicit ref_idx column so nothing can silently get shuffled later.
    """
    ex = load_examples(kuusi_csv)
    de = load_definitions(defs_csv)
    frame = pd.concat([ex, de], ignore_index=True)
    frame.insert(0, "ref_idx", range(len(frame)))      # 0..N-1, the canonical order
    # Reorder columns so the saved CSV is easy to read.
    frame = frame[["ref_idx", "source_type", "text",
                   "theme", "main_class", "subgroup_code", "kuusi_id"]]
    return frame


# =============================================================================
# PART D -- embed every reference text and save the index
# =============================================================================
def main():
    # --- choose the encoder from the command line (default LaBSE) ---
    ap = argparse.ArgumentParser(description="Build the Kuusi reference index.")
    ap.add_argument("--encoder", choices=list(ENCODERS), default="labse",
                    help="which embedding model to use (default: labse)")
    args = ap.parse_args()
    hf_id, prefix = ENCODERS[args.encoder]

    os.makedirs(NPY_DIR, exist_ok=True)

    # 1) Build the labeled table (no embeddings yet). Pure bookkeeping.
    frame = build_reference_frame()
    print(f"\nTOTAL reference points: {len(frame)}")

    # 2) Embed the 'text' column. The encoder maps each proverb / definition string
    #    to a vector that captures MEANING (not words), so cross-language matches
    #    work. We keep the output order identical to `frame` so the i-th vector
    #    lines up with the i-th metadata row. The per-encoder prefix (e.g. e5's
    #    "query: ") is applied here and MUST be reused when embedding the 21k.
    print(f"\nencoder: {args.encoder} -> {hf_id}  (prefix={prefix!r})")
    model = SentenceTransformer(hf_id)
    texts = [prefix + t for t in frame["text"].tolist()]
    print("embedding reference texts (this is the slow part)...")
    emb = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=False,   # we L2-normalize later in 02, like Method A
    )

    # 3) Hard guarantee: vectors and labels must be the same length and order.
    #    If this ever fails, the whole pipeline is invalid, so we crash loudly.
    assert emb.shape[0] == len(frame), (
        f"embeddings ({emb.shape[0]}) != metadata rows ({len(frame)})")

    # 4) Save both halves of the index + the encoder choice (so downstream steps
    #    can refuse to mix encoders).
    np.save(EMB_OUT, emb)
    frame.to_csv(META_OUT, index=False, encoding="utf-8-sig")
    with open(ENC_FILE, "w", encoding="utf-8") as f:
        f.write(f"{args.encoder}\t{hf_id}\t{prefix}\n")
    print(f"\nsaved embeddings -> {EMB_OUT}  shape={emb.shape}")
    print(f"saved metadata   -> {META_OUT}  rows={len(frame)}")
    print(f"recorded encoder -> {ENC_FILE}")
    print("\nreference index built. re-run 02 to score this encoder, "
          "or run 03 for the all-LLM classifier.")


if __name__ == "__main__":
    main()
