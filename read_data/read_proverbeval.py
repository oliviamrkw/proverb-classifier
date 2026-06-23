"""
read_proverbeval.py  --  ProverbEval (Ethiopian low-resource languages)
=======================================================================
Source: https://huggingface.co/datasets/israel/ProverbEval
License: not stated on the dataset page -> fine for your own research, but
verify before REDISTRIBUTING the text in a public dataset.

This dataset has many task-specific subsets. We only pull the plain
per-language proverb lists (the bare config names), not the quiz/task files.

What this fills: proverb_native for all three languages, AND proverb_en for
the Afaan Oromo (om) rows that need it -- see note (1).

------------------------------------------------------------------------
NOTE (1) -- Oromo embedded English:
  The base Oromo (`orm`) subset stores everything in a single "Proverb"
  column, and for ~70 rows the English translation is jammed into that one
  field after the proverb (inconsistent delimiters). There is NO separate
  translation column to map, so for Oromo only we split the trailing English
  out into proverb_en. Amharic/Tigrinya (Ethiopic script) are clean and
  untouched. The split is heuristic -> spot-check the extracted translations;
  a few are mis-aligned/typo'd in the SOURCE itself.

NOTE (2) -- near-duplicate variants:
  The Oromo subset also contains hundreds of near-duplicate rows: the same
  proverb repeated with different trailing punctuation ("X", "X.", "X!",
  "X !"), capitalisation, or spacing. Exact-match dedup misses these. We now
  dedup on a normalised key (case-fold + NFC + collapse whitespace + strip
  edge punctuation). The key only ignores case/spacing/EDGE punctuation, so
  genuinely different proverbs are never merged. When a variant group includes
  a row that carries an English translation, that row is the one kept.
  (These are attested variant surface forms -- collapsing them is the right
  call for ML/embedding work, but worth a footnote in your write-up.)
------------------------------------------------------------------------

Run:  python read_proverbeval.py
"""

import re
import unicodedata
import pandas as pd
from datasets import load_dataset
from proverb_schema import blank_frame, clean, save

URL = "https://huggingface.co/datasets/israel/ProverbEval"

CONFIGS = {
    "amh": "am",   # Amharic
    "orm": "om",   # Afaan Oromo
    "tir": "ti",   # Tigrinya
}

# Only Afaan Oromo (Latin script) has the embedded-English problem.
SPLIT_LANGS = {"om"}

# English words that don't collide with Afaan Oromo, for detecting embedded English.
_EN = re.compile(
    r"\b(the|is|are|was|were|has|have|had|who|will|would|when|never|always|"
    r"nothing|everything|said|says|better|than|cannot|himself|themselves|"
    r"land|those|walk|person|people|rich|poor|horse|donkey|thorn|body|money|"
    r"bones|meat|ground|cattle|woman|market|honey|disease|nose|tree|relatives?)\b",
    re.I,
)

# Punctuation stripped from the ends when building a near-duplicate key.
_DEDUP_PUNCT = " !?.,;:፣።፥፤፦…-–—'\"“”‘’()[]"


def split_embedded_english(text: str):
    """Return (native, english). english is "" if no embedded English found.
    Cuts at the sentence boundary just before the first clearly-English word."""
    m = _EN.search(text)
    if not m:
        return text, ""
    cut = text.rfind(".", 0, m.start())
    if cut == -1:
        return text, ""
    return clean(text[:cut + 1]), clean(text[cut + 1:])


def norm_key(text: str) -> str:
    """Comparison key for near-duplicate detection: case-fold + NFC + collapse
    whitespace + drop spaces-before-punctuation + strip EDGE punctuation.
    Ignores only case/spacing/edge-punctuation, never internal wording, so it
    won't merge genuinely different proverbs. Used for grouping only; the
    stored text is left intact."""
    s = unicodedata.normalize("NFC", str(text)).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([!?.,;:፣።፥፤፦])", r"\1", s)
    return s.strip(_DEDUP_PUNCT)


def main():
    frames = []
    for cfg, lang in CONFIGS.items():
        ds = load_dataset("israel/ProverbEval", cfg)
        df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
        prov_col = next(c for c in df.columns if "roverb" in c.lower())

        sub = blank_frame(len(df))
        sub["language"] = lang
        sub["resource_level"] = "low"

        cleaned = df[prov_col].map(clean)
        if lang in SPLIT_LANGS:
            pairs = cleaned.map(split_embedded_english)
            sub["proverb_native"] = [p[0] for p in pairs]
            sub["proverb_en"] = [p[1] for p in pairs]   # source-provided -> human_machine_labelled stays blank
        else:
            sub["proverb_native"] = cleaned

        sub["source_dataset"] = "ProverbEval"
        sub["source_url"] = URL
        sub["license"] = "unspecified-verify"
        frames.append(sub)

        extracted = int((sub["proverb_en"] != "").sum())
        note = f"  ({extracted} embedded English translations split out)" if extracted else ""
        print(f"  {lang}: {len(sub)} rows{note}")

    out = pd.concat(frames, ignore_index=True)

    # Near-duplicate dedup (see NOTE 2). Keep the variant that has an English
    # translation, if any; otherwise keep the first occurrence.
    out["_key"] = out["proverb_native"].map(norm_key)
    out["_has_en"] = (out["proverb_en"] != "").astype(int)
    before = len(out)
    out = (out.sort_values(["language", "_key", "_has_en"],
                           ascending=[True, True, False], kind="stable")
              .drop_duplicates(subset=["language", "_key"], keep="first")
              .sort_index()
              .drop(columns=["_key", "_has_en"])
              .reset_index(drop=True))
    print(f"  near-duplicate variants removed: {before - len(out)}")

    out["id"] = [f"{out['language'][i]}_proverbeval_{i+1:05d}" for i in range(len(out))]
    save(out, "data/processed/proverbeval.csv")


if __name__ == "__main__":
    main()