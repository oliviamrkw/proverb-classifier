"""
proverb_schema.py
=================
Shared definitions for the proverb master dataset.

Every reader script imports from here, so all per-dataset CSVs have the EXACT
same 15 columns in the EXACT same order. That is what makes combining them at
the end a one-line operation instead of a debugging nightmare.

You normally never run this file directly — the reader scripts use it.
"""

import os
import unicodedata
import pandas as pd

# The 15 columns you specified, in fixed order.
COLUMNS = [
    "id",
    "language",
    "resource_level",
    "proverb_native",
    "proverb_en",
    "fig_or_literal",
    "mk_theme",
    "proverb_type",
    "prescriptive_or_descriptive",
    "structural_pattern",
    "crosslingual_type_id",
    "source_dataset",
    "source_url",
    "license",
    "human_machine_labelled",
]


def clean(text) -> str:
    """Normalize unicode (NFC) and strip whitespace. Never changes wording.
    This is the same fix that prevents the 'funny characters' problem."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text)).strip()


def blank_frame(n: int) -> pd.DataFrame:
    """An empty table with the right columns and n rows (all empty strings).
    A reader fills in only the columns it has data for; the rest stay blank,
    which is exactly what you want — ML fills them later."""
    return pd.DataFrame({c: [""] * n for c in COLUMNS})


def save(df: pd.DataFrame, path: str) -> None:
    """Write a per-dataset CSV in the canonical column order.
    utf-8-sig = opens correctly in Excel AND reads fine in pandas."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = df[COLUMNS]  # enforce column order
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  wrote {path}  ({len(df)} rows)")


def validate(df: pd.DataFrame) -> bool:
    """Sanity checks. Prints PASS/FAIL and any problems."""
    problems = []
    for c in COLUMNS:
        if c not in df.columns:
            problems.append(f"missing column: {c}")
    for c in ["id", "language", "proverb_native", "source_dataset"]:
        if c in df.columns:
            empty = (df[c].astype(str).str.strip() == "").sum()
            if empty:
                problems.append(f"{empty} empty value(s) in required column '{c}'")
    if "id" in df.columns and df["id"].duplicated().any():
        problems.append(f"{df['id'].duplicated().sum()} duplicate id(s)")
    print("VALIDATION:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("   -", p)
    return not problems
