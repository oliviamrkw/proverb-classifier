"""
read_oweyor.py  --  OWE-YOR (Yoruba proverbs)
=============================================
Source: https://huggingface.co/datasets/LingoJr/OWEYOR   (license: apache-2.0)

The dataset has two columns: a proverb text column and a 0/1 Label, where
1 = proverb and 0 = non-proverb. We keep only the proverbs (Label == 1).

What this fills: proverb_native (Yoruba) only. The public version has NO English
translations or meanings, so every other column stays blank for you to fill later.

Run:  python read_oweyor.py
First run downloads the data automatically and caches it.
"""

import pandas as pd
from datasets import load_dataset
from proverb_schema import blank_frame, clean, save

URL = "https://huggingface.co/datasets/LingoJr/OWEYOR"


def main():
    ds = load_dataset("LingoJr/OWEYOR")               # downloads automatically
    df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)

    # find the proverb-text column and the label column without hard-coding names
    prov_col = next(c for c in df.columns if "roverb" in c.lower())
    label_col = next(c for c in df.columns if "label" in c.lower())

    df = df[df[label_col] == 1].reset_index(drop=True)   # keep proverbs only
    df = df.drop_duplicates(subset=[prov_col]).reset_index(drop=True)

    out = blank_frame(len(df))
    out["language"] = "yo"
    out["resource_level"] = "low"
    out["proverb_native"] = df[prov_col].map(clean)
    out["source_dataset"] = "OWE-YOR"
    out["source_url"] = URL
    out["license"] = "apache-2.0"
    out["id"] = [f"yo_oweyor_{i+1:05d}" for i in range(len(out))]
    save(out, "processed/oweyor.csv")


if __name__ == "__main__":
    main()
