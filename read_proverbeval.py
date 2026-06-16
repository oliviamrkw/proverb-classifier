"""
read_proverbeval.py  --  ProverbEval (Ethiopian low-resource languages)
=======================================================================
Source: https://huggingface.co/datasets/israel/ProverbEval
License: not stated on the dataset page -> fine for your own research, but
verify before REDISTRIBUTING the text in a public dataset.

This dataset has many task-specific subsets. We only pull the plain
per-language proverb lists (the bare config names), not the quiz/task files.

What this fills: proverb_native only. (English meanings exist but are buried in
the 'generation' task files; skip for now.)

Run:  python read_proverbeval.py
"""

import pandas as pd
from datasets import load_dataset
from proverb_schema import blank_frame, clean, save

URL = "https://huggingface.co/datasets/israel/ProverbEval"

# HuggingFace config name -> your language code. These bare configs are the
# plain proverb lists. ("eng" also exists but overlaps your other English
# sources, so it's left out; add "eng": "en" if you want it.)
CONFIGS = {
    "amh": "am",   # Amharic
    "orm": "om",   # Afaan Oromo
    "tir": "ti",   # Tigrinya
}


def main():
    frames = []
    for cfg, lang in CONFIGS.items():
        ds = load_dataset("israel/ProverbEval", cfg)
        df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
        prov_col = next(c for c in df.columns if "roverb" in c.lower())

        sub = blank_frame(len(df))
        sub["language"] = lang
        sub["resource_level"] = "low"
        sub["proverb_native"] = df[prov_col].map(clean)
        sub["source_dataset"] = "ProverbEval"
        sub["source_url"] = URL
        sub["license"] = "unspecified-verify"
        frames.append(sub)
        print(f"  {lang}: {len(sub)} rows")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["language", "proverb_native"]).reset_index(drop=True)
    out["id"] = [f"{out['language'][i]}_proverbeval_{i+1:05d}" for i in range(len(out))]
    save(out, "processed/proverbeval.csv")


if __name__ == "__main__":
    main()
