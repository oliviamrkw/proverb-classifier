"""
combine.py  --  stack every per-dataset CSV into one master
===========================================================
Reads every processed/*.csv (except master.csv itself) and stacks them into
processed/master.csv in your 15-column schema. Run this AFTER the readers.

Run:  python combine.py
"""

import glob
import pandas as pd
from proverb_schema import COLUMNS, validate

OUT = "processed/master.csv"


def main():
    files = [f for f in sorted(glob.glob("processed/*.csv")) if not f.endswith("master.csv")]
    if not files:
        print("No processed/*.csv files found. Run the read_*.py scripts first.")
        return

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
        frames.append(df)
        print(f"  + {f}: {len(df)} rows")

    master = pd.concat(frames, ignore_index=True)[COLUMNS]
    master.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\ncombined {len(files)} files -> {OUT}  ({len(master)} rows)\n")

    validate(master)

    print("\nRows by source and language:")
    print(master.groupby(["source_dataset", "language"]).size().to_string())


if __name__ == "__main__":
    main()
