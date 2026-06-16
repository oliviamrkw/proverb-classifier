"""
read_maps.py  --  MAPS (the CSV you already built)
==================================================
You already have MAPS in the earlier format. This just remaps it into the new
15-column schema. Point INPUT at your existing combined MAPS file.

What this fills: proverb_native, proverb_en, AND fig_or_literal (MAPS is your
only source with real figurative/literal labels).

LICENSE NOTE: MAPS is gated (request form) and is NOT redistributable. Keep
these rows for your own research/ML, but for the PUBLIC release you'd drop the
MAPS text and ship IDs + a fetch script instead.

Run:  python read_maps.py
"""

import pandas as pd
from proverb_schema import blank_frame, clean, save

# >>> EDIT THIS to point at the MAPS file you built earlier <<<
INPUT = "data/processed/proverbs_master.csv"
URL = "https://github.com/UKPLab/maps"


def main():
    src = pd.read_csv(INPUT, encoding="utf-8-sig", dtype=str).fillna("")

    out = blank_frame(len(src))
    out["language"] = src.get("lang", src.get("language", ""))
    out["resource_level"] = src.get("resource_level", "")
    out["proverb_native"] = src["proverb_native"].map(clean)
    out["proverb_en"] = src.get("proverb_en", "").map(clean)
    out["fig_or_literal"] = src.get("label", "")          # 'figurative'/'literal'
    out["source_dataset"] = "MAPS"
    out["source_url"] = URL
    out["license"] = "gated-research-only-do-not-redistribute"
    out["human_machine_labelled"] = "human"               # MAPS labels are human
    out["id"] = [f"{out['language'][i]}_maps_{i+1:05d}" for i in range(len(out))]
    save(out, "processed/maps.csv")


if __name__ == "__main__":
    main()
