"""
build_dataset.py
================
Turn raw MAPS Excel files into one CSV per language, plus a
combined master CSV across all languages.

UNIFIED SCHEMA (one row = one proverb)
--------------------------------------
    id              zh_0001            stable key  {lang}_{4-digit index}
    lang            zh                 ISO code
    resource_level  high               high / medium / low  (you set this)
    proverb_native  失败乃成功之母       original script
    proverb_en      Failure is ...     English (human translation when available)
    context_native  A: ... B: ...      the MAPS conversation, native language
    context_en      A: ... B: ...      the MAPS conversation, English
    label           figurative         your prediction target (figurative/literal)
    source          MAPS               MAPS / Owomoyela2005 / ...
    notes                              free text; annotator comments, edge cases

USAGE
-----
    python build_dataset.py

    
"""
 
import argparse
import os
import sys
import pandas as pd

# handles duplicates
bn = pd.read_csv("data/processed/bn.csv", encoding="utf-8-sig")
print(bn[bn.duplicated("proverb_native", keep=False)].sort_values("proverb_native"))

CONFIG = {
    "zh": {
        "resource_level": "high",
        "folder": "MAPS/zh",
        "label_file": "test_proverbs.xlsx",
        "en_file": "human_translation_2en.xlsx", 
        "source": "MAPS",
    },
    "bn": {
        "resource_level": "medium",
        "folder": "bn",
        "label_file": "test_proverbs.xlsx",
        "en_file": "human_translation_2en.xlsx", 
        "source": "MAPS",
    },
    "id": {
        "resource_level": "medium",
        "folder": "id",
        "label_file": "test_proverbs.xlsx",
        "en_file": "human_translation_2en.xlsx", 
        "source": "MAPS",
    }
}

REQUIRED_COLUMNS = [
    "id", "lang", "resource_level", "proverb_native", "proverb_en",
    "context_native", "context_en", "label", "source", "notes",
]

LABEL_MAP = {1: "figurative", 0: "literal"}



def load_language(lang: str, cfg: dict, data_dir: str) -> pd.DataFrame:
    """Read one language's MAPS files and return rows in the unified schema."""
    folder = os.path.join(data_dir, cfg["folder"])
    label_path = os.path.join(folder, cfg["label_file"])
    native = pd.read_excel(label_path)

    en = None   # no need to translate english
    if cfg.get("en_file"):
        en_path = os.path.join(folder, cfg["en_file"])
        if os.path.exists(en_path):
            en = pd.read_excel(en_path)

    # check data format
    if en is not None: 
        # check row counts the same
        if len(en) != len(native):  
            raise ValueError(
                f"[{lang}] native ({len(native)}) and English ({len(en)}) files "
                f"have different row counts."
            )
        # check that answer keys are the same
        if "answer_key" in native and "answer_key" in en:
            if not (native["answer_key"].values == en["answer_key"].values).all():
                raise ValueError(
                    f"[{lang}] native and English files are not row-aligned "
                    f"(answer_key sequence differs). Pair by a shared key instead."
                )
        # notify mismatch if native is different than translated
        if "is_figurative" in en:
            n_mis = int((native["is_figurative"].values != en["is_figurative"].values).sum())
            if n_mis:
                print(f"  [warn] {lang}: {n_mis} label mismatch(es) between native "
                      f"and English files; using NATIVE labels.")
    
    out = pd.DataFrame()
    out["lang"] = [lang] * len(native)
    out["resource_level"] = cfg["resource_level"]
    out["proverb_native"] = native["proverb"].astype(str).str.strip()
    out["proverb_en"] = (en["proverb"].astype(str).str.strip()
                         if en is not None else "")
    out["context_native"] = native["conversation"].astype(str).str.strip()
    out["context_en"] = (en["conversation"].astype(str).str.strip()
                         if en is not None else "")
    out["label"] = native["is_figurative"].map(LABEL_MAP)
    out["source"] = cfg["source"]
    out["notes"] = ""
 
    # stable id: zh_0001 ... in file order
    out.insert(0, "id", [f"{lang}_{i+1:04d}" for i in range(len(out))])
    return out[REQUIRED_COLUMNS]



def validate(df: pd.DataFrame) -> bool:
    ok = True
    print("\n=== VALIDATION ===")
 
    # 1. schema
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [FAIL] missing columns: {missing}"); ok = False
 
    # 2. required fields non-null
    for col in ["id", "lang", "proverb_native", "label"]:
        n = df[col].isnull().sum() + (df[col].astype(str).str.strip() == "").sum()
        if n:
            print(f"  [FAIL] {n} empty value(s) in required column '{col}'"); ok = False
 
    # 3. label values
    bad = set(df["label"].dropna().unique()) - {"figurative", "literal"}
    if bad:
        print(f"  [FAIL] unexpected label value(s): {bad}"); ok = False
 
    # 4. unique ids
    dup_ids = df["id"].duplicated().sum()
    if dup_ids:
        print(f"  [FAIL] {dup_ids} duplicate id(s)"); ok = False
 
    # 5. duplicate proverbs within a language
    for lang, g in df.groupby("lang"):
        d = g["proverb_native"].duplicated().sum()
        if d:
            print(f"  [warn] {lang}: {d} duplicate proverb(s) within language")
 
    # 6. per-language class balance
    print("\n  Per-language summary:")
    print(f"    {'lang':<6}{'n':>6}{'fig':>6}{'lit':>6}{'%fig':>7}")
    for lang, g in df.groupby("lang"):
        n = len(g); fig = (g["label"] == "figurative").sum(); lit = n - fig
        pct = 100 * fig / n if n else 0
        flag = "  <-- imbalanced" if (pct < 30 or pct > 70) else ""
        print(f"    {lang:<6}{n:>6}{fig:>6}{lit:>6}{pct:>6.0f}%{flag}")
 
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}  (total rows: {len(df)})")
    print("==================\n")
    return ok



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
 
    frames = []
    for lang, cfg in CONFIG.items():
        print(f"Loading {lang} ...")
        df = load_language(lang, cfg, args.data_dir)
        per_lang_path = os.path.join(args.out_dir, f"{lang}.csv")
        df.to_csv(per_lang_path, index=False, encoding="utf-8")
        print(f"  wrote {per_lang_path}  ({len(df)} rows)")
        frames.append(df)
 
    master = pd.concat(frames, ignore_index=True)
    master_path = os.path.join(args.out_dir, "proverbs_master.csv")
    master.to_csv(master_path, index=False, encoding="utf-8")
    print(f"\nwrote {master_path}  ({len(master)} rows across {master['lang'].nunique()} language(s))")
 
    ok = validate(master)
    sys.exit(0 if ok else 1)



if __name__ == "__main__":
    main()