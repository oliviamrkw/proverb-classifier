"""
clean_master.py  --  final cleanup stage for the combined proverb master.
Run AFTER combine.py:   python clean_master.py
Requires:  pip install wordfreq

Fixes (in order), all verified safe:
  1. NFC + whitespace normalization (idempotent safety net).
  2. Oromo embedded-English split. ProverbEval Afaan Oromo rows store the
     English translation inside the proverb field. Detected at SENTENCE level
     by English-word fraction. Applied ONLY to language 'om' and ONLY where
     proverb_en is empty. Other languages are deliberately NOT touched:
     French/Spanish/Yoruba share Latin vocabulary with English and would
     false-positive; their English already lives in proverb_en.
  3. Drop genuine junk rows: empty, pure punctuation, or a lone English word
     (e.g. "Proverbs", "spoiled."). Chinese/Amharic etc. are safe -- the rule
     keys on "single ASCII English word", not on word count.
  4. Near-duplicate dedup on a normalized native key (case + whitespace + edge
     punctuation), WITHIN A LANGUAGE BUT ACROSS SOURCES. This removes the same
     proverb appearing in two datasets (e.g. German in both MAPS and Gutenberg).
     Keep priority within each duplicate group: a row WITH a fig/literal label
     wins (keeps the MAPS gold label), then a row WITH English. Dedup is NOT
     done across languages -- two languages can coincidentally share an
     identical short Latin-script string, and merging those would corrupt data.
"""
import re, unicodedata, sys
import pandas as pd
from wordfreq import top_n_list

IN  = sys.argv[1] if len(sys.argv) > 1 else "data/processed/master.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/processed/master.csv"

_EN = set(w for w in top_n_list("en", 40000) if len(w) >= 3)
_PUNCT = " !?.,;:፣።፥፤፦…-–—'\"“”‘’()[]"

def normspace(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(t))).strip()

def _sentences(s):
    s = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", s)
    return [p for p in re.split(r"(?<=[.!?])\s+", s.strip()) if p.strip()]

def _en_frac(seg):
    toks = [t for t in re.findall(r"[A-Za-z']+", seg.lower()) if len(t) >= 3]
    return (sum(t in _EN for t in toks) / len(toks)) if len(toks) >= 2 else 0.0

def split_english(text, thr=0.6):
    s = _sentences(text); i = len(s)
    while i > 0 and _en_frac(s[i-1]) >= thr:
        i -= 1
    if i in (0, len(s)):
        return text, ""
    return " ".join(s[:i]).strip(), " ".join(s[i:]).strip()

def norm_key(s):
    s = unicodedata.normalize("NFC", str(s)).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([!?.,;:፣።፥፤፦])", r"\1", s)
    return s.strip(_PUNCT)

def is_junk(nat):
    if nat == "" or re.fullmatch(r"[\W_]+", nat):
        return True
    m = re.fullmatch(r"([A-Za-z]+)\.?", nat)      # a single ASCII word (= not a native proverb)
    return bool(m) and m.group(1).lower() in _EN

def main():
    df = pd.read_csv(IN, encoding="utf-8-sig", dtype=str).fillna("")
    n0 = len(df)

    for c in ("proverb_native", "proverb_en"):
        df[c] = df[c].map(normspace)

    split_n = 0
    for idx in df.index[(df.language == "om") & (df.proverb_en == "")]:
        nat, en = split_english(df.at[idx, "proverb_native"])
        if en:
            df.at[idx, "proverb_native"], df.at[idx, "proverb_en"] = nat, en
            split_n += 1

    junk = df.proverb_native.map(is_junk)
    df = df[~junk].copy()

    # --- near-duplicate dedup: within language, ACROSS sources -------------
    df["_k"] = df.proverb_native.map(norm_key)
    df["_l"] = (df["fig_or_literal"].str.strip() != "").astype(int)   # has gold label
    df["_e"] = (df["proverb_en"] != "").astype(int)                   # has English
    before = len(df)
    # sort so the most informative row in each (language, key) group is first:
    # labelled rows beat unlabelled; among those, English-bearing beat empty.
    df = (df.sort_values(["language", "_k", "_l", "_e"],
                         ascending=[True, True, False, False], kind="stable")
            .drop_duplicates(subset=["language", "_k"], keep="first")
            .sort_index().drop(columns=["_k", "_l", "_e"]))

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"rows: {n0} -> {len(df)}")
    print(f"  Oromo embedded-English split : {split_n}")
    print(f"  junk rows dropped            : {int(junk.sum())}")
    print(f"  near-duplicate rows removed  : {before - len(df)}  (within-language, cross-source)")
    print(f"  Oromo rows now with English  : {int(((df.language=='om')&(df.proverb_en!='')).sum())}")

if __name__ == "__main__":
    main()