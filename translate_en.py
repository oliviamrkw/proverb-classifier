"""
translate_en.py  --  fill missing English translations (machine translation)
============================================================================
Fills `proverb_en` for rows that don't have one yet, using NLLB-200, a free
local translation model (no API key). For every row it translates, it sets
`human_machine_labelled = 1`  (1 = machine-generated).

This version calls the model directly (AutoModelForSeq2SeqLM) instead of the
high-level "translation" pipeline, because some transformers versions don't
register that pipeline task. This approach works regardless of version.

Rows that already have a translation (Gutenberg, MAPS) are left untouched.
Languages with no model mapped are skipped with a warning.

QUALITY CAVEAT: proverbs are figurative and these are low-resource languages --
exactly where machine translation is weakest. Treat these as rough reference
glosses, not gold translations. (That's why they're flagged with a 1.)

INSTALL (one time):
    pip install transformers torch sentencepiece sacremoses pandas

RUN:
    python translate_en.py --limit 5       # test on 5 rows per language first
    python translate_en.py                 # then the whole file

NOTE: first run downloads the model (~2.5 GB) and caches it. On a laptop (CPU)
it is slow; it will use your GPU automatically if you have one.
"""

import argparse
import pandas as pd

IN_DEFAULT = "processed/master.csv"
OUT_DEFAULT = "processed/master_translated.csv"
MODEL = "facebook/nllb-200-distilled-600M"
TGT = "eng_Latn"

# your language code -> NLLB language code
NLLB = {
    "yo": "yor_Latn",   # Yoruba
    "am": "amh_Ethi",   # Amharic
    "om": "gaz_Latn",   # Afaan Oromo
    "ti": "tir_Ethi",   # Tigrinya
}


class Translator:
    """Loads NLLB once for one source language and translates batches to English.
    Callable interface matches what run() expects: returns a list of
    {'translation_text': ...} dicts, like the old pipeline did."""

    def __init__(self, src_code, tgt=TGT):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.src_lang = src_code
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.bos = self.tok.convert_tokens_to_ids(tgt)   # target-language token

    def __call__(self, texts, batch_size=16, max_length=400):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=self.bos,
                                          max_length=max_length)
            for d in self.tok.batch_decode(gen, skip_special_tokens=True):
                out.append({"translation_text": d})
        return out


def get_translator(src_code):
    return Translator(src_code)


def needs_translation(df):
    """Rows that have native text but no English yet."""
    return (df["proverb_en"].astype(str).str.strip() == "") & \
           (df["proverb_native"].astype(str).str.strip() != "")


def run(inp, out, limit=None, batch_size=16, translator_factory=get_translator):
    df = pd.read_csv(inp, encoding="utf-8-sig", dtype=str).fillna("")

    todo = df[needs_translation(df)]
    print(f"{len(todo)} row(s) missing an English translation")

    for lang, grp in todo.groupby("language"):
        if lang not in NLLB:
            print(f"  skip '{lang}' -- no translation model mapped ({len(grp)} rows left blank)")
            continue

        idxs = list(grp.index)
        if limit:
            idxs = idxs[:limit]
        texts = df.loc[idxs, "proverb_native"].tolist()

        print(f"  translating {len(idxs)} '{lang}' row(s) ...")
        translator = translator_factory(NLLB[lang])
        results = translator(texts, batch_size=batch_size, max_length=400)
        translations = [r["translation_text"].strip() for r in results]

        df.loc[idxs, "proverb_en"] = translations
        df.loc[idxs, "human_machine_labelled"] = "1"   # 1 = machine-generated

    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"wrote {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=None,
                    help="translate at most N rows per language (for a quick test)")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    run(a.inp, a.out, a.limit, a.batch_size)


if __name__ == "__main__":
    main()