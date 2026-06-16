"""
read_gutenberg.py  --  'A Polyglot of Foreign Proverbs' (Bohn, 1857)
====================================================================
Source: https://www.gutenberg.org/cache/epub/51090/pg51090.txt
License: public domain (free to use and redistribute)

Seven European languages with English translations. Each entry in the book
looks like:   native proverb. _English translation._
The English part is wrapped in _underscores_ in the plain-text file.

What this fills: proverb_native AND proverb_en.

Run:  python read_gutenberg.py
"""

import re
import urllib.request
import pandas as pd
from proverb_schema import blank_frame, clean, save

URL = "https://www.gutenberg.org/cache/epub/51090/pg51090.txt"

# section header word -> (language code, resource level)
SECTIONS = [
    ("FRENCH",     "fr", "high"),
    ("ITALIAN",    "it", "high"),
    ("GERMAN",     "de", "high"),
    ("SPANISH",    "es", "high"),
    ("PORTUGUESE", "pt", "high"),
    ("DUTCH",      "nl", "high"),
    ("DANISH",     "da", "medium"),
]


def fetch_text() -> str:
    with urllib.request.urlopen(URL) as r:
        return r.read().decode("utf-8")


def parse(text: str):
    """Return a list of (lang, level, native, english) tuples."""
    rows = []

    # find where each language section starts (header 'FRENCH PROVERBS.' etc.)
    positions = []
    for name, lang, level in SECTIONS:
        m = re.search(rf"\n{name} PROVERBS\.", text)
        if m:
            positions.append((m.start(), name, lang, level))
    positions.sort()

    # proverbs end where the English index begins
    idx = re.search(r"\nENGLISH INDEX", text)
    end_all = idx.start() if idx else len(text)

    for i, (start, name, lang, level) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else end_all
        section = text[start:end]

        # each proverb is its own paragraph (separated by a blank line)
        for para in re.split(r"\n\s*\n", section):
            p = " ".join(para.split())            # collapse wrapped lines
            if not p:
                continue
            if p.startswith(f"{name} PROVERBS"):   # skip the section header
                continue
            if re.fullmatch(r"[A-Z]\.", p):        # skip alphabet headers "A." "B."
                continue

            m = re.search(r"_(.+?)_", p)           # first _italic_ span = English
            if not m:
                continue
            english = m.group(1).strip()
            native = p[:m.start()].strip().rstrip(".").strip()
            if native and english:
                rows.append((lang, level, native, english))
    return rows


def main():
    rows = parse(fetch_text())
    out = blank_frame(len(rows))
    out["language"] = [r[0] for r in rows]
    out["resource_level"] = [r[1] for r in rows]
    out["proverb_native"] = [clean(r[2]) for r in rows]
    out["proverb_en"] = [clean(r[3]) for r in rows]
    out["source_dataset"] = "Gutenberg-Polyglot-Bohn-1857"
    out["source_url"] = URL
    out["license"] = "public-domain"
    out["id"] = [f"gutenberg_{i+1:05d}" for i in range(len(out))]
    save(out, "processed/gutenberg.csv")


if __name__ == "__main__":
    main()
