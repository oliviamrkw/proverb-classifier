"""
read_gutenberg.py  --  'A Polyglot of Foreign Proverbs' (Bohn, 1857)
====================================================================
Source: https://www.gutenberg.org/cache/epub/51090/pg51090.txt
License: public domain (free to use and redistribute)

Seven European languages with English translations, each a "monograph,"
followed by a back-of-book alphabetical English index.

Run:  python read_gutenberg.py
"""

import re
import urllib.request
from collections import Counter

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

# Cross-reference tokens that appear (italicised) in the book's INDEX, never
# as real translations. Used only as a per-row safety net for fix (3).
INDEX_TOKENS = {
    "see", "see also", "i.e", "i. e", "e.g", "cf", "or", "we say",
    "viz", "ditto", "meaning", "so", "scotch", "scotice", "hamburg",
    "moses", "knows", "of", "give", "ironical",
}


def fetch_text() -> str:
    with urllib.request.urlopen(URL) as r:
        return r.read().decode("utf-8")


def find_proverbs_end(text: str, after: int) -> int:
    """Where the proverb body ends (start of the index / Gutenberg footer).

    Searched only from `after` (the last section's start) onward so it can
    never match the 'ENGLISH INDEX' line in the table of contents, which
    appears near the top of the file with trailing page numbers.
    """
    candidates = []
    # Stand-alone heading line: "GENERAL INDEX", "ENGLISH INDEX", or "INDEX".
    # `$` end-of-line anchor means the contents entry ("ENGLISH INDEX  405-579")
    # is NOT matched, because it has page numbers after it.
    m = re.search(r"^[ \t]*(?:GENERAL |ENGLISH )?INDEX\.?[ \t]*$",
                  text[after:], re.MULTILINE)
    if m:
        candidates.append(after + m.start())
    # Backstop: Project Gutenberg end marker.
    m = re.search(r"\*\*\*\s*END OF TH", text[after:])
    if m:
        candidates.append(after + m.start())
    return min(candidates) if candidates else len(text)


def looks_like_index_row(native: str, english: str) -> bool:
    """Safety net: True if a parsed row is actually a book-index entry."""
    e = english.strip().lower().rstrip(".")
    if e in INDEX_TOKENS:
        return True
    # Index entries carry a trailing page number, e.g. "... biter, 240—"
    if re.search(r",\s*\d{1,3}\s*[—–-]?\s*$", native):
        return True
    return False


def split_plain(p: str):
    """German format: 'Native sentence. English sentence.'

    Split at the first sentence-final [.?!] followed by whitespace and the
    start of the English clause (allowing an optional opening quote/bracket
    before the capital letter). Heuristic -- spot-check German output.
    """
    m = re.search(r'[.?!]["”’»)\]]?\s+(?=["“‘«(\[]?[A-Z])', p)
    if not m:
        return None
    native = p[:m.end()].strip()
    english = p[m.end():].strip()
    # Drop a trailing editorial note in (parentheses), e.g. "(So Shakspeare ...)"
    english = re.sub(r"\s*\([^)]*\)\s*$", "", english).strip()
    native = native.rstrip(".").strip()
    return (native, english) if native and english else None


def parse(text: str):
    """Return a list of (lang, level, native, english) tuples."""
    rows = []

    # Locate each language section (header 'FRENCH PROVERBS.' etc.). The
    # required trailing period means the contents lines ('FRENCH PROVERBS
    # 1- 64') are not matched.
    positions = []
    for name, lang, level in SECTIONS:
        m = re.search(rf"\n{name} PROVERBS\.", text)
        if m:
            positions.append((m.start(), name, lang, level))
    positions.sort()

    for i, (start, name, lang, level) in enumerate(positions):
        if i + 1 < len(positions):
            end = positions[i + 1][0]
        else:
            end = find_proverbs_end(text, start)   # last (Danish) section
        section = text[start:end]

        for para in re.split(r"\n\s*\n", section):
            p = " ".join(para.split())             # collapse wrapped lines
            if not p:
                continue
            if p.startswith(f"{name} PROVERBS"):    # section header
                continue
            if re.fullmatch(r"[A-Z]\.", p):         # alphabet divider "A." "B."
                continue

            if lang == "de":                        # plain "Native. English."
                parsed = split_plain(p)
                if not parsed:
                    continue
                native, english = parsed
            else:                                   # "Native. _English._"
                m = re.search(r"_(.+?)_", p)        # first italic span = English
                if not m:
                    continue
                english = m.group(1).strip()
                native = p[:m.start()].strip().rstrip(".").strip()

            if not (native and english):
                continue
            if looks_like_index_row(native, english):   # fix (3) safety net
                continue
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

    # Built-in sanity report -- eyeball this every run.
    print(f"total rows: {len(out)}")
    print("by language:", dict(sorted(Counter(r[0] for r in rows).items())))

    save(out, "data/processed/gutenberg.csv")


if __name__ == "__main__":
    main()