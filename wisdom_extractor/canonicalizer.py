"""
wisdom_extractor/canonicalizer.py
==============================================================================
CANONICALIZATION ENGINE — ported from the Wisdom Extractor + extended

PROVENANCE
----------
The engine design (first-match-wins regex cascade -> minimal proposition) and
the rules marked [WE] below are ported verbatim from:
    Belciug, V. & Pelican, E. (2025). "The Wisdom Extractor: Mining
    Cross-Cultural Proverbs to Elicit Time-Tested Heuristics." ConsILR-2025.
    Code: https://github.com/ovladon/wisdom-extractor (core/canonicalize.py,
    v10_working rule set). Cite them in the manuscript if this ships.

WHY WE EXTENDED IT (measured, not assumed)
------------------------------------------
On our real 100 hand-labeled proverbs, the stock [WE] rules matched only
6/100 — 94 passed through unchanged. Their rule set is narrow English
templates. So this module keeps their engine + rules, and ADDS [EXT] rules
targeting OUR implicit-prescription mechanisms, so that the canonical prefix
becomes a machine-checkable signal:

    canonical prefix        rule_family     maps to our column
    "Better X than Y."      comparison   -> has_valuation_or_comparison
    "X is valued as Y."     valuation    -> has_valuation_or_comparison
    "Avoid X."              prohibition  -> is_direct_command (avoid)
    "Do X."                 imperative   -> is_direct_command (do)
    "If X, then Y."         conditional  -> consequential cause->result
    "X causes Y."           causal       -> consequential cause->result
    (no match)              none         -> no structural signal

The function returns (canonical_text, rule_family, rule_id) so downstream
code can build heuristic columns from the FAMILY, not by re-parsing text.

HONEST LIMIT: rule-based canonicalization is high-precision / LOW-RECALL.
Proverbs whose prescription lives purely in imagery ("A golden hammer breaks
an iron gate") match no structural template and return family='none'. This
module narrows the "needs human/model judgment" pile; it does not eliminate it.
==============================================================================
"""
import re

QUOTES = "\"'\u201c\u201d\u2018\u2019`"

# ---- RULES ------------------------------------------------------------------
# Format: (rule_id, family, compiled_pattern, replacement)
# First match wins (same engine behavior as the Wisdom Extractor).
# [WE]  = verbatim from wisdom-extractor core/canonicalize.py (v10_working)
# [EXT] = our additions targeting implicit-prescription mechanisms
_RAW_RULES = [
    # ---------------- comparisons and preferences [WE] ----------------
    ("we_better_1", "comparison",
     r"(?i)^(?:it'?s )?better (?:to )?(.+?) than (?:to )?(.+?)\.?$", r"Better \1 than \2."),
    ("we_better_2", "comparison",
     r"(?i)^(.+?) is better than (.+?)\.?$", r"Better \1 than \2."),
    ("we_prefer", "comparison",
     r"(?i)^prefer (.+?) (?:over|to) (.+?)\.?$", r"Better \1 than \2."),
    # [EXT] comparison variants common in our data
    ("ext_better_3", "comparison",
     r"(?i)^(.+?) (?:is|are) (?:worth more|more valuable) than (.+?)\.?$", r"Better \1 than \2."),
    ("ext_as_good", "comparison",
     r"(?i)^(?:one )?(.+?) is as good as (.+?)\.?$", r"Equal \1 and \2."),
    ("ext_no_x_y_good", "comparison",
     r"(?i)^(?:if |when )?(?:there (?:are|is) )?no (.+?),\s*(.+?) (?:is|are) (?:also )?good\.?$",
     r"Better \2 than nothing when no \1."),
    ("ext_best", "valuation",
     r"(?i)^(?:the )?best (.+?) (?:is|are) (.+?)\.?$", r"\2 is valued as best \1."),
    ("ext_is_rich", "valuation",
     r"(?i)^(?:a |an )?(.+?) is (?:a |an )?(rich|precious|valuable|great|good) (.+?)\.?$",
     r"\1 is valued as \2 \3."),

    # ---------------- prohibitions and avoidance [WE] ----------------
    ("we_avoid_1", "prohibition",
     r"(?i)^(?:you should )?(?:never|don'?t|do not|cannot|can'?t|avoid) (.+?)\.?$", r"Avoid \1."),
    ("we_avoid_2", "prohibition",
     r"(?i)^(?:it'?s )?(?:bad|wrong|dangerous) to (.+?)\.?$", r"Avoid \1."),

    # ---------------- conditional wisdom [WE] ----------------
    ("we_cond_1", "conditional",
     r"(?i)^(?:when|if) (.+?), (?:then )?(.+?)\.?$", r"If \1, then \2."),
    ("we_cond_2", "conditional",
     r"(?i)^where (?:there'?s|you (?:have|find)) (.+?), (?:there'?s|you (?:have|find)) (.+?)\.?$",
     r"If there is \1, there is \2."),
    # [EXT] "he who X, Y" — extremely common proverb structure, absent from [WE]
    ("ext_hewho", "conditional",
     r"(?i)^(?:he|she|they|one) who (.+?)(?:,| shall| will) (.+?)\.?$",
     r"If a person \1, then that person \2."),
    ("ext_cause", "causal",
     r"(?i)^(.+?) (?:causes?|brings?|breeds?|draws?|makes?|leads? to) (.+?)\.?$",
     r"\1 causes \2."),

    # ---------------- timing/process templates [WE] ----------------
    ("we_early", "causal",
     r"(?i)^(?:the )?early (.+?) (?:gets?|catches?) (?:the )?(.+?)\.?$", r"Early action brings \2."),
    ("we_practice", "causal",
     r"(?i)^(?:practice|repetition) makes? (?:perfect|improvement)\.?$", r"Practice improves skill."),
    ("we_time", "valuation",
     r"(?i)^time (?:is|equals?) (?:money|wealth|value)\.?$", r"Time has value."),
    ("we_hands", "causal",
     r"(?i)^(?:many|multiple|several) hands (?:make|create) (?:light work|easy work)\.?$",
     r"Cooperation reduces effort."),
    ("we_unity", "causal",
     r"(?i)^(?:together|unity) (?:we|is) (?:stand|strength)(?:, (?:divided|apart) (?:we )?fall)?\.?$",
     r"Unity creates strength."),
    ("we_excess", "causal",
     r"(?i)^too (?:many|much) (.+?) (?:spoils?|ruins?) (?:the )?(.+?)\.?$", r"Excess of \1 harms \2."),
    ("we_balance", "prohibition",
     r"(?i)^(?:all work|only work) and no play makes? (.+?) (?:a )?(?:dull|boring) (?:boy|person)\.?$",
     r"Balance work and rest."),
    ("we_haste", "causal",
     r"(?i)^(?:haste|rushing|hurrying) (?:makes|creates|causes) (?:waste|mistakes|errors)\.?$",
     r"Haste causes problems."),
    ("we_steady", "causal",
     r"(?i)^(?:slow and )?steady wins (?:the )?race\.?$", r"Consistency beats speed."),
    ("we_patience", "causal",
     r"(?i)^(?:good things|patience) (?:comes?|rewards) (?:to )?(?:those who|people who) wait\.?$",
     r"Patience brings rewards."),

    # ---------------- [EXT] bare imperatives (our biggest gap) ----------------
    # A sentence starting with a base-form advice verb and no subject = command.
    # Kept LAST so structural rules above win first (engine is first-match-wins).
    ("ext_imperative", "imperative",
     r"(?i)^((?:make|keep|look|take|give|put|let|strike|cut|count|cast|spare|seize"
     r"|know|mind|measure|hope|live|save|beware|trust|speak|act|think|leave|hold"
     r"|walk|run|judge|praise|help|prepare|waste|want|marry|choose|learn|honou?r"
     r"|respect|obey|forgive|remember|ask|answer|strike|catch) .+?)\.?$",
     r"Do: \1."),
]

_COMPILED = [(rid, fam, re.compile(pat), rep) for rid, fam, pat, rep in _RAW_RULES]


def canonicalize_with_family(s):
    """
    Returns (canonical_text, rule_family, rule_id).
    rule_family in {comparison, valuation, prohibition, conditional, causal,
                    imperative, none}.
    Engine behavior mirrors the Wisdom Extractor: cleanup -> first matching
    rule rewrites -> stop -> ensure terminal punctuation.
    """
    # -- cleanup (verbatim from Wisdom Extractor canonicalize()) --
    t = str(s).strip().strip(QUOTES).strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\u201c\u201d\u2018\u2019`]", '"', t)
    t = re.sub(r"[\u2013\u2014]", "-", t)
    t = re.sub(r"^(old saying:|proverb:|they say:?|it is said:?|english equivalent:?)\s*",
               "", t, flags=re.I)
    t = re.sub(r"\s*(- \w+ proverb|- \w+ saying)\.?$", "", t, flags=re.I)

    family, rule_id = "none", ""
    for rid, fam, rx, rep in _COMPILED:
        if rx.search(t):
            t = rx.sub(rep, t)
            family, rule_id = fam, rid
            break                       # first-match-wins, same as [WE]

    t = t.strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t, family, rule_id
