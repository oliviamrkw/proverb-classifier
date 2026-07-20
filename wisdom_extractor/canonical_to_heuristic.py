"""
wisdom_extractor/canonical_to_heuristic.py
==============================================================================
CANONICAL FORM -> IMPLICIT-PRESCRIPTION HEURISTIC COLUMNS

Once a proverb has been canonicalized (see canonicalizer.py), the RULE FAMILY
that fired is a machine-checkable structural signal. This module converts that
family into hint columns aligned with our annotation scheme:

    family        canon_is_command  canon_has_valuation  canon_has_cause_result
    imperative           1                  0                     0
    prohibition          1                  0                     0
    comparison           0                  1                     0
    valuation            0                  1                     0
    conditional          0                  0                     1
    causal               0                  0                     1
    none                 0                  0                     0

and derives:
    canon_implicit_hint = 1 if ANY of the three hints is 1, else 0

IMPORTANT — what this column IS and IS NOT:
- It is a STRUCTURAL hint: "the surface form matches a prescription-bearing
  template." High precision on what it flags.
- It is NOT a full implicit-prescription label. family='none' does NOT mean
  "no prescription" — it means "no structural template matched", which
  includes all imagery-only proverbs (the golden-hammer class). Treat
  canon_implicit_hint=0 as "unknown, needs the valence/agency judgment",
  not as a negative label. This asymmetry is why the column is named _hint.
- For the conditional/causal family, a real prescription additionally needs
  the result to be non-neutral and the cause controllable (our existing
  scheme). The hint says the STRUCTURE is present, not that the full test
  passes. Downstream models should get the family + hints as FEATURES, not
  as ground truth.
==============================================================================
"""

FAMILY_TO_HINTS = {
    #  family        (is_command, has_valuation, has_cause_result)
    "imperative":   (1, 0, 0),
    "prohibition":  (1, 0, 0),
    "comparison":   (0, 1, 0),
    "valuation":    (0, 1, 0),
    "conditional":  (0, 0, 1),
    "causal":       (0, 0, 1),
    "none":         (0, 0, 0),
}


def family_to_columns(family):
    """Returns dict of the four heuristic hint columns for a rule family."""
    cmd, val, cr = FAMILY_TO_HINTS.get(family, (0, 0, 0))
    return {
        "canon_is_command": cmd,
        "canon_has_valuation": val,
        "canon_has_cause_result": cr,
        "canon_implicit_hint": 1 if (cmd or val or cr) else 0,
    }