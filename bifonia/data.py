"""
IPA mappings and context wordlists for Portuguese heterophonic bifoniaaph disambiguation.

IPA data is loaded from tugalex's heterophonic_homographs.csv (sibling package).
"""

import csv
import pathlib

from bifonia.vocab import voc

_CSV = pathlib.Path(__file__).parent / "data" / "heterophonic_homographs.csv"

# word → {POS: IPA}  e.g. {"para": {"ADP": "ˈpɐɾɐ", "VERB": "ˈpaɾɐ"}}
HOMOGRAPHS: dict = {}
with _CSV.open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        HOMOGRAPHS.setdefault(row["word"], {})[row["pos"]] = row["ipa"]

# "sobre" ADP (about/over) shares its closed-o IPA with the NOUN (nautical sail);
# if an older CSV lacks the ADP row, fall back to the NOUN reading.
HOMOGRAPHS["sobre"].setdefault("ADP", HOMOGRAPHS["sobre"].get("NOUN", "ˈsobɾɨ"))

# derived per-POS dicts (backward-compatible)
VERBS_IPA: dict = {w: d["VERB"] for w, d in HOMOGRAPHS.items() if "VERB" in d}
NOUNS_IPA: dict = {w: d["NOUN"] for w, d in HOMOGRAPHS.items() if "NOUN" in d}
ADP_IPA: dict = {w: d["ADP"] for w, d in HOMOGRAPHS.items() if "ADP" in d}
ADJ_IPA: dict = {w: d["ADJ"] for w, d in HOMOGRAPHS.items() if "ADJ" in d}

AMBIGUOUS_WORDS: set = set(HOMOGRAPHS)

# sensible linguistic defaults when context scoring yields no winner
DEFAULT_POS: dict = {
    "para": "ADP",
    "pelo": "ADP",
    "sobre": "ADP",
    "seco": "ADJ",
    "tola": "ADJ",
    "porto": "NOUN",
    "acordo": "NOUN",
    "acerto": "NOUN",
    "cerro": "NOUN",
    "choro": "NOUN",
    "colher": "NOUN",
    "começo": "NOUN",
    "conserto": "NOUN",
    "coro": "NOUN",
    "corte": "NOUN",
    # "forma" NOUN IPA (closed-o) = baking mold only (rare).  All other NOUN
    # senses (manner, shape, geometric form) share the open-o VERB IPA.
    # Default to VERB so "uma forma geométrica" / "desta forma" get open-o.
    "forma": "VERB",
    "gozo": "NOUN",
    "gosto": "NOUN",
    "jogo": "NOUN",
    "molho": "NOUN",
    "olho": "NOUN",
    "rego": "NOUN",
    "sede": "NOUN",
    "torre": "NOUN",
    "transtorno": "NOUN",
    "peso": "NOUN",
    "posto": "NOUN",
}
assert set(DEFAULT_POS) == AMBIGUOUS_WORDS, (
    f"DEFAULT_POS missing: {AMBIGUOUS_WORDS - set(DEFAULT_POS)}"
)

# ── per-word POS bias (European Portuguese frequency priors) ──────────────────
# Integer bonus added to each POS score before signal scoring begins.
# Use to encode corpus-level frequency priors when the default tiebreaker
# (DEFAULT_POS) is not granular enough.  Tune by running benchmark_tagger.py
# and inspecting per-word accuracy.  Values are intentionally small (1–3) so
# they lose to any explicit context signal.
BASE_SCORE: dict = {
    # Small frequency priors to break near-ties.
    # Keep values ≤ 1 to avoid overriding any explicit context signal.
    # "sobre" ADP prior is intentionally 0: the DET-before+7 NOUN boost already
    # gives score_noun=12 for unambiguous "a/um sobre" (nautical) sentences, and a
    # non-zero ADP prior would create ties that degrade NOUN accuracy.
    # "pelo" most commonly ADP (por+o) in EP prose — +1 to break bare-context ties.
    "pelo": {"ADP": 1, "NOUN": 0, "VERB": 0},
    # three-way and two-way words: no safe non-zero prior without corpus tuning.
}

# ── context wordlists (loaded from locale/<lang>/*.voc — see vocab.py) ─────────
# Edit the .voc files to extend these without touching code.

STOPPABLE_THINGS = set(voc("stoppable_things"))   # "para"=parar (VERB) subjects/objects
DET   = set(voc("determiners"))                    # determiners + contracted prep+article
QUANT = set(voc("quantifiers"))                    # quantifier determiners
PRON  = set(voc("pronouns"))                        # personal / relative / clitic pronouns
AUX_VERBS = set(voc("aux_verbs"))                  # auxiliary / movement verbs
NUMERIC   = set(voc("numeric"))                    # cardinal number words
SOBRE_GOV = set(voc("sobre_governors"))            # govern "sobre" as preposition
NEVER_AFTER_PREP = set(voc("never_after_prep"))    # contracted forms invalid after ADP

PREP = set(voc("ambiguous_preps"))

BEFORE_PREP = set(voc("before_prep_extra")) | AUX_VERBS

# Tokens that may follow a preposition (ADP complements). Built from the .voc
# leaf list plus DET|PRON, minus contracted article forms (those signal NOUN).
_AFTER_PREP_EXCL = set(voc("after_prep_exclude"))
AFTER_PREP = (set(voc("after_prep_extra")) | DET | PRON) - _AFTER_PREP_EXCL
