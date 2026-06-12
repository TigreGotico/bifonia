"""
IPA mappings and context wordlists for Portuguese heterophonic bifoniaaph disambiguation.

IPA data is loaded from tugalex's heterophonic_homographs.csv (sibling package).
"""

import csv
import pathlib

from bifonia.vocab import voc

_CSV = pathlib.Path(__file__).parent / "data" / "heterophonic_homographs.csv"

# The dataset is keyed on MEANING (`sense`), not POS: a word's distinct readings
# are identified by a meaning slug, because two senses can share a POS (e.g.
# "sede" thirst and seat are both nouns).  `pos` is a descriptive attribute (the
# dominant grammatical reading) used by the context scorer, and may repeat.
#
#   HOMOGRAPHS  : word → {sense: IPA}      e.g. {"sede": {"thirst": "ˈsedɨ", "seat": "ˈsɛdɨ"}}
#   SENSE_POS   : word → {sense: pos}      the descriptive POS of each sense
#   POS_SENSES  : word → {pos: [sense,…]}  senses the scorer's POS guess maps to
HOMOGRAPHS: dict = {}
SENSE_POS: dict = {}
POS_SENSES: dict = {}
with _CSV.open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        w, s, p, ipa = row["word"], row["sense"], row["pos"], row["ipa"]
        HOMOGRAPHS.setdefault(w, {})[s] = ipa
        SENSE_POS.setdefault(w, {})[s] = p
        POS_SENSES.setdefault(w, {}).setdefault(p, []).append(s)

AMBIGUOUS_WORDS: set = set(HOMOGRAPHS)


# Membership helpers for the POS scorer (word → representative IPA for that POS).
# A word "has" a POS candidate iff some sense carries it; the scorer then narrows
# to a single sense via POS_SENSES (+ resolve_sense when a POS maps to several).
def _ipa_by_pos(pos: str) -> dict:
    return {w: HOMOGRAPHS[w][senses[0]]
            for w, pmap in POS_SENSES.items()
            for p, senses in pmap.items() if p == pos}


VERBS_IPA: dict = _ipa_by_pos("VERB")
NOUNS_IPA: dict = _ipa_by_pos("NOUN")
ADP_IPA: dict = _ipa_by_pos("ADP")
ADJ_IPA: dict = _ipa_by_pos("ADJ")

# Default sense when a word's guessed POS maps to several senses and no context
# cue decides (only "sede": thirst is the prototypical reading).  See
# scoring.resolve_sense.
DEFAULT_SENSE: dict = {
    "sede": "thirst",
}

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
