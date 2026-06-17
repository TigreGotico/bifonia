"""
IPA mappings and context wordlists for Portuguese heterophonic bifoniaaph disambiguation.

IPA data is loaded from tugalex's heterophonic_homographs.csv (sibling package).
"""

import csv
import pathlib

from bifonia.vocab import voc

_CSV = pathlib.Path(__file__).parent / "data" / "heterophonic_homographs.csv"

# The dataset is keyed on (MEANING, POS) combos: a word's distinct readings are
# identified by a meaning slug, and each slug carries the set of parts of speech
# it can realise at its pronunciation.  Most readings are single-POS, but a slug
# may carry several when a noun reading and a homographic verb share one vowel —
# e.g. «forma» (shape) is the noun "a forma" AND the verb «formar» (3sg "forma"),
# both open-o ˈfɔɾmɐ → pos = {NOUN, VERB}.  The CSV encodes a multi-POS slug as
# "NOUN|VERB".  A word is POS-ambiguous (a tagger cannot pick the reading) iff
# some POS maps to more than one sense — that is the set the hybrid ensemble
# routes to the rules instead of to a tagger.
#
#   HOMOGRAPHS   : word → {sense: IPA}       e.g. {"sede": {"thirst": "ˈsedɨ", "seat": "ˈsɛdɨ"}}
#   SENSE_POS    : word → {sense: pos}        primary (first) POS of each sense
#   SENSE_POSES  : word → {sense: (pos,…)}    full POS set of each (sense,POS) combo
#   POS_SENSES   : word → {pos: [sense,…]}    senses a tagger's POS maps to
#   POS_AMBIGUOUS: words where some POS maps to ≥2 senses (postag cannot resolve)
HOMOGRAPHS: dict = {}
SENSE_POS: dict = {}
SENSE_POSES: dict = {}
POS_SENSES: dict = {}
with _CSV.open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        w, s, ipa = row["word"], row["sense"], row["ipa"]
        poses = tuple(p for p in row["pos"].split("|") if p)
        HOMOGRAPHS.setdefault(w, {})[s] = ipa
        SENSE_POS.setdefault(w, {})[s] = poses[0]
        SENSE_POSES.setdefault(w, {})[s] = poses
        for p in poses:
            POS_SENSES.setdefault(w, {}).setdefault(p, []).append(s)

AMBIGUOUS_WORDS: set = set(HOMOGRAPHS)

# Words a part-of-speech tag cannot disambiguate (some POS maps to ≥2 senses):
# the genuine same-POS / overlapping-POS heterophones (e.g. sede, molho,
# corte, forma, tola, bola, cor, lobo, polo).
POS_AMBIGUOUS: set = {w for w, pm in POS_SENSES.items()
                      if any(len(senses) > 1 for senses in pm.values())}


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
    "bola": "ball", "cor": "colour", "lobo": "wolf", "polo": "pole",
    "corte": "cut", "forma": "shape", "molho": "sauce", "tola": "foolish",
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
    # noun/verb -o alternation words (closed-o noun citation form is the default;
    # explicit context cues flip to the open-ɔ verb reading).
    "torno": "NOUN",      # «o torno» / «em torno de» (closed) vs «eu torno» (open)
    "troco": "NOUN",      # «o troco» (closed) vs «eu troco» (open)
    "toco": "NOUN",       # «o toco» (closed) vs «eu toco» (open)
    "rogo": "NOUN",       # «o rogo» / «a rogo de» (closed) vs «eu rogo» (open)
    "contorno": "NOUN",   # «o contorno» (closed) vs «eu contorno» (open)
    "entorno": "NOUN",    # «o entorno» (closed) vs «eu entorno» (open)
    # choco: adjective (ovo choco) and noun (cuttlefish) share closed-o; the
    # verb «eu choco» is the only open-ɔ reading. Default to the adjective.
    "choco": "ADJ",
    # deverbal noun (closed) vs 1sg -ar/-er verb (open) — IPA from infopedia.
    "almoço": "NOUN", "rolo": "NOUN", "soco": "NOUN",
    "esforço": "NOUN", "conforto": "NOUN", "aborto": "NOUN", "adorno": "NOUN",
    "reforço": "NOUN", "soldo": "NOUN", "esboço": "NOUN",
    "governo": "NOUN", "emprego": "NOUN", "selo": "NOUN", "gelo": "NOUN",
    # wave 3 (agent-discovered, infopedia-validated; n/v slugs)
    "sopro": "NOUN",
    "forro": "NOUN",
    "dobro": "NOUN",
    "abono": "NOUN",
    "logro": "NOUN",
    "topo": "NOUN",
    "jorro": "NOUN",
    "golfo": "NOUN",
    "colmo": "NOUN",
    "fosso": "NOUN",
    "toldo": "NOUN",
    "arrojo": "NOUN",
    "despojo": "NOUN",
    "cobro": "NOUN",
    "domo": "NOUN",
    "sobro": "NOUN",
    "retorno": "NOUN",
    "desgosto": "NOUN",
    "desconforto": "NOUN",
    "desdobro": "NOUN",
    "redobro": "NOUN",
    "reboco": "NOUN",
    "decoro": "NOUN",
    "estofo": "NOUN",
    "destroço": "NOUN",
    "desafogo": "NOUN",
    "arroto": "NOUN",
    "consolo": "NOUN",
    "desconsolo": "NOUN",
    "desacordo": "NOUN",
    "engodo": "NOUN",
    "escorço": "NOUN",
    "desaforo": "NOUN",
    "abrolho": "NOUN",
    "rola": "NOUN",
    "solto": "ADJ",
    "encosto": "NOUN",
    "apelo": "NOUN",
    "desprezo": "NOUN",
    "enredo": "NOUN",
    "desenredo": "NOUN",
    "espeto": "NOUN",
    "aperto": "NOUN",
    "desvelo": "NOUN",
    "repelo": "NOUN",
    "arrepelo": "NOUN",
    "congelo": "NOUN",
    "arremesso": "NOUN",
    "arremedo": "NOUN",
    "despego": "NOUN",
    "desapego": "NOUN",
    "apego": "NOUN",
    "desemprego": "NOUN",
    "desmantelo": "NOUN",
    "atropelo": "NOUN",
    "degelo": "NOUN",
    "desgelo": "NOUN",
    "desespero": "NOUN",
    "escabelo": "NOUN",
    "novelo": "NOUN",
    "relevo": "NOUN",
    "aceno": "NOUN",
    "sopeso": "NOUN",
    "empeno": "NOUN",
    "desempeno": "NOUN",
    "sossego": "NOUN",
    "desassossego": "NOUN",
    "esmero": "NOUN",
    "azedo": "ADJ",
    "desempeço": "NOUN",
    "desemperro": "NOUN",
    "emperro": "NOUN",
    "zelo": "NOUN", "cerco": "NOUN", "erro": "NOUN", "tempero": "NOUN",
    # diacritic-collapse + deverbal additions
    "bola": "NOUN", "cor": "NOUN", "lobo": "NOUN", "polo": "NOUN",
    "renovo": "NOUN", "soma": "NOUN", "força": "NOUN",
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
