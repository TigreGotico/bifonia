"""
IPA mappings and context wordlists for Portuguese heterophonic homograph disambiguation.

IPA data is loaded from bifonia's own data/heterophonic_homographs.csv and is
European Portuguese. Per-lect marking overrides are loaded from
data/lect_overrides.csv, and the marks that need no context from
data/context_free_marks.csv.
"""

import csv
import pathlib
import unicodedata

from linguatag import with_default_region

from bifonia.vocab import voc

_CSV = pathlib.Path(__file__).parent / "data" / "heterophonic_homographs.csv"
_LECT_CSV = pathlib.Path(__file__).parent / "data" / "lect_overrides.csv"
_MARKS_CSV = pathlib.Path(__file__).parent / "data" / "context_free_marks.csv"

# Each reading of a word is a meaning slug with the parts of speech it takes at
# that pronunciation. A slug carries several when a noun and a homographic verb
# share the vowel: «forma» (shape) is the noun and 3sg «formar», both ˈfɔɾmɐ,
# written "NOUN|VERB" in the CSV.
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

# Words a POS tag cannot disambiguate (sede, molho, corte, forma, tola, bola, …).
POS_AMBIGUOUS: set = {w for w, pm in POS_SENSES.items()
                      if any(len(senses) > 1 for senses in pm.values())}


def _form(path: pathlib.Path, line: int, word: str, form: str) -> str:
    """Return *form* in NFC, or raise if a letter of *word* has no code point."""
    form = unicodedata.normalize("NFC", form)
    where = f"{path}, line {line}: the form {form!r} of {word!r}"
    if form and len(form) != len(word):
        raise ValueError(f"{where} has {len(form)} code points in NFC, "
                         f"but the word has {len(word)}")
    if any(unicodedata.category(c).startswith("M") for c in form):
        raise ValueError(f"{where} keeps a combining mark in NFC")
    return form


# Per-lect marking overrides: lect tag → {word: form}. The form replaces the
# diacritized form of every sense of the word; an empty form leaves the word
# unmarked. A lect needs rows only where its stressed vowel differs from the
# European one. add_extra_diacritics copies the case of each letter onto the code
# point at the same position, so _form requires one NFC code point per letter.
LECT_OVERRIDES: dict = {}
with _LECT_CSV.open(encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        word = row["word"]
        form = _form(_LECT_CSV, reader.line_num, word, row["form"])
        LECT_OVERRIDES.setdefault(row["lang"], {})[word] = form

# Marks that need no context: lect code → {word: form}. Each row names the
# orthography2ipa lect codes where it passed the checks of
# scripts/build_context_free_marks.py ("pt-BR|pt-BR-x-cuiaba|..."), which builds
# the file and records the evidence. The context table owns its words, so a row
# for one of them is rejected.
CONTEXT_FREE_MARKS: dict = {}
with _MARKS_CSV.open(encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        word = row["word"]
        if word in AMBIGUOUS_WORDS:
            raise ValueError(f"{_MARKS_CSV}, line {reader.line_num}: {word!r} "
                             "belongs to the context table")
        form = _form(_MARKS_CSV, reader.line_num, word, row["form"])
        for lect in row["lects"].split("|"):
            CONTEXT_FREE_MARKS.setdefault(lect, {})[word] = form


def _for_lect(table: dict, lang: str) -> dict:
    """Return the entry of *table* for the longest lect that is *lang* or a parent
    of it."""
    tag = with_default_region(lang.strip())
    matches = [lect for lect in table if tag == lect or tag.startswith(lect + "-")]
    return table[max(matches, key=len)] if matches else {}


def lect_overrides(lang: str) -> dict:
    """Return the {word: form} overrides for the lect tag *lang*.

    A tag matches an override lect when it is that lect or one of its sub-lects,
    case-insensitively and with ``_`` read as ``-``: ``pt-BR``, ``pt_br`` and
    ``pt-BR-x-rj`` all match ``pt-BR``. Surrounding whitespace is ignored, and a
    script subtag is dropped, so ``pt-Latn-BR`` also matches ``pt-BR``. ``None``,
    an empty tag and a tag that matches no lect return an empty dict, so they keep
    the default European Portuguese marking.
    """
    return _for_lect(LECT_OVERRIDES, lang) if lang else {}


def context_free_marks(lang: str) -> dict:
    """Return the {word: form} marks that need no context for the lect tag *lang*.

    A tag is read as in :func:`lect_overrides` and takes the rows of the longest
    lect code of the table that is the tag or its parent: ``pt-BR-x-sp`` takes
    the rows that passed for ``pt-BR-x-sp``, and ``pt-BR-x-foo`` the rows of
    ``pt-BR``. ``None`` and an empty tag mean ``pt-PT``, the default lect, and a tag
    without a region takes the default region of its language: ``pt`` is ``pt-PT``. The table has codes of pt-PT and pt-BR only, so every other
    tag, ``pt-AO`` and ``pt-UY`` included, returns an empty dict: no
    transcription of those lects was checked.
    """
    return _for_lect(CONTEXT_FREE_MARKS, lang or "pt-PT")


# word → IPA of its first sense with that POS. The scorer scores a POS only for
# the words in its table.
def _ipa_by_pos(pos: str) -> dict:
    return {w: HOMOGRAPHS[w][senses[0]]
            for w, pmap in POS_SENSES.items()
            for p, senses in pmap.items() if p == pos}


VERBS_IPA: dict = _ipa_by_pos("VERB")
NOUNS_IPA: dict = _ipa_by_pos("NOUN")
ADP_IPA: dict = _ipa_by_pos("ADP")
ADJ_IPA: dict = _ipa_by_pos("ADJ")

# The dominant sense of each same-POS word; in the library only scoring._resolve_sede reads it.
DEFAULT_SENSE: dict = {
    "sede": "thirst",
    "bola": "ball", "cor": "colour", "lobo": "wolf", "polo": "pole",
    "corte": "cut", "forma": "shape", "molho": "sauce", "tola": "foolish",
}

# The POS when context scoring gives no single winner.
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
    # The noun IPA (closed o) is the rare baking mould; shape and manner share
    # the open-o VERB IPA ("uma forma geométrica", "desta forma").
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
    # Closed-o noun by default; a context cue flips to the open-ɔ 1sg verb.
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
    # IPA checked against Infopédia.
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
    "bola": "NOUN", "cor": "NOUN", "lobo": "NOUN", "polo": "NOUN",
    "renovo": "NOUN", "soma": "NOUN", "força": "NOUN",
    # adverb (open) vs the pluperfect of ser/ir (closed). The adverb is the
    # frequent reading, and only a cue of the verb overrides it (scoring.py).
    "fora": "ADV",
}
assert set(DEFAULT_POS) == AMBIGUOUS_WORDS, (
    f"DEFAULT_POS missing: {AMBIGUOUS_WORDS - set(DEFAULT_POS)}"
)

# European Portuguese frequency priors, added to each POS score before the
# context signals; kept at 1 or less so any context signal wins (tuned with
# scripts/benchmark_tagger.py). "sobre" has no ADP prior: score_noun already
# gives "a/um sobre" (the sail) 12, and an ADP prior would make ties that cost
# NOUN accuracy. "pelo" is mostly por+o in EP prose, so ADP gets +1 to break
# bare-context ties. Other words have no safe prior without corpus tuning.
BASE_SCORE: dict = {
    "pelo": {"ADP": 1, "NOUN": 0, "VERB": 0},
}

# Context wordlists from locale/<lang>/*.voc (see vocab.py).

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

# Tokens that may follow a preposition, minus the contracted article forms
# (those signal NOUN).
_AFTER_PREP_EXCL = set(voc("after_prep_exclude"))
AFTER_PREP = (set(voc("after_prep_extra")) | DET | PRON) - _AFTER_PREP_EXCL
