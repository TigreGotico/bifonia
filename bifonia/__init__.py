"""
bifonia — Portuguese heterophonic homograph disambiguation.

Identifies the correct IPA pronunciation of words whose spelling is identical
but whose phonology depends on their part of speech (e.g. NOUN vs VERB).

Quick start::

    from bifonia import tokenize, is_ambiguous, disambiguate

    sentence = "O autocarro para em frente ao hospital."
    words = tokenize(sentence)
    for i, w in enumerate(words):
        if is_ambiguous(w):
            ipa = disambiguate(words, i)
            print(f"{w} → [{ipa}]")
"""

import re
from functools import lru_cache as _lru_cache

from bifonia.data import (
    AMBIGUOUS_WORDS, HOMOGRAPHS, DEFAULT_POS, POS_SENSES, SENSE_POS,
)
from bifonia.scoring import (
    guess_pos as _scoring_guess_pos,
    resolve_sense as _resolve_sense,
)
from bifonia.version import VERSION_STR

__version__ = VERSION_STR
__all__ = [
    "tokenize",
    "is_ambiguous",
    "guess_pos",
    "guess_sense",
    "disambiguate",
    "add_extra_diacritics",
    "AMBIGUOUS_WORDS",
    "HOMOGRAPHS",
]


def tokenize(text: str) -> list:
    """Lowercase and split *text* into word tokens (strips punctuation)."""
    return re.findall(r"\w+", text.lower(), re.UNICODE)


def is_ambiguous(word: str) -> bool:
    """Return True if *word* is a known heterophonic homograph."""
    return word.lower() in AMBIGUOUS_WORDS


def disambiguate(words: list, idx: int, pos: str = None, sense: str = None) -> str:
    """Return the IPA transcription for the ambiguous word at position *idx*.

    The reading is selected by MEANING (`sense`), not POS: most words have one
    sense per POS, so a POS guess resolves directly, but words like ``sede``
    (thirst vs seat — both nouns) are disambiguated by context meaning cues.

    Parameters
    ----------
    words:
        Tokenized (lowercased) sentence as returned by :func:`tokenize`.
    idx:
        Index of the ambiguous word within *words*.
    pos:
        Override POS guessing with an explicit UDEP tag (``"ADP"``, ``"NOUN"``,
        ``"VERB"``, ``"ADJ"``); the sense is then resolved within that POS.
    sense:
        Override entirely with an explicit meaning slug (see the CSV / HOMOGRAPHS).

    Raises
    ------
    ValueError
        If the word is not a known homograph.
    KeyError
        If *sense* is given but has no IPA entry for this word.
    """
    token = words[idx]
    # Accept pre-AO1990 / diacritized tokens by normalising to base form first.
    word = _DIACRITIZED_TO_BASE.get(token, token)
    if word not in AMBIGUOUS_WORDS:
        raise ValueError(f"{token!r} is not a known heterophonic homograph")

    if sense is None:
        sense = guess_sense(words, idx, pos=pos)
    return HOMOGRAPHS[word][sense]


# ── diacritics lookup ─────────────────────────────────────────────────────────
# Non-canonical, non-AO1990 forms that force the correct vowel quality in a
# downstream rule-based G2P (convention: acute = open, circumflex = closed).
# Words with no meaningful diacritic change (ADP para/pelo) are omitted so
# add_extra_diacritics returns them unchanged.
_DIACRITIZED: dict = {
    ("acordo",     "agreement"):  "acôrdo",
    ("acordo",     "wake"):       "acórdo",
    ("acerto",     "settlement"): "acêrto",
    ("acerto",     "adjust"):     "acérto",
    ("cerro",      "hill"):       "cêrro",
    ("cerro",      "shut"):       "cérro",
    ("choro",      "weeping"):    "chôro",
    ("choro",      "weep"):       "chóro",
    ("colher",     "spoon"):      "colhér",   # noun = open ɛ
    ("colher",     "harvest"):    "colhêr",   # verb = closed e
    ("começo",     "beginning"):  "comêço",
    ("começo",     "begin"):      "coméço",
    ("conserto",   "repair"):     "consêrto",
    ("conserto",   "mend"):       "consérto",
    ("coro",       "choir"):      "côro",
    ("coro",       "blush"):      "córo",
    ("corte",      "court"):      "côrte",
    ("corte",      "cut"):        "córte",
    ("forma",      "mould"):      "fôrma",
    ("forma",      "shape"):      "fórma",
    ("gosto",      "taste"):      "gôsto",
    ("gosto",      "like"):       "gósto",
    ("gozo",       "enjoyment"):  "gôzo",
    ("gozo",       "enjoy"):      "gózo",
    ("jogo",       "game"):       "jôgo",
    ("jogo",       "play"):       "jógo",
    ("molho",      "sauce"):      "môlho",
    ("molho",      "bundle"):     "mólho",
    ("olho",       "eye"):        "ôlho",
    ("olho",       "look"):       "ólho",
    ("para",       "stop"):       "pára",
    ("pelo",       "hair"):       "pêlo",
    ("pelo",       "peel"):       "pélo",
    ("peso",       "weight"):     "pêso",
    ("peso",       "weigh"):      "péso",
    ("porto",      "harbour"):    "pôrto",
    ("porto",      "carry"):      "pórto",
    ("posto",      "station"):    "pôsto",
    ("posto",      "post"):       "pósto",
    ("rego",       "furrow"):     "rêgo",
    ("rego",       "water"):      "régo",
    ("seco",       "dry"):        "sêco",
    ("seco",       "dry_vb"):     "séco",
    ("sede",       "seat"):       "séde",     # seat (HQ) = open ɛ
    ("sede",       "thirst"):     "sêde",     # thirst    = closed e
    ("sobre",      "sail"):       "sôbre",
    ("sobre",      "leftover"):   "sóbre",
    ("tola",       "head"):       "tóla",     # head/wood = open ɔ
    ("tola",       "foolish"):    "tôla",     # adj       = closed o
    ("torre",      "tower"):      "tôrre",
    ("torre",      "roast"):      "tórre",
    ("transtorno", "disorder"):   "transtôrno",
    ("transtorno", "upset"):      "transtórno",
}


# Reverse maps from diacritized form:
#   → base word   (for test normalisation and pre-AO1990 input stripping)
#   → sense slug  (for unambiguous pre-AO1990 / already-diacritized input)
_DIACRITIZED_TO_BASE:  dict = {v: k[0] for k, v in _DIACRITIZED.items()}
_DIACRITIZED_TO_SENSE: dict = {v: k[1] for k, v in _DIACRITIZED.items()}


@_lru_cache(maxsize=1)
def _sense_model():
    """Lazy-load the learned sense model; None if absent (→ rule fallback)."""
    try:
        from bifonia.model import SenseModel
        return SenseModel.load()
    except Exception:
        return None


def _rule_sense(base: str, words: list, idx: int, pos: str = None) -> str:
    resolved_pos = pos or _scoring_guess_pos(words, idx)
    return _resolve_sense(base, words, idx, resolved_pos)


def guess_sense(words: list, idx: int, pos: str = None) -> str:
    """Return the most likely MEANING slug for the ambiguous word at *idx*.

    If the token is already a diacritized form (pre-AO1990 orthography or output
    of :func:`add_extra_diacritics`), the sense is read straight off the diacritic
    (e.g. *séde* → seat, *sêde* → thirst, *pára* → stop).

    Otherwise, where a trained statistical model covers the word and is flagged to
    win over the rules (``route == "model"`` with sufficient margin), the model
    predicts the sense; every other case falls back to the corpus-free rule engine.
    An explicit *pos* override always uses the rule resolver.
    """
    token = words[idx]
    if token in _DIACRITIZED_TO_SENSE:
        return _DIACRITIZED_TO_SENSE[token]
    base = _DIACRITIZED_TO_BASE.get(token, token)
    normalised = words
    if base != token:                       # normalise a diacritized base form
        normalised = list(words)
        normalised[idx] = base
    if pos is not None:
        return _rule_sense(base, normalised, idx, pos)
    model = _sense_model()
    if (model is not None and model.has(base) and model.route(base) == "model"
            and model.margin(base, normalised, idx) >= model.margin_tau(base)):
        return model.predict(base, normalised, idx)
    return _rule_sense(base, normalised, idx)


def guess_pos(words: list, idx: int) -> str:
    """Return the most likely UDEP POS tag for the ambiguous word at *idx*.

    Thin wrapper over :func:`guess_sense` that maps the resolved meaning back to
    its descriptive POS, preserving the pre-existing POS-tagging interface.
    Diacritized input (e.g. *pára*, *acôrdo*) is resolved without context scoring.
    """
    token = words[idx]
    base = _DIACRITIZED_TO_BASE.get(token, token)
    if token in _DIACRITIZED_TO_SENSE:
        return SENSE_POS[base][_DIACRITIZED_TO_SENSE[token]]
    if base != token:
        normalised = list(words)
        normalised[idx] = base
        return _scoring_guess_pos(normalised, idx)
    return _scoring_guess_pos(words, idx)


def add_extra_diacritics(sentence: str) -> str:
    """Return *sentence* with non-canonical diacritics on heterophonic homographs.

    The inserted diacritics are not AO1990-compliant but force a downstream
    rule-based G2P to emit the correct vowel quality:

    - acute  (ó/é) → open  /ɔ/ or /ɛ/  (typically verb reading)
    - circumflex (ô/ê) → closed /o/ or /e/  (typically noun reading)

    Words that are not ambiguous, or whose POS reading needs no extra diacritic
    (e.g. ADP *para*, ADP *pelo*), are returned unchanged. Original capitalisation
    is preserved, so a sentence-initial homograph keeps its capital (*Acórdo …*).
    """
    words = tokenize(sentence)
    replacements = {}
    for i, word in enumerate(words):
        if is_ambiguous(word):
            diacritized = _DIACRITIZED.get((word, guess_sense(words, i)))
            if diacritized:
                replacements[i] = diacritized
    if not replacements:
        return sentence

    # Walk the same \w+ tokens over the original string so each replacement lands
    # on the right occurrence with its case preserved (a plain str.replace would
    # miss a capitalised token and could hit a substring of another word).
    out, pos, idx = [], 0, 0
    for m in re.finditer(r"\w+", sentence, re.UNICODE):
        out.append(sentence[pos:m.start()])
        tok = m.group(0)
        diac = replacements.get(idx)
        if diac is not None and tok[:1].isupper():
            diac = diac[:1].upper() + diac[1:]
        out.append(diac if diac is not None else tok)
        pos, idx = m.end(), idx + 1
    out.append(sentence[pos:])
    return "".join(out)
