"""
bifonia — Portuguese heterophonic bifoniaaph disambiguation.

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

from bifonia.data import AMBIGUOUS_WORDS, HOMOGRAPHS, DEFAULT_POS
from bifonia.scoring import guess_pos
from bifonia.version import VERSION_STR

__version__ = VERSION_STR
__all__ = [
    "tokenize",
    "is_ambiguous",
    "guess_pos",
    "disambiguate",
    "add_extra_diacritics",
    "AMBIGUOUS_WORDS",
    "HOMOGRAPHS",
]


def tokenize(text: str) -> list:
    """Lowercase and split *text* into word tokens (strips punctuation)."""
    return re.findall(r"\w+", text.lower(), re.UNICODE)


def is_ambiguous(word: str) -> bool:
    """Return True if *word* is a known heterophonic bifoniaaph."""
    return word.lower() in AMBIGUOUS_WORDS


def disambiguate(words: list, idx: int, pos: str = None) -> str:
    """Return the IPA transcription for the ambiguous word at position *idx*.

    Parameters
    ----------
    words:
        Tokenized (lowercased) sentence as returned by :func:`tokenize`.
    idx:
        Index of the ambiguous word within *words*.
    pos:
        Override the POS guessing with an explicit UDEP tag
        (``"ADP"``, ``"NOUN"``, ``"VERB"``, ``"ADJ"``).

    Raises
    ------
    ValueError
        If the word is not a known bifoniaaph.
    KeyError
        If *pos* is given but has no IPA entry for this word.
    """
    word = words[idx]
    if not is_ambiguous(word):
        raise ValueError(f"{word!r} is not a known heterophonic bifoniaaph")

    resolved_pos = pos or guess_pos(words, idx)
    return HOMOGRAPHS[word][resolved_pos]


# ── diacritics lookup ─────────────────────────────────────────────────────────
# Non-canonical, non-AO1990 forms that force the correct vowel quality in a
# downstream rule-based G2P (convention: acute = open, circumflex = closed).
# Words with no meaningful diacritic change (ADP para/pelo) are omitted so
# add_extra_diacritics returns them unchanged.
_DIACRITIZED: dict = {
    ("acordo",     "NOUN"): "acôrdo",
    ("acordo",     "VERB"): "acórdo",
    ("acerto",     "NOUN"): "acêrto",
    ("acerto",     "VERB"): "acérto",
    ("cerro",      "NOUN"): "cêrro",
    ("cerro",      "VERB"): "cérro",
    ("choro",      "NOUN"): "chôro",
    ("choro",      "VERB"): "chóro",
    ("colher",     "NOUN"): "colhér",   # noun = open ɛ
    ("colher",     "VERB"): "colhêr",   # verb = closed e
    ("começo",     "NOUN"): "comêço",
    ("começo",     "VERB"): "coméço",
    ("conserto",   "NOUN"): "consêrto",
    ("conserto",   "VERB"): "consérto",
    ("coro",       "NOUN"): "côro",
    ("coro",       "VERB"): "córo",
    ("corte",      "NOUN"): "côrte",
    ("corte",      "VERB"): "córte",
    ("forma",      "NOUN"): "fôrma",
    ("forma",      "VERB"): "fórma",
    ("gosto",      "NOUN"): "gôsto",
    ("gosto",      "VERB"): "gósto",
    ("gozo",       "NOUN"): "gôzo",
    ("gozo",       "VERB"): "gózo",
    ("jogo",       "NOUN"): "jôgo",
    ("jogo",       "VERB"): "jógo",
    ("molho",      "NOUN"): "môlho",
    ("molho",      "VERB"): "mólho",
    ("olho",       "NOUN"): "ôlho",
    ("olho",       "VERB"): "ólho",
    ("para",       "VERB"): "pára",
    ("pelo",       "NOUN"): "pêlo",
    ("pelo",       "VERB"): "pélo",
    ("peso",       "NOUN"): "pêso",
    ("peso",       "VERB"): "péso",
    ("porto",      "NOUN"): "pôrto",
    ("porto",      "VERB"): "pórto",
    ("posto",      "NOUN"): "pôsto",
    ("posto",      "VERB"): "pósto",
    ("rego",       "NOUN"): "rêgo",
    ("rego",       "VERB"): "régo",
    ("seco",       "ADJ"):  "sêco",
    ("seco",       "VERB"): "séco",
    ("sede",       "NOUN"): "séde",     # noun = open ɛ
    ("sede",       "VERB"): "sêde",     # verb = closed e
    ("sobre",      "NOUN"): "sôbre",
    ("sobre",      "VERB"): "sóbre",
    ("tola",       "NOUN"): "tóla",     # noun = open ɔ
    ("tola",       "ADJ"):  "tôla",     # adj  = closed o
    ("torre",      "NOUN"): "tôrre",
    ("torre",      "VERB"): "tórre",
    ("transtorno", "NOUN"): "transtôrno",
    ("transtorno", "VERB"): "transtórno",
}


# Reverse map: diacritized form → base word (used for test normalization and
# for accepting already-diacritized input back through the pipeline).
_DIACRITIZED_TO_BASE: dict = {v: k[0] for k, v in _DIACRITIZED.items()}


def add_extra_diacritics(sentence: str) -> str:
    """Return *sentence* with non-canonical diacritics on heterophonic bifoniaaphs.

    The inserted diacritics are not AO1990-compliant but force a downstream
    rule-based G2P to emit the correct vowel quality:

    - acute  (ó/é) → open  /ɔ/ or /ɛ/  (typically verb reading)
    - circumflex (ô/ê) → closed /o/ or /e/  (typically noun reading)

    Words that are not ambiguous, or whose POS reading needs no extra diacritic
    (e.g. ADP *para*, ADP *pelo*), are returned unchanged.
    """
    words = tokenize(sentence)
    output = sentence
    for i, word in enumerate(words):
        if is_ambiguous(word):
            pos = guess_pos(words, i)
            diacritized = _DIACRITIZED.get((word, pos))
            if diacritized:
                output = output.replace(word, diacritized, 1)
    return output
