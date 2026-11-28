"""Text helpers shared by the rule engine and the model feature extractor.

Both :mod:`bifonia.scoring` and :mod:`bifonia.features` use these, so a cue
gives the same evidence to a rule and to a learned feature. Pure stdlib, no
``bifonia`` imports. Tokens arrive lowercased, so nothing here changes case.
"""
from __future__ import annotations

import functools
import unicodedata

# Sentence and clause punctuation and quotation marks only. The enclitic hyphen
# ("torno-me") stays: the scorers read it as verb evidence.
_PUNCT = str.maketrans("", "", ".,;:!?\"'()[]{}«»–—")

# Endings of a nominalised verb phrase ("para análise", "para revisão").
DEVERBAL_SUFFIXES = ("ção", "são", "gem", "ura", "ência", "ância", "mento", "ismo", "ise")

_VERB_ENDINGS = ("ou", "eu", "iu", "ei", "ava", "ara", "era")
_VOWELS = frozenset("aeiouáéíóúâêîôûãõàèìòùäëïöü")


def strip_punct(token: str) -> str:
    """Return *token* with every mark of ``_PUNCT`` removed, wherever it is; case is kept."""
    return token.translate(_PUNCT)


def fold(text: str) -> str:
    """Return *text* without combining accents, for accent-insensitive cue matching.

    Real text often drops accents ("Pascoa", "chourico"). Fold only content cues,
    never grammatical words, where the accent is meaningful ("está"/"esta").
    """
    return "".join(ch for ch in unicodedata.normalize("NFD", text)
                   if unicodedata.category(ch) != "Mn")


@functools.lru_cache(maxsize=None)
def folded(cues: frozenset[str]) -> frozenset[str]:
    """Return an accent-folded copy of a cue set (cached: the sets are module constants)."""
    return frozenset(fold(c) for c in cues)


def cue_score(words: list[str], idx: int, cues: frozenset[str], near: int = 4) -> int:
    """Return the proximity-weighted count of *cues* in *words*, the target at *idx* excluded.

    A hit within *near* tokens of *idx* counts 2 and a farther hit counts 1, so a
    near cue beats a distant one while recipe prose can still place its cue far
    away. *cues* must already be accent-folded with :func:`folded`.
    """
    score = 0
    for j, word in enumerate(words):
        if j != idx and fold(strip_punct(word)) in cues:
            score += 2 if abs(j - idx) <= near else 1
    return score


def looks_verb(word: str) -> bool:
    """Return True when *word* has a finite-verb ending: a past tense, or an
    imperfect "-ia" after a consonant ("comia", but not "areia")."""
    return (word.endswith(_VERB_ENDINGS)
            or (word.endswith("ia") and len(word) >= 4 and word[-3] not in _VOWELS))
