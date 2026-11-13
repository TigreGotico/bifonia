"""Shared, dependency-free text utilities for tokenized-context analysis.

This is the lowest layer of the package: pure stdlib, no ``bifonia`` imports.
Both the rule engine (:mod:`bifonia.scoring`) and the model feature extractor
(:mod:`bifonia.features`) import these helpers, which guarantees they apply
*identical* token normalization and cue scoring — so a cue behaves the same
whether it drives a hand-written rule or a learned model feature.

Tokens reaching these helpers are assumed already lowercased (the pipeline
lowercases each sentence before tokenizing), so nothing here changes case.
"""
from __future__ import annotations

import functools
import unicodedata

# Punctuation removed from a token before any comparison.  Deliberately narrow:
# sentence/clause punctuation and quotation marks only — it never strips the
# enclitic hyphen ("torno-me", "vejo-o"), which is load-bearing POS evidence the
# scorers rely on elsewhere.
_PUNCT = str.maketrans("", "", ".,;:!?\"'()[]{}«»–—")


def strip_punct(token: str) -> str:
    """Return *token* with surrounding/embedded punctuation removed.

    Case is left untouched (inputs are already lowercased upstream).
    """
    return token.translate(_PUNCT)


def fold(text: str) -> str:
    """Return *text* with combining accents removed, for accent-insensitive cue
    matching.

    Real-world text routinely drops diacritics ("Pascoa" for "páscoa",
    "chourico" for "chouriço"), so semantic-cue lookups fold both the token and
    the cue list to the same accent-free form.  This is applied **only** to
    content cues — never to grammatical comparisons, where accents are
    meaningful ("está" vs "esta", "pôr" vs "por").
    """
    return "".join(ch for ch in unicodedata.normalize("NFD", text)
                   if unicodedata.category(ch) != "Mn")


@functools.lru_cache(maxsize=None)
def folded(cues: frozenset[str]) -> frozenset[str]:
    """Accent-folded copy of a cue set.

    Cached because the inputs are module-level constant vocabularies, so each
    distinct set is folded exactly once.
    """
    return frozenset(fold(c) for c in cues)


def cue_score(words: list[str], idx: int, cues: frozenset[str], near: int = 4) -> int:
    """Proximity-weighted count of accent-folded *cues* occurring in *words*.

    A hit within *near* tokens of the target at *idx* counts double; a farther
    hit counts once.  This lets a decisive nearby cue outweigh an incidental
    distant one while still reading sentence-wide context — recipe prose, say,
    places the disambiguating word far from the homograph.

    *cues* must already be accent-folded (pass the raw voc through
    :func:`folded` once and reuse the result).
    """
    score = 0
    for j, word in enumerate(words):
        if j != idx and fold(strip_punct(word)) in cues:
            score += 2 if abs(j - idx) <= near else 1
    return score
