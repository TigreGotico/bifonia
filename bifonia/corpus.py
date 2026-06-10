"""
Canonical labeled sentence corpus for Portuguese heterophonic bifoniaaph disambiguation.

CORPUS maps each ambiguous word to a dict of {POS: [sentences]} where the sentence
uses that word in that POS reading (with non-standard diacritics on the target word
to show the expected pronunciation, e.g. "pára" = VERB, "pêlo" = NOUN fur).

Usage::

    from bifonia.corpus import CORPUS, iter_records
    for word, pos, sentence in iter_records():
        ...
"""

import ast
import pathlib

_DATA = pathlib.Path(__file__).parent / "data"


def _load(name: str) -> dict:
    text = (_DATA / name).read_text(encoding="utf-8")
    return ast.literal_eval(text)


# Load agent-generated sentence groups
_GROUPS = [
    _load("grp_a.py"),
    _load("grp_b.py"),
    _load("grp_c.py"),
    _load("grp_d.py"),
    _load("grp_pps.py"),
]
# append extra_*.py files if present
for _extra_path in sorted(_DATA.glob("extra_*.py")):
    _GROUPS.append(_load(_extra_path.name))

CORPUS: dict[str, dict[str, list[str]]] = {}
for _group in _GROUPS:
    for _word, _pos_sents in _group.items():
        if _word not in CORPUS:
            CORPUS[_word] = {}
        for _pos, _sents in _pos_sents.items():
            CORPUS[_word].setdefault(_pos, []).extend(_sents)


def iter_records():
    """Yield (word, pos, sentence) triples for every sentence in CORPUS."""
    for word, pos_sents in sorted(CORPUS.items()):
        for pos, sents in sorted(pos_sents.items()):
            for sent in sents:
                yield word, pos, sent


def stats() -> dict:
    """Return per-word sentence counts."""
    return {
        word: {pos: len(sents) for pos, sents in pos_sents.items()}
        for word, pos_sents in sorted(CORPUS.items())
    }
