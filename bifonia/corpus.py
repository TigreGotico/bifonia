"""
Canonical labeled sentence corpus for Portuguese heterophonic homograph disambiguation.

The corpus lives in ``data/corpus.jsonl`` — one JSON record per line::

    {"word": "molho", "pos": "NOUN", "ipa": "ˈmoʎu", "sentence": "O molho de tomate ..."}

``CORPUS`` maps each ambiguous word to ``{POS: [sentences]}``; ``IPA`` maps
``(word, POS)`` to the European-Portuguese transcription that POS reading carries.

Usage::

    from bifonia.corpus import CORPUS, iter_records
    for word, pos, sentence in iter_records():
        ...
"""

import json
import pathlib

_JSONL = pathlib.Path(__file__).parent / "data" / "corpus.jsonl"

CORPUS: dict[str, dict[str, list[str]]] = {}
IPA: dict[tuple[str, str], str] = {}

with _JSONL.open(encoding="utf-8") as _fh:
    for _line in _fh:
        _line = _line.strip()
        if not _line:
            continue
        _r = json.loads(_line)
        _word, _pos, _sent = _r["word"], _r["pos"], _r["sentence"]
        CORPUS.setdefault(_word, {}).setdefault(_pos, []).append(_sent)
        if _r.get("ipa"):
            IPA[(_word, _pos)] = _r["ipa"]


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
