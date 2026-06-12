"""
Canonical labeled sentence corpus for Portuguese heterophonic homograph disambiguation.

The corpus lives in ``data/corpus.jsonl`` — one JSON record per line::

    {"word": "molho", "pos": "NOUN", "ipa": "ˈmoʎu", "sentence": "O molho de tomate ..."}

``CORPUS`` maps each ambiguous word to ``{sense: [sentences]}``; ``IPA`` maps
``(word, sense)`` to the European-Portuguese transcription that reading carries.
The bucket key is MEANING (`sense`), not POS, so two senses sharing a POS
(``sede`` thirst vs seat — both nouns) each get their own bucket.

Usage::

    from bifonia.corpus import CORPUS, iter_records
    for word, sense, sentence in iter_records():
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
        _word, _sense, _sent = _r["word"], _r["sense"], _r["sentence"]
        CORPUS.setdefault(_word, {}).setdefault(_sense, []).append(_sent)
        if _r.get("ipa"):
            IPA[(_word, _sense)] = _r["ipa"]


def iter_records():
    """Yield (word, sense, sentence) triples for every sentence in CORPUS."""
    for word, sense_sents in sorted(CORPUS.items()):
        for sense, sents in sorted(sense_sents.items()):
            for sent in sents:
                yield word, sense, sent


def stats() -> dict:
    """Return per-word sentence counts, keyed by sense."""
    return {
        word: {sense: len(sents) for sense, sents in sense_sents.items()}
        for word, sense_sents in sorted(CORPUS.items())
    }
