"""
Parametrised tests for bifonia disambiguation.

Test cases are loaded from tests/test_sentences.jsonl.
Each record: {"sentence": str, "word": str, "pos": str, "xfail": str|null}

Add new test cases by appending lines to that file — no Python edits needed.
"""

import json
import pytest
from pathlib import Path

from bifonia import tokenize, disambiguate, HOMOGRAPHS, add_extra_diacritics


# ── helpers ───────────────────────────────────────────────────────────────────

def _dis(sentence: str, word: str) -> str:
    words = tokenize(sentence.lower())
    idx = words.index(word)
    return disambiguate(words, idx)


def _expect(sentence: str, word: str, sense: str) -> None:
    ipa = _dis(sentence, word)
    expected = HOMOGRAPHS[word][sense]
    assert ipa == expected, (
        f"  sentence : {sentence!r}\n"
        f"  got      : [{ipa}]\n"
        f"  expected : {sense} = [{expected}]"
    )


# ── load test cases ───────────────────────────────────────────────────────────

_DATA_FILE = Path(__file__).parent / "test_sentences.jsonl"
_ALL = [json.loads(l) for l in _DATA_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]

_NORMAL = [c for c in _ALL if not c["xfail"]]
_XFAIL  = [c for c in _ALL if c["xfail"]]


# ── main test (regular cases) ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "case",
    _NORMAL,
    ids=[f"{c['word']}/{c['sense']}:{c['sentence'][:40]}" for c in _NORMAL],
)
def test_disambiguate(case):
    _expect(case["sentence"], case["word"], case["sense"])


# ── known-hard xfail cases ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, marks=pytest.mark.xfail(reason=c["xfail"], strict=False))
        for c in _XFAIL
    ],
    ids=[f"{c['word']}/{c['sense']}:{c['sentence'][:40]}" for c in _XFAIL],
)
def test_disambiguate_xfail(case):
    _expect(case["sentence"], case["word"], case["sense"])


# ── add_extra_diacritics smoke tests ─────────────────────────────────────────

def test_diacritics_verb_para():
    assert "pára" in add_extra_diacritics("O autocarro para em frente ao hospital.")

def test_diacritics_noun_gosto():
    assert "gôsto" in add_extra_diacritics("O gosto do vinho é excelente.")

def test_diacritics_verb_gosto():
    assert "gósto" in add_extra_diacritics("Eu gosto de música clássica.")

def test_diacritics_noun_acordo():
    assert "acôrdo" in add_extra_diacritics("O acordo foi assinado ontem.")

def test_diacritics_verb_acordo():
    assert "acórdo" in add_extra_diacritics("Eu acordo cedo todos os dias.")
