"""
add_extra_diacritics adds marks and keeps every letter: removing the marks from
its output gives the input back, capitals included. orthography2ipa states this
contract for a ``mark`` stage and checks it at run time.

The expected strings are written by hand from the sense that the same sentence
gets in lower case. The table checks read ``base`` from Unicode, not from bifonia.
"""

import unicodedata

import pytest

from bifonia import _DIACRITIZED, add_extra_diacritics
from bifonia.data import LECT_OVERRIDES


def base(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


@pytest.mark.parametrize("sentence, expected", [
    ("A SEDE DO PARTIDO FICA NO PORTO.", "A SÉDE DO PARTIDO FICA NO PÔRTO."),
    ("TENHO MUITA SEDE", "TENHO MUITA SÊDE"),
    ("O ACORDO FOI ASSINADO", "O ACÔRDO FOI ASSINADO"),
])
def test_capitals_stay_capitals(sentence, expected):
    assert add_extra_diacritics(sentence) == expected


def test_each_letter_keeps_its_own_case():
    assert add_extra_diacritics("a SeDe do partido") == "a SéDe do partido"


@pytest.mark.parametrize("lang", ["pt-PT", "pt-BR", "pt-UY"])
def test_every_table_word_keeps_its_letters_in_capitals(lang):
    words = {word for word, _ in _DIACRITIZED}
    words |= {word for overrides in LECT_OVERRIDES.values() for word in overrides}
    outputs = {word.upper(): add_extra_diacritics(word.upper(), lang) for word in words}

    assert sum(word != out for word, out in outputs.items()) > 100
    assert {word: out for word, out in outputs.items() if base(out) != base(word)} == {}
