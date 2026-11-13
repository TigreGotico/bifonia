"""
add_extra_diacritics adds marks and keeps every letter: removing the marks from
its output gives the input back, capitals included. orthography2ipa states this
contract for a ``mark`` stage and checks it at run time.

The expected strings are written by hand from the sense that the same sentence
gets in lower case. The table checks read ``base`` from Unicode, not from bifonia.
"""

import csv
import unicodedata

import pytest

from bifonia import _DIACRITIZED, add_extra_diacritics
from bifonia.data import _LECT_CSV, LECT_OVERRIDES


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


def test_every_table_form_has_the_letters_of_its_word():
    # add_extra_diacritics copies the case of each letter of the word onto the
    # code point of the form at the same position, so a form must be in NFC and
    # as long as its word. The C++ port reads the override file as it is, so the
    # file rows are checked, not the forms that bifonia loads from them.
    forms = {(word, form) for (word, _), form in _DIACRITIZED.items()}
    with _LECT_CSV.open(encoding="utf-8") as fh:
        forms |= {(row["word"], row["form"]) for row in csv.DictReader(fh) if row["form"]}

    assert len(forms) > 250
    assert {(word, form) for word, form in forms
            if base(form) != base(word)
            or unicodedata.normalize("NFC", form) != form
            or len(form) != len(word)} == set()
