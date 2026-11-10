"""Tests for the proper-noun flags that bias add_extra_diacritics."""
import json
from pathlib import Path

import pytest

from bifonia import (HOMOGRAPHS, _strip_edge, add_extra_diacritics, guess_sense, proper_flags,
                     tokenize)


@pytest.mark.parametrize("abbrev", ["n.º", "N.º"])
def test_numero_with_dot_matches_numero_without_dot(abbrev):
    # "n.º" and "nº" are spellings of the same abbreviation, so a title such as
    # "Choro n.º 13" must get the same first-token flag and reading as "Choro nº 13".
    assert proper_flags(f"Choro {abbrev} 13")[0] is True
    assert add_extra_diacritics(f"Choro {abbrev} 13") == f"Chôro {abbrev} 13"
    assert add_extra_diacritics("Choro nº 13") == "Chôro nº 13"


def _gold_rows_before_no():
    # Gold rows whose sentence starts with the homograph followed by "no".
    root = Path(__file__).parent
    rows = []
    for path in (root / "test_sentences.jsonl", root.parent / "hf" / "test.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            words = tokenize(case["sentence"])
            if len(words) > 1 and words[0] == case["word"] and words[1] == "no":
                rows.append(case)
    return rows


_BEFORE_NO = _gold_rows_before_no()


def test_gold_rows_before_no_are_present():
    assert len(_BEFORE_NO) == 67


@pytest.mark.parametrize("case", _BEFORE_NO, ids=[c["sentence"][:40] for c in _BEFORE_NO])
def test_contraction_no_does_not_mark_a_title(case):
    # "no" is the contraction em + o, not a number abbreviation.
    sentence = case["sentence"]
    words, flags = tokenize(sentence), proper_flags(sentence)
    sense = guess_sense(words, 0, proper=flags[0])
    assert HOMOGRAPHS[case["word"]][sense] == HOMOGRAPHS[case["word"]][case["sense"]]


@pytest.mark.parametrize("sentence, expected", [
    ("Jogo no mesmo clube de futebol desde os sete anos de idade.",
     "Jógo no mesmo clube de futebol desde os sete anos de idade."),
    ("Choro nr 13", "Chôro nr 13"),
])
def test_title_abbreviations(sentence, expected):
    assert add_extra_diacritics(sentence) == expected


def _gold_rows():
    path = Path(__file__).parent / "test_sentences.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [row for row in rows if not row["xfail"]]


@pytest.mark.parametrize("change_case", [str.upper, str.swapcase], ids=["upper", "swapcase"])
def test_letter_case_does_not_change_the_gold_reading(change_case):
    # A word written all in capitals has no case signal, so each gold sentence
    # takes the reading of its gold sense in capitals and in swapped case too.
    misses = []
    for case in _gold_rows():
        sentence = change_case(case["sentence"])
        words, flags = tokenize(sentence), proper_flags(sentence)
        idx = next(i for i, w in enumerate(words) if _strip_edge(w) == case["word"])
        readings = HOMOGRAPHS[case["word"]]
        if readings[guess_sense(words, idx, proper=flags[idx])] != readings[case["sense"]]:
            misses.append(sentence)
    assert len(misses) == 0, (len(misses), misses[:3])


@pytest.mark.parametrize("sentence, expected", [
    ("ELE ENTROU PELO PORTÃO", "ELE ENTROU PELO PORTÃO"),
    ("OLHO PARA TI", "ÓLHO PARA TI"),
    ("EU ALMOÇO às duas", "EU ALMÓÇO às duas"),
    ("CHORO Nº 13", "CHÔRO Nº 13"),
])
def test_capitals_are_no_proper_noun_signal(sentence, expected):
    assert add_extra_diacritics(sentence) == expected


@pytest.mark.parametrize("sentence, expected", [
    ("Vivo perto de Cerro Corá", [False, False, False, True, True]),
    ("VIVO PERTO DE CERRO CORÁ", [False, False, False, False, False]),
    ("Vivo perto de CERRO Corá", [False, False, False, False, True]),
    ("Cerro Corá fica longe", [True, True, False, False]),
    ("CERRO CORÁ FICA LONGE", [False, False, False, False]),
])
def test_only_a_capital_before_lower_case_letters_flags_a_name(sentence, expected):
    assert proper_flags(sentence) == expected


def test_a_name_in_mixed_case_text_keeps_the_noun_reading():
    # Cerro Corá is a place name: the noun "cerro" (hill) has the closed vowel.
    assert add_extra_diacritics("Cerro Corá fica longe") == "Cêrro Corá fica longe"
