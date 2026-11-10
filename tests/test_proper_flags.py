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


@pytest.mark.parametrize("sentence, expected", [
    ("Governo PS aprova orçamento", "Govêrno PS aprova orçamento"),
    ("Acordo UE-Mercosul entra em vigor", "Acôrdo UE-Mercosul entra em vigor"),
    ("Choro IV", "Chôro IV"),
    ("Torre D. Fernando", "Tôrre D. Fernando"),
])
def test_an_all_caps_token_after_a_title_case_word_marks_a_name(sentence, expected):
    # After a first word in title case, an all-caps token is an acronym, a roman
    # numeral or an initial. The first word is then a name: the noun, closed vowel.
    assert add_extra_diacritics(sentence) == expected


def test_a_sentence_in_capitals_has_no_name_signal():
    assert proper_flags("GOVERNO PS APROVA ORÇAMENTO") == [False, False, False, False]


@pytest.mark.parametrize("sentence", [
    "Jogo. Depois descanso.",
    "Jogo! Depois descanso.",
    "Jogo? Depois descanso.",
    "Jogo... Depois descanso.",
])
def test_a_first_word_that_ends_its_sentence_is_no_name(sentence):
    # "Jogo" is a sentence of one word and "Depois" starts the next sentence, so
    # no word is a capitalised word inside a sentence.
    assert proper_flags(sentence) == [False, False, False]


@pytest.mark.parametrize("sentence, expected", [
    # Verb "jogar", first person: "I play. Then I rest."
    ("Jogo. Depois descanso.", "Jógo. Depois descanso."),
    ("Jogo? Sim, jogo todos os sábados.", "Jógo? Sim, jógo todos os sábados."),
    # Verb "chorar": "Do I cry? I never cry in public."
    ("Choro? Nunca choro em público.", "Chóro? Nunca chóro em público."),
    # Verb "acordar": "I wake up. Then I take a shower."
    ("Acordo. Depois tomo banho.", "Acórdo. Depois tomo banho."),
])
def test_a_verb_that_ends_its_sentence_keeps_the_verb_reading(sentence, expected):
    # The verb readings have the open vowel (jogo /ˈʒɔɡu/, choro /ˈʃɔɾu/,
    # acordo /ɐˈkɔɾdu/), so they take the acute accent.
    assert add_extra_diacritics(sentence) == expected


@pytest.mark.parametrize("sentence, expected", [
    # A colon does not end a sentence: the first word labels what follows, and the
    # label is the noun. Government ɡuˈveɾnu, agreement ɐˈkoɾdu, game ˈʒoɡu.
    ("Governo: PS aprova orçamento.", "Govêrno: PS aprova orçamento."),
    ("Acordo: PS e PSD aprovam.", "Acôrdo: PS e PSD aprovam."),
    # The club "o Porto" is named after the city, the noun harbour ˈpoɾtu.
    ("Jogo: Benfica vence o Porto.", "Jôgo: Benfica vence o Pôrto."),
])
def test_a_first_word_before_a_colon_keeps_the_noun_reading(sentence, expected):
    assert add_extra_diacritics(sentence) == expected


# Readings from bifonia/data/heterophonic_homographs.csv: a noun with the closed
# vowel takes the circumflex, a verb with the open vowel takes the acute.
#   governo,government,NOUN,ɡuˈveɾnu      jogo,play,VERB,ˈʒɔɡu
#   acordo,agreement,NOUN,ɐˈkoɾdu         jogo,game,NOUN,ˈʒoɡu
#   acordo,wake,VERB,ɐˈkɔɾdu              choro,weep,VERB,ˈʃɔɾu
#   choro,weeping,NOUN,ˈʃoɾu


@pytest.mark.parametrize("sentence, expected", [
    # "Agreement: Government and unions sign." Government is a name here.
    ("Acordo: Governo e sindicatos assinam.", "Acôrdo: Govêrno e sindicatos assinam."),
    ("Atenção: Governo aprova orçamento.", "Atenção: Govêrno aprova orçamento."),
])
def test_a_capitalised_word_after_a_colon_is_inside_the_sentence(sentence, expected):
    # A colon does not end a Portuguese sentence, so the capital of "Governo"
    # marks a name, and the name takes the noun reading.
    assert add_extra_diacritics(sentence) == expected
    assert proper_flags(sentence)[1] is True


@pytest.mark.parametrize("sentence, expected", [
    # "I play… Then I rest."
    ("Jogo… Depois descanso.", "Jógo… Depois descanso."),
    # "I wake up… Then I take a shower."
    ("Acordo… Depois tomo banho.", "Acórdo… Depois tomo banho."),
    # "I cry… I never cry in public."
    ("Choro… Nunca choro em público.", "Chóro… Nunca chóro em público."),
])
def test_an_ellipsis_character_ends_a_sentence(sentence, expected):
    assert add_extra_diacritics(sentence) == expected
    assert proper_flags(sentence)[:2] == [False, False]


def test_an_ellipsis_character_gives_the_tokens_of_three_full_stops():
    assert tokenize("Jogo… Depois descanso.") == ["jogo...", "depois", "descanso."]
    assert tokenize("Jogo… Depois descanso.") == tokenize("Jogo... Depois descanso.")


@pytest.mark.parametrize("sentence, expected", [
    ("«Jogo». Depois descanso.", "«Jógo». Depois descanso."),
    ("Jogo». Depois descanso.", "Jógo». Depois descanso."),
    ("(Jogo). Depois descanso.", "(Jógo). Depois descanso."),
    ('"Jogo". Depois descanso.', '"Jógo". Depois descanso.'),
    ("“Jogo”. Depois descanso.", "“Jógo”. Depois descanso."),
    ("«Acordo». Depois tomo banho.", "«Acórdo». Depois tomo banho."),
])
def test_a_full_stop_after_a_closing_quote_or_bracket_ends_a_sentence(sentence, expected):
    # The same reading and the same flags as "Jogo. Depois descanso.".
    assert add_extra_diacritics(sentence) == expected
    assert proper_flags(sentence) == [False] * len(tokenize(sentence))


@pytest.mark.parametrize("sentence, expected, flags", [
    # "Today. Jogo Bonito won." Jogo Bonito is a name: the noun "game".
    ("Hoje. Jogo Bonito venceu.", "Hoje. Jôgo Bonito venceu.", [False, True, True, False]),
    ("Ganhámos ontem! Governo PS aprova orçamento.",
     "Ganhámos ontem! Govêrno PS aprova orçamento.",
     [False, False, True, False, False, False]),
    # Choro IV is the title of a piece of music: the noun "weeping".
    ("Ouvi o concerto. Choro IV é a minha peça preferida.",
     "Ouvi o concerto. Chôro IV é a minha peça preferida.",
     [False, False, False, True, False, False, False, False, False, False]),
    ("Ouvi o concerto. Choro n.º 13 é a minha peça preferida.",
     "Ouvi o concerto. Chôro n.º 13 é a minha peça preferida.",
     [False, False, False, True, False, False, False, False, False, False, False, False]),
])
def test_the_first_word_of_every_sentence_can_start_a_name(sentence, expected, flags):
    assert add_extra_diacritics(sentence) == expected
    assert proper_flags(sentence) == flags
