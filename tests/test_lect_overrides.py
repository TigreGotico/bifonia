"""
Tests for per-lect marking in add_extra_diacritics (``lang=``).

The pt-BR expectations come from Brazilian sources, not from bifonia's table:
Wiktionary gives Brazil a closed nasal stressed vowel for soma, abono, aceno and
domo in every sense (/ˈsõ.mɐ/, /aˈbõ.nu/, /aˈsẽ.nu/, /ˈdõ.mu/), and the same rule
applies to empeno and desempeno. soma is written closed (sôma), because an
unmarked soma takes the open vowel of the tugalex rjx and spx lexicons. tola keeps
the European marks: bifonia's head sense also covers the timber, which no
Brazilian source covers.

pt-UY takes the Brazilian marks. No source states the rule for Uruguayan
Portuguese; it is inferred from its rural Rio Grande do Sul vowel system.
"""

import csv
import importlib.util
import json
import re
import shutil
import unicodedata
from pathlib import Path

import pytest

import bifonia.data
from bifonia import add_extra_diacritics, AMBIGUOUS_WORDS

BR_LANGS = ["pt-BR", "pt-BR-x-rj", "pt-BR-x-sp", "pt_br", " pt-BR ", "pt-Latn-BR"]
UY_LANGS = ["pt-UY", "pt-UY-x-rivera", "pt_uy", " pt-UY ", "pt-Latn-UY"]
EUROPEAN_LANGS = [None, "", "pt-PT", "pt-PT-x-lbx", "pt-AO", "pt-brx", "pt-uyx",
                  "pt-Latn", "en"]

# (sentence, European output, Brazilian output)
CASES = [
    ("Ele soma os números de cabeça em poucos segundos.",
     "Ele sóma os números de cabeça em poucos segundos.",
     "Ele sôma os números de cabeça em poucos segundos."),
    ("A soma das parcelas dá cem euros certos.",
     "A sôma das parcelas dá cem euros certos.",
     "A sôma das parcelas dá cem euros certos."),
    ("O abono de transporte foi pedido no início do mês.",
     "O abôno de transporte foi pedido no início do mês.",
     "O abono de transporte foi pedido no início do mês."),
    ("abono a fatura do fornecedor",
     "abóno a fatura do fornecedor",
     "abono a fatura do fornecedor"),
    ("Dei um aceno de despedida enquanto o comboio se afastava lentamente na estação.",
     "Dei um acêno de despedida enquanto o comboio se afastava lentamente na estação.",
     "Dei um aceno de despedida enquanto o comboio se afastava lentamente na estação."),
    ("aceno-te da janela todas as manhãs.",
     "acéno-te da janela todas as manhãs.",
     "aceno-te da janela todas as manhãs."),
    ("O domo da sinagoga tem um design octogonal.",
     "O dômo da sinagoga tem um design octogonal.",
     "O domo da sinagoga tem um design octogonal."),
    ("Domo o touro campeão que venceu todas as corridas.",
     "Dómo o touro campeão que venceu todas as corridas.",
     "Domo o touro campeão que venceu todas as corridas."),
    ("O empeno da madeira deixou a porta torta.",
     "O empêno da madeira deixou a porta torta.",
     "O empeno da madeira deixou a porta torta."),
    ("Empeno a tampa do frasco com a humidade.",
     "Empéno a tampa do frasco com a humidade.",
     "Empeno a tampa do frasco com a humidade."),
    ("O processo de desempeno das vigas de aço é demorado.",
     "O processo de desempêno das vigas de aço é demorado.",
     "O processo de desempeno das vigas de aço é demorado."),
    ("Eu desempeno o meu dever com cuidado.",
     "Eu desempéno o meu dever com cuidado.",
     "Eu desempeno o meu dever com cuidado."),
    ("Bateu com a tola na porta do carro ao sair e ficou com uma nódoa negra.",
     "Bateu com a tóla na porta do carro ao sair e ficou com uma nódoa negra.",
     "Bateu com a tóla na porta do carro ao sair e ficou com uma nódoa negra."),
    ("A madeira de tola é leve e resistente, ideal para móveis de jardim.",
     "A madeira de tóla é leve e resistente, ideal para móveis de jardim.",
     "A madeira de tóla é leve e resistente, ideal para móveis de jardim."),
    ("Não sejas tola e pensa bem antes de decidir.",
     "Não sejas tôla e pensa bem antes de decidir.",
     "Não sejas tôla e pensa bem antes de decidir."),
    # para and pelo keep the European marks: the marked senses have the same
    # vowel in Brazil, and the prepositions stay unmarked in both lects.
    ("O autocarro para em frente ao hospital.",
     "O autocarro pára em frente ao hospital.",
     "O autocarro pára em frente ao hospital."),
    ("Vou para casa depois do trabalho.",
     "Vou para casa depois do trabalho.",
     "Vou para casa depois do trabalho."),
    ("O cão tem o pelo muito macio.",
     "O cão tem o pêlo muito macio.",
     "O cão tem o pêlo muito macio."),
    ("Eu pelo as batatas antes de as cozer.",
     "Eu pélo as batatas antes de as cozer.",
     "Eu pélo as batatas antes de as cozer."),
    ("Passou pelo parque a correr.",
     "Passou pelo parque a correr.",
     "Passou pelo parque a correr."),
]
_IDS = [c[0][:40] for c in CASES]

# Words whose Brazilian marking differs from the European one.
BR_WORDS = {"soma", "abono", "aceno", "domo", "empeno", "desempeno"}


@pytest.mark.parametrize("lang", BR_LANGS)
@pytest.mark.parametrize("sentence,_european,brazilian", CASES, ids=_IDS)
def test_pt_br_marking(sentence, _european, brazilian, lang):
    assert add_extra_diacritics(sentence, lang=lang) == brazilian


@pytest.mark.parametrize("lang", UY_LANGS)
@pytest.mark.parametrize("sentence,_european,brazilian", CASES, ids=_IDS)
def test_pt_uy_takes_brazilian_marking(sentence, _european, brazilian, lang):
    assert add_extra_diacritics(sentence, lang=lang) == brazilian


@pytest.mark.parametrize("sentence,european,uruguayan", [
    ("Ele soma os números.", "Ele sóma os números.", "Ele sôma os números."),
    ("Eu aceno da janela.", "Eu acéno da janela.", "Eu aceno da janela."),
])
def test_pt_uy_closes_the_vowel_pt_pt_keeps_it_open(sentence, european, uruguayan):
    assert add_extra_diacritics(sentence, lang="pt-UY") == uruguayan
    assert add_extra_diacritics(sentence, lang="pt-PT") == european


@pytest.mark.parametrize("lang", EUROPEAN_LANGS)
@pytest.mark.parametrize("sentence,european,_brazilian", CASES, ids=_IDS)
def test_other_lects_keep_european_marking(sentence, european, _brazilian, lang):
    assert add_extra_diacritics(sentence) == european
    assert add_extra_diacritics(sentence, lang=lang) == european


def test_pt_br_differs_only_on_override_words():
    corpus = Path(__file__).parent / "test_sentences.jsonl"
    sentences = [json.loads(line)["sentence"]
                 for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for sentence in sentences:
        source = re.findall(r"\w+", sentence)
        european = re.findall(r"\w+", add_extra_diacritics(sentence))
        brazilian = re.findall(r"\w+", add_extra_diacritics(sentence, lang="pt-BR"))
        assert len(source) == len(european) == len(brazilian)
        for src, eu, br in zip(source, european, brazilian):
            if eu == br:
                continue
            changed += 1
            assert src.lower() in BR_WORDS, (sentence, src, eu, br)
            assert br.lower() in (src.lower(), "sôma"), (sentence, src, eu, br)
    assert changed > 0


def test_override_rows_are_known_homographs_with_a_source():
    from bifonia.data import LECT_OVERRIDES

    path = Path(__file__).parent.parent / "bifonia" / "data" / "lect_overrides.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["word"] for r in rows if r["lang"] == "pt-BR"} == BR_WORDS
    assert LECT_OVERRIDES["pt-UY"] == LECT_OVERRIDES["pt-BR"]
    for row in rows:
        assert row["word"] in AMBIGUOUS_WORDS
        assert row["source"].strip()
    assert LECT_OVERRIDES["pt-BR"]["soma"] == "sôma"


def _load_data_with(tmp_path, extra_rows):
    """Load bifonia/data.py next to a copy of its tables, with *extra_rows*
    appended to lect_overrides.csv. Return the module and the csv path."""
    package = Path(bifonia.data.__file__).parent
    (tmp_path / "data").mkdir()
    shutil.copy(package / "data" / "heterophonic_homographs.csv", tmp_path / "data")
    rows = (package / "data" / "lect_overrides.csv").read_text(encoding="utf-8")
    lect_csv = tmp_path / "data" / "lect_overrides.csv"
    lect_csv.write_text(rows + extra_rows, encoding="utf-8")
    shutil.copy(package / "data.py", tmp_path)
    spec = importlib.util.spec_from_file_location("data_copy", tmp_path / "data.py")
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    return data, lect_csv


def test_an_override_form_in_nfd_keeps_every_letter(tmp_path, monkeypatch):
    # One more row whose form is in NFD: "o" and U+0302 in place of "ô". A text
    # editor can save it so.
    data, _ = _load_data_with(tmp_path, "pt-AO,soma,so\u0302ma,probe,probe\n")

    monkeypatch.setitem(bifonia.data.LECT_OVERRIDES, "pt-AO", data.LECT_OVERRIDES["pt-AO"])
    assert add_extra_diacritics("A SOMA DE DOIS", "pt-AO") == "A S\u00d4MA DE DOIS"
    assert add_extra_diacritics("soma", "pt-AO") == "s\u00f4ma"
    assert data.LECT_OVERRIDES["pt-AO"] == {"soma": "s\u00f4ma"}, ascii(data.LECT_OVERRIDES)


def test_every_shipped_override_row_has_one_code_point_per_letter():
    # The rule of the loader: in NFC, a form has as many code points as its word
    # has letters, and no combining mark is left over. An empty form is valid.
    path = Path(bifonia.data.__file__).parent / "data" / "lect_overrides.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        form = unicodedata.normalize("NFC", row["form"])
        assert not form or len(form) == len(row["word"]), row
        assert not any(unicodedata.combining(c) for c in form), row
    assert sum(len(forms) for forms in bifonia.data.LECT_OVERRIDES.values()) == len(rows)


# A form that is still longer or shorter than its word in NFC. add_extra_diacritics
# copies the case of each letter of the word onto the form position by position,
# so such a form would drop letters: "A soma dá dez." would become "A só̂m dá dez.".
# "so" + U+0301 + U+0302 + "ma" is "s", "ó", U+0302, "m", "a" in NFC: 5 code points
# for the 4 letters of soma.
@pytest.mark.parametrize("row,form,word,form_len,word_len", [
    ("pt-CV,soma,so\u0301\u0302ma,probe,probe\n", "s\u00f3\u0302ma", "soma", 5, 4),
    ("pt-CV,acordo,aco\u0308\u0301rdo,probe,probe\n", "ac\u00f6\u0301rdo", "acordo", 7, 6),
    ("pt-CV,jogo,j\u00f4g,probe,probe\n", "j\u00f4g", "jogo", 3, 4),
])
def test_an_override_form_of_another_length_is_rejected_at_load(
        tmp_path, row, form, word, form_len, word_len):
    shipped = (Path(bifonia.data.__file__).parent / "data" / "lect_overrides.csv")
    line = len(shipped.read_text(encoding="utf-8").splitlines()) + 1
    lect_csv = tmp_path / "data" / "lect_overrides.csv"
    message = (f"{lect_csv}, line {line}: the form {form!r} of {word!r} has "
               f"{form_len} code points in NFC, but the word has {word_len}")
    with pytest.raises(ValueError) as err:
        _load_data_with(tmp_path, row)
    assert str(err.value) == message


def test_an_override_form_that_keeps_a_combining_mark_is_rejected_at_load(tmp_path):
    # "so" + U+0301 + U+0302 + "m" is "s", "ó", U+0302, "m" in NFC: 4 code points,
    # as many as the letters of soma, because no precomposed "ó̂" exists. The mark
    # takes the place of a letter, so "A soma dá dez." would become "A só̂m dá dez.".
    shipped = (Path(bifonia.data.__file__).parent / "data" / "lect_overrides.csv")
    line = len(shipped.read_text(encoding="utf-8").splitlines()) + 1
    lect_csv = tmp_path / "data" / "lect_overrides.csv"
    form = "s\u00f3\u0302m"
    message = f"{lect_csv}, line {line}: the form {form!r} of 'soma' keeps a combining mark in NFC"
    with pytest.raises(ValueError) as err:
        _load_data_with(tmp_path, "pt-CV,soma,so\u0301\u0302m,probe,probe\n")
    assert str(err.value) == message


@pytest.mark.parametrize("raw", ["s\u1ed1ma", "so\u0302\u0301ma"])
def test_an_override_form_with_a_precomposed_double_mark_loads(tmp_path, raw):
    # U+1ED1 is "o" with circumflex and acute in one code point; the NFD spelling
    # composes to it.
    data, _ = _load_data_with(tmp_path, f"pt-CV,soma,{raw},probe,probe\n")
    assert data.LECT_OVERRIDES["pt-CV"] == {"soma": "s\u1ed1ma"}
