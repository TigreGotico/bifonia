"""
Tests for per-lect marking in add_extra_diacritics (``lang=``).

The pt-BR expectations come from Brazilian sources, not from bifonia's table:
Wiktionary gives Brazil a closed nasal stressed vowel for soma, abono, aceno and
domo in every sense (/ˈsõ.mɐ/, /aˈbõ.nu/, /aˈsẽ.nu/, /ˈdõ.mu/), and the same rule
applies to empeno and desempeno. soma is written closed (sôma), because an
unmarked soma takes the open vowel of the tugalex rjx and spx lexicons. tola keeps
the European marks: bifonia's head sense also covers the timber, which no
Brazilian source covers.
"""

import csv
import json
import re
from pathlib import Path

import pytest

from bifonia import add_extra_diacritics, AMBIGUOUS_WORDS

BR_LANGS = ["pt-BR", "pt-BR-x-rj", "pt-BR-x-sp", "pt_br", " pt-BR ", "pt-Latn-BR"]
EUROPEAN_LANGS = [None, "", "pt-PT", "pt-PT-x-lbx", "pt-AO", "pt-brx", "pt-Latn", "en"]

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
    for row in rows:
        assert row["word"] in AMBIGUOUS_WORDS
        assert row["source"].strip()
    assert LECT_OVERRIDES["pt-BR"]["soma"] == "sôma"
