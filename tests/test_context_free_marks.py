"""
Tests for the marks that need no context (data/context_free_marks.csv).

The expected marks come from the norms and the transcriptions, not from bifonia
or orthography2ipa:

- pêra: Acordo de 1945 Base XXII, «pêra, substantivo, e pera, preposição
  arcaica»; Infopédia pera [ˈperɐ] for the fruit.
- vôo, enjôo: AO1990 Base IX 8.º drops the circumflex of «enjoo» and «voo»;
  Infopédia voo [ˈvou], enjoo [ẽˈʒou].
- combóio: the pre-AO pt-BR spelling of the verb form (tugalex pt-BR map);
  Infopédia comboio [kõˈbɔju], Michaelis comboio [komˈbojo].
- idéia, heróico, asteróide: AO1990 Base IX 3.º drops the accent of ⟨ei oi⟩ in
  «ideia», «heroico» and «alcaloide»; Michaelis ideia [iˈdɛjɐ], heroico
  [eˈɾɔjku], asteroide [asteˈɾɔjdi]. Infopédia ideia [iˈdɐjɐ] has another vowel,
  so pt-PT keeps «ideia».
- lingüiça, tranqüilo: AO1990 Base XIV 1.º drops the trema; Michaelis linguiça
  and tranquilo say [ɡw] and [kw].
- areia: Michaelis areia [aˈɾejɐ] has a closed vowel, so pt-BR keeps «areia»
  although the pt-BR norm wrote «aréia».
"""

import csv
import importlib.util
import shutil
import unicodedata
from pathlib import Path

import pytest

import bifonia.data
from bifonia import AMBIGUOUS_WORDS, add_extra_diacritics

PT_LANGS = [None, "", "pt-PT", "pt_pt", "pt-PT-x-lbx", " pt-PT ", "pt-Latn-PT"]
BR_LANGS = ["pt-BR", "pt_br", "pt-BR-x-rj", "pt-BR-x-sp", "pt-Latn-BR"]
UNCHECKED_LANGS = ["pt-AO", "pt-MZ", "pt-UY", "pt-brx", "en"]

# (sentence, pt-PT output, pt-BR output)
CASES = [
    ("Comi uma pera madura ao lanche.",
     "Comi uma pêra madura ao lanche.",
     "Comi uma pera madura ao lanche."),
    ("O voo foi cancelado por causa do nevoeiro.",
     "O vôo foi cancelado por causa do nevoeiro.",
     "O voo foi cancelado por causa do nevoeiro."),
    ("Tomei um comprimido para o enjoo.",
     "Tomei um comprimido para o enjôo.",
     "Tomei um comprimido para o enjoo."),
    ("O comboio chegou atrasado.",
     "O combóio chegou atrasado.",
     "O comboio chegou atrasado."),
    ("Tive uma ideia brilhante.",
     "Tive uma ideia brilhante.",
     "Tive uma idéia brilhante."),
    ("Foi um gesto heroico.",
     "Foi um gesto heroico.",
     "Foi um gesto heróico."),
    ("O asteroide passou perto da Terra.",
     "O asteroide passou perto da Terra.",
     "O asteróide passou perto da Terra."),
    ("Comprei linguiça e pão.",
     "Comprei linguiça e pão.",
     "Comprei lingüiça e pão."),
    ("O mar está tranquilo.",
     "O mar está tranquilo.",
     "O mar está tranqüilo."),
    ("A areia da praia está quente.",
     "A areia da praia está quente.",
     "A areia da praia está quente."),
]
_IDS = [c[0][:30] for c in CASES]


@pytest.mark.parametrize("lang", PT_LANGS)
@pytest.mark.parametrize("sentence,european,_brazilian", CASES, ids=_IDS)
def test_pt_pt_marks(sentence, european, _brazilian, lang):
    assert add_extra_diacritics(sentence, lang=lang) == european


def test_default_lect_is_pt_pt():
    for sentence, european, _ in CASES:
        assert add_extra_diacritics(sentence) == european


@pytest.mark.parametrize("lang", BR_LANGS)
@pytest.mark.parametrize("sentence,_european,brazilian", CASES, ids=_IDS)
def test_pt_br_marks(sentence, _european, brazilian, lang):
    assert add_extra_diacritics(sentence, lang=lang) == brazilian


@pytest.mark.parametrize("lang", UNCHECKED_LANGS)
@pytest.mark.parametrize("sentence", [c[0] for c in CASES], ids=_IDS)
def test_unchecked_lects_get_no_context_free_mark(sentence, lang):
    assert add_extra_diacritics(sentence, lang=lang) == sentence


@pytest.mark.parametrize("sentence,lang,expected", [
    ("PERA", "pt-PT", "PÊRA"),
    ("Pera e maçã.", "pt-PT", "Pêra e maçã."),
    ("IDEIA", "pt-BR", "IDÉIA"),
    ("Ideia nova.", "pt-BR", "Idéia nova."),
    ("LingUiça", "pt-BR", "LingÜiça"),
])
def test_case_is_kept(sentence, lang, expected):
    assert add_extra_diacritics(sentence, lang=lang) == expected


@pytest.mark.parametrize("sentence,lang,expected", [
    ("Que ideia!", "pt-BR", "Que idéia!"),
    ("Uma pera, duas maçãs.", "pt-PT", "Uma pêra, duas maçãs."),
    ("«Ideia»", "pt-BR", "«Idéia»"),
    ("ideia-chave", "pt-BR", "idéia-chave"),
])
def test_punctuation_and_hyphen_around_the_word(sentence, lang, expected):
    assert add_extra_diacritics(sentence, lang=lang) == expected


@pytest.mark.parametrize("sentence,lang", [
    ("Comi uma pêra.", "pt-PT"),
    ("Tive uma idéia.", "pt-BR"),
    ("As peras estão maduras.", "pt-PT"),  # 1945 Base XXII: «o plural, peras, sem acento»
    ("A ideiazinha.", "pt-BR"),
])
def test_other_spellings_are_left_alone(sentence, lang):
    assert add_extra_diacritics(sentence, lang=lang) == sentence


def test_context_table_still_decides_its_words():
    # pelo and polo are context words: the context-free table never marks them.
    assert add_extra_diacritics("Passou pelo parque com a pera.") == \
        "Passou pelo parque com a pêra."
    assert add_extra_diacritics("O cão tem o pelo macio e uma ideia.", lang="pt-BR") == \
        "O cão tem o pêlo macio e uma idéia."
    assert add_extra_diacritics("A pela de pera.", lang="pt-PT") == "A pela de pêra."


def _rows():
    path = Path(bifonia.data.__file__).parent / "data" / "context_free_marks.csv"
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_every_row_cites_its_evidence():
    rows = _rows()
    assert len(rows) > 100
    for row in rows:
        lects = row["lects"].split("|")
        assert set(lects) <= {"pt-PT", "pt-BR"}, row
        assert row["word"] not in AMBIGUOUS_WORDS, row
        assert row["norm_source"].startswith("tugalex acordo_ortografico_pt_"), row
        for lect in lects:
            assert f"{lect}: " in row["pronunciation_source"], row
            assert f"{lect}: " in row["transcription"], row
        word, form = row["word"], unicodedata.normalize("NFC", row["form"])
        assert len(word) == len(form) and word != form, row
        changed = [(w, f) for w, f in zip(word, form) if w != f]
        assert all(f in "áéêíóôúü" and unicodedata.normalize("NFD", f)[0] == w
                   for w, f in changed), row


def test_loaded_table_matches_the_csv():
    from bifonia.data import CONTEXT_FREE_MARKS

    expected = {}
    for row in _rows():
        for lect in row["lects"].split("|"):
            expected.setdefault(lect, {})[row["word"]] = row["form"]
    assert CONTEXT_FREE_MARKS == expected
    assert CONTEXT_FREE_MARKS["pt-PT"]["pera"] == "pêra"
    assert CONTEXT_FREE_MARKS["pt-BR"]["ideia"] == "idéia"
    assert "ideia" not in CONTEXT_FREE_MARKS["pt-PT"]
    assert "pera" not in CONTEXT_FREE_MARKS["pt-BR"]


def _load_data_with(tmp_path, extra_rows):
    """Load bifonia/data.py next to a copy of its tables, with *extra_rows*
    appended to context_free_marks.csv."""
    package = Path(bifonia.data.__file__).parent
    (tmp_path / "data").mkdir()
    for name in ("heterophonic_homographs.csv", "lect_overrides.csv"):
        shutil.copy(package / "data" / name, tmp_path / "data")
    rows = (package / "data" / "context_free_marks.csv").read_text(encoding="utf-8")
    (tmp_path / "data" / "context_free_marks.csv").write_text(rows + extra_rows, encoding="utf-8")
    shutil.copy(package / "data.py", tmp_path)
    spec = importlib.util.spec_from_file_location("data_copy", tmp_path / "data.py")
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    return data


def test_loader_rejects_a_context_word(tmp_path):
    with pytest.raises(ValueError, match="context table"):
        _load_data_with(tmp_path, "sede,séde,pt-PT,probe,probe,probe\n")


def test_loader_reads_an_nfd_form_in_nfc(tmp_path):
    data = _load_data_with(tmp_path, "jabuti,jabúti,pt-PT,probe,probe,probe\n")
    assert data.CONTEXT_FREE_MARKS["pt-PT"]["jabuti"] == "jabúti"


def test_loader_rejects_a_form_of_another_length(tmp_path):
    with pytest.raises(ValueError, match="code points"):
        _load_data_with(tmp_path, "jabuti,jabutí-,pt-PT,probe,probe,probe\n")
