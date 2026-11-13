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
- assembléia, heróico, asteróide: AO1990 Base IX 3.º drops the accent of ⟨ei
  oi⟩ in «assembleia», «heroico» and «alcaloide»; Michaelis assembleia
  [asẽˈblɛjɐ], heroico [eˈɾɔjku], asteroide [asteˈɾɔjdi]. Infopédia assembleia
  [ɐsẽˈblɐjɐ] has another vowel, so pt-PT keeps «assembleia».
- lingüiça, tranqüilo: AO1990 Base XIV 1.º drops the trema; Michaelis linguiça
  and tranquilo say [ɡw] and [kw].
- areia: Michaelis areia [aˈɾejɐ] has a closed vowel, so pt-BR keeps «areia»
  although the pt-BR norm wrote «aréia».
- ideia, estreia: also forms of idear and estrear, a second reading
  (EAR_VERB_RULE in scripts/build_context_free_marks.py), so pt-BR leaves them
  unmarked.
- ateia: pt.wiktionary gives the verb forms of atear /ɐˈtej.ɐ/ for Brazil, so
  pt-BR leaves «ateia» unmarked.

The trema is not written for the pt-BR sub-lects whose orthography2ipa spec
reads ⟨ü⟩ as a full vowel (pt-BR-x-sp gives tranqüilo [tɾɐ̃kuˈilu]); the accent
rows are.
"""

import csv
import importlib.util
import shutil
import unicodedata
from pathlib import Path

import pytest

import bifonia.data
from bifonia import AMBIGUOUS_WORDS, add_extra_diacritics

PT_LANGS = [None, "", "pt", "PT", "pt-PT", "pt_pt", "pt-PT-x-lisbon", "pt-PT-x-unknown",
            " pt-PT ", "pt-Latn-PT"]
BR_LANGS = ["pt-BR", "pt_br", "pt-Latn-BR", "pt-BR-x-cuiaba", "pt-BR-x-floripa",
            "pt-BR-x-unknown"]
# orthography2ipa reads ⟨ü⟩ as a full vowel in these 11 specs. The last three tags
# are orthography2ipa aliases of pt-BR-x-sp, pt-BR-x-rj and pt-BR-x-fluminen.
BR_NO_TREMA_LANGS = ["pt-BR-x-bahia", "pt-BR-x-caipira", "pt-BR-x-ce", "pt-BR-x-fluminen",
                     "pt-BR-x-mg", "pt-BR-x-norte", "pt-BR-x-pr", "pt-BR-x-recife", "pt-BR-x-rj",
                     "pt-BR-x-sp", "pt-BR-x-sul",
                     "pt-BR-x-sao-paulo", "pt-BR-x-rio-janeiro", "pt-BR-x-fluminense"]
UNCHECKED_LANGS = ["pt-AO", "pt-MZ", "pt-UY", "pt-brx", "en"]

# (sentence, pt-PT output, pt-BR output, output of a pt-BR sub-lect with no trema)
CASES = [
    ("Comi uma pera madura ao lanche.",
     "Comi uma pêra madura ao lanche.",
     "Comi uma pera madura ao lanche.",
     "Comi uma pera madura ao lanche."),
    ("O voo foi cancelado por causa do nevoeiro.",
     "O vôo foi cancelado por causa do nevoeiro.",
     "O voo foi cancelado por causa do nevoeiro.",
     "O voo foi cancelado por causa do nevoeiro."),
    ("Tomei um comprimido para o enjoo.",
     "Tomei um comprimido para o enjôo.",
     "Tomei um comprimido para o enjoo.",
     "Tomei um comprimido para o enjoo."),
    ("O comboio chegou atrasado.",
     "O combóio chegou atrasado.",
     "O comboio chegou atrasado.",
     "O comboio chegou atrasado."),
    ("A assembleia votou a proposta.",
     "A assembleia votou a proposta.",
     "A assembléia votou a proposta.",
     "A assembléia votou a proposta."),
    ("Foi um gesto heroico.",
     "Foi um gesto heroico.",
     "Foi um gesto heróico.",
     "Foi um gesto heróico."),
    ("O asteroide passou perto da Terra.",
     "O asteroide passou perto da Terra.",
     "O asteróide passou perto da Terra.",
     "O asteróide passou perto da Terra."),
    ("Comprei linguiça e pão.",
     "Comprei linguiça e pão.",
     "Comprei lingüiça e pão.",
     "Comprei linguiça e pão."),
    ("O mar está tranquilo.",
     "O mar está tranquilo.",
     "O mar está tranqüilo.",
     "O mar está tranquilo."),
    ("A areia da praia está quente.",
     "A areia da praia está quente.",
     "A areia da praia está quente.",
     "A areia da praia está quente."),
    ("Tive uma ideia para a estreia.",
     "Tive uma ideia para a estreia.",
     "Tive uma ideia para a estreia.",
     "Tive uma ideia para a estreia."),
    ("Ele ateia o fogo.",
     "Ele ateia o fogo.",
     "Ele ateia o fogo.",
     "Ele ateia o fogo."),
]
_IDS = [c[0][:30] for c in CASES]


@pytest.mark.parametrize("lang", PT_LANGS)
@pytest.mark.parametrize("sentence,european,_brazilian,_no_trema", CASES, ids=_IDS)
def test_pt_pt_marks(sentence, european, _brazilian, _no_trema, lang):
    assert add_extra_diacritics(sentence, lang=lang) == european


def test_default_lect_is_pt_pt():
    for sentence, european, _, _ in CASES:
        assert add_extra_diacritics(sentence) == european


@pytest.mark.parametrize("lang", BR_LANGS)
@pytest.mark.parametrize("sentence,_european,brazilian,_no_trema", CASES, ids=_IDS)
def test_pt_br_marks(sentence, _european, brazilian, _no_trema, lang):
    assert add_extra_diacritics(sentence, lang=lang) == brazilian


@pytest.mark.parametrize("lang", BR_NO_TREMA_LANGS)
@pytest.mark.parametrize("sentence,_european,_brazilian,no_trema", CASES, ids=_IDS)
def test_pt_br_sub_lects_with_no_trema(sentence, _european, _brazilian, no_trema, lang):
    assert add_extra_diacritics(sentence, lang=lang) == no_trema


def test_pt_br_x_sp_gets_no_trema():
    from bifonia.data import context_free_marks

    marks = context_free_marks("pt-BR-x-sp")
    assert marks and not any("ü" in form for form in marks.values())
    assert add_extra_diacritics("Tranquilo, comi linguiça.", lang="pt-BR-x-sp") == \
        "Tranquilo, comi linguiça."


@pytest.mark.parametrize("lang", UNCHECKED_LANGS)
@pytest.mark.parametrize("sentence", [c[0] for c in CASES], ids=_IDS)
def test_unchecked_lects_get_no_context_free_mark(sentence, lang):
    assert add_extra_diacritics(sentence, lang=lang) == sentence


@pytest.mark.parametrize("sentence,lang,expected", [
    ("PERA", "pt-PT", "PÊRA"),
    ("Pera e maçã.", "pt-PT", "Pêra e maçã."),
    ("ASSEMBLEIA", "pt-BR", "ASSEMBLÉIA"),
    ("Assembleia geral.", "pt-BR", "Assembléia geral."),
    ("LingUiça", "pt-BR", "LingÜiça"),
])
def test_case_is_kept(sentence, lang, expected):
    assert add_extra_diacritics(sentence, lang=lang) == expected


@pytest.mark.parametrize("sentence,lang,expected", [
    ("Que plateia!", "pt-BR", "Que platéia!"),
    ("Uma pera, duas maçãs.", "pt-PT", "Uma pêra, duas maçãs."),
    ("«Odisseia»", "pt-BR", "«Odisséia»"),
    ("assembleia-geral", "pt-BR", "assembléia-geral"),
])
def test_punctuation_and_hyphen_around_the_word(sentence, lang, expected):
    assert add_extra_diacritics(sentence, lang=lang) == expected


@pytest.mark.parametrize("sentence,lang", [
    ("Comi uma pêra.", "pt-PT"),
    ("Uma assembléia.", "pt-BR"),
    ("As peras estão maduras.", "pt-PT"),  # 1945 Base XXII: «o plural, peras, sem acento»
    ("A assembleiazinha.", "pt-BR"),
])
def test_other_spellings_are_left_alone(sentence, lang):
    assert add_extra_diacritics(sentence, lang=lang) == sentence


def test_context_table_still_decides_its_words():
    # pelo and polo are context words: the context-free table never marks them.
    assert add_extra_diacritics("Passou pelo parque com a pera.") == \
        "Passou pelo parque com a pêra."
    assert add_extra_diacritics("O cão tem o pelo macio e uma joia.", lang="pt-BR") == \
        "O cão tem o pêlo macio e uma jóia."
    assert add_extra_diacritics("A pela de pera.", lang="pt-PT") == "A pela de pêra."


def _rows():
    path = Path(bifonia.data.__file__).parent / "data" / "context_free_marks.csv"
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_every_row_cites_its_evidence():
    rows = _rows()
    assert len(rows) > 100
    for row in rows:
        codes = row["lects"].split("|")
        parents = {code[:5] for code in codes}
        assert parents <= {"pt-PT", "pt-BR"} and all(
            code[:5] == code or code[5:8] == "-x-" for code in codes), row
        assert row["word"] not in AMBIGUOUS_WORDS, row
        assert row["norm_source"].startswith("tugalex acordo_ortografico_pt_"), row
        for parent in parents:
            assert f"{parent}: " in row["pronunciation_source"], row
            assert f"{parent}: " in row["transcription"], row
        word, form = row["word"], unicodedata.normalize("NFC", row["form"])
        assert len(word) == len(form) and word != form, row
        changed = [(w, f) for w, f in zip(word, form) if w != f]
        assert all(f in "áéêíóôúü" and unicodedata.normalize("NFD", f)[0] == w
                   for w, f in changed), row


def test_loaded_table_matches_the_csv():
    from bifonia.data import CONTEXT_FREE_MARKS

    expected = {}
    for row in _rows():
        for code in row["lects"].split("|"):
            expected.setdefault(code, {})[row["word"]] = row["form"]
    assert CONTEXT_FREE_MARKS == expected
    assert CONTEXT_FREE_MARKS["pt-PT"]["pera"] == "pêra"
    assert CONTEXT_FREE_MARKS["pt-BR"]["assembleia"] == "assembléia"
    assert "assembleia" not in CONTEXT_FREE_MARKS["pt-PT"]
    assert "pera" not in CONTEXT_FREE_MARKS["pt-BR"]
    for word in ("ideia", "estreia", "ateia", "taoismo"):
        assert word not in CONTEXT_FREE_MARKS["pt-BR"] and word not in CONTEXT_FREE_MARKS["pt-PT"]


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
