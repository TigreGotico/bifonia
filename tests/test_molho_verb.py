"""The verb molhar ("eu molho") takes the open reading, like the bundle noun."""
import pytest

from bifonia import add_extra_diacritics, tokenize, guess_pos, guess_sense, disambiguate
from bifonia.data import POS_SENSES


def test_verb_maps_to_open_reading():
    assert POS_SENSES["molho"]["VERB"] == ["bundle"]
    assert "sauce" not in POS_SENSES["molho"]["VERB"]


@pytest.mark.parametrize("sentence", [
    "Molho os pés na bacia com água morna.",
    "Molho o pão no café antes de o comer.",
])
def test_verb_molho_gets_no_closed_mark(sentence):
    out = add_extra_diacritics(sentence)
    assert "ô" not in out
    assert out == "Mólho" + sentence[5:]


def test_verb_molho_ipa_is_open():
    words = tokenize("Molho os pés na bacia com água morna.")
    assert disambiguate(words, 0, pos="VERB") == "ˈmɔʎu"
    assert guess_sense(words, 0, postag="VERB") == "bundle"


def test_sauce_noun_stays_closed():
    assert add_extra_diacritics("O molho de tomate ficou ótimo.") == "O môlho de tomate ficou ótimo."


# Sauce nouns that the generic verb cues (sentence start, "que"/"qual" before)
# would read as the verb. Gold: labelled sauce in the bundled corpus.
_SAUCE_NOUNS = [
    "Falta molho neste prato.",
    "Que molho escolhes para a pasta?",
    "Qual molho vais usar para o arroz?",
    "Qual molho de compra recomendas mesmo?",
    "Molho de amêndoa é tradicional no bacalhau.",
    "Molho de hortelã é perfeito para o borrego.",
    "Molho de alho francês e batata é cremoso.",
    "Molho de trufas é considerado um luxo na cozinha.",
    "Molho de alho e coentros é típico no Alentejo.",
    "Molho de cebola caramelizada deu sabor ao hambúrguer.",
    "Molho de mostarda acompanha o espeto na perfeição.",
    "Molho de mostarda acompanha o espêto na perfeição.",
]

_VERBS = [
    "Caso sobre água da rega, molho também as roseiras.",
    "Eu aperto o gatilho da pistola de água e molho-o todo.",
    "Que molho as plantas?",
    "Eu molho as plantas todas as manhãs.",
    "Molho-as de manhã.",
]


def _molho(sentence):
    words = tokenize(sentence)
    return words, next(i for i, t in enumerate(words) if t.strip(".,;:!?").startswith("molho"))


@pytest.mark.parametrize("sentence", _SAUCE_NOUNS)
def test_sauce_noun_is_not_read_as_verb(sentence):
    words, idx = _molho(sentence)
    assert guess_pos(words, idx) == "NOUN"
    assert disambiguate(words, idx) == "ˈmoʎu"


@pytest.mark.parametrize("sentence", _VERBS)
def test_verb_molho_stays_verb(sentence):
    words, idx = _molho(sentence)
    assert guess_pos(words, idx) == "VERB"
    assert disambiguate(words, idx) == "ˈmɔʎu"
