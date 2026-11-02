"""The verb molhar ("eu molho") takes the open reading, like the bundle noun."""
import pytest

from bifonia import add_extra_diacritics, tokenize, guess_sense, disambiguate
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
