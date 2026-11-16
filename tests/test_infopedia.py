"""Tests for the bundled Infopédia IPA lookup."""
import pytest

from bifonia import infopedia


@pytest.mark.parametrize("word, pos, expected", [
    ("colher", "NOUN", "kuˈʎɛɾ"),   # spoon: open e
    ("colher", "VERB", "kuˈʎeɾ"),   # harvest: closed e
    # The verb reading carries the combined tag NOUN|VERB in the data.
    ("corte", "VERB", "ˈkɔɾtɨ"),    # (que eu) corte: open o
    ("forma", "VERB", "ˈfɔɾmɐ"),    # (ele) forma: open o
    ("tola", "NOUN", "ˈtɔlɐ"),      # head: open o; ADJ|NOUN row is closed
])
def test_ipa_selects_reading_by_ud_tag(word, pos, expected):
    assert infopedia.ipa(word, pos) == expected


def test_ipa_unknown_word_is_none():
    assert infopedia.ipa("inexistente", "NOUN") is None


def test_molho_verb_is_open():
    # 1sg of molhar is open (Portugal /ˈmɔ.ʎu/); only the sauce noun is closed.
    assert infopedia.ipa("molho", "VERB") == "ˈmɔʎu"
    assert infopedia.ipa("molho", "NOUN") == "ˈmoʎu"   # NOUN-only row: sauce
