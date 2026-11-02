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
