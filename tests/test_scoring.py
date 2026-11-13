"""Tests for helpers in the rule-based context scorer."""
import pytest

from bifonia import tokenize
from bifonia.scoring import _is_numeric, score_adp


@pytest.mark.parametrize("word", ["5", "2024", "000", "-5", "cinco", "vinte"])
def test_numeric_tokens(word):
    assert _is_numeric(word)


# float() parses these, but none is a number in Portuguese text.
@pytest.mark.parametrize("word", ["nan", "inf", "infinity", "1_000", "1e5", "xyz", ""])
def test_non_numeric_tokens(word):
    assert not _is_numeric(word)


@pytest.mark.parametrize("word", ["nan", "inf", "1_000", "1e5"])
def test_float_spellings_get_no_numeral_credit(word):
    def score(nxt):
        words = tokenize(f"O autocarro para {nxt} vezes por dia")
        return score_adp(words, words.index("para"))

    assert score(word) == score("xyz")
    assert score("5") > score("xyz")
