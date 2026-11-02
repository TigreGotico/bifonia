"""Tests for the proper-noun flags that bias add_extra_diacritics."""
import pytest

from bifonia import add_extra_diacritics, proper_flags


@pytest.mark.parametrize("abbrev", ["n.º", "N.º"])
def test_numero_with_dot_matches_numero_without_dot(abbrev):
    # "n.º" and "nº" are spellings of the same abbreviation, so a title such as
    # "Choro n.º 13" must get the same first-token flag and reading as "Choro nº 13".
    assert proper_flags(f"Choro {abbrev} 13")[0] is True
    assert add_extra_diacritics(f"Choro {abbrev} 13") == f"Chôro {abbrev} 13"
    assert add_extra_diacritics("Choro nº 13") == "Chôro nº 13"
