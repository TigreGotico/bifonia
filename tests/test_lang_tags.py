"""Language tags are read by linguatag, so deprecated codes, ISO 639-2 codes and
underscores reach the same lect and locale as the canonical tag."""
import pytest

from bifonia import add_extra_diacritics, data
from bifonia.features import load_structural_vocs
from bifonia.vocab import voc

PT_BR = ["pt-BR", "pt_br", "PT-br", "por-BR", "por_BR", "pt-Latn-BR"]


@pytest.mark.parametrize("tag", PT_BR)
def test_brazilian_spellings_take_the_pt_br_rows(tag):
    assert data.lect_overrides(tag) == data.LECT_OVERRIDES["pt-BR"]
    assert data.context_free_marks(tag) == data.CONTEXT_FREE_MARKS["pt-BR"]


@pytest.mark.parametrize("tag", ["pt", "por", "pt-x-foo", "pt-Latn"])
def test_tag_without_region_takes_the_default_region_rows(tag):
    assert data.context_free_marks(tag) == data.CONTEXT_FREE_MARKS["pt-PT"]


def test_iso_639_2_tag_marks_a_sentence_like_its_two_letter_tag():
    assert add_extra_diacritics("o abono", "por-BR") == add_extra_diacritics("o abono", "pt-BR")


@pytest.mark.parametrize("tag", ["pt-PT", "pt_PT", "pt", "por", "pt-BR"])
def test_locale_vocabularies_follow_the_closest_locale(tag):
    assert load_structural_vocs(tag) == load_structural_vocs("pt-pt") != {}
    assert voc("articles", tag) == voc("articles", "pt-pt")
