"""The two readings of "fora".

The adverb, the preposition and the interjection have an open stressed o; the
pluperfect of ser and ir has a closed one. The expected values come from these
sources, not from the engine:

- Formulário Ortográfico 1943, rule 14.ª: "fôra (v.) e fora (adv. interj. ou
  s.m.)".
- Acordo Ortográfico 1945: "fora (ô), flexão de ser e ir, e fora (ó), advérbio,
  interjeição e substantivo".
- Infopédia, fora: "ˈfɔrɐ advérbio"; Priberam, fora: "fora |ó| ... advérbio".
- Wiktionary, fora: the adverb is pt-IPA "fóra" and the verb form pt-IPA
  "fôra", for Portugal and Brazil alike.
"""
import pytest

from bifonia import add_extra_diacritics, disambiguate, tokenize
from bifonia.scoring import score_verb

OPEN = "ˈfɔɾɐ"     # Infopédia ˈfɔrɐ, with the tap ɾ that the roster writes
CLOSED = "ˈfoɾɐ"


def _ipa(sentence: str) -> str:
    words = tokenize(sentence)
    idx = next(i for i, t in enumerate(words) if t.strip(".,;:!?") == "fora")
    return disambiguate(words, idx)


@pytest.mark.parametrize("sentence", [
    "Ele está lá fora.",
    "Fora!",
    "Fora daqui, já!",
    "Hoje jantamos fora.",
    "Deitou o lixo fora antes de sair.",
    "Todos vieram à festa, fora o João.",
    "Passei a tarde fora de casa.",
    "O gato saltou para fora da caixa.",
    # a verb cue next to an adverb veto: the veto wins
    "Disse que fora de casa ninguém fuma.",
    "Ficaram lá fora sentados no muro.",
    # a noun in -ado/-ada after it is no participle
    "Jantamos fora sábado.",
    "Ficámos fora madrugada adentro.",
    # punctuation between the cue word and fora breaks the cue
    "Ela percebeu que, fora a Netflix, nada mais via.",
])
def test_adverb_preposition_interjection_is_open(sentence):
    assert _ipa(sentence) == OPEN


@pytest.mark.parametrize("sentence", [
    "Assim fora decidido.",
    "Se ela fora rica, teria ido.",
    "A carta fora escrita na véspera.",
    "Nunca fora tão feliz como naquele verão.",
    "Já não era o homem que fora.",
    "Ele fora ao Porto na semana anterior.",
    "O dia fora longo e cansativo.",
    "A ponte fora então construída pelos romanos.",
    # an adverb between the subject and fora
    "Ele já fora rei.",
    # an adverb between fora and the participle, with no other cue
    "Tudo fora já decidido.",
    # an enclitic pronoun: only a verb hosts one
    "Fora-lhe dito que esperasse.",
    "Fora-se embora sem avisar.",
])
def test_pluperfect_of_ser_and_ir_is_closed(sentence):
    assert _ipa(sentence) == CLOSED


@pytest.mark.parametrize("lang", ["pt-PT", "pt-BR"])
def test_marks_match_the_reading(lang):
    assert add_extra_diacritics("Ele está lá fora.", lang=lang) == "Ele está lá fóra."
    assert add_extra_diacritics("Assim fora decidido.", lang=lang) == "Assim fôra decidido."


@pytest.mark.parametrize("sentence, expected", [
    # the pluperfect has a subject cue before it: 8
    ("Ele já fora.", 8),
    ("Ele já fora!", 8),
    ("Nunca fora, dizia.", 8),
    # the adverb has no verb cue, or an adverb cue: 0
    ("Saiu porta fora!", 0),
    ("Saiu porta fora.", 0),
    ("Ele está lá fora.", 0),
    ("Fora!", 0),
])
def test_verb_score_ignores_attached_punctuation(sentence, expected):
    """A token with attached punctuation scores as the bare word: 8 with a
    pluperfect cue (Formulário Ortográfico 1943, fôra v.), 0 for the adverb
    (Priberam, fora |ó| advérbio)."""
    words = tokenize(sentence)
    idx = next(i for i, t in enumerate(words) if t.strip(".,;:!?") == "fora")
    assert score_verb(words, idx) == expected
    bare = list(words)
    bare[idx] = words[idx].strip(".,;:!?")
    assert score_verb(words, idx) == score_verb(bare, idx)
