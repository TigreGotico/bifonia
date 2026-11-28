"""An adverb after a clause-initial verb does not make "para" the verb parar (#108)."""
import pytest

from bifonia import guess_sense, tokenize


@pytest.mark.parametrize("sentence, sense", [
    ("Olho sempre para o horizonte quando estou no mar ou numa montanha.", "purpose"),
    ("Vou sempre para casa.", "purpose"),
    ("Corre constantemente para fora do Sol.", "purpose"),
    ("Ele sempre para o carro aqui.", "stop"),
    ("O comboio sempre para aqui.", "stop"),
    ("Ela nunca para de falar.", "stop"),
])
def test_para_after_an_adverb(sentence, sense):
    words = tokenize(sentence)
    i = next(k for k, w in enumerate(words) if w.strip(".,").lower() == "para")
    assert guess_sense(words, i) == sense
