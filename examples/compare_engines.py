"""
Compare the two engines behind bifonia on the same sentences.

`guess_sense` serves each word with a per-word ensemble — the learned model where
it wins on held-out data, the rule engine otherwise. This example shows all three
predictions side by side so the routing is visible.
"""
from bifonia import tokenize, is_ambiguous, guess_sense
from bifonia.scoring import guess_pos as _rule_pos, resolve_sense as _rule_resolve
from bifonia.model import SenseModel, PERCEPTRON_PATH

_model = SenseModel.load(str(PERCEPTRON_PATH))


def rule_sense(words, i):
    return _rule_resolve(words[i], words, i, _rule_pos(words, i))


def model_sense(words, i):
    w = words[i]
    return _model.predict(w, words, i) if _model.has(w) else "—"


sentences = [
    "A sede da empresa fica em Lisboa.",          # sede → seat   (rule-routed)
    "Tinha tanta sede que bebi água fresca.",      # sede → thirst (rule-routed)
    "O jogo de futebol foi emocionante.",          # jogo → game   (model-routed)
    "Eu jogo futebol todos os domingos.",          # jogo → play   (model-routed)
    "O olho esquerdo ficou inchado.",              # olho → eye    (model-routed)
    "Eu olho sempre para os dois lados.",          # olho → look   (model-routed)
]

print(f"{'word':<8} {'rules':<9} {'model':<9} {'ensemble':<9}  sentence")
for sentence in sentences:
    words = tokenize(sentence)
    for i, w in enumerate(words):
        if is_ambiguous(w):
            r, m, e = rule_sense(words, i), model_sense(words, i), guess_sense(words, i)
            route = _model.route(w) if _model.has(w) else "rules"
            print(f"{w:<8} {r:<9} {m:<9} {e:<9}  ({route})  {sentence}")
            break
