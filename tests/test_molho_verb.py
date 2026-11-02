"""The verb molhar ("eu molho") takes the open reading, like the bundle noun."""
import pytest

from bifonia import add_extra_diacritics, disambiguate, guess_pos, guess_sense, tokenize
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


# A clause-initial bare sauce noun before a modifier is not the verb.
_CLAUSE_INITIAL_SAUCE = [
    "Molho picante é o meu preferido.",
    "Molho bechamel leva farinha e leite.",
    "Molho branco ou vermelho?",
    "Molho verde acompanha o peixe.",
    "Molho inglês dá sabor à carne.",
    "Molho caseiro é sempre melhor.",
    "Molho, sal e pimenta a gosto.",
    "Molho pesto com massa fresca.",
    "Molho agridoce combina com frango.",
    "Qual molho, o branco?",
    "Qual molho, o de tomate?",
    "Com o peixe, molho verde.",
]

# An adverbial "de" locution after the verb does not make it the noun.
_VERB_WITH_ADVERBIAL_DE = [
    "Molho de manhã as plantas.",
    "Molho de novo os pés.",
    "Molho de propósito o chão.",
    "Molho de seguida as plantas.",
    "Já molho de manhã, já rego à tarde.",
    "Eu molho de manhã as plantas.",
    "Molho-as de manhã.",
]


@pytest.mark.parametrize("sentence", _CLAUSE_INITIAL_SAUCE)
def test_clause_initial_sauce_keeps_closed_mark(sentence):
    out = add_extra_diacritics(sentence)
    assert "ôlho" in out and "ólho" not in out


@pytest.mark.parametrize("sentence", _VERB_WITH_ADVERBIAL_DE)
def test_verb_before_adverbial_de_keeps_open_mark(sentence):
    out = add_extra_diacritics(sentence)
    assert "ólho" in out and "ôlho" not in out


# Clause context decides a clause-initial "molho". Each class lists sauce nouns
# (closed ô) and verbs of molhar (open ó). A clause-initial verb needs its object
# right after it, or after an adverb or a fixed locution.
_CLAUSE_CASES = {
    "relative qual": (
        ["Com qual molho fica melhor o arroz?", "Qual molho preferes?",
         "Quais molhos? Qual molho, o verde?"],
        ["A mangueira com a qual molho as plantas está rota.",
         "As toalhas com as quais molho a cara são novas.",
         "O regador com o qual molho a horta é velho.",
         "O balde no qual molho a esfregona está partido.",
         "Os vasos nos quais molho a terra são grandes.",
         "Esta é a razão pela qual molho as plantas à noite.",
         "A esponja pela qual molho o pano é amarela."],
    ),
    "quantifier or intensifier": (
        ["Molho muito picante faz-me mal.", "Molho bastante picante…",
         "Molho pouco salgado é melhor.", "Molho nenhum me agrada.",
         "Molho algum me convence.", "Molho demasiado salgado estraga tudo.",
         "Molho muito doce não combina com peixe."],
        ["Molho muito as plantas no verão.", "Molho pouco a relva.",
         "Molho bastante os vasos de barro.", "Molho muitas vezes as plantas."],
    ),
    "contracted preposition": (
        ["Molho à bolonhesa é o meu preferido.", "Molho ao lado, por favor.",
         "Molho no prato, não na mesa.", "Molho dessa marca não compro.",
         "Molho a mais estraga o prato.", "Molho nas batatas é bom.",
         "Molho desses é caro.", "Molho à parte, por favor.",
         "Molho à espanhola leva pimento.", "Molho da casa é servido à parte.",
         "Molho do assado fica no tacho."],
        ["Molho à pressa as plantas.", "Molho as roupas na bacia.",
         "Molho os pés no mar."],
    ),
    "adverbial de": (
        ["Molho de véspera sabe sempre melhor.", "Molho de noite não faço.",
         "Molho de véspera fica mais apurado.", "Molho de manhã não preparo.",
         "Molho de noite é pesado."],
        ["Molho de véspera o bacalhau.", "Molho de noite as sementes.",
         "Molho de manhã cedo os vasos.", "Molho de manhã as plantas."],
    ),
    "preposition, coordination or adverb": (
        ["Molho com natas é ótimo.", "Molho sem glúten é caro.",
         "Molho e azeite são a base.", "Molho mais picante é melhor.",
         "Molho também é comida.", "Molho para massa fica melhor com natas."],
        ["Molho agora os pés.", "Molho hoje as plantas.",
         "Molho aqui os pés.", "Molho sozinho as plantas.", "Molho mais as plantas no verão.",
         "Molho cedo as plantas.",
         "Molho sozinha o jardim."],
    ),
    "object pronoun, numeral or verb list": (
        ["Molho, sal e pimenta a gosto.", "Molho verde acompanha o peixe.",
         "Molho inglês dá sabor à carne."],
        ["Molho tudo.", "Molho isto primeiro.", "Molho isso já.", "Molho tudo de uma vez.",
         "Molho duas vezes as plantas.", "Molho cada planta com cuidado.", "Molho cada vaso à vez."],
    ),
    "de + noun before any verb": (
        ["Molho de soja tempera o arroz.", "Molho de natas enriquece o prato.",
         "Molho de cogumelos acompanhou o bife.", "Molho de caril domina o sabor.",
         "Molho de tomate, o clássico.", "Molho de natas e o arroz, por favor.",
         "Molho de limão realça o peixe.", "Molho de queijo cobre a massa.",
         "Molho de tomate engrossa o caldo."],
        ["Molho de manhã as plantas.", "Molho de véspera o bacalhau.",
         "Molho de leve a toalha.", "Molho de joelhos o chão da cozinha.",
         "Molho de hora a hora as plantas.", "Molho de cima as folhas da alface.",
         "Molho de dia a relva.", "Molho de graça os vasos do vizinho."],
    ),
    "possessive after a contracted preposition": (
        ["Molho da minha avó é o melhor.", "Molho de tomate da minha mãe é famoso.",
         "Molho igual ao meu não há.", "Molho do meu pai leva vinho.",
         "Molho das nossas tias é secreto.", "Molho do nosso restaurante tem fama.",
         "Molho dos meus sonhos é este."],
        ["Molho os meus pés.", "Molho a minha toalha na bacia.",
         "Molho as minhas plantas na varanda."],
    ),
    "determiner after a coordinator": (
        ["Molho picante e a cerveja chegaram.", "Molho branco ou o vermelho?",
         "Molho como o da avó não há.", "Molho verde e o pão chegaram.",
         "Molho rosa ou a maionese?", "Molho doce como o da tia não há."],
        ["Molho os pés e as mãos.", "Molho as toalhas e os lençóis.",
         "Molho o pão e como-o."],
    ),
    "subordinate clause": (
        ["Molho que sobrou é para amanhã.", "Molho picante que arde é bom.",
         "Molho quando arrefece fica espesso.",
         "Molho picante quando está frio sabe melhor."],
        ["Molho as plantas quando está calor.", "Molho sempre as plantas que comprei.",
         "Molho os vasos porque a terra está seca."],
    ),
}
_NOUN_CASES = [(k, t) for k, (nouns, _) in _CLAUSE_CASES.items() for t in nouns]
_VERB_CASES = [(k, t) for k, (_, verbs) in _CLAUSE_CASES.items() for t in verbs]


def test_every_class_has_three_nouns_and_three_verbs():
    assert all(len(n) >= 3 and len(v) >= 3 for n, v in _CLAUSE_CASES.values())


@pytest.mark.parametrize("kind, sentence", _NOUN_CASES)
def test_clause_context_noun(kind, sentence):
    out = add_extra_diacritics(sentence)
    assert "ôlho" in out and "ólho" not in out


@pytest.mark.parametrize("kind, sentence", _VERB_CASES)
def test_clause_context_verb(kind, sentence):
    out = add_extra_diacritics(sentence)
    assert "ólho" in out and "ôlho" not in out


# The verb molhar is rare next to the sauce noun, so a clause-initial "molho"
# is the noun unless its object follows. Headings and sauce subjects with any
# verb stay closed. Verbs with no adjacent object take the noun reading: these
# are the accepted cost of that default.
_NOUN_DEFAULT_HEADINGS = [
    "Molho sem lactose.", "Molho para salada.", "Molho com natas e cogumelos.",
    "Molho com natas engorda.", "Molho picante engrossa o caldo.",
    "Molho caseiro tempera a carne.", "Molho verde, a minha especialidade.",
    "Molho bechamel, o mais fácil de fazer.", "Molho cada vez mais caro, é o que é.",
]
_VERBS_READ_AS_NOUN = [
    "Molho de vez em quando as plantas.",
    "Molho no lavatório a roupa.",
    "Molho nas mãos a cara.",
    "Molho na bacia os pés cansados.",
    "Molho em água morna as sementes.",
    "Molho com cuidado.",
    "Molho com cuidado as folhas.",
    "Molho para não secar.",
    "Molho e seco a loiça.",
    "Molho em água fria o bacalhau.",
    "Molho com a mangueira o jardim.",
    "Molho para amaciar a carne.",
    "Molho e escorro a massa.",
    "Molho três vezes por dia.",
    "Molho, rego e podo as plantas.",
    "Molho, lavo e seco a roupa.",
    "Molho no meu lavatório a camisola.",
    "Molho na minha bacia os pés.",
    "Molho com a minha mangueira o jardim.",
    "Molho ou rego as plantas.",
    "Molho e a seguir seco a loiça.",
    "Molho sempre que me lembro.",
    "Molho sempre que faz calor.",
    "Molho logo que me levanto.",
    "Molho já que está calor.",
]


@pytest.mark.parametrize("sentence", _NOUN_DEFAULT_HEADINGS + _VERBS_READ_AS_NOUN)
def test_clause_initial_molho_without_object_is_the_noun(sentence):
    out = add_extra_diacritics(sentence)
    assert "ôlho" in out and "ólho" not in out


@pytest.mark.parametrize("sentence", [
    "Molho as plantas todos os dias.", "Eu molho o pão no café.", "Molho-as de manhã.",
    "A mangueira com a qual molho as plantas está rota.", "Molho de manhã as plantas.",
    "Não molho as plantas.", "Molho sempre as plantas.",
    "Molho todas as plantas de manhã.", "Molho todos os dias as plantas.",
])
def test_clear_verb_signal_keeps_the_verb(sentence):
    out = add_extra_diacritics(sentence)
    assert "ólho" in out and "ôlho" not in out
