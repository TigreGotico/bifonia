"""
Basic usage of homogr — Portuguese heterophonic homograph disambiguation.
"""

from homogr import tokenize, is_ambiguous, disambiguate, add_extra_diacritics

sentences = [
    "Vou para casa depois do trabalho.",          # para = ADP
    "O autocarro para em frente ao hospital.",    # para = VERB
    "O gosto do vinho é excelente.",              # gosto = NOUN (closed-o)
    "Eu gosto de música clássica.",               # gosto = VERB (open-ɔ)
    "Comprei um pelo de gato para o museu.",      # pelo = NOUN + para = ADP
]

for sentence in sentences:
    print(f"\n  {sentence}")
    words = tokenize(sentence)
    for i, word in enumerate(words):
        if is_ambiguous(word):
            ipa = disambiguate(words, i)
            print(f"    {word!r:12s} → [{ipa}]")

    diacritized = add_extra_diacritics(sentence)
    if diacritized != sentence.lower():
        print(f"    diacritized: {diacritized}")
