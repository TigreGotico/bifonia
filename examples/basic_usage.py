"""
Basic usage of bifonia — Portuguese heterophonic homograph disambiguation.
"""

from bifonia import tokenize, is_ambiguous, disambiguate, add_extra_diacritics

sentences = [
    "Vou para casa depois do trabalho.",            # para = ADP
    "O autocarro para em frente ao hospital.",      # para = VERB (stops)
    "O gosto do vinho é excelente.",                # gosto = NOUN (closed-o, taste)
    "Eu gosto de música clássica.",                 # gosto = VERB (open-ɔ, I like)
    "O gato perdeu muito pelo no sofá.",            # pelo = NOUN (fur)
    "Passou pelo parque a caminho de casa.",        # pelo = ADP (por+o)
    "Seco as mãos antes de tocar nos alimentos.",   # seco = VERB (I dry)
    "O amendoim seco é vendido em feiras.",         # seco = ADJ (dry)
    "O posto de saúde fica ao fundo da rua.",       # posto = NOUN (health post)
    "Posto fotos de viagem nas redes sociais.",     # posto = VERB (I post/upload)
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
