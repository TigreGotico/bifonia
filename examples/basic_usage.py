"""
Basic usage of bifonia — Portuguese heterophonic homograph disambiguation.

Each homograph is resolved by MEANING (`sense`), which selects the IPA. Two
senses can share a part of speech (sede thirst/seat are both nouns), so the
meaning — not the POS — is what determines the pronunciation.
"""

from bifonia import tokenize, is_ambiguous, guess_sense, disambiguate, add_extra_diacritics

sentences = [
    "Vou para casa depois do trabalho.",            # para  → purpose (ˈpɐɾɐ)
    "O autocarro para em frente ao hospital.",      # para  → stop    (ˈpaɾɐ)
    "O gosto do vinho é excelente.",                # gosto → taste   (ˈgoʃtu)
    "Eu gosto de música clássica.",                 # gosto → like    (ˈgɔʃtu)
    "A sede da empresa fica em Lisboa.",            # sede  → seat    (ˈsɛdɨ)
    "Tinha tanta sede que bebi a garrafa toda.",    # sede  → thirst  (ˈsedɨ) — same POS!
    "O corte de cabelo ficou perfeito.",            # corte → cut     (ˈkɔɾtɨ)
    "A corte do rei reunia-se no salão.",           # corte → court   (ˈkoɾtɨ)
    "Untou a forma antes de deitar a massa.",       # forma → mould   (ˈfoɾmɐ)
    "Resolveu o problema desta forma simples.",     # forma → shape   (ˈfɔɾmɐ)
]

for sentence in sentences:
    print(f"\n  {sentence}")
    words = tokenize(sentence)
    for i, word in enumerate(words):
        if is_ambiguous(word):
            sense = guess_sense(words, i)
            ipa = disambiguate(words, i)
            print(f"    {word!r:10s} → {sense:<8} [{ipa}]")

    diacritized = add_extra_diacritics(sentence)
    if diacritized != sentence.lower():
        print(f"    diacritized: {diacritized}")
