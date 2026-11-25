# Lects

<!-- agent-hint
covers: how bifonia treats Brazilian Portuguese, Uruguayan Portuguese and the African and Asian lects
source: `bifonia/data/lect_overrides.csv`, `bifonia/__init__.py:add_extra_diacritics`
generated: no
verify: diff <(sed -n '/^\x60\x60\x60python/,/^\x60\x60\x60/p' docs/lects.md | grep -vP '^\x60{3}' | python -I -) <(sed -n '/^\x60\x60\x60python/,/^\x60\x60\x60/p' docs/lects.md | grep -oP '(print\(.*\)\s+|^)# \K.*') && python -m pytest -q tests/test_lect_overrides.py tests/test_lang_tags.py
see: `docs/context_free_marks.md`, `docs/word_references.md`
-->

## Brazilian Portuguese

bifonia's pronunciations are European Portuguese, taken from Infopédia. A separate Brazilian table is not needed. For 123 of the 132 words, Brazilian Portuguese keeps the same contrast, with the same sense on the same stressed vowel. Many of these words were checked against the Brazilian transcription in Wiktionary, but the repository keeps no list of the checked words. The others follow the general rule: nouns have a closed vowel and the homographic verb forms have an open one, as in *o gosto* [o] and *eu gosto* [ɔ]. One of the 123 words varies: Brazilian speakers say the noun *esmero* with [e] or [ɛ]. bifonia marks it *esmêro*, which is an accepted Brazilian form. *fora* has the opposite pattern, an open adverb and a closed verb form, and Wiktionary gives the same two vowels for Brazil.

`add_extra_diacritics` marks the stressed-vowel quality with é/ê/ó/ô. The pt-BR lect of the phonemizer builds the other segments, such as final [ɪ], palatalised *t* and *d*, coda *s* and *r*, and vocalised *l*.

Nine words differ in Brazil:

- *para* has [a] in both senses. bifonia marks the verb *pára* and leaves the preposition unmarked. This is also correct for Brazil.
- *pelo* has [e] as both contraction and noun. bifonia marks the noun *pêlo* and the verb *pélo* and leaves the contraction unmarked. This is also correct for Brazil.
- *soma*, *abono*, *aceno*, *domo*, *empeno* and *desempeno* have a closed nasal stressed vowel in both senses. Before *m* or *n*, Brazilian Portuguese for the most part has no open mid vowel.
- *tola* is closed in Brazil in the colloquial sense 'judgement, head' (Michaelis). bifonia files that sense and the timber under one `head` sense, and no Brazilian source covers the timber. So pt-BR keeps the European marks for *tola*.

Pass `lang="pt-BR"`, or a sub-lect such as `pt-BR-x-rj`, to `add_extra_diacritics` to apply these differences. *abono*, *aceno*, *domo*, *empeno* and *desempeno* then stay unmarked. *soma* is always *sôma*, because an unmarked *soma* gets an open vowel from the tugalex pt-BR lexicons (`rjx` and `spx`). Every lect tag other than pt-BR and pt-UY keeps the European marks, and `None` means the default. The rows are in `bifonia/data/lect_overrides.csv`, each with its source. For *empeno* and *desempeno* the source is the nasal rule and related verb forms, not a dictionary entry.

`pt-UY` and its sub-lects take the same rows as pt-BR. This is an inference, not a sourced rule: no source states how Uruguayan Portuguese pronounces a stressed mid vowel before a nasal. Its vowel system is that of rural Rio Grande do Sul Portuguese (Rona 1965a, as reported by Meirelles 2006, p. 32; Meirelles 2006, pp. 148-149), and Brazilian Portuguese closes these vowels (Câmara Jr. 1977a, as reported by Lenzi & Brenner 2008, p. 59). The one recorded token, the Spanish place name *Canelones*, has a closed [õ] (Meirelles 2006, p. 143), and contact with Spanish gives no open [ɛ] or [ɔ].

```python
from bifonia import add_extra_diacritics

print(add_extra_diacritics("Ele soma os números."))                # Ele sóma os números.
print(add_extra_diacritics("Ele soma os números.", lang="pt-BR"))  # Ele sôma os números.
```

Do not use the IPA column of `heterophonic_homographs.csv` as a transcription for Brazilian, African or Timorese lects.

## African and Asian lects

The African and Asian lects pt-AO, pt-MZ, pt-CV, pt-GW, pt-ST, pt-TL and pt-MO keep the European marks, and bifonia has no override rows for them. No source found describes the Brazilian closing of a stressed mid vowel before *m* or *n* for any of these lects. For Angola and East Timor the sources point the other way: Luanda Portuguese has open nasal vowels (Cambolo & Osório 2022, pp. 203-205), and Timorese Portuguese does not nasalise a vowel before a nasal consonant (Albuquerque 2010, p. 278) and *ontem* is recorded with an open [ɔ] (Albuquerque 2011, p. 235). This evidence leads to an inference, not a sourced rule: the pt-BR rows do not apply to any of the seven lects. For pt-MZ, pt-CV, pt-GW, pt-ST and pt-MO the inference rests only on the absence of a source. Undolo (2014, pp. 183-185) and Cambolo & Osório (2022, pp. 198-200, with the summary on p. 210) report that Angolan speakers tend to open a stressed /e/ or /o/ where European Portuguese has a closed vowel, as in the noun *desgosto* and in *sede* 'thirst'. Both describe a variable tendency, not a categorical rule, so bifonia keeps the European marks for pt-AO.

---
[← Marks that need no context](context_free_marks.md) · [Index](index.md) · [Word references →](word_references.md)
