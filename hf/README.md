---
license: apache-2.0
language:
- pt
task_categories:
- token-classification
tags:
- portuguese
- european-portuguese
- pos-tagging
- homographs
- grapheme-to-phoneme
- text-to-speech
- ipa
pretty_name: bifonia — Portuguese Heterophonic Homographs
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: test
    path: test.jsonl
---

# bifonia: Portuguese Heterophonic Homograph Disambiguation

Labelled European-Portuguese (pt-PT) sentences for **131 heterophonic homographs**:
words with identical spelling whose pronunciation (IPA) depends on part of speech
or meaning, e.g. `para` (preposition `ˈpɐɾɐ` vs verb `ˈpaɾɐ`), `molho`
(sauce `ˈmoʎu` vs bundle `ˈmɔʎu`), `corte` (royal court `ˈkoɾtɨ` vs cut `ˈkɔɾtɨ`).

Useful for grapheme-to-phoneme / TTS front-ends and for POS disambiguation.

## Schema

| field | description |
|-------|-------------|
| `word` | the ambiguous headword |
| `sense` | the **meaning** slug that selects the IPA (e.g. `thirst` / `seat`, `mould` / `shape`), the bucket key |
| `pos` | descriptive part of speech of this reading (`NOUN` / `VERB` / `ADP` / `ADJ`); may repeat across senses |
| `ipa` | European-Portuguese transcription of `word` in this reading |
| `sentence` | a natural sentence using `word` in that meaning |
| `diacritized` | `word` with the disambiguating diacritic restored (acute = open vowel, circumflex = closed): `sêde`/`séde`, `fôrma`/`fórma` |
| `diacritized_sentence` | the sentence with that diacritic restored on `word`, the target for a diacritics-restoration model |

The reading is keyed on **meaning, not POS**: two senses can share a part of
speech (`sede` thirst and seat are both nouns, distinguished only by their open/
closed vowel), so `sense`, not `pos`, is the label a model should predict.

## Splits

- `train`: 80% (124,339 sentences), `test`: 20% (30,944 sentences)
- shuffled and stratified per `(word, sense)` with a fixed seed, so each bucket
  is represented i.i.d. in both splits (no tail/ordering skew).

## Construction & validation

Sentences were generated with an LLM and then **validated for semantics and
grammaticality by a stronger independent model** plus rule-based audits against
[infopédia](https://www.infopedia.pt). Pronunciation labels were verified against
the dictionary's open/closed-vowel marks. Sentences whose homograph was in the
wrong reading, ungrammatical, or not European Portuguese were removed. Figurative
and idiomatic uses of the same reading (same pronunciation) are kept.

## Words

The 131 words and their 297 readings are in
[`bifonia/data/heterophonic_homographs.csv`](https://github.com/TigreGotico/bifonia/blob/dev/bifonia/data/heterophonic_homographs.csv),
and [`docs/word_references.md`](https://github.com/TigreGotico/bifonia/blob/dev/docs/word_references.md)
describes each one.

## Citation

```bibtex
@misc{bifonia2025,
  title  = {bifonia: Portuguese Heterophonic Homograph Disambiguation},
  author = {JarbasAI / TigreGotico},
  year   = {2025},
  url    = {https://github.com/TigreGotico/bifonia},
  license = {Apache-2.0},
}
```
