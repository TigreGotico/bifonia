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
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: test
    path: test.jsonl
---

# bifonia — Portuguese Heterophonic Homograph Disambiguation

Labelled European-Portuguese (pt-PT) sentences for **27 heterophonic homographs** —
words with identical spelling whose pronunciation (IPA) depends on part of speech
or meaning, e.g. `para` (preposition `ˈpɐɾɐ` vs verb `ˈpaɾɐ`), `molho`
(sauce `ˈmoʎu` vs bundle `ˈmɔʎu`), `corte` (royal court `ˈkoɾtɨ` vs cut `ˈkɔɾtɨ`).

Useful for grapheme-to-phoneme / TTS front-ends and for POS disambiguation.

## Schema

| field | description |
|-------|-------------|
| `word` | the ambiguous headword |
| `pos` | the reading label (`NOUN` / `VERB` / `ADP` / `ADJ`) — the bucket that selects the IPA |
| `ipa` | European-Portuguese transcription of `word` in this reading |
| `sentence` | a natural sentence using `word` in that reading |
| `diacritized` | `word` with the disambiguating diacritic restored (e.g. `pêlo`, `fôrma`) |
| `diacritized_sentence` | the sentence with the diacritic restored on `word` |

## Splits

- `train`: 80% (stratified per word+POS)
- `test`: 20% (stratified per word+POS)

## Construction & validation

Sentences were generated with an LLM and then **validated for semantics and
grammaticality by a stronger independent model** plus rule-based audits against
[infopédia](https://www.infopedia.pt). Pronunciation labels were verified against
the dictionary's open/closed-vowel marks. Sentences whose homograph was in the
wrong reading, ungrammatical, or not European Portuguese were removed. Figurative
and idiomatic uses of the same reading (same pronunciation) are kept.

## Words

`acordo`, `acerto`, `cerro`, `choro`, `colher`, `começo`, `conserto`, `coro`,
`corte`, `forma`, `gosto`, `gozo`, `jogo`, `molho`, `olho`, `para`, `pelo`,
`peso`, `porto`, `posto`, `rego`, `seco`, `sede`, `sobre`, `tola`, `torre`,
`transtorno`.

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
