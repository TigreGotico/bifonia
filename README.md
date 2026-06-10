# bifonia

Rule-based Portuguese heterophonic homograph disambiguation for TTS.

Identifies the correct IPA pronunciation of 27 words whose orthography is identical
but whose phonology depends on part of speech (NOUN vs VERB vs ADP vs ADJ).

```python
from bifonia import tokenize, is_ambiguous, disambiguate

words = tokenize("O autocarro para em frente ao hospital.")
for i, w in enumerate(words):
    if is_ambiguous(w):
        print(w, "→", disambiguate(words, i))
# para → ˈpaɾɐ  (VERB reading — stops)
```

## Accuracy

98.46 % on a corpus of ~13 500 labelled sentences — vs 81.9 % for Stanza and 66.5 %
for spaCy on the same test set.  (Three-way homographs `para`/`pelo`/`sobre` are harder:
90–95 % each; all other words score 98–100 %.)

## How it works

Context scoring assigns integer points to each candidate POS based on the ±4-word
window: determiners, pronouns, passive auxiliaries, infinitive markers, copular verbs,
degree adverbs.  The highest-scoring POS wins; ties fall back to a per-word default.

See [`docs/methodology.md`](docs/methodology.md) for the full algorithm description and
benchmark comparison.

## Install

```bash
pip install -e . --no-deps
```

## API

```python
from bifonia import tokenize, is_ambiguous, guess_pos, disambiguate, add_extra_diacritics

words = tokenize("Vou para casa depois do trabalho.")
for i, word in enumerate(words):
    if is_ambiguous(word):
        pos  = guess_pos(words, i)          # "ADP"
        ipa  = disambiguate(words, i)       # "ˈpɐɾɐ"
        rich = add_extra_diacritics("Vou para casa depois do trabalho.")

print(rich)  # "Vou para casa depois do trabalho."  (unchanged — ADP needs no diacritic)
```

## Word coverage

27 words across NOUN / VERB / ADP / ADJ:
`acordo`, `acerto`, `cerro`, `choro`, `colher`, `começo`, `conserto`, `coro`, `corte`,
`forma`, `gosto`, `gozo`, `jogo`, `molho`, `olho`, `para`, `pelo`, `peso`, `porto`,
`posto`, `rego`, `seco`, `sede`, `sobre`, `tola`, `torre`, `transtorno`.

See [`docs/words.md`](docs/words.md) for IPA, diacritized forms, and usage notes per word.

## See also

- [`docs/methodology.md`](docs/methodology.md) — dataset construction, algorithm, benchmark
- [`docs/usage.md`](docs/usage.md) — full API reference
- [`docs/diacritics_restoration.md`](docs/diacritics_restoration.md) — framing as a diacritics-restoration task and ML model suggestions
- [`examples/basic_usage.py`](examples/basic_usage.py) — runnable demo
- [`dataset.py`](dataset.py) — export corpus to CSV / JSON for HuggingFace publication
- [`benchmark_tagger.py`](benchmark_tagger.py) — reproduce the accuracy comparison
