# bifonia

Pronunciation disambiguation for European-Portuguese **heterophonic homographs** —
words spelled identically whose pronunciation depends on **meaning**.

`sede` is *thirst* (`ˈsedɨ`, closed e) or a *headquarters* (`ˈsɛdɨ`, open e); `forma`
is a *mould* (`ˈfoɾmɐ`) or a *shape* (`ˈfɔɾmɐ`); `molho` is *sauce* (`ˈmoʎu`) or a
*bundle* (`ˈmɔʎu`). A text-to-speech front-end that guesses wrong says the wrong word
out loud. bifonia picks the right reading — and therefore the right IPA — from context.

```python
from bifonia import tokenize, is_ambiguous, guess_sense, disambiguate

words = tokenize("Tinha tanta sede que bebi a garrafa toda.")
i = words.index("sede")
guess_sense(words, i)    # 'thirst'
disambiguate(words, i)   # 'ˈsedɨ'
```

## Why meaning, not part of speech

The obvious approach — tag the part of speech and pick the pronunciation from it —
cannot work when two readings share a POS. `sede` thirst and seat are **both nouns**;
`corte` cut and court are both nominal; `forma` mould and shape likewise. A POS tagger
labels them identically and is wrong on the minority reading by construction. bifonia
keys every reading on **`sense`** (a meaning slug) and resolves the meaning directly.

## Two interchangeable engines

| engine | needs a corpus? | how it decides |
|--------|-----------------|----------------|
| **rules** | no | hand-written context rules + wordlists (`.voc`) |
| **learned** | yes | per-word Naive-Bayes / averaged-perceptron over context features |

The rule engine is self-contained and needs no training data — the right fit for a fork
of a low-resource language. The learned models are trained from the labelled corpus and
generalise better where enough data exists. `guess_sense` uses a **per-word ensemble**:
each word is served by whichever engine scores at least as well on held-out data, so the
combined system never does worse than the rules alone. Both are pure Python with no heavy
runtime dependencies.

## Accuracy

Sense prediction, measured two ways:

| approach | synthetic test | real-world (OOD) |
|----------|:--------------:|:----------------:|
| most-common baseline | 52.7% | 47.5% |
| spaCy POS → sense | 65.7% | 81.4% |
| Stanza POS → sense | 75.5% | 82.5% |
| rules (no corpus) | 94.0% | 83.2% |
| Naive-Bayes | 98.1% | 86.7% |
| averaged perceptron | 99.0% | 89.6% |
| **shipped ensemble** | **95.7%** | **89.1%** |

The *synthetic* column is the held-out split of the generated training corpus, balanced
across senses; the *OOD* column is real sentences from
[`bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild).
The two columns answer different questions. The synthetic set is balanced, so it exposes
how badly POS tagging handles minority readings (a tagger cannot separate two senses that
share a part of speech — both score 0% on `sede`/thirst). Real text is skewed toward the
majority readings POS taggers do get right, which lifts them to ~82% — yet the
meaning-aware models still win, and the perceptron leads by ~7 points. Reproduce with
`python benchmark_tagger.py` (synthetic) and `python benchmark_ood.py` (OOD).

## Install

```bash
pip install -e . --no-deps
```

No required dependencies. `ovos_spec_tools` is used for locale resolution when present and
falls back to the standard library otherwise.

## API

```python
from bifonia import (tokenize, is_ambiguous, guess_sense, guess_pos,
                     disambiguate, add_extra_diacritics)

sentence = "Resolveu o problema desta forma simples."
words = tokenize(sentence)
i = words.index("forma")

guess_sense(words, i)              # 'shape'
guess_pos(words, i)                # 'NOUN'   (descriptive)
disambiguate(words, i)             # 'ˈfɔɾmɐ'
disambiguate(words, i, sense="mould")   # 'ˈfoɾmɐ'  (override)
add_extra_diacritics(sentence)     # '...desta fórma simples.'  (acute = open vowel)
```

`add_extra_diacritics` rewrites each homograph with a disambiguating diacritic
(acute → open vowel, circumflex → closed) that a downstream grapheme-to-phoneme stage
can read directly.

## Datasets

Both on the Hugging Face Hub, schema `{word, sense, pos, ipa, sentence}`:

- [`bifonia-pt-homographs`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs)
  — 56,891 labelled sentences over 27 words, with stratified train/test splits, for
  training and synthetic evaluation.
- [`bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild)
  — real Wikipedia and web sentences, an out-of-distribution test set.

## Word coverage

27 homographs: `acordo`, `acerto`, `cerro`, `choro`, `colher`, `começo`, `conserto`,
`coro`, `corte`, `forma`, `gosto`, `gozo`, `jogo`, `molho`, `olho`, `para`, `pelo`,
`peso`, `porto`, `posto`, `rego`, `seco`, `sede`, `sobre`, `tola`, `torre`, `transtorno`.

Per-word IPA, senses, and diacritized forms are in [`docs/words.md`](docs/words.md).

## Project layout

- `bifonia/data/corpus.jsonl` — the labelled corpus (single source of truth).
- `bifonia/data/heterophonic_homographs.csv` — the `word,sense,pos,ipa` table.
- `bifonia/data/sense_model_{nb,perceptron}.json` — trained models (JSON weights).
- `bifonia/locale/<lang>/*.voc` — context wordlists, one term per line, editable.
- `bifonia/features.py` — language-agnostic feature extraction (shared by train and inference).

Porting to a related language means supplying a corpus and `.voc` files and retraining —
the algorithm carries no hardcoded Portuguese.

## See also

- [`docs/methodology.md`](docs/methodology.md) — algorithm, features, and benchmarks
- [`docs/usage.md`](docs/usage.md) — full API reference
- [`docs/words.md`](docs/words.md) — per-word pronunciation notes
- [`docs/diacritics_restoration.md`](docs/diacritics_restoration.md) — the diacritics-restoration task
- [`examples/basic_usage.py`](examples/basic_usage.py) — runnable demo
- [`train.py`](train.py) · [`benchmark_tagger.py`](benchmark_tagger.py) · [`benchmark_ood.py`](benchmark_ood.py)
