# bifonia

**Pronunciation disambiguation for European-Portuguese heterophonic homographs** —
words spelled identically whose pronunciation depends on **meaning**, not just part
of speech. Zero runtime dependencies, pure standard library.

`sede` is *thirst* (`ˈsedɨ`, closed e) or a *headquarters* (`ˈsɛdɨ`, open e); `forma`
is a *mould* (`ˈfoɾmɐ`) or a *shape* (`ˈfɔɾmɐ`); `molho` is *sauce* (`ˈmoʎu`) or a
*bundle* (`ˈmɔʎu`). A text-to-speech front-end that guesses wrong says the wrong word
out loud. bifonia picks the right reading — and therefore the right IPA — from context.

```python
from bifonia import tokenize, guess_sense, disambiguate

words = tokenize("Tinha tanta sede que bebi a garrafa toda.")
i = words.index("sede")
guess_sense(words, i)    # 'thirst'
disambiguate(words, i)   # 'ˈsedɨ'   (closed e)

words = tokenize("A sede da empresa fica em Lisboa.")
i = words.index("sede")
disambiguate(words, i)   # 'ˈsɛdɨ'   (open e — same spelling, different word)
```

## Why this is hard (and why POS-tagging isn't enough)

The obvious approach — tag the part of speech and pick the pronunciation from it —
**cannot work when two readings share a POS.** `sede` thirst and seat are *both
nouns*; `forma` mould and shape are *both nouns*; `corte` cut and court are both
nominal. A part-of-speech tagger labels them identically and is wrong on the
minority reading by construction.

It is easy to *look* like you've solved this and not have. Encyclopedic text
(Wikipedia) is **noun-skewed** — one reading dominates and the minority readings
barely occur — so a POS-tagger that just predicts the majority sense scores well.
The honest test is a **sense-balanced** set where the minority readings actually
appear.

bifonia ships that test: [**`bifonia-pt-homographs-gold`**](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-gold)
— 2,346 real sentences (OpenSubtitles, Wikipedia, web), ~40% of them on
POS-ambiguous readings, labelled independently of the engine. To our knowledge it
is the first real-text heterophonic-homograph benchmark for European Portuguese.

## Accuracy on the balanced real benchmark

| approach | all | POS-ambiguous (same-POS) |
|---|:---:|:---:|
| most-common (majority sense) | 48.2% | 72.0% |
| cue-only (curated wordlists, no POS, no learning) | 53.2% | 87.3% |
| Yarowsky decision list (trained) | 88.6% | 86.5% |
| spaCy `pt_core_news_lg` (POS → sense) | 85.8% | 74.0% |
| **bifonia rules (zero-dependency)** | **95.8%** | **97.6%** |

![Accuracy by approach on the balanced real benchmark](docs/img/realgold_approaches.png)

The headline: a strong neural POS-tagger drops to **chance-like 74% on the
POS-ambiguous half** — it has no signal to separate two senses that share a part of
speech — while a zero-dependency, meaning-keyed rule engine holds **97.6%**. On the
POS-*separable* half a tagger does fine (~93%); on the half that needs *meaning*,
only meaning works.

![Rules vs spaCy by subset](docs/img/realgold_subset.png)

A second finding worth its own line: **curated word-lists alone reach 87.3%** on the
hard cases, before any POS reasoning or learning — meaning lives in collocations.

## Two engines, one ensemble

| engine | needs a corpus? | how it decides |
|---|---|---|
| **rules** | no | context rules over part-of-speech + meaning cues in `.voc` wordlists |
| **learned** | yes | per-word Naive-Bayes / averaged perceptron over context features |

The rule engine is self-contained and needs no training data — the right fit for a
low-resource language. The learned models are trained from the labelled corpus and
serve as a full-roster baseline and a corpus-QC engine. `guess_sense` is a **per-word
ensemble**: each word is served by whichever engine wins on held-out behavioural
data, so the combined system never does worse than the rules.

A consistent result across this project: **models trained on the corpus do not beat
the rules out-of-distribution.** Because the corpus is rule-labelled, a model can at
best mimic the rules in-distribution and generalises worse on real text — true for
Naive-Bayes, the averaged perceptron, and the classic Yarowsky decision list alike.
The rules remain the predictor; an external POS tag can be fused on top:

```python
# hybrid: trust an external tagger only where POS uniquely picks a sense, else rules
guess_sense(words, i, postag="NOUN")
```

On the encyclopedic wild set this hybrid ensemble reaches **95.3%** (rules 90.7%,
spaCy 91.6%) — the tagger helps on POS-separable words, the rules carry the rest.

## Install

```bash
pip install bifonia          # zero dependencies, pure standard library
```

## API

```python
from bifonia import (tokenize, is_ambiguous, guess_sense, guess_pos,
                     disambiguate, add_extra_diacritics)

sentence = "Resolveu o problema desta forma simples."
words = tokenize(sentence)
i = words.index("forma")

guess_sense(words, i)                    # 'shape'
disambiguate(words, i)                   # 'ˈfɔɾmɐ'
disambiguate(words, i, sense="mould")    # 'ˈfoɾmɐ'  (override)
add_extra_diacritics(sentence)           # '...desta fórma simples.'  (acute = open vowel)
```

`add_extra_diacritics` rewrites each homograph with a disambiguating diacritic
(acute → open vowel, circumflex → closed) that a downstream grapheme-to-phoneme
stage can read directly.

## Datasets

All on the Hugging Face Hub, schema `{word, sense, pos, ipa, sentence}`:

- [**`bifonia-pt-homographs-gold`**](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-gold)
  — 2,346 real, sense-balanced, independently-labelled sentences (43 words). The
  honest evaluation set.
- [`bifonia-pt-homographs`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs)
  — the labelled training/synthetic corpus with stratified splits.
- [`bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild)
  — the encyclopedic (noun-skewed) wild set.

## Architecture

Layered and dependency-free, so a cue behaves identically in the rules and in the
model features:

```
bifonia/text.py      strip / accent-fold / proximity-weighted cue scoring   (shared, stdlib)
bifonia/cues.py      SENSE_CUES + FEATURE_CUES — the cue registry            (single source of truth)
bifonia/scoring.py   context rules + one generic cue resolver
bifonia/features.py  language-agnostic features, incl. CUE:<sense> from the registry
bifonia/locale/<lang>/*.voc   editable context + meaning wordlists
baselines.py         a Baseline protocol + zero-dep baselines for benchmarking
```

The meaning cues that disambiguate same-spelling readings live in **one declarative
registry** (`bifonia/cues.py`) read by both the rule resolver and the model feature
extractor. **Adding a homograph is data-only**: drop the `.voc` wordlist(s) and add
one registry line — no new code. Porting to a related language means supplying
`.voc` files and (optionally) a corpus; the algorithm carries no hardcoded
Portuguese.

## Coverage

**131 homographs, 265 sense readings.** The disambiguation axis is **vowel quality**
(open ɔ/ɛ vs closed o/e). The hard core is the same-POS pairs a tagger cannot touch:
`sede`, `corte`, `forma`, `molho`, `bola`, `cor`, `lobo`, `polo`, `tola`. Per-word
IPA, senses, and diacritized forms are in [`docs/word_references.md`](docs/word_references.md).

## See also

- [`docs/benchmarks.md`](docs/benchmarks.md) — full benchmarks, subsets, and plots
- [`docs/methodology.md`](docs/methodology.md) — algorithm and features
- [`docs/usage.md`](docs/usage.md) — full API reference
- [`baselines.py`](baselines.py) · [`benchmark_ood.py`](benchmark_ood.py) — reproduce the numbers
- [`examples/basic_usage.py`](examples/basic_usage.py) — runnable demo
