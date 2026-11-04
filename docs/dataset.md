# bifonia: Portuguese Homograph Disambiguation Dataset

## Overview

A labeled sentence corpus for **heterophonic homograph disambiguation** in European Portuguese. Heterophonic homographs are words spelled identically but pronounced differently depending on their **meaning** (`sense`). Correct disambiguation matters for TTS pipelines.

Example: **"sede"**, both senses are nouns, distinguished only by vowel quality:
- *thirst* (closed e): /ˈsedɨ/, as in "Tenho *sede* depois do exercício."
- *seat / HQ* (open ɛ): /ˈsɛdɨ/, as in "A *sede* da empresa fica em Lisboa."

Example: **"para"**
- *purpose* (ADP, preposition to/for): /ˈpɐɾɐ/, as in "Vou *para* casa."
- *stop* (VERB, parar): /ˈpaɾɐ/, as in "O autocarro *pára*."

The bucket key is the **meaning** (`sense`), not part of speech. Each record also carries a `pos` attribute, the dominant grammatical reading of that sense, which is descriptive only and may repeat across senses of the same word (both senses of `sede` are `NOUN`).

---

## Statistics

| Metric | Value |
|--------|-------|
| Total sentences (`bifonia/data/corpus.jsonl`) | 155,204 |
| Train / test split (`hf/`) | 124,339 / 30,944 |
| Ambiguous words | 131 |
| Senses with sentences | 286 of the 297 in `heterophonic_homographs.csv` |
| POS attributes | NOUN, VERB, ADP, ADJ |
| Language | European Portuguese (pt-PT) |
| License | Apache-2.0 |

The committed `hf/` split holds 79 sentences that `corpus.jsonl` does not hold ([#44](https://github.com/TigreGotico/bifonia/issues/44)). `python scripts/dataset.py --out <dir>` prints the corpus totals. The snippet under [Usage](#usage) prints the sentence count of each sense of a word.

---

## Schema

`bifonia/data/corpus.jsonl` is the single source of truth: one JSON record per line.

| Field | Type | Description |
|-------|------|-------------|
| `word` | str | The ambiguous word (lowercase, no diacritics) |
| `sense` | str | Meaning slug, the label to predict (e.g. `thirst`, `seat`) |
| `pos` | str | Descriptive UPOS attribute of that sense: NOUN, VERB, ADP, or ADJ |
| `ipa` | str | IPA transcription this `(word, sense)` reading carries |
| `sentence` | str | Full sentence context using the plain (undiacritized) word form |

`bifonia/data/heterophonic_homographs.csv` carries the per-reading IPA table with columns `word,sense,pos,ipa` (one row per `(word, sense)`).

---

## ML tasks

This dataset supports:

1. **Sense disambiguation**: predict the `sense` of an ambiguous word given its sentence context. Input: `(sentence, word, word_index)`. Output: `sense` (or `ipa`).
2. **TTS pronunciation prediction**: given a sentence, predict the IPA of each ambiguous word. `bifonia` ships both a corpus-free rule engine and corpus-trained learned models (Naive-Bayes and an averaged perceptron) trained from this corpus. See `docs/methodology.md` for the accuracy comparison on a synthetic split and on a real-text OOD set.
3. **Portuguese sense disambiguation**: a targeted sub-task for the 131 homograph types in the roster.

---

## Splits

`python scripts/dataset.py --hf` generates stratified 80/20 train/test splits under `hf/`:

```
hf/
  train.jsonl   # 80% per (word, sense) stratum
  test.jsonl    # 20% per (word, sense) stratum
```

Sentences are shuffled and stratified per `(word, sense)` with a fixed seed, so train and test are independently and identically distributed. This property matters for downstream consumers such as a BiLSTM sense classifier.

---

## Usage

```python
from bifonia.corpus import CORPUS, iter_records

# Iterate all labeled sentences
for word, sense, sentence in iter_records():
    print(word, sense, sentence[:60])

# Access by word
for sense, sentences in CORPUS["sede"].items():
    print(f"sede/{sense}: {len(sentences)} sentences")
```

Disambiguation:
```python
from bifonia import tokenize, is_ambiguous, guess_sense, disambiguate

words = tokenize("A sede de conhecimento move-nos.")
for i, w in enumerate(words):
    if is_ambiguous(w):
        print(f"{w} → {guess_sense(words, i)} → [{disambiguate(words, i)}]")
# sede → thirst → [ˈsedɨ]
```

---

## Hugging Face datasets

Three datasets on the Hub. Each record has the fields `word`, `sense`, `pos`, `ipa` and `sentence`:

- [`TigreGotico/bifonia-pt-homographs`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs): the committed `hf/` split, 124,339 train and 30,944 test sentences over 131 words, for training and synthetic evaluation. It adds the `diacritized` and `diacritized_sentence` fields.
- [`TigreGotico/bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild): 6,640 real Wikipedia and web sentences over 110 words, forming an out-of-distribution (OOD) test set, labels annotated by an LLM, licensed CC-BY-SA-4.0. `scripts/benchmark_ood.py` evaluates against it.
- [`TigreGotico/bifonia-pt-homographs-gold`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-gold): 2,233 real, sense-balanced sentences over 40 words, labelled independently of the rules, with a `source` field. No script in this repository evaluates against it.

After generating splits, push the synthetic dataset:

```bash
python scripts/dataset.py --hf --out .

# Then push to HF Hub (requires huggingface-cli login):
huggingface-cli upload TigreGotico/bifonia-pt-homographs hf/ --repo-type dataset
```

---

## Citation

```bibtex
@misc{bifonia,
  title  = {bifonia: Portuguese Heterophonic Homograph Disambiguation},
  author = {JarbasAI / TigreGotico},
  url    = {https://github.com/TigreGotico/bifonia},
  license = {Apache-2.0},
}
```

---
[← Data quality](data_quality.md) · [Home](../README.md) · [Diacritics restoration →](diacritics_restoration.md)
