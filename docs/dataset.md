# bifonia — Portuguese Homograph Disambiguation Dataset

## Overview

A labeled sentence corpus for **heterophonic homograph disambiguation** in European Portuguese. Heterophonic homographs are words spelled identically but pronounced differently depending on their **meaning** (`sense`). Correct disambiguation matters for TTS pipelines.

Example: **"sede"**, both senses are nouns, distinguished only by vowel quality:
- *thirst* (closed e): /ˈsedɨ/ — "Tenho *sede* depois do exercício."
- *seat / HQ* (open ɛ): /ˈsɛdɨ/ — "A *sede* da empresa fica em Lisboa."

Example: **"para"**
- *purpose* (ADP, preposition to/for): /ˈpɐɾɐ/ — "Vou *para* casa."
- *stop* (VERB, parar): /ˈpaɾɐ/ — "O autocarro *pára*."

The bucket key is the **meaning** (`sense`), not part of speech. Each record also carries a `pos` attribute, the dominant grammatical reading of that sense, which is descriptive only and may repeat across senses of the same word (both senses of `sede` are `NOUN`).

---

## Statistics

| Metric | Value |
|--------|-------|
| Total sentences | 56 891 |
| Train / test split | 45 492 / 11 400 |
| Ambiguous words | 27 |
| POS attributes | NOUN, VERB, ADP, ADJ |
| Language | European Portuguese (pt-PT) |
| License | Apache-2.0 |

### Per-word sentence counts

| Word | Senses (pos) | Sentences |
|------|-------------|-----------|
| acerto | settlement (NOUN) / adjust (VERB) | 2403 |
| acordo | agreement (NOUN) / wake (VERB) | 2021 |
| cerro | hill (NOUN) / shut (VERB) | 2024 |
| choro | weeping (NOUN) / weep (VERB) | 1956 |
| colher | spoon (NOUN) / harvest (VERB) | 2033 |
| começo | beginning (NOUN) / begin (VERB) | 2050 |
| conserto | repair (NOUN) / mend (VERB) | 2010 |
| coro | choir (NOUN) / blush (VERB) | 1973 |
| corte | court (NOUN) / cut (VERB) | 2248 |
| forma | mould (NOUN) / shape (VERB) | 2107 |
| gosto | taste (NOUN) / like (VERB) | 2089 |
| gozo | enjoyment (NOUN) / enjoy (VERB) | 1874 |
| jogo | game (NOUN) / play (VERB) | 1986 |
| molho | sauce (NOUN) / bundle (VERB) | 1815 |
| olho | eye (NOUN) / look (VERB) | 1967 |
| para | purpose (ADP) / stop (VERB) | 2087 |
| pelo | by_the (ADP) / hair (NOUN) / peel (VERB) | 2256 |
| peso | weight (NOUN) / weigh (VERB) | 2406 |
| porto | harbour (NOUN) / carry (VERB) | 3010 |
| posto | station (NOUN) / post (VERB) | 2709 |
| rego | furrow (NOUN) / water (VERB) | 1967 |
| seco | dry (ADJ) / dry_vb (VERB) | 1889 |
| sede | thirst (NOUN) / seat (NOUN) | 1445 |
| sobre | about (ADP) / sail (NOUN) / leftover (VERB) | 1645 |
| tola | foolish (ADJ) / head (NOUN) | 1921 |
| torre | tower (NOUN) / roast (VERB) | 2527 |
| transtorno | disorder (NOUN) / upset (VERB) | 2473 |

---

## Schema

`bifonia/data/corpus.jsonl` is the single source of truth: one JSON record per line.

| Field | Type | Description |
|-------|------|-------------|
| `word` | str | The ambiguous word (lowercase, no diacritics) |
| `sense` | str | Meaning slug — the label to predict (e.g. `thirst`, `seat`) |
| `pos` | str | Descriptive UPOS attribute of that sense: NOUN, VERB, ADP, or ADJ |
| `ipa` | str | IPA transcription this `(word, sense)` reading carries |
| `sentence` | str | Full sentence context using the plain (undiacritized) word form |

`bifonia/data/heterophonic_homographs.csv` carries the per-reading IPA table with columns `word,sense,pos,ipa` (one row per `(word, sense)`).

---

## ML tasks

This dataset supports:

1. **Sense disambiguation**: predict the `sense` of an ambiguous word given its sentence context. Input: `(sentence, word, word_index)`. Output: `sense` (or `ipa`).
2. **TTS pronunciation prediction**: given a sentence, predict the IPA of each ambiguous word. `bifonia` ships both a corpus-free rule engine and corpus-trained learned models (Naive-Bayes and an averaged perceptron) trained from this corpus. See `docs/methodology.md` for the accuracy comparison on a synthetic split and on a real-text OOD set.
3. **Portuguese sense disambiguation**: a targeted sub-task for the 27 homograph types listed above.

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

Two datasets on the Hub, both with schema `{word, sense, pos, ipa, sentence}`:

- [`TigreGotico/bifonia-pt-homographs`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs): 56 891 sentences over 27 words, with stratified train/test splits, for training and synthetic evaluation.
- [`TigreGotico/bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild): real Wikipedia and web sentences forming an out-of-distribution (OOD) test set, labels annotated by an LLM, licensed CC-BY-SA-4.0. `scripts/benchmark_ood.py` evaluates against it.

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
