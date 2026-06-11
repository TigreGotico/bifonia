# bifonia — Portuguese Homograph Disambiguation Dataset

## Overview

A labeled sentence corpus for **heterophonic homograph disambiguation** in European Portuguese.
Heterophonic homographs are words spelled identically but pronounced differently depending on
their syntactic function (POS). Correct disambiguation is critical for TTS pipelines.

Example: **"para"**
- as ADP (preposition to/for): /ˈpɐɾɐ/ — "Vou *para* casa."
- as VERB (parar, to stop): /ˈpaɾɐ/ — "O autocarro *pára*."

---

## Statistics (current corpus)

| Metric | Value |
|--------|-------|
| Total sentences | ~13 900 |
| Ambiguous words | 27 |
| POS classes | NOUN, VERB, ADP, ADJ |
| Language | European Portuguese (pt-PT) |
| License | Apache-2.0 |

### Per-word sentence counts

| Word | POS variants | Sentences |
|------|-------------|-----------|
| acerto | NOUN / VERB | ~440 |
| acordo | NOUN / VERB | ~435 |
| cerro | NOUN / VERB | ~425 |
| choro | NOUN / VERB | ~415 |
| colher | NOUN / VERB | ~438 |
| começo | NOUN / VERB | ~418 |
| conserto | NOUN / VERB | ~450 |
| coro | NOUN / VERB | ~433 |
| corte | NOUN / VERB | ~585 |
| forma | NOUN / VERB | ~636 |
| gosto | NOUN / VERB | ~434 |
| gozo | NOUN / VERB | ~640 |
| jogo | NOUN / VERB | ~429 |
| molho | NOUN / VERB | ~450 |
| olho | NOUN / VERB | ~418 |
| para | ADP / VERB | ~636 |
| pelo | ADP / NOUN / VERB | ~860 |
| peso | NOUN / VERB | ~427 |
| porto | NOUN / VERB | ~420 |
| posto | NOUN / VERB | ~669 |
| rego | NOUN / VERB | ~446 |
| seco | ADJ / VERB | ~445 |
| sede | NOUN / VERB | ~596 |
| sobre | ADP / NOUN / VERB | ~870 |
| tola | NOUN / ADJ | ~430 |
| torre | NOUN / VERB | ~425 |
| transtorno | NOUN / VERB | ~637 |

---

## Schema

Each record contains:

| Field | Type | Description |
|-------|------|-------------|
| `word` | str | The ambiguous word (lowercase, no diacritics) |
| `pos` | str | UPOS tag: NOUN, VERB, ADP, or ADJ |
| `ipa` | str | IPA transcription for this word in this POS reading |
| `diacritized` | str | Orthographic form with non-standard diacritics marking the reading (e.g. `pára` for VERB, `para` for ADP) |
| `sentence` | str | Full sentence context using the plain (undiacritized) word form |
| `diacritized_sentence` | str | Same sentence with `word` replaced by `diacritized` |

---

## ML Tasks

This dataset is suitable for:

1. **Homograph disambiguation** — classify the POS/reading of an ambiguous word given its
   sentence context. Input: `(sentence, word, word_index)`. Output: `pos` (or `ipa`).

2. **TTS pronunciation prediction** — given a sentence, predict the IPA of each ambiguous
   word. Baseline: the rule-based `bifonia` scorer reaches **98.8% accuracy** on this corpus.

3. **Portuguese POS disambiguation** — a targeted sub-task of POS tagging for the 27
   homograph types listed above.

---

## Splits

Use `python dataset.py --hf` to generate stratified 80/20 train/test splits under `hf/`:

```
hf/
  train.jsonl   # 80% per (word, pos) stratum
  test.jsonl    # 20% per (word, pos) stratum
```

---

## Data Generation

Sentences were collected from two sources:
- **LLM-generated**: produced by free coding agents (opencode-free, antigravity-flash-low)
  via `corpus_gen.py`, then manually reviewed before merging
- **Human-curated**: the initial seed set in `bifonia/data/grp_*.py`

Generation prompts enforce:
- Unambiguous usage of the target word in the specified POS
- Varied register, sentence length (6–20 words), and vocabulary
- No repetition of existing corpus sentences

---

## Usage

```python
from bifonia.corpus import CORPUS, iter_records

# Iterate all labeled sentences
for word, pos, sentence in iter_records():
    print(word, pos, sentence[:60])

# Access by word
for pos, sentences in CORPUS["para"].items():
    print(f"para/{pos}: {len(sentences)} sentences")
```

Disambiguation:
```python
from bifonia import tokenize, is_ambiguous, disambiguate

words = tokenize("O autocarro para em frente ao hospital.")
for i, w in enumerate(words):
    if is_ambiguous(w):
        ipa = disambiguate(words, i)
        print(f"{w} → [{ipa}]")
# para → [ˈpaɾɐ]  (VERB reading)
```

---

## HuggingFace Upload

After generating splits:

```bash
python dataset.py --hf --out .

# Then push to HF Hub (requires huggingface-cli login):
huggingface-cli upload TigreGotico/bifonia-pt-homographs hf/ --repo-type dataset
```

---

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
