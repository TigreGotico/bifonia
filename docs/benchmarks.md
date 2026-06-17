# Benchmarks

## Out-of-distribution (real Wikipedia sentences)

[`TigreGotico/bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild)
— **7,119 real sentences across 112 words**.

| approach | OOD accuracy |
|---|---|
| most-common (majority sense per word) | 74.6% |
| **rules (corpus-free)** | **83.8%** |
| shipped ensemble (model ⊕ rules) | **84.5%** |
| **spaCy `pt_core_news_lg` (POS → sense)** | **93.2%** |
| Naive-Bayes / perceptron | ~38% \* |

### Reading these numbers honestly
- **A strong neural POS-tagger (spaCy) wins here (93%)** because the expanded
  roster is largely **POS-separable** (deverbal noun vs 1sg verb): tagging the
  homograph's POS correctly resolves the reading. The rule engine trails because
  of bare-object / minimal-context sentences.
- bifonia's rule engine is **offline, zero-dependency and deterministic**, and —
  unlike a POS-tagger — it disambiguates **same-POS lexical pairs**
  (`sede` thirst/seat, `corte` cut/court, `forma` mould/shape, `molho`
  sauce/bundle), where a POS-tagger can only fall back to the majority sense.
- \* The trained NB/perceptron are diluted: the corpus is rich for the original
  27 words but thin for many rare new words, and statistical models generalise
  poorly OOD. The ensemble leans on rules for the new words.
- ⚠️ `benchmark_ood.py`'s `most-common` derives the baseline from the 27-word
  train split, so it under-reports on the expanded roster (new words → no
  baseline). Use the full-coverage figure above.

## Synthetic (in-distribution)
Rule-engine accuracy on the labelled corpus: **96.5%** (124 words, 68k records);
original-27 subset **95.7%** (up from 94.6% with the tokenizer + scorer rules).

## Why POS-tagging isn't the whole story
The disambiguation axis is **vowel quality** (open ɔ/ɛ vs closed o/e). POS predicts
it for noun/verb pairs, but the lexical/same-POS pairs need **meaning** cues — the
reason bifonia is meaning-keyed.
