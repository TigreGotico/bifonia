# Benchmarks

## Out-of-distribution (real Wikipedia sentences)

[`TigreGotico/bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild)
— **7,119 real sentences across 112 words**.

| approach | OOD accuracy |
|---|---|
| most-common (majority sense per word) | 74.6% |
| **rules (corpus-free)** | **89.8%** |
| shipped ensemble (model ⊕ rules) | **89.9%** |
| **spaCy `pt_core_news_lg` (POS → sense)** | **93.2%** |
| Naive-Bayes / perceptron | ~38% \* |

### Reading these numbers honestly
- **A strong neural POS-tagger (spaCy) wins here (93%)** because the expanded
  roster is largely **POS-separable** (deverbal noun vs 1sg verb): tagging the
  homograph's POS correctly resolves the reading. After adding conservative VERB rules (coordination/comparison → noun), the
  rule engine reaches 87.2% — the residual gap is mostly **proper nouns**
  (place/team names like *Cerro Corá*) that need capitalization, a documented
  next step.
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

## The skew caveat & the balanced hard-subset test

The wild OOD set is **sense-skewed** — Wikipedia is encyclopedic (noun-heavy), so
for most words one reading dominates and minority senses barely occur. On the
"POS-no-lift" subset (72 words) **most-common alone scores 98%**, so high spaCy /
shipped numbers there mostly reflect *predicting the dominant sense*, not
disambiguation. spaCy's apparent edge on wild is largely this artifact.

The fair test is **balanced** (equal senses per word), on words a POS-tagger
**cannot** separate (same-POS or POS-unreliable readings):

| approach | balanced hard subset (8 words, n=1280) |
|---|---|
| most-common | 50% |
| spaCy `pt_core_news_lg` (POS→sense) | 59% |
| **bifonia rules** | **94%** |

Per word (rules): `sede` 88, `forma` 78, `molho` 94, `gosto` 99, `corte` 100,
`gozo` 98, `coro` 99, `posto` 94. On the same-POS pairs (`sede`, `molho`, `coro`,
`gozo`) spaCy is pinned at ~50% (chance) — it has no POS signal to use — while the
meaning-keyed rules resolve them. **This is bifonia's core value:
meaning-based disambiguation where part-of-speech is uninformative.**
