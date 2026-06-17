# Benchmarks

## Out-of-distribution (real Wikipedia sentences)

[`TigreGotico/bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild)
— **7,119 real sentences across 112 words**.

| approach | OOD accuracy |
|---|---|
| most-common (corpus-derived majority sense) | 46.0% † |
| Naive-Bayes (all 124 words) | 85.0% |
| perceptron (all 124 words) | 86.5% |
| **rules (corpus-free)** | **89.8%** |
| shipped (rules + model only where it beats rules OOD-proxy) | **89.8%** |
| **spaCy `pt_core_news_lg` (POS → sense)** | **93.2%** |

### Reading these numbers honestly
- **A strong neural POS-tagger (spaCy) wins here (93%)** because the expanded
  roster is largely **POS-separable** (deverbal noun vs 1sg verb): tagging the
  homograph's POS correctly resolves the reading. The rule engine reaches 89.8%
  — the residual gap is mostly **proper nouns** (place/team names like *Cerro
  Corá*) and minority verb readings.
- bifonia's rule engine is **offline, zero-dependency and deterministic**, and —
  unlike a POS-tagger — it disambiguates **same-POS lexical pairs**
  (`sede` thirst/seat, `corte` cut/court, `forma` mould/shape, `molho`
  sauce/bundle), where a POS-tagger can only fall back to the majority sense.
- **The learned models now cover all 124 words** (trained on a regenerated
  stratified 80/20 split) but still trail the rules OOD (85–86% vs 89.8%). This
  is **circularity**: the corpus labels were assigned by the rule engine, so a
  model can at best mimic the rules in-distribution and generalises *worse* on
  real sentences. Consequently the shipped route-gate adopts a word's model only
  when it **strictly beats the rules on the hand-curated behavioral set** (the
  OOD proxy) — true for just 1 word — so **shipped ≡ rules (89.8%)**. The models
  are retained as a full-roster baseline and as the corpus QC engine
  (see [data_quality.md](data_quality.md)), not as the shipped predictor.
- † `most-common` here is derived from the (balanced) bundled corpus, so it picks
  each word's corpus-majority sense — which often isn't the wild-dominant sense,
  hence 46%. The wild set's *own* dominant-sense baseline is ~75% (it is
  noun-skewed; see the skew caveat below).

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
