# Benchmarks

## Out-of-distribution (real Wikipedia sentences)

Evaluated on [`TigreGotico/bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild)
— **7,119 real sentences across 112 words** (expanded from the original 27).
Run with `python benchmark_ood.py`.

| approach | OOD accuracy |
|---|---|
| most-common (majority per word) | 20.6% |
| spaCy / Stanza (POS → sense) | ≈ POS-tagging does not resolve the open/closed-vowel contrast on this set |
| **rules (corpus-free)** | **83.8%** |
| Naive-Bayes | 37.6% \* |
| averaged perceptron | 38.6% \* |
| **shipped ensemble (model⊕rules)** | **84.5%** |

\* The trained models are diluted here: the corpus is rich for the original 27
words but thin for many of the ~85 newly-added (rare) words, and statistical
models generalise poorly out-of-distribution. The **rule engine** carries the
new words, so the shipped ensemble (model where confident, else rules) is the
honest production number. On the original-27 subset the perceptron still scores
~90% OOD.

## Synthetic (in-distribution)

Rule-engine accuracy on the labelled corpus (original 27 words): **95.7%**
(up from 94.6% as the scorer gained the punctuation/clitic-aware tokenizer and
the prenominal-adjective / complement-preposition rules).

## Notes
- The disambiguation axis is **vowel quality** (open ɔ/ɛ vs closed o/e), not POS,
  which is why generic POS-taggers add little here and the meaning-keyed rule
  engine does well.
