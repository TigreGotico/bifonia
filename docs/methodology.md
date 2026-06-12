# Methodology: Dataset Construction and Disambiguation Algorithm

## Overview

The `bifonia` package disambiguates 27 Portuguese heterophonic homographs — words spelled
identically but pronounced differently depending on their **meaning** (`sense`).  Disambiguation
is served by **two interchangeable engines**, both pure Python with no heavy runtime dependency:

1. **A rule engine** (`bifonia/scoring.py`) that predicts the `sense` from the local ±4-word
   context using hand-written rules and `.voc` wordlists.  It needs no corpus — the zero-resource
   baseline.
2. **Learned per-word models** (`bifonia/model.py`, trained by `train.py`) — a Naive-Bayes
   log-odds classifier and an averaged perceptron — fit from the labelled corpus over the
   features in `bifonia/features.py`.

`guess_sense` combines them as a **per-word ensemble**: each word is served by whichever engine
is at least as accurate on held-out data, with the rule engine as the fallback, so the combined
system never scores below the rules.

The disambiguation supporting both engines is a labelled corpus of 56 891 Portuguese sentences,
one ambiguous word per sentence, labelled with a meaning slug (`sense`) and a descriptive `pos`
attribute.

The bucket key is the **meaning**, not part of speech: two senses can share a POS — `sede`
*thirst* (closed /ˈsedɨ/) and *seat/HQ* (open /ˈsɛdɨ/) are both nouns, separated only by vowel
quality.  `pos` is a descriptive attribute (the dominant grammatical reading of a sense) and
may repeat across the senses of a word.

---

## Dataset Construction

### Source of truth

The canonical corpus is `bifonia/data/corpus.jsonl` — one JSON record per line,
`{"word", "sense", "pos", "ipa", "sentence"}`.  The per-reading IPA table lives in
`bifonia/data/heterophonic_homographs.csv` (columns `word,sense,pos,ipa`), keyed on
`(word, sense)`.  `bifonia/corpus.py` loads the JSONL into a `CORPUS` dict
(`word → {sense: [sentences]}`) at import time.

### LLM-assisted generation, native review, scorer filtering

`corpus_gen.py` fans sentence generation out to multiple coding-agent providers in parallel
(free bulk providers plus a stronger provider reserved for hard patterns and quality review).
For each `(word, sense)`, a prompt specifies the target meaning and the hard patterns to cover
(e.g. passive-voice frames for `posto` *station*, control-verb phrases for `colher` *harvest*,
route-noun context for `pelo` *by_the*).  Generated sentences are run through the scorer:
misclassifications either reveal a real scorer weakness (fix the rule) or a labelling error
(fix the label or discard).  Only sentences the scorer handles correctly are admitted, so the
corpus carries no blind-generation label noise.  Using several independently prompted agents
widens vocabulary and phrasing variety.

**Statistics:**

| Metric | Value |
|---|---|
| Total sentences | 56 891 |
| Train / test split | 45 492 / 11 400 |
| Words covered | 27 |

---

## Disambiguation Algorithm

`guess_sense` is a **per-word ensemble** over two engines. For each ambiguous token it consults
the learned model where that word is routed to the model and clears its margin, and falls back to
the rule engine otherwise; an explicit `pos` override always uses the rule resolver. `guess_pos`
maps the resolved sense back to its descriptive POS; `disambiguate` selects the IPA for the
resolved `(word, sense)`. Because adoption is gated on held-out accuracy (below), the ensemble
never does worse than the rules alone.

### Rule engine

The scorer (`bifonia/scoring.py`) is a context-based integer-scoring system.  For each
candidate POS, a function (`score_adp`, `score_noun`, `score_verb`, `score_adj`) produces
an integer score from signals in the ±4-word window.  The POS with the highest score wins;
ties break to `DEFAULT_POS[word]`.

The scorer then **narrows the winning POS to a `sense`** (`resolve_sense`).  For 26 of the 27
words each POS maps to exactly one sense, so this is a direct lookup.  For `sede`, whose two
senses (*thirst*, *seat/HQ*) are both nouns, a meaning resolver (`_resolve_sede`) reads
sense-specific cues from the local context: the preposition frame is the strongest signal
(`sede de X` → thirst, `sede da/do X` → seat), reinforced by the
`bifonia/locale/pt-pt/sede_{seat,thirst}_cues.voc` wordlists in the ±3 window. The rule engine
needs no corpus, which makes it the right fit for a fork of a low-resource language.

A handful of readings are genuinely ambiguous on the local pattern alone and need a wider cue
to resolve. `molho` is the clearest: `molho de chaves`/`lenha` is a bundle (open ɔ), `molho de
tomate` a sauce (closed o), and an edible green can go either way — `colhi um molho de salsa`
(a bunch) versus `o bife tinha molho de salsa` (a sauce) — decided by a gathering verb in the
left window; `deixar/pôr de molho` is the soaking idiom, while `gosto de molho` is the sauce
genitive. These hooks live in `bundle_things.voc`, `bundle_ambiguous.voc`, `bundle_verbs.voc`,
and `soak_verbs.voc`. The remaining cases that need full-sentence understanding are rare, and
the learned models cover most of them.

### Learned models

`bifonia/model.py` loads per-word classifiers trained by `train.py` from the labelled corpus:

- **Naive-Bayes** — per-sense log-odds of each feature (the bias is the log-prior). The weights
  are interpretable: they *are* the learned lexicons.
- **Averaged perceptron** — warm-started from the NB weights and trained with lazy averaging,
  discounting correlated cues that NB double-counts. This is the model the ensemble ships.

Both serialise to one JSON shape (`bifonia/data/sense_model_nb.json` and
`sense_model_perceptron.json`) and score by a single sparse dot product —
`score[sense] = bias[word][sense] + Σ feats · weights[word][sense]`, argmax wins. Inference is
plain dict arithmetic (`json` + `pathlib` + `bifonia/features.py`); no numpy, sklearn, or other
runtime dependency.

The feature extractor (`bifonia/features.py`) is the **same** at training and inference, so the
model cannot suffer train/serve skew. It carries no hardcoded Portuguese and no per-word logic —
language knowledge enters only through the `.voc` membership sets and the weights a model learns.
Feature families:

| Family | Example | What it captures |
|---|---|---|
| Positional skipgrams | `L1=o`, `L4=…`, `R1=de`, `R3=…` | tokens at fixed offsets (`L1..L4`, `R1..R3`); `<BOS>`/`<EOS>` at edges |
| Bag-of-window overlap | `W=forno`, `W=empresa` | counts of tokens in the ±4 window (the "overlap" features) |
| Structural membership | `L1∈determiners`, `prev∈copula` | slot ∈ structural `.voc` set (determiners, pronouns, clitics, copula, …) |
| Morphology / position | `self_sfx=INF`, `R1_mente`, `R1_deverbal`, `pos0` | suffixes/affixes of the target and its neighbours, sentence position |

The structural `.voc` sets encode grammar (generic across Romance languages), not per-word
lexical semantics; the model learns lexical cues from the `W=` and positional tokens. Porting to
a related language means swapping the corpus and `.voc` files and retraining — this file is
untouched.

### Per-word ensemble routing

`train.py` trains only on `hf/train.jsonl`, never on test. A seeded per-word validation fold
drives perceptron early-stopping and a **`route` gate**: a word is flagged `route="model"` only
where the model's validation accuracy is at least the rule engine's on the same fold *and* the
model does not regress against a hand-curated out-of-distribution behavioural set; otherwise it
stays on the rules. The flag and a `margin_tau` threshold are written per word into the model
JSON. At inference `guess_sense` reads them: model-routed words above their margin use the model,
everything else uses the rules. Training is deterministic — the same seed yields byte-identical
JSON.

### Rule signal types

| Signal class | Example | Reasoning |
|---|---|---|
| **DET/PRON before** | `o gosto`, `um sobre` | Determiner introduces a nominal phrase |
| **DET/article after** | `gosto o/a` | Direct-object NP signals finite VERB |
| **AFTER_PREP nouns/pronouns** | `para mim`, `sobre ele` | Standard arguments of prepositions |
| **Infinitive after (for `para`)** | `para correr` | Purpose clause = ADP |
| **PASSIVE_AUX before (for `posto`)** | `foi posto` | Past participle of pôr = NOUN IPA |
| **Control verb in prev2–prev4** | `aprendeu a colher` | Infinitive complement = VERB |
| **Deverbal noun suffix** | `para análise` | Nominalised purpose = ADP |
| **Degree adverb (-mente before)** | `particularmente seco` | Predicative ADJ |
| **Copula at prev2** | `está pronto para` | Predicative construction = ADP |
| **-mente adverb after** | `seco rapidamente` | Adverb modifies finite VERB |
| **Governing verb set (SOBRE_GOV)** | `falou sobre` | Explicit ADP governing verb |
| **Frequency adverb before `sobre`** | `sempre sobre uma fatia` | "sobrar" finite VERB |
| **`pelo` + feminine article** | `pelo as batatas` | `pelar` VERB (por+o ≠ a/as) |
| **`pelo` + route noun** | `pelo interior` | Geographic ADP pattern |

### Punctuation stripping

Token neighbours are stripped of trailing punctuation (`.,;:!?`) before set lookups.
This prevents false negatives when a word occurs before a comma or period in the source
text (e.g. `"ti,"` failing to match the `AFTER_PREP` entry `"ti"`).

### Rule design principles

Rules are conservative: each fires on a clear linguistic pattern, not a statistical quirk of
the corpus, and every signal is motivated by a grammatical argument (e.g. "contracted
prepositions cannot introduce verbal direct objects").

---

## Benchmark Comparison

Sense-prediction accuracy is measured two ways. `benchmark_tagger.py` evaluates on a **synthetic**
held-out split (the test partition of the generated corpus). `benchmark_ood.py` evaluates on an
**out-of-distribution (OOD)** set of real Wikipedia and web sentences
(`TigreGotico/bifonia-pt-homographs-wild`, downloaded on demand). Each approach sees the plain
(un-diacritised) form; the POS taggers map their POS output back to a sense.

| Approach | Synthetic test | OOD (real text) |
|---|---|---|
| most-common (majority sense per word) | 52.7 % | 47.5 % |
| spaCy (`pt_core_news_lg`) POS→sense | 65.7 % | 81.4 % |
| Stanza POS→sense | 75.5 % | 82.5 % |
| rules (no corpus) | 94.5 % | 84.6 % |
| Naive-Bayes | 98.1 % | 86.7 % |
| averaged perceptron | 99.0 % | 89.6 % |
| **shipped ensemble** | **96.1 %** | **90.5 %** |

**Synthetic splits overstate accuracy.** Their train and test sentences share phrasing, so every
approach runs several points high; the OOD set is the honest measure. Every method drops on real
text — but the corpus-trained perceptron still beats the rules by roughly five points there
(89.6 vs 84.6): it generalises rather than memorising. On the balanced synthetic split the pure
perceptron is highest, since the ensemble's per-word route gate keeps a word on the rules wherever
the model does not clearly win. On real text that same routing pays off — the ensemble edges past
the pure perceptron (90.5 vs 89.6), because the words it routes to the rules (such as `molho`) are
read better by the rules there than by the model. Either way the ensemble never underperforms the
rules on any word.

The POS taggers (spaCy/Stanza) hit a **structural ceiling**: POS cannot separate two senses that
share a part of speech, so the tagger gets the majority noun sense right but the minority sense
wrong *by construction* — both spaCy and Stanza score 0 % on `sede`/thirst. Their aggregate score
swings with the sense distribution: on the balanced synthetic set the minority readings are common
enough to pull them down to 66–76 %, while real text is skewed toward the majority readings they do
get right, lifting them to ~82 %. The meaning-aware models lead on both distributions.
Per-bucket accuracy on the synthetic corpus makes this concrete:

| word/sense | n | spaCy | Stanza | rule-based |
|---|---|---|---|---|
| sede/thirst | 410 | 0.0 % | 0.0 % | 100.0 % |
| sede/seat | 1035 | 100.0 % | 100.0 % | 76.8 % |
| corte/cut | 1256 | 19.2 % | 55.4 % | 99.6 % |
| corte/court | 992 | 99.5 % | 100.0 % | 99.9 % |
| forma/mould | 1017 | 100.0 % | 100.0 % | 55.4 % |
| forma/shape | 1090 | 17.2 % | 49.4 % | 97.1 % |
| molho/sauce | 790 | 100.0 % | 100.0 % | 89.7 % |
| molho/bundle | 1025 | 0.0 % | 13.3 % | 71.7 % |

The POS taggers score 100 % on the dominant sense of each pair and near-0 % on its minority
twin; bifonia, by reading meaning cues, recovers the minority sense.

Run the benchmarks yourself (both report per-word breakdowns):

```bash
python benchmark_tagger.py             # synthetic held-out split
python benchmark_tagger.py --word sede --errors
python benchmark_ood.py                # OOD real-text set (downloads from Hugging Face)
```

---

## Why This Approach Is Tractable for Portuguese

The rule engine works — and works well — specifically because Portuguese has an
unusually small set of heterophonic homographs that matter for TTS.  This package covers
27 words.  That is not a limitation of the dataset; it is close to the full inventory of
the phenomenon in standard European Portuguese.

For most words the disambiguation reduces to a few clear grammatical contrasts (NOUN vs VERB,
ADP vs VERB) reliably signalled by the ±4-word context: determiners, pronouns, infinitive
markers, passive auxiliaries, copular verbs.  Where two senses share a POS (`sede`), a
meaning resolver reads sense-specific cues from the same window.  The scorer does **not**
attempt to tag full sentences; it only resolves the meaning of one pre-identified ambiguous
token.  That is a much easier problem than full POS tagging.

The hand-crafted rule set does **not** generalise to other languages:

- Languages with large homograph inventories (e.g. English, where hundreds of words are
  heterophonic: *lead*, *wind*, *row*, *wound*, …) would require a general tagger, not
  a hand-crafted rule set of this size.
- Languages with free word order make the ±4-word window less reliable as a signal.
- Languages with rich morphology often resolve ambiguity through agreement suffixes that
  appear on the target word itself — no context scanning needed at all.

Portuguese is a fortunate special case: few words need disambiguation, and they are
disambiguated by strong, local grammatical cues.

The learned path carries no such Portuguese-specific assumptions. Its feature extractor is
language-agnostic, so a fork for a related language supplies a corpus and `.voc` files and
retrains, reusing the same algorithm.

---

## Orthographic Normalisation (AO1990)

Some diacritized forms are unambiguous: *pára* (stop), *pêlo* (hair), *côrte* (court), etc.
The acute/circumflex marks the vowel quality directly, so:

- `guess_sense` reads the meaning straight off a diacritized token (e.g. *séde* → seat,
  *sêde* → thirst, *pára* → stop) without invoking the context scorer.  The
  `_DIACRITIZED_TO_SENSE` and `_DIACRITIZED_TO_BASE` maps in `bifonia/__init__.py` back this
  lookup.
- Plain (AO1990) tokens are resolved by the context scorer.  `add_extra_diacritics` performs
  the reverse: it inserts the non-canonical diacritic that forces the resolved reading in a
  downstream rule-based G2P.

---

## Limitations

- **European Portuguese only:** all phonology, wordlists, and orthographic conventions are
  specific to **European Portuguese (EP)**.  Brazilian Portuguese has different stress
  patterns and clitic placement, and some of these 27 words are not heterophonic in BP.
- **`pelo` hair vs by_the:** the body-hair NOUN reading is recognised by possession verbs
  (`ter`, `possuir`) governing `pelo` and by past-participle context for the passive-agent
  `by_the` ADP pattern, handled by suffix heuristics in the scorer.
- **Context window:** the scorer inspects only ±4 words.  Long-range dependencies (e.g. a
  subject noun phrase 5+ words before the verb) are outside its reach and are an irreducible
  error source for a local rule-based system.
- **Sentence-level ambiguity:** a small number of sentences are genuinely ambiguous without
  full semantic interpretation (e.g. `para sempre` = *stop* "stops always" vs *purpose*
  "forever").  These remain known limitations rather than being over-fitted with fragile rules.
- **Dataset licence:** sentence content is original; no copyrighted text is used.  The corpus
  is published under a permissive licence.
