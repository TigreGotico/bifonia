# Diacritics Restoration as a Homograph Disambiguation Task

## The problem

Portuguese has heterophonic homographs — words whose spelling is identical but whose
pronunciation depends on their **meaning** (`sense`). A rule-based G2P engine that does not
disambiguate the sense will systematically mispronounce them.

The classic example:

| Written | Sense | IPA |
|---------|-------|-----|
| para | purpose (*goes to*) | `ˈpɐɾɐ` |
| para | stop (*halts*) | `ˈpaɾɐ` |

Two senses can also share a part of speech — `sede` *thirst* (`ˈsedɨ`) and *seat/HQ*
(`ˈsɛdɨ`) are both nouns, distinguished only by vowel quality — so the disambiguation key is
the meaning, not POS. A G2P that sees `para` or `sede` cannot know which IPA to emit.

## Framing as diacritics restoration

### Diacritics are standard Portuguese phonology

The acute and circumflex accents used here are standard Portuguese orthographic markers
for vowel quality and are part of the language:

- **Acute (´)** marks an open vowel: *ó* = /ɔ/, *é* = /ɛ/
- **Circumflex (^)** marks a closed vowel: *ô* = /o/, *ê* = /e/

Every Portuguese G2P engine, phonemiser, and TTS system handles these accents correctly —
they appear in thousands of unambiguous words (*ótimo*, *ônibus*, *êxito*, *pé*, *pó*, …).
The disambiguation output is therefore **drop-in compatible** with any Portuguese TTS
pipeline: insert the diacritised form before G2P and the correct vowel is produced without
any changes to the downstream system.

AO1990 (the 1990 Orthographic Agreement) writes a small set of formerly-heterophonic words
without these accents — the words this package covers. Under AO1990, *pára* (stops) is
written *para*, orthographically identical to the preposition. Placing the diacritic on the
verbal reading supplies the phonological signal the plain spelling lacks.

### Mapping to a sequence-labelling problem

The task: for each ambiguous token, decide whether to insert a diacritic and which one.
All other tokens pass through unchanged.

| Canonical | Restored | Sense | Vowel |
|-----------|----------|-------|-------|
| para | para (unchanged) | purpose (ADP) | ɐ |
| para | **pára** | stop (VERB) | open a |
| pelo | **pêlo** | hair (NOUN) | closed e (fur) |
| pelo | pelo (unchanged) | by_the (ADP) | — |
| sede | **sêde** | thirst (NOUN) | closed e |
| sede | **séde** | seat (NOUN) | open ɛ |
| sobre | **sôbre** | sail (NOUN) | closed o (nautical sail) |
| sobre | **sóbre** | leftover (VERB) | open ɔ (to be left over) |

```
Input:  O autocarro para em frente ao hospital.
Output: O autocarro pára em frente ao hospital.
         ↓ standard Portuguese G2P / TTS
IPA:    u ɐwtukɐˈʁu ˈpaɾɐ ẽj ˈfɾẽtɨ ɐu uʃpiˈtaɫ
```

Because the output uses only standard orthographic conventions, no TTS model retraining
or phoneme-table modification is required.

### Diacritic convention

| Diacritic | Vowel quality | Typical reading |
|-----------|--------------|-----------------|
| acute (ó/é) | open /ɔ/ or /ɛ/ | verb / minority noun sense |
| circumflex (ô/ê) | closed /o/ or /e/ | dominant noun sense |

The polarity tracks **vowel quality**, not POS: where a word's senses share a POS (`sede`,
both nouns) or invert the usual mapping (`colher`, `tola`), the diacritic still encodes the
correct open/closed vowel. The full `(word, sense) → diacritized` table is in
`bifonia/__init__.py`.

## Engines and accuracy

`bifonia` resolves the sense with **two interchangeable engines**, both pure Python (see
`docs/methodology.md`):

- a **rule engine** (`scoring.py`) — a hand-crafted context scorer that inspects a ±4-word window
  using integer signals for determiners, pronouns, passive auxiliaries, copular verbs, infinitive
  markers, degree adverbs, and governing verbs, then narrows the winning POS to a `sense`. It needs
  no corpus.
- **learned per-word models** (`model.py`, trained by `train.py`) — a Naive-Bayes log-odds
  classifier and an averaged perceptron, fit from the labelled corpus over the language-agnostic
  features in `features.py`.

`guess_sense` is a per-word ensemble that routes each word to whichever engine is at least as
accurate on held-out data, with the rule engine as the fallback.

Sense-prediction accuracy, measured on a **synthetic** held-out split (`benchmark_tagger.py`) and
on an **out-of-distribution (OOD)** set of real Wikipedia/web sentences (`benchmark_ood.py`,
`TigreGotico/bifonia-pt-homographs-wild`):

| Approach | Synthetic test | OOD (real text) |
|---|---|---|
| most-common (majority sense per word) | 52.7% | 47.5% |
| spaCy (`pt_core_news_lg`) POS→sense | 65.7% | 81.4% |
| Stanza POS→sense | 75.5% | 82.5% |
| rules (no corpus) | 94.5% | 84.6% |
| Naive-Bayes | 98.1% | 86.7% |
| averaged perceptron | 99.0% | 89.6% |
| **ensemble** | **96.1%** | **90.5%** |

The OOD column is the more representative measure: synthetic train and test sentences share
phrasing, so synthetic accuracy runs higher. On real text the corpus-trained perceptron beats the
rules by about five points (89.6 vs 84.6): it generalises rather than memorising. The POS taggers
plateau because they cannot separate two senses that share a POS (e.g. `sede`): they nail the
dominant noun sense and miss the minority one by construction.

Run the comparison:

```bash
python benchmark_tagger.py            # synthetic held-out split
python benchmark_tagger.py --word sede --errors
python benchmark_ood.py               # OOD real-text set
```

### When the scorer succeeds

The scorer inspects a **±4-word window**. Representative signals:

- DET or QUANT immediately before → NOUN (+5)
- PRON immediately before → VERB (+5)
- Subjunctive conjunction (caso/embora/conquanto) before → VERB (+4)
- Negation/frequency adverb (não/nunca/sempre) before → VERB (+4)
- Passive auxiliary (foi/foram/fosse) before `posto` → NOUN (+8) — PPT of *pôr* = closed-o
- Governing verb before `sobre` (falou, discutiu, …) → ADP (+6)
- `-mente` adverb directly after → VERB (+3); directly before → ADJ (+4)
- Copular verb before (é, está, ficou) → ADJ (+5)
- `DET NOUN ADJ` attributive pattern (DET at −2) → ADJ (+4); clause-boundary–guarded
- Control verb (aprendeu, começou, …) in prev2–prev4 before `a colher` → VERB (+6)

### Where it struggles

Common failure modes:

1. **No context**: sentence-initial or between-clause position where no signal fires.
2. **Locally indistinguishable pairs**: ADP `AFTER_PREP` (+5) beats VERB prev2-DET (+3), so
   "pára a medicação" and "para a casa" are the same locally — telling them apart needs
   semantic comprehension (animate agent? destination noun?).
3. **Long-range dependency**: the governing verb is three or more tokens away
   (e.g. "A nebulosa forma novas estrelas", subject 3 tokens back).
4. **Shared-POS minority sense**: where two senses share a POS (`sede` thirst vs seat,
   `forma` mould vs shape, `molho` sauce vs bundle), the minority reading rides on a small
   set of meaning cues and is the largest single error source — see the per-bucket table in
   `methodology.md`.

### The `forma` / `molho` mould-vs-shape problem

`forma` carries two senses on two IPA forms: *mould* (`ˈfoɾmɐ`, closed-o, `NOUN`) and *shape*
(`ˈfɔɾmɐ`, open-o, `VERB`), where *shape* absorbs manner/way/geometric-form readings and 3sg
*formar*. The closed-o mould reading fires only on explicit baking-tin cues; everything else
defaults to open-o. `molho` is parallel: *sauce* (`ˈmoʎu`, `NOUN`) vs *bundle* (`ˈmɔʎu`,
`VERB`). Separating "baking mould" from "manner/way" — or "sauce" from "bundle of keys" —
requires comprehension of the noun phrase that local context cannot always supply.

These are the cases the learned models address: trained over the corpus, the perceptron beats the
rule engine on real text (see the accuracy table above), recovering some of the harder shared-POS
and long-range cases the rules cannot reach.

## Learned statistical models

`train.py` fits per-word classifiers from the corpus using the language-agnostic features in
`features.py` — positional skipgrams, a bag-of-window overlap, structural `.voc` membership, and
morphology/position cues — and serialises them to `bifonia/data/sense_model_{nb,perceptron}.json`:

- **Naive-Bayes** — per-sense log-odds of each feature; interpretable, the weights *are* the
  learned lexicons.
- **Averaged perceptron** — warm-started from the NB weights, discounting correlated cues NB
  double-counts. This is the model the ensemble ships.

Inference is a sparse dot product in pure stdlib (no numpy/sklearn), so the learned engine runs
under the same zero-dependency install as the rules. The rule engine is the corpus-free baseline
and the fallback for any word the model is not routed to.

The rule scorer is a manual decision tree with soft (additive) edges: it degrades gracefully when
signals conflict, and each of its signals corresponds to a learnable feature (DET before, PRON
before, CONJ_SUBJ before) that the learned models pick up automatically from the corpus.

## Corpus

The labeled corpus is `bifonia/data/corpus.jsonl`, loaded by `bifonia/corpus.py` and exported
by `dataset.py`:

```
python dataset.py --out data/
```

### Statistics

| Metric | Value |
|--------|-------|
| Total sentences | 56 891 |
| Train / test split | 45 492 / 11 400 |
| Unique ambiguous words | 27 |
| POS attributes | NOUN, VERB, ADP, ADJ |
| Domain coverage | science, medicine, engineering, biology, animals, objects, day-to-day, chit-chat, news, books/literature |

Sense-prediction accuracy for every approach is in the [Engines and accuracy](#engines-and-accuracy)
table above.

### Record schema (`corpus.jsonl`)

| Field | Description |
|--------|-------------|
| `word` | Base (canonical) ambiguous word |
| `sense` | Meaning slug — the label to predict |
| `pos` | Descriptive UDEP POS attribute of that sense |
| `ipa` | IPA transcription this `(word, sense)` reading carries |
| `sentence` | Source sentence using the plain (undiacritized) word form |

The IPA table `bifonia/data/heterophonic_homographs.csv` carries the same fields minus
`sentence` (columns `word,sense,pos,ipa`), one row per `(word, sense)`.

## Heavier model architectures

The learned engine is a lightweight per-word linear model. Larger neural models can push
past it — particularly on the long-range and shared-POS cases — at the cost of the zero-dependency
runtime. The corpus and `{word, sense, ipa}` labels support all of them:

### BiLSTM (sequence labeller)

```
Input:  token sequence (lowercased, subword BPE or character CNN)
Output: per-token label ∈ {UNCHANGED, ACUTE, CIRCUMFLEX}
        (only applied at positions of known ambiguous words)
Loss:   cross-entropy, ignore non-ambiguous positions
```

### BERTimbau fine-tune

BERTimbau (neuralmind/bert-base-portuguese-cased) provides rich contextual embeddings.
Fine-tune with a token classification head on the corpus records. The model has seen the
ambiguous words in diverse contexts, so it generalises well even on the harder
(long-range dependency) cases.

Input: the `sentence` field.  
Label: the position of the ambiguous word is tagged with its `sense` class.

### Sequence-to-sequence (diacritisation)

Cast the full task as character-level seq2seq:

```
Input:  "O autocarro para em frente ao hospital."
Target: "O autocarro pára em frente ao hospital."
```

A small encoder-decoder (T5-small or NLLB variant) fine-tuned on the corpus learns to insert
diacritics end-to-end without an explicit sense intermediate step.

## Evaluation

Run the test suite as a sanity check:

```bash
python -m pytest tests/test_disambiguate.py -v
```

The `hf/train.jsonl` / `hf/test.jsonl` split is shuffled and stratified per
`(word, sense)` with a fixed seed, so train and test are i.i.d. — use it directly, or build
stratified k-folds over the corpus records by `(word, sense)` for cross-validation.

Key metrics: sense accuracy, precision/recall per `sense`, and the per-bucket breakdown that
isolates the minority shared-POS senses (`sede`, `forma`, `molho`) — the hardest cases.

## Hugging Face datasets

The `sense` label and per-reading `ipa` make this a heterophonic-homograph disambiguation
benchmark, published as two datasets (schema `{word, sense, pos, ipa, sentence}`):

- [`TigreGotico/bifonia-pt-homographs`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs)
  — the synthetic corpus with stratified train/test splits, for training and synthetic evaluation.
- [`TigreGotico/bifonia-pt-homographs-wild`](https://huggingface.co/datasets/TigreGotico/bifonia-pt-homographs-wild)
  — real Wikipedia and web sentences forming an OOD test set, labels annotated by an LLM, licensed
  CC-BY-SA-4.0.

```python
from datasets import load_dataset
synthetic = load_dataset("TigreGotico/bifonia-pt-homographs")
wild = load_dataset("TigreGotico/bifonia-pt-homographs-wild")
```

Together they fill a gap: no other PT-PT benchmark targets heterophonic-homograph pronunciation
disambiguation at the sense level, and the wild set supplies the out-of-distribution measure that
synthetic splits cannot.
