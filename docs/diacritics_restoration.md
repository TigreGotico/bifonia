# Diacritics Restoration as a Homograph Disambiguation Task

## The problem

Portuguese has 27 heterophonic homographs — words whose spelling is identical but whose
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

The acute and circumflex accents used here are **not** project-specific notation.
They are the standard Portuguese orthographic markers for vowel quality and are already
part of the language:

- **Acute (´)** marks an open vowel: *ó* = /ɔ/, *é* = /ɛ/
- **Circumflex (^)** marks a closed vowel: *ô* = /o/, *ê* = /e/

Every Portuguese G2P engine, phonemiser, and TTS system already handles these accents
correctly — they appear in thousands of unambiguous words (*ótimo*, *ônibus*, *êxito*,
*pé*, *pó*, …).  The disambiguation output is therefore **drop-in compatible** with any
existing Portuguese TTS pipeline: insert the diacritised form before G2P and the correct
vowel is produced without any changes to the downstream system.

The issue is only that AO1990 (the 1990 Orthographic Agreement) removed these accents
from a small set of words that were heterophonic — the very words this package covers.
Post-reform, *pára* (stops) became *para*, making it orthographically identical to the
preposition.  Restoring the diacritic on the verbal reading re-establishes the phonological
signal that the reform erased.

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

## Rule-based scorer baseline

`bifonia` ships a hand-crafted context scorer (`scoring.py`) that inspects a ±4-word
window using integer signals for determiners, pronouns, passive auxiliaries, copular
verbs, infinitive markers, degree adverbs, and governing verbs, then narrows the winning POS
to a `sense`.  Sense-prediction accuracy on the full 56 891-sentence corpus:

| Approach | Full-corpus accuracy |
|---|---|
| **rule-based (bifonia)** | **94.17%** |
| Stanza POS→sense | 75.47% |
| spaCy (`pt_core_news_lg`) POS→sense | 65.74% |
| most-common (majority sense per word) | 52.66% |

The POS taggers plateau because they cannot separate two senses that share a POS (e.g.
`sede`): they nail the dominant noun sense and miss the minority one by construction.

Run the comparison yourself:

```bash
python benchmark_tagger.py
python benchmark_tagger.py --word sede --errors
```

### When the scorer succeeds

The scorer inspects a **±4-word window**.  Representative signals:

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

These are precisely the cases where a sequence model should excel.

## Decision tree vs. scoring function

The current scorer is effectively a manual decision tree with soft (additive) edges.
A hard decision tree would be more transparent but more brittle — a single missing node
breaks the subtree, whereas the scorer degrades gracefully when signals conflict.

The scoring approach *validates* the machine-learning path:

- Each signal corresponds to a learnable feature (DET before, PRON before, CONJ_SUBJ before).
- A BiLSTM or transformer with character and word embeddings can learn these patterns
  automatically from the corpus, and generalise to long-range dependencies the rule
  system cannot reach.
- The scorer provides a strong interpretable baseline and an upper-bound estimate of
  what context-local signals can achieve.

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
| Rule-based sense accuracy | **94.17%** (full corpus) |
| Stanza POS→sense | 75.47% |
| spaCy (pt_core_news_lg) POS→sense | 65.74% |
| most-common (majority sense) | 52.66% |
| Domain coverage | science, medicine, engineering, biology, animals, objects, day-to-day, chit-chat, news, books/literature |

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

## Suggested model architectures

### BiLSTM (sequence labeller)

```
Input:  token sequence (lowercased, subword BPE or character CNN)
Output: per-token label ∈ {UNCHANGED, ACUTE, CIRCUMFLEX}
        (only applied at positions of known ambiguous words)
Loss:   cross-entropy, ignore non-ambiguous positions
```

Baseline expected to exceed 95 % accuracy with 1 000+ training sentences per word.

### BERTimbau fine-tune

BERTimbau (neuralmind/bert-base-portuguese-cased) provides rich contextual embeddings.
Fine-tune with a token classification head on the corpus records. The model has already
seen the ambiguous words in diverse contexts, so it should generalise well even on the
harder (long-range dependency) cases.

Input: the `sentence` field.  
Label: the position of the ambiguous word is tagged with its `sense` class.

### Sequence-to-sequence (diacritisation)

Cast the full task as character-level seq2seq:

```
Input:  "O autocarro para em frente ao hospital."
Target: "O autocarro pára em frente ao hospital."
```

A small encoder-decoder (T5-small or NLLB variant) fine-tuned on the corpus can learn
to insert diacritics end-to-end without an explicit sense intermediate step.

## Evaluation

Run the test suite as a sanity check:

```bash
python -m pytest tests/test_disambiguate.py -v
```

The shipped `hf/train.jsonl` / `hf/test.jsonl` split is shuffled and stratified per
`(word, sense)` with a fixed seed, so train and test are i.i.d. — use it directly, or build
stratified k-folds over the corpus records by `(word, sense)` for cross-validation.

Key metrics: sense accuracy, precision/recall per `sense`, and the per-bucket breakdown that
isolates the minority shared-POS senses (`sede`, `forma`, `molho`) — the hardest cases.

## Potential Hugging Face dataset

The `sense` label and per-reading `ipa` make this dataset directly publishable as a
heterophonic-homograph disambiguation benchmark:

```python
from datasets import load_dataset
ds = load_dataset("json", data_files={"train": "hf/train.jsonl", "test": "hf/test.jsonl"})
```

It fills a gap: no existing PT-PT benchmark targets heterophonic-homograph pronunciation
disambiguation at the sense level. The corpus is entirely synthetic but covers realistic
domain diversity.
