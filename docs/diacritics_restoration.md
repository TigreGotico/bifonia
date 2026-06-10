# Diacritics Restoration as a Homograph Disambiguation Task

## The problem

Portuguese has 27 heterophonic homographs — words whose spelling is identical but whose
pronunciation depends on part of speech. A rule-based G2P engine that does not perform
POS disambiguation will systematically mispronounce them.

The classic example:

| Written | Intended | POS | IPA |
|---------|----------|-----|-----|
| para | destination (*goes to*) | ADP | `ˈpɐɾɐ` |
| para | stops (*halts*) | VERB | `ˈpaɾɐ` |

The vowel difference is the AO1990-prohibited acute accent: the canonical written form
`para` is ambiguous. A G2P that sees `para` cannot know which IPA to emit.

## Framing as diacritics restoration

A natural formulation: insert a **non-standard diacritic** that encodes the correct vowel
quality, then let a phonemiser that honours those diacritics emit the right IPA.

| Canonical | Restored | POS | Vowel |
|-----------|----------|-----|-------|
| para | para (unchanged) | ADP | closed ɐ |
| para | **pára** | VERB | open a |
| pelo | **pêlo** | NOUN | closed e (fur) |
| pelo | pelo (unchanged) | ADP | — |
| sobre | **sôbre** | NOUN | closed o (envelope) |
| sobre | **sóbre** | VERB | open ɔ (to be left over) |

This maps disambiguation onto a **token-level sequence labelling** problem:

```
Input:  O autocarro para em frente ao hospital.
Output: O autocarro pára em frente ao hospital.
```

The model need only decide, for each occurrence of an ambiguous token, whether to insert
a diacritic and which one. All other tokens pass through unchanged.

## Convention

| Diacritic | Vowel quality | Typical POS |
|-----------|--------------|-------------|
| acute (ó/é) | open /ɔ/ or /ɛ/ | VERB |
| circumflex (ô/ê) | closed /o/ or /e/ | NOUN |

The exceptions (sede, colher, tola) follow the same acute/circumflex convention but
the NOUN/VERB polarity is reversed for phonological reasons documented in `data.py`.

## Rule-based scorer baseline

`homogr` ships a hand-crafted context scorer (`scoring.py`) that inspects a ±2-word
window and applies integer signals for determiners, pronouns, conjunctions, and
preposition-governing nouns.  It achieves **97%+ accuracy** on the 11 000-sentence
corpus vs **53%** for a generic Portuguese POS tagger (TugaTagger, which defaults
to a single class per word).

Run the comparison yourself:

```bash
python benchmark_tagger.py
python benchmark_tagger.py --tagger stanza  # if stanza/pt model available
```

### When the scorer succeeds

- DET or QUANT immediately before → NOUN (+5)
- PRON immediately before → VERB (+5)
- Subjunctive conjunction (caso/embora/conquanto) before → VERB (+4)
- Negation adverb (não/nunca/jamais) before → VERB (+4)
- Passive auxiliary (foi/foram/fosse) before → VERB (+4)
- SOBRE_GOV governing noun before → ADP (+6)
- Intensifier (muito/pouco/bastante) before "sobre" → ADP (+5)
- Infinitive following → ADP (−5 on VERB)

### Where it struggles (xfail catalogue)

The scorer fails on 15 sentences in the current test suite. Common failure modes:

1. **No context**: sentence-initial or between-clause position where no signal fires.
2. **DET NOUN VERB DET NOUN**: ADP AFTER_PREP (+5) beats VERB prev2-DET (+3).
   "pára a medicação" and "para a casa" are locally indistinguishable — semantic
   comprehension (animate agent? destination noun?) is needed.
3. **Long-range dependency**: governing verb is three or more tokens away
   (e.g. "A nebulosa forma novas estrelas" where subject is 3 tokens back).

### The `forma` ambiguity problem

`forma` is exceptional: it has three semantic readings sharing just two IPA forms.

| Semantic reading | IPA | CSV label |
|---|---|---|
| manner, way ("desta forma") | ˈfɔɾmɐ | VERB |
| to form/shape (verb, 3rd pers.) | ˈfɔɾmɐ | VERB |
| baking mold / pan ("fôrma de bolo") | ˈfoɾmɐ | NOUN |

The IPA distinction is only between mold (closed-o, NOUN) and everything else
(open-o, VERB). Disambiguating "baking mold" from "manner/way" requires semantic
comprehension of the noun phrase — local context signals cannot do it reliably.
The compound "pão-de-forma" is worth hard-coding as a fixed phrase → NOUN.

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

The labeled corpus lives in `homogr/corpus.py` and is exported by `dataset.py`:

```
python dataset.py --out data/
```

### Statistics (current)

| Metric | Value |
|--------|-------|
| Total sentences | ~11 000 |
| Unique ambiguous words | 27 |
| POS classes | 4 (NOUN, VERB, ADP, ADJ) |
| Sentences per word | ~180–420 (varies by word) |
| Rule-based scorer accuracy | 97.2% overall |
| Generic tagger (TugaTagger) | 53.1% — barely above majority class |
| Domain coverage | science, medicine, engineering, biology, animals, objects, day-to-day, chit-chat, news, books/literature |

### CSV schema

| Column | Description |
|--------|-------------|
| `word` | Base (canonical) ambiguous word |
| `pos` | UDEP POS tag |
| `ipa` | IPA transcription for this POS reading |
| `diacritized` | Non-standard diacritized form of the word alone |
| `sentence` | Source sentence (may already contain diacritized form) |
| `diacritized_sentence` | Sentence with the target word replaced by its diacritized form |

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

Input: the `sentence` column.  
Label: the position of the ambiguous word is tagged with its `pos` class.

### Sequence-to-sequence (diacritisation)

Cast the full task as character-level seq2seq:

```
Input:  "O autocarro para em frente ao hospital."
Target: "O autocarro pára em frente ao hospital."
```

A small encoder-decoder (T5-small or NLLB variant) fine-tuned on the corpus can learn
to insert diacritics end-to-end without an explicit POS intermediate step.

## Evaluation

Run the test suite as a sanity check:

```bash
python -m pytest tests/test_disambiguate.py -v
```

For the ML models, use stratified 5-fold cross-validation over the corpus records,
stratified by `(word, pos)` to ensure each fold has balanced examples.

Key metrics: accuracy, precision/recall per POS class, and error analysis on the 14
currently xfailed rule-based patterns (the hard cases).

## Potential Hugging Face dataset

The `diacritized_sentence` column makes this dataset directly publishable as a
diacritics-restoration benchmark:

```python
from datasets import load_dataset
ds = load_dataset("csv", data_files="dataset.csv")
```

It fills a gap: no existing PT-PT benchmark targets heterophonic homograph pronunciation
disambiguation at the grapheme level. The corpus is entirely synthetic but covers
realistic domain diversity.
