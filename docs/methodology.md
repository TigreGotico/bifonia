# Methodology: Dataset Construction and Disambiguation Algorithm

## Overview

The `bifonia` package disambiguates 27 Portuguese heterophonic homographs — words spelled
identically but pronounced differently depending on their **meaning** (`sense`).  The package
comprises two tightly coupled artefacts:

1. **A labelled corpus** of 56 891 Portuguese sentences, one ambiguous word per sentence,
   labelled with a meaning slug (`sense`) and a descriptive `pos` attribute.
2. **A rule-based scorer** that predicts the `sense` from the local ±4-word context.

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
| Rule-based sense accuracy on full corpus | **94.17 %** |

---

## Disambiguation Algorithm

The scorer (`bifonia/scoring.py`) is a context-based integer-scoring system.  For each
candidate POS, a function (`score_adp`, `score_noun`, `score_verb`, `score_adj`) produces
an integer score from signals in the ±4-word window.  The POS with the highest score wins;
ties break to `DEFAULT_POS[word]`.

The scorer then **narrows the winning POS to a `sense`** (`resolve_sense`).  For 26 of the 27
words each POS maps to exactly one sense, so this is a direct lookup.  For `sede`, whose two
senses (*thirst*, *seat/HQ*) are both nouns, a meaning resolver (`_resolve_sede`) reads
sense-specific cues from the local context: the preposition frame is the strongest signal
(`sede de X` → thirst, `sede da/do X` → seat), reinforced by the
`bifonia/locale/pt-pt/sede_{seat,thirst}_cues.voc` wordlists in the ±3 window.  `guess_sense`
returns the meaning slug; `guess_pos` maps it back to its descriptive POS; `disambiguate`
selects the IPA for the resolved `(word, sense)`.

### Signal types

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
prepositions cannot introduce verbal direct objects").  `benchmark_tagger.py` measures
per-word and per-sense accuracy so a new signal can be validated against the full corpus and
the test suite before it is kept.

---

## Benchmark Comparison

`benchmark_tagger.py` runs a four-way comparison of sense-prediction accuracy on the full
56 891-sentence corpus (train 45 492 / test 11 400).  Each tagger sees the plain
(un-diacritised) form; the POS taggers map their POS output back to a sense.

| Approach | Train | Test | Full |
|---|---|---|---|
| most-common (majority sense per word) | 52.66 % | 52.64 % | 52.66 % |
| spaCy (`pt_core_news_lg`) POS→sense | 65.64 % | 66.11 % | 65.74 % |
| Stanza POS→sense | 75.44 % | 75.62 % | 75.47 % |
| **rule-based (bifonia)** | **94.08 %** | **94.50 %** | **94.17 %** |

The key insight: a POS tagger hits a **structural ceiling** on senses that share a POS.  It
gets the majority noun sense right but the minority sense wrong *by construction*, because POS
carries no information that separates them.  Per-bucket accuracy on the full corpus makes this
concrete:

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
twin; the rule-based scorer, by reading meaning cues, recovers the minority sense.

Run the benchmark yourself:

```bash
python benchmark_tagger.py
python benchmark_tagger.py --word sede --errors
```

---

## Why This Approach Is Tractable for Portuguese

The rule-based method works — and works well — specifically because Portuguese has an
unusually small set of heterophonic homographs that matter for TTS.  This package covers
27 words.  That is not a limitation of the dataset; it is close to the full inventory of
the phenomenon in standard European Portuguese.

For most words the disambiguation reduces to a few clear grammatical contrasts (NOUN vs VERB,
ADP vs VERB) reliably signalled by the ±4-word context: determiners, pronouns, infinitive
markers, passive auxiliaries, copular verbs.  Where two senses share a POS (`sede`), a
meaning resolver reads sense-specific cues from the same window.  The scorer does **not**
attempt to tag full sentences; it only resolves the meaning of one pre-identified ambiguous
token.  That is a much easier problem than full POS tagging.

This approach does **not** generalise to other languages:

- Languages with large homograph inventories (e.g. English, where hundreds of words are
  heterophonic: *lead*, *wind*, *row*, *wound*, …) would require a general tagger, not
  a hand-crafted rule set of this size.
- Languages with free word order make the ±4-word window less reliable as a signal.
- Languages with rich morphology often resolve ambiguity through agreement suffixes that
  appear on the target word itself — no context scanning needed at all.

Portuguese is a fortunate special case: few words need disambiguation, and they are
disambiguated by strong, local grammatical cues.

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
