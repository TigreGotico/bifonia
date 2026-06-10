# Methodology: Dataset Construction and Disambiguation Algorithm

## Overview

The `bifonia` package disambiguates 27 Portuguese heterophonic homographs — words spelled
identically but pronounced differently depending on part of speech.  The work involves two
tightly coupled artefacts:

1. **A labelled corpus** of ~11 000 Portuguese sentences, one ambiguous word per sentence,
   with a UDEP POS label (NOUN / VERB / ADP / ADJ).
2. **A rule-based scorer** that predicts the POS label from the local ±4-word context.

The two improve together in an iterative loop: add hard sentences → find scorer weaknesses →
fix rules → add more hard sentences → repeat.

---

## Dataset Construction

### Seed sentences (human-written, ~500)

The corpus was seeded with approximately 500 sentences written by hand by a native speaker.
These cover the canonical disambiguation patterns and serve as the ground truth that all
subsequent work is validated against.  They are stored in `bifonia/data/grp_*.py`.

### LLM-assisted expansion with agentpipe (guided and reviewed)

After the human seed, the corpus was expanded using
[**agentpipe**](https://github.com/TigreGoticoLda/agentpipe) — an open-source Python
library that fans work out to multiple coding-agent providers in parallel.  Three
providers were used:

| Provider | Model | Cost |
|---|---|---|
| `opencode-free` | DeepSeek Coder | free |
| `antigravity-flash-low` | Gemini Flash | free |
| `claude` (Claude Code) | Claude Sonnet / Haiku | paid, monitor usage |

Each provider received the same prompt and returned independent candidate sentences;
results were deduplicated and staged for human review before being committed.  When
running `corpus_gen.py` with Claude-based providers, watch token consumption — the
free providers are suitable for bulk generation; Claude is best reserved for hard
patterns or quality review.

The generation process was:

1. A human provided the target word, the target POS, and a description of the hard patterns
   to cover (e.g. "passive-voice sentences for `posto` NOUN", "control-verb phrases for
   `colher` VERB", "route-noun context for `pelo` ADP").
2. `agentpipe` dispatched the prompt concurrently to all selected providers and merged the
   results.
3. All generated sentences were run through the scorer.  Misclassified sentences either
   revealed real scorer weaknesses (fix the rules) or were labelling errors (fix the label
   or discard).
4. Only sentences that the scorer handles correctly after any necessary rule improvement
   were committed to the corpus.

Using multiple independently prompted agents increases vocabulary variety — each model
has different priors on phrasing and lexical choice.  The generation script
(`corpus_gen.py`) is included in the repository and can be used to extend the corpus
further.

This human-guided, automatically-filtered pipeline means the final corpus is both large
enough for statistically meaningful evaluation and free from label noise caused by blind
bulk generation.

### Corpus structure

| File pattern | Contents |
|---|---|
| `bifonia/data/grp_a.py` … `grp_pps.py` | Original human-curated batches |
| `bifonia/data/extra_<word>.py` | Per-word extension packs (LLM-assisted) |

Each file is a Python module containing a JSON-serialisable dict
`{word: {POS: [sentence, …]}}`.  `bifonia/corpus.py` merges all sources into a single
`CORPUS` dict at import time.

**Statistics (as of latest build):**

| Metric | Value |
|---|---|
| Total sentences | ~13 570 |
| Words covered | 27 |
| Human-written seed sentences | ~500 |
| LLM-assisted (human-guided) | ~13 070 |
| Rule-based accuracy on full corpus | **98.46 %** |
| Hard three-way words (`para`/`pelo`/`sobre`) | 90–95 % each |

---

## Disambiguation Algorithm

The scorer (`bifonia/scoring.py`) is a context-based integer-scoring system.  For each
candidate POS, a function (`score_adp`, `score_noun`, `score_verb`, `score_adj`) produces
an integer score from signals in the ±4-word window.  The POS with the highest score wins;
ties break to `DEFAULT_POS[word]`.

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

### Iterative refinement methodology

Each round follows the pattern:

```
1. Identify the worst-performing words (lowest accuracy on current corpus).
2. Inspect misclassified sentences: what local signal is missing or firing incorrectly?
3. Formulate a rule hypothesis.
4. Implement and test.  Run full corpus accuracy + pytest.
5. If accuracy improves without regressions, commit the rule.
6. Add hard sentences that exercise the new/fixed rule.
7. Return to step 1.
```

Rules are kept conservative: they must fire on clear linguistic patterns, not statistical
quirks of the current training set.  Every added signal is motivated by a grammatical
argument (e.g. "contracted prepositions cannot introduce verbal direct objects").

---

## Benchmark Comparison

The rule-based scorer was compared against three external taggers using the same corpus
(taggers see the un-diacritised form, an equal-footing test):

| System | Accuracy |
|---|---|
| **bifonia rule-based** | **99.51 %** |
| Stanza (neural, `pt`) | 81.9 % |
| spaCy (`pt_core_news_lg`) | 66.5 % |
| TugaTagger (spaCy backend) | 66.5 % |
| TugaTagger (Brill backend) | 53.1 % |

The rule-based system dominates because it was designed specifically for this narrow task;
general-purpose neural taggers are not trained with heterophonic homograph disambiguation
as an objective.

Run the benchmark yourself:

```bash
python benchmark_tagger.py --tagger all
python benchmark_tagger.py --word para --errors
```

---

## Why This Approach Is Tractable for Portuguese

The rule-based method works — and works well — specifically because Portuguese has an
unusually small set of heterophonic homographs that matter for TTS.  This package covers
27 words.  That is not a limitation of the dataset; it is close to the full inventory of
the phenomenon in standard European Portuguese.

For each word, the disambiguation reduces to a few clear grammatical contrasts (NOUN vs VERB,
ADP vs VERB) that are reliably signalled by the ±4-word context: determiners, pronouns,
infinitive markers, passive auxiliaries, copular verbs.  The scorer does **not** attempt to
tag full sentences; it only resolves the POS of one pre-identified ambiguous token.  That is
a much easier problem than full POS tagging.

This approach does **not** generalise to other languages:

- Languages with large homograph inventories (e.g. English, where hundreds of words are
  heterophonic: *lead*, *wind*, *row*, *wound*, …) would require a general POS tagger, not
  a hand-crafted rule set of this size.
- Languages with free word order make the ±4-word window less reliable as a signal.
- Languages with rich morphology often resolve ambiguity through agreement suffixes that
  appear on the target word itself — no context scanning needed at all.

Portuguese is a fortunate special case: few words need disambiguation, and they are
disambiguated by strong, local grammatical cues.

---

## Orthographic Normalisation (AO1990)

Pre-AO1990 (pre-1990 Orthographic Agreement) text uses diacritics on words that are
unambiguous in modern Portuguese: *pára* (now *para*), *pêlo* (now *pelo*),
*côrte* (now *corte*), etc.  For the purpose of this package:

- Text that contains these forms is simply **pre-AO1990 text** and requires no special
  handling — the scorer will naturally see the pre-reform spelling and score it correctly
  because the context signals are the same.
- For **normalisation pipelines** that want to enforce AO1990, the correct approach is to
  replace unambiguous pre-reform forms with their post-reform equivalents before calling
  the scorer.  The `_DIACRITIZED_TO_BASE` mapping in `bifonia/__init__.py` covers the
  relevant substitutions.
- The scorer's `_DIACRITIZED_TO_BASE` map is intentionally **not** applied globally inside
  `guess_pos`; it is the caller's responsibility to normalise input if required.  This
  keeps the scorer stateless and transparent.

---

## Transparency and Limitations

- **LLM usage:** approximately 96 % of corpus sentences were generated by LLMs via
  [agentpipe](https://github.com/TigreGoticoLda/agentpipe) under close human supervision
  (providers: `opencode-free` / DeepSeek; `antigravity-flash-low` / Gemini Flash; and
  Claude Sonnet / Haiku via Claude Code for hard patterns and quality review).  No
  sentence was admitted to the corpus without passing the scorer (and any necessary rule
  fix).  The human seed (~500 sentences) defines the disambiguation gold standard; LLM
  sentences expand coverage of hard patterns.
- **European Portuguese only:** all phonology, wordlists, and orthographic conventions
  are specific to **European Portuguese (EP)**.  Brazilian Portuguese (BP) has different
  stress patterns, clitic placement rules, and some of these 27 words may not be
  heterophonic in BP at all.  A BP dialect round is planned as a future extension.
- **tugamorph integration (planned):** morphological analysis from
  [tugamorph](https://github.com/TigreGoticoLda/tugamorph) could improve disambiguation
  of `pelo NOUN` ("body hair") by recognising possession verbs (`ter`, `possuir`) that
  govern nominal `pelo`, and by identifying past participles robustly for the passive-agent
  `pelo ADP` pattern.  Currently handled by suffix heuristics.
- **Context window:** the scorer only inspects ±4 words.  Long-range dependencies (e.g.
  a subject noun phrase 5+ words before the verb) are outside the model's reach and
  represent an irreducible error source for a local rule-based system.
- **Sentence-level ambiguity:** a small number of sentences are genuinely ambiguous without
  full semantic interpretation (e.g. `para sempre` = VERB "stops always" vs ADP "forever").
  These are left as known limitations, not over-fitted with fragile rules.
- **Dialect:** the corpus reflects European Portuguese phonology and orthographic conventions.
  Brazilian Portuguese may differ in some disambiguation patterns (e.g. verbal clitic
  placement).
- **Dataset licence:** sentence content is original; no copyrighted text was used.  The
  corpus is intended for HuggingFace publication under a permissive licence (CC-BY or
  similar).
