# bifonia — usage guide

## Install

```bash
pip install -e /path/to/bifonia --no-deps
```

No dependencies — pure standard library.

## Quick start

```python
from bifonia import tokenize, is_ambiguous, guess_sense, disambiguate, add_extra_diacritics

words = tokenize("A sede de conhecimento move-nos.")
i = words.index("sede")
guess_sense(words, i)     # "thirst"
disambiguate(words, i)    # "ˈsedɨ"
```

```python
words = tokenize("Vou para casa depois do trabalho.")
for i, word in enumerate(words):
    if is_ambiguous(word):
        print(word, "→", disambiguate(words, i))
# para → ˈpɐɾɐ  (purpose / ADP reading)
```

## API

### `tokenize(text) → list[str]`
Lowercase and split text into word tokens, stripping punctuation.

### `is_ambiguous(word) → bool`
True if the word is a known heterophonic homograph.

### `guess_sense(words, idx, pos=None, postag=None) → str`
Primary entry point. Return the most likely **meaning slug** (the label to predict) for the
word at position *idx* based on its context — e.g. `"thirst"`, `"seat"`, `"stop"`. This uses
the **per-word ensemble**: the learned model serves words it is routed to and clears its margin
on, and the rule engine serves the rest (and is always the fallback), so a caller gets the best
available reading without choosing an engine. Pass `pos=` to force the rule resolver within that
POS. Pass `postag=` to supply an external POS tag (e.g. from spaCy or Stanza) as a
hybrid-ensemble hint that restricts the candidate senses.

### `guess_pos(words, idx) → str`
Return the **lexical class that selects the word's pronunciation** — `"ADP"`, `"NOUN"`,
`"VERB"`, `"ADJ"`. A thin wrapper over `guess_sense` that maps the resolved meaning back to
its lexical class, used only to feed `guess_sense`/`disambiguate`.

> **This is not a part-of-speech tagger and must not be used as one.** The tag is chosen to
> land on the correct IPA, and can deliberately disagree with the word's syntactic function.
> The clearest case is an **adjective used as a noun**: in *"aquela tola comprou um carro"*,
> `tola` is syntactically a NOUN (the subject), but keeps the *adjective* reading (`foolish`,
> closed ˈtolɐ) — so `guess_pos` returns `"ADJ"`. The roster even tags that sense `ADJ|NOUN`
> so an external NOUN tag defers to the rules rather than forcing the concrete `head` noun
> (open ˈtɔlɐ). For genuine POS tagging use a real tagger (spaCy, Stanza); for pronunciation
> use `guess_sense`/`disambiguate`.

### `disambiguate(words, idx, pos=None, sense=None, postag=None) → str`
Return the IPA transcription for the word at *idx*, selected by **meaning**. Pass `sense=` to
choose the reading directly (`disambiguate(words, i, sense="seat")`), `pos=` to fix the POS
before the sense is resolved within it, or `postag=` to pass an external POS-tag hint through to
the ensemble.

### `add_extra_diacritics(sentence) → str`
Return the sentence with non-canonical diacritics inserted on ambiguous words to
force correct downstream G2P output.

| Diacritic | IPA | Vowel quality |
|-----------|-----|---------------|
| `ó` / `é` | /ɔ/ / /ɛ/ | open vowel |
| `ô` / `ê` | /o/ / /e/ | closed vowel |

## Word coverage

IPA data is sourced from `bifonia/data/heterophonic_homographs.csv`, keyed on `(word, sense)`.

| Word | Senses (pos) |
|------|-------------|
| para | purpose (ADP) / stop (VERB) |
| pelo | by_the (ADP) / hair (NOUN) / peel (VERB) |
| tola | foolish (ADJ) / head (NOUN) |
| seco | dry (ADJ) / dry_vb (VERB) |
| acordo | agreement (NOUN) / wake (VERB) |
| acerto | settlement (NOUN) / adjust (VERB) |
| cerro | hill (NOUN) / shut (VERB) |
| choro | weeping (NOUN) / weep (VERB) |
| colher | spoon (NOUN) / harvest (VERB) |
| começo | beginning (NOUN) / begin (VERB) |
| conserto | repair (NOUN) / mend (VERB) |
| coro | choir (NOUN) / blush (VERB) |
| corte | court (NOUN) / cut (VERB) |
| forma | mould (NOUN) / shape (VERB) |
| gozo | enjoyment (NOUN) / enjoy (VERB) |
| gosto | taste (NOUN) / like (VERB) |
| jogo | game (NOUN) / play (VERB) |
| molho | sauce (NOUN) / bundle (VERB) |
| olho | eye (NOUN) / look (VERB) |
| rego | furrow (NOUN) / water (VERB) |
| sede | thirst (NOUN) / seat (NOUN) |
| sobre | about (ADP) / sail (NOUN) / leftover (VERB) |
| torre | tower (NOUN) / roast (VERB) |
| transtorno | disorder (NOUN) / upset (VERB) |
| peso | weight (NOUN) / weigh (VERB) |
| porto | harbour (NOUN) / carry (VERB) |
| posto | station (NOUN) / post (VERB) |

## Scoring heuristic

The disambiguator assigns an integer score to each applicable POS using signals derived from
the ±4-word context. The POS with the highest score wins; ties fall back to `DEFAULT_POS[word]`.
The winning POS is then **narrowed to a sense**: for most words each POS maps to exactly one
sense, so the lookup is direct; for `sede` (whose two senses are both nouns) `resolve_sense`
reads sense-specific cues (`bifonia/locale/pt-pt/sede_{seat,thirst}_cues.voc`) — e.g.
`sede de X` → thirst, `sede da empresa` → seat.

**ADP signals (score_adp)**
- Governing verb before (`falou sobre`, `discutir sobre`, …) → +6
- `AFTER_PREP` pronoun/noun after (`para mim`, `sobre ele`) → +5
- `NEVER_AFTER_PREP` token immediately after → strongly negative

**NOUN signals (score_noun)**
- DET or QUANT immediately before → +5
- `do` / `da` genitive contraction after → +3
- Passive auxiliary before `posto` (`foi posto`) → +8 (PPT of *pôr* = closed-o)

**VERB signals (score_verb)**
- PRON immediately before → +5 (+2 if sentence position 1)
- DET/QUANT directly after (direct object) → +3 (with exclusions for contracted preps)
- Negation/frequency adverb before (`não`, `nunca`, `sempre`) → +4
- `-mente` adverb directly after → +3
- Passive auxiliary before → +4
- Infinitive immediately before → −5 (nominal context); guarded against clause boundaries
- Temporal conjunction before (`quando`, `enquanto`) → +2
- Subjunctive conjunction before (`caso`, `embora`) → +4

**ADJ signals (score_adj)**
- Copular verb before (`é`, `está`, `ficou`, …) → +5
- Degree intensifier before (`muito`, `bastante`, `completamente`) → +2
- `-mente` adverb before → +4 (predicative: "particularmente seco")
- `DET NOUN ADJ` attributive pattern (DET at −2, content word at −1) → +4; suppressed
  across clause boundaries (comma/period on raw −1 token)
- Post-positive: bare content noun immediately before, no clause boundary, next is not
  a direct-object DET → +3

See `bifonia/scoring.py` for the complete signal table and per-word overrides.

## Punctuation handling

All context lookups strip trailing punctuation (`.,;:!?`) from neighbour tokens before
set membership tests. This prevents false negatives when a word appears immediately
before a comma or sentence boundary.

## Orthographic normalisation

The scorer operates on plain (AO1990) orthography. Diacritized input (`pára`, `pêlo`,
`côrte`, …) is handled directly: `guess_sense` reads the sense straight off the diacritic
(e.g. *séde* → seat, *sêde* → thirst, *pára* → stop) without context scoring. The
`bifonia._DIACRITIZED_TO_BASE` and `_DIACRITIZED_TO_SENSE` maps back this lookup.

## Engines, training, and benchmarks

`guess_sense` draws on two interchangeable engines (see `docs/methodology.md`): the corpus-free
rule engine and corpus-trained learned models (Naive-Bayes and an averaged perceptron). The
model artefacts are `bifonia/data/sense_model_{nb,perceptron}.json`.

Retrain the learned models from the labelled corpus:

```bash
python scripts/train.py --model both          # rebuilds both JSON artefacts
python scripts/train.py --model perceptron --min-count 3 --seed 1337
```

Measure sense-prediction accuracy two ways:

```bash
python scripts/benchmark_tagger.py            # synthetic held-out split (per-word breakdown)
python scripts/benchmark_ood.py               # out-of-distribution real-text set (downloads from Hugging Face)
```
