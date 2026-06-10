# bifonia — usage guide

## Install

```bash
pip install -e /path/to/bifonia --no-deps
```

## Quick start

```python
from bifonia import tokenize, is_ambiguous, disambiguate, add_extra_diacritics

words = tokenize("Vou para casa depois do trabalho.")
for i, word in enumerate(words):
    if is_ambiguous(word):
        print(word, "→", disambiguate(words, i))
# para → ˈpɐɾɐ  (ADP reading)
```

## API

### `tokenize(text) → list[str]`
Lowercase and split text into word tokens, stripping punctuation.

### `is_ambiguous(word) → bool`
True if the word is a known heterophonic homograph.

### `guess_pos(words, idx) → str`
Return the most likely UDEP POS tag (`"ADP"`, `"NOUN"`, `"VERB"`, `"ADJ"`) for the
word at position *idx* based on its context.

### `disambiguate(words, idx, pos=None) → str`
Return the IPA transcription for the word at *idx*.  Pass `pos=` to override the
automatic POS guess.

### `add_extra_diacritics(sentence) → str`
Return the sentence with non-canonical diacritics inserted on ambiguous words to
force correct downstream G2P output.

| Diacritic | IPA | Typical reading |
|-----------|-----|-----------------|
| `ó` / `é` | /ɔ/ / /ɛ/ | open vowel (often VERB) |
| `ô` / `ê` | /o/ / /e/ | closed vowel (often NOUN) |

## Word coverage

IPA data is sourced from `bifonia/data/heterophonic_homographs.csv`.

| Word | POS readings |
|------|-------------|
| para | ADP, VERB |
| pelo | ADP, NOUN, VERB |
| tola | NOUN, ADJ |
| seco | ADJ, VERB |
| acordo | NOUN, VERB |
| acerto | NOUN, VERB |
| cerro | NOUN, VERB |
| choro | NOUN, VERB |
| colher | NOUN, VERB |
| começo | NOUN, VERB |
| conserto | NOUN, VERB |
| coro | NOUN, VERB |
| corte | NOUN, VERB |
| forma | NOUN, VERB |
| gozo | NOUN, VERB |
| gosto | NOUN, VERB |
| jogo | NOUN, VERB |
| molho | NOUN, VERB |
| olho | NOUN, VERB |
| rego | NOUN, VERB |
| sede | NOUN, VERB |
| sobre | ADP, NOUN, VERB |
| torre | NOUN, VERB |
| transtorno | NOUN, VERB |
| peso | NOUN, VERB |
| porto | NOUN, VERB |
| posto | NOUN, VERB |

## Scoring heuristic

The disambiguator assigns an integer score to each applicable POS using signals
derived from the ±4-word context.  The POS with the highest score wins; ties fall
back to `DEFAULT_POS[word]`.

**ADP signals (score_adp)**
- Governing verb before (`falou sobre`, `discutir sobre`, …) → +6
- `AFTER_PREP` pronoun/noun after (`para mim`, `sobre ele`) → +5
- Sentence-initial position → −4 (rare for bare preposition)
- `NEVER_AFTER_PREP` token immediately after → strongly negative

**NOUN signals (score_noun)**
- DET or QUANT immediately before → +5
- `de` / genitive prep after → +3
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
set membership tests.  This prevents false negatives when a word appears immediately
before a comma or sentence boundary.

## Orthographic normalisation

The scorer operates on plain (post-AO1990) orthography.  Pre-reform text with
diacritics (*pára*, *pêlo*, *côrte*, …) should be normalised before calling `guess_pos`.
`bifonia._DIACRITIZED_TO_BASE` provides the mapping:

```python
from bifonia import _DIACRITIZED_TO_BASE

def normalise(sentence: str) -> str:
    s = sentence.lower()
    for d, b in _DIACRITIZED_TO_BASE.items():
        s = s.replace(d, b)
    return s
```
