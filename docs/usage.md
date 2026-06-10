# homogr — usage guide

## Install

```bash
pip install -e /path/to/homogr --no-deps
```

## Quick start

```python
from homogr import tokenize, is_ambiguous, disambiguate, add_extra_diacritics

words = tokenize("Vou para casa depois do trabalho.")
for i, word in enumerate(words):
    if is_ambiguous(word):
        print(word, "→", disambiguate(words, i))
# para → ˈpɐɾɐ  (ADP reading)
```

## API

### `tokenize(text) → list[str]`
Lowercase and split text into word tokens (strips punctuation).

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

| Diacritic | IPA | Reading |
|-----------|-----|---------|
| `ó` | /ɔ/ | open-o (verb) |
| `é` | /ɛ/ | open-e (verb) |
| `ô` | /o/ | closed-o (noun) |
| `ê` | /e/ | closed-e (noun) |

## Word coverage

IPA data is sourced from `tugalex/data/heterophonic_homographs.csv`.

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
| sobre | NOUN, VERB |
| torre | NOUN, VERB |
| transtorno | NOUN, VERB |
| peso | NOUN, VERB |
| porto | NOUN, VERB |
| posto | NOUN, VERB |

## Scoring heuristic

The disambiguator assigns a non-negative score to each applicable POS using
surrounding context tokens:

**ADP**: +5 if preceded by a motion/auxiliary verb; +5 if followed by a determiner,
pronoun, or time adverb; +5 if followed by a numeral; +2 if followed by an infinitive.

**NOUN**: +5 if preceded by a determiner; +3 if followed by a genitive preposition
(`de`, `do`, `da` …).

**VERB**: +5 if preceded by a pronoun; +3 if followed by a determiner (direct object);
+5 if the word itself is an infinitive form; -5 if two infinitives appear in sequence.

**ADJ**: +5 if preceded by a determiner or pronoun; +3 if followed by a conjunction or
genitive preposition.

The POS with the highest positive score wins.  Ties and zero-score cases fall back to
`DEFAULT_POS`.
